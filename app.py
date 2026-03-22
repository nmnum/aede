"""
AEDE Streamlit Dashboard
=========================
Run with: streamlit run app.py
"""

import sys
import os
import json
import time
import threading
import numpy as np
from dataclasses import asdict
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go

from aede import run_campaign, ProblemSpec, Objective, BudgetSpec, StoppingCriterion
from aede.campaign import _update_pareto_front

st.set_page_config(
    page_title="AEDE — Adaptive Experimental Decision Engine",
    page_icon="⚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .finding-box {
        background: #1e2d4a;
        border-left: 4px solid #4a90d9;
        padding: 0.75rem 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.4rem 0;
        font-size: 0.9rem;
        color: #e8edf5 !important;
    }
    .warning-box {
        background: #3a2a0a;
        border-left: 4px solid #e6a817;
        padding: 0.75rem 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.4rem 0;
        font-size: 0.9rem;
        color: #f5e6c0 !important;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f3a6e;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #c5d0e8;
    }
</style>
""", unsafe_allow_html=True)

# ── Persistent store ───────────────────────────────────────────────────────────
# Stored in session_state so it survives reruns.
# Background thread writes ONLY to _store (plain dict), never session_state directly.
if "store" not in st.session_state:
    st.session_state.store = {
        "live_hv": [],
        "live_iter": 0,
        "live_weights": [],
        "stop_triggered": False,
        "running": False,
        "results": None,
        "logger": None,
        "campaign_hash": None,
        "upload_X": None,
        "upload_Y": None,
        "upload_feasible": None,
        "upload_n_dims": None,
    }

_store = st.session_state.store  # local alias — same object, survives reruns

# ── Constants ──────────────────────────────────────────────────────────────────
ACCENT = "#1f3a6e"
STRATEGY_COLORS = {"bo": "#1f77b4", "nsga": "#e67e22", "diversity": "#27ae60"}

# ── Figure helpers ─────────────────────────────────────────────────────────────
def make_pareto_fig(Y_all, pareto_front, obj_names):
    fig = go.Figure()
    if len(Y_all) > 0:
        fig.add_trace(go.Scatter(
            x=Y_all[:, 0], y=Y_all[:, 1], mode="markers",
            marker=dict(color="#aab4c8", size=6, opacity=0.5),
            name="All observations",
        ))
    if len(pareto_front) > 0:
        pf = pareto_front[np.argsort(pareto_front[:, 0])]
        fig.add_trace(go.Scatter(
            x=pf[:, 0], y=pf[:, 1], mode="markers+lines",
            marker=dict(color=ACCENT, size=10, symbol="diamond"),
            line=dict(color=ACCENT, width=1.5, dash="dot"),
            name="Pareto front",
        ))
    fig.update_layout(
        xaxis_title=obj_names[0], yaxis_title=obj_names[1],
        height=360, margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e", font=dict(color="#e0e0e0"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2a2a4a")
    fig.update_yaxes(showgrid=True, gridcolor="#2a2a4a")
    return fig


def make_hv_fig(hv_history):
    if not hv_history:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(hv_history))), y=hv_history,
        mode="lines+markers",
        line=dict(color=ACCENT, width=2), marker=dict(size=6),
        fill="tozeroy", fillcolor="rgba(31,58,110,0.08)",
        name="Hypervolume",
    ))
    fig.update_layout(
        xaxis_title="Iteration", yaxis_title="Hypervolume",
        height=240, margin=dict(l=40, r=20, t=20, b=40),
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e", font=dict(color="#e0e0e0"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2a2a4a")
    fig.update_yaxes(showgrid=True, gridcolor="#2a2a4a")
    return fig


def make_eig_fig(hv_data, threshold):
    if not hv_data:
        return go.Figure()
    eig_vals = [h.get("eig_per_cost", 0.0) for h in hv_data]
    iters = [h.get("iteration", i) for i, h in enumerate(hv_data)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=eig_vals, mode="lines+markers",
        line=dict(color="#e67e22", width=2), marker=dict(size=5),
        name="EIG/cost",
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#e74c3c",
                  annotation_text=f"Threshold ({threshold})",
                  annotation_position="top right")
    fig.update_layout(
        xaxis_title="Iteration", yaxis_title="EIG per cost",
        height=240, margin=dict(l=40, r=20, t=20, b=40),
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e", font=dict(color="#e0e0e0"),
    )
    return fig


def make_weights_fig(weight_history):
    if not weight_history:
        return go.Figure()
    iters = [w.get("iteration", i) for i, w in enumerate(weight_history)]
    strategies = [k for k in weight_history[0] if k != "iteration"]
    fig = go.Figure()
    for s in strategies:
        vals = [w.get(s, 0.0) for w in weight_history]
        fig.add_trace(go.Scatter(
            x=iters, y=vals, mode="lines", stackgroup="one", name=s,
            line=dict(color=STRATEGY_COLORS.get(s, "#888"), width=1),
            fillcolor=STRATEGY_COLORS.get(s, "#888"),
        ))
    fig.update_layout(
        xaxis_title="Iteration", yaxis_title="Strategy weight",
        yaxis_range=[0, 1], height=220,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e", font=dict(color="#e0e0e0"),
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗ AEDE")
    st.caption("Adaptive Experimental Decision Engine")
    st.divider()

    st.markdown("### Benchmark")
    benchmark = st.selectbox(
        "Landscape",
        ["vlmop2", "molecular", "high_constraint"],
        format_func=lambda x: {
            "vlmop2": "VLMOP2 (bi-objective)",
            "molecular": "Molecular Proxy",
            "high_constraint": "High Constraint (tri-obj)",
        }[x],
        label_visibility="collapsed",
    )
    n_dims = st.slider("Input dimensions", 2, 10, 2 if benchmark == "vlmop2" else 5)

    st.markdown("### Objectives")
    n_obj = 3 if benchmark == "high_constraint" else 2
    defaults_names = (
        [("binding_affinity", "maximize"), ("safety_score", "maximize"), ("metabolic_stability", "maximize")]
        if benchmark != "vlmop2"
        else [("objective_1", "maximize"), ("objective_2", "maximize")]
    )
    obj_names, obj_dirs = [], []
    for i in range(n_obj):
        c1, c2 = st.columns([3, 2])
        obj_names.append(c1.text_input(f"Obj {i+1}", defaults_names[i][0], key=f"oname_{i}"))
        obj_dirs.append(c2.selectbox("Dir", ["maximize", "minimize"],
                                     index=0 if defaults_names[i][1] == "maximize" else 1,
                                     key=f"odir_{i}", label_visibility="collapsed"))

    st.markdown("### Budget")
    n_exp = st.slider("Total experiments", 20, 300, 80)
    batch_size = st.slider("Per iteration", 2, 20, 5)
    pool_size = st.slider("Candidate pool", 20, 200, 60)

    st.markdown("### Stopping")
    eig_thresh = st.number_input("EIG/cost threshold", min_value=0.0001, max_value=0.1,
                                  value=0.005, step=0.001, format="%.4f")
    consec = st.slider("Consecutive iters below threshold", 2, 5, 3)
    min_exp = st.slider("Min experiments", 5, 50, 15)

    st.markdown("### Surrogate")
    n_mc = st.slider("MC samples", 16, 256, 64)
    seed = st.number_input("Random seed", value=42, step=1)

    st.divider()
    if _store["results"]:
        _r = _store["results"]
        _hv = _r["hv_history"][-1] if _r["hv_history"] else 0
        st.metric("Final HV", f"{_hv:.4f}")
        st.metric("Experiments", _r["n_experiments"])
        st.metric("Pareto front size", len(_r["pareto_front"]))
        if _store["stop_triggered"]:
            st.success("✓ Stopped early")


# ── Main page ──────────────────────────────────────────────────────────────────
st.title("⚗ AEDE — Adaptive Experimental Decision Engine")
st.caption("Configure in the sidebar, then launch. Results appear below as the campaign runs.")
st.divider()

# ── Custom data upload ─────────────────────────────────────────────────────────
with st.expander("📂 Upload your own experimental data (optional)", expanded=False):
    st.markdown("""
**Use this if you already have real experimental data** and want AEDE to fit its surrogate
to your results and recommend the next experiments to run.

**Required CSV format:**
- Columns `x_0, x_1, ... x_N` — your input variables (one per dimension)
- Columns matching your objective names exactly (e.g. `yield`, `cost`)
- Optional column `feasible` — 1 if experiment succeeded, 0 if it violated constraints

**Example (2 inputs, 2 objectives):**
```
x_0,x_1,yield,selectivity,feasible
0.12,0.55,0.81,0.63,1
0.43,0.21,0.74,0.71,1
0.88,0.90,0.51,0.82,0
```
When data is uploaded, AEDE skips random initialisation and warm-starts its surrogate from your measurements.
""")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="data_upload")
    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(upload_df)} rows × {len(upload_df.columns)} columns")
            st.dataframe(upload_df.head(5), use_container_width=True, hide_index=True)
            x_cols = [c for c in upload_df.columns if c.startswith("x_")]
            obj_cols = [c for c in upload_df.columns if c in obj_names]
            feasible_col = "feasible" if "feasible" in upload_df.columns else None
            if not x_cols:
                st.warning("No input columns found — name them x_0, x_1, x_2, …")
            elif not obj_cols:
                st.warning(f"No objective columns matched. Your objective names are: {obj_names}")
            else:
                st.info(f"Detected **{len(x_cols)} inputs** ({', '.join(x_cols)}), "
                        f"**{len(obj_cols)} objectives** ({', '.join(obj_cols)})"
                        + (" + feasibility column ✓" if feasible_col else ""))
                _store["upload_X"] = upload_df[x_cols].values.astype(float)
                _store["upload_Y"] = upload_df[obj_cols].values.astype(float)
                _store["upload_feasible"] = (
                    upload_df[feasible_col].values.astype(int)
                    if feasible_col else np.ones(len(upload_df), dtype=int)
                )
                _store["upload_n_dims"] = len(x_cols)
                st.success("Data ready — click Launch Campaign to warm-start from your data.")
        except Exception as e:
            st.error(f"Could not parse file: {e}")

st.divider()

# ── Launch button ──────────────────────────────────────────────────────────────
col_btn, col_status = st.columns([1, 4])
with col_btn:
    go_btn = st.button(
        "▶ Launch Campaign", type="primary",
        disabled=_store["running"], use_container_width=True,
    )
with col_status:
    status_box = st.empty()

if _store["running"]:
    if _store["live_hv"]:
        status_box.info(f"⏳ Running — iteration **{_store['live_iter']}**, HV = **{_store['live_hv'][-1]:.4f}**")
    else:
        status_box.info("⏳ Campaign starting…")
elif _store["results"]:
    status_box.success("✅ Campaign complete — results below.")

# ── Launch handler ─────────────────────────────────────────────────────────────
if go_btn and not _store["running"]:
    objectives = [Objective(name=n, direction=d) for n, d in zip(obj_names, obj_dirs)]
    spec = ProblemSpec(
        name=f"{benchmark}_{seed}",
        objectives=objectives,
        budget=BudgetSpec(experiments=n_exp, per_iteration_cap=batch_size),
        stopping_criterion=StoppingCriterion(
            eig_per_cost_threshold=eig_thresh,
            consecutive_iterations=consec,
            min_experiments=min_exp,
        ),
        benchmark=benchmark,
        n_input_dims=n_dims,
        candidate_pool_size=pool_size,
        random_seed=int(seed),
    )

    _store["live_hv"] = []
    _store["live_iter"] = 0
    _store["live_weights"] = []
    _store["stop_triggered"] = False
    _store["results"] = None
    _store["logger"] = None
    _store["campaign_hash"] = None
    _store["running"] = True

    db = "/tmp/aede_campaign.db"
    if os.path.exists(db):
        os.remove(db)

    _n_exp = n_exp
    _batch_size = batch_size
    _eig_thresh = eig_thresh
    _upload_X = _store.get("upload_X")
    _upload_Y = _store.get("upload_Y")
    _upload_feasible = _store.get("upload_feasible")

    def _run():
        """Runs in background thread. ONLY writes to _store — never session_state."""
        def on_iter(iteration, manifest, hv):
            _store["live_hv"].append(hv)
            _store["live_iter"] = iteration
            weights = manifest.strategy_weights.copy()
            weights["iteration"] = iteration
            _store["live_weights"].append(weights)
            if manifest.should_stop:
                _store["stop_triggered"] = True

        try:
            campaign_results = run_campaign(
                spec, db_path=db, verbose=False, n_mc=n_mc, on_iteration=on_iter,
                seed_X=_upload_X, seed_Y=_upload_Y, seed_feasible=_upload_feasible,
            )
            _store["results"] = campaign_results
            _store["logger"] = campaign_results["logger"]
            _store["campaign_hash"] = campaign_results["campaign_hash"]
        finally:
            _store["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    st.rerun()

# ── Live progress ──────────────────────────────────────────────────────────────
if _store["running"]:
    _live_hv = _store["live_hv"]
    _live_iter = _store["live_iter"]
    if _live_hv:
        estimated_total = max(n_exp // batch_size, 1)
        st.progress(
            min(_live_iter / estimated_total, 1.0),
            text=f"Iteration {_live_iter} of ~{estimated_total} — HV = {_live_hv[-1]:.4f}"
        )
        st.markdown('<div class="section-header">Hypervolume (live)</div>', unsafe_allow_html=True)
        st.plotly_chart(make_hv_fig(_live_hv), use_container_width=True)
    else:
        st.info("Initialising surrogate and running first batch…")
    time.sleep(0.4)
    st.rerun()

# ── Results ────────────────────────────────────────────────────────────────────
campaign_results = _store["results"]
if campaign_results is None:
    st.stop()

hv_hist = campaign_results["hv_history"]
pareto = campaign_results["pareto_front"]
Y_all = campaign_results["Y_all"]
n_obj_res = pareto.shape[1] if len(pareto) > 0 else 2
logger = _store["logger"]
campaign_hash = _store["campaign_hash"]

disp_obj_names = (
    obj_names[:n_obj_res]
    if len(obj_names) >= n_obj_res
    else [f"Objective {i+1}" for i in range(n_obj_res)]
)

if _store["stop_triggered"]:
    st.success("✓ Campaign converged early — stopping criterion met before budget exhaustion.")

st.divider()
m1, m2, m3, m4 = st.columns(4)
final_hv = hv_hist[-1] if hv_hist else 0
m1.metric("Final HV", f"{final_hv:.4f}",
          delta=f"+{final_hv - hv_hist[0]:.4f}" if len(hv_hist) > 1 else None)
m2.metric("Iterations", len(hv_hist))
m3.metric("Experiments used", campaign_results["n_experiments"])
m4.metric("Pareto front size", len(pareto))

st.divider()
st.markdown('<div class="section-header">📈 Results</div>', unsafe_allow_html=True)
col_pf, col_hv = st.columns([1.3, 1])

with col_pf:
    st.markdown("**Pareto Front**")
    if n_obj_res == 2:
        st.plotly_chart(make_pareto_fig(Y_all, pareto, disp_obj_names[:2]), use_container_width=True)
    else:
        fig_3d = go.Figure(data=[
            go.Scatter3d(
                x=Y_all[:, 0], y=Y_all[:, 1], z=Y_all[:, 2],
                mode="markers", marker=dict(size=4, color="#aab4c8", opacity=0.4), name="All",
            ),
            go.Scatter3d(
                x=pareto[:, 0], y=pareto[:, 1], z=pareto[:, 2],
                mode="markers", marker=dict(size=8, color=ACCENT, symbol="diamond"),
                name="Pareto front",
            ) if len(pareto) > 0 else go.Scatter3d(x=[], y=[], z=[]),
        ])
        fig_3d.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_3d, use_container_width=True)

with col_hv:
    st.markdown("**Hypervolume over iterations**")
    st.plotly_chart(make_hv_fig(hv_hist), use_container_width=True)

    if logger:
        hv_data = logger.get_hv_history(campaign_hash or "")
        st.markdown("**EIG/cost vs stopping threshold**")
        st.plotly_chart(make_eig_fig(hv_data, eig_thresh), use_container_width=True)

weight_hist = _store["live_weights"]
if not weight_hist and logger:
    weight_hist = logger.get_strategy_weight_history(campaign_hash or "")
if weight_hist:
    st.markdown("**Portfolio strategy weights over time**")
    st.plotly_chart(make_weights_fig(weight_hist), use_container_width=True)

if logger and campaign_hash:
    decisions = logger.get_recent_decisions(campaign_hash, n=15)
    if decisions:
        st.markdown("**Recent selected experiments**")
        df = pd.DataFrame(decisions)
        display_cols = ["iteration", "candidate_id", "provenance", "decision_score",
                        "expected_pareto_improvement", "feasibility_probability",
                        "experiment_cost", "rank"]
        df_disp = df[[c for c in display_cols if c in df.columns]].copy()
        for col in ["decision_score", "expected_pareto_improvement"]:
            if col in df_disp:
                df_disp[col] = df_disp[col].round(4)
        for col in ["feasibility_probability", "experiment_cost"]:
            if col in df_disp:
                df_disp[col] = df_disp[col].round(3)
        st.dataframe(df_disp, use_container_width=True, hide_index=True)

# ── Insight Report ─────────────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-header">🔍 Insight Report</div>', unsafe_allow_html=True)

insight = campaign_results["insight_report"]
cost_eff = insight.cost_efficiency

for finding in insight.key_findings:
    box_class = "warning-box" if "WARNING" in finding else "finding-box"
    st.markdown(f'<div class="{box_class}">{finding}</div>', unsafe_allow_html=True)

st.divider()
col1, col2, col3, col4 = st.columns(4)
col1.metric("HV yield / experiment", f"{cost_eff.hv_per_experiment:.4f}")
col2.metric("Informative fraction", f"{cost_eff.informative_fraction:.0%}")
col3.metric("Budget efficiency", f"{cost_eff.budget_efficiency:.0%}")
col4.metric("Cost / HV unit", f"{cost_eff.cost_per_hv_unit:.2f}")

st.divider()
col_phase, col_calib = st.columns(2)

with col_phase:
    st.markdown("**Strategy Phase Transitions**")
    if insight.phase_transitions:
        pt_df = pd.DataFrame([
            {"Iteration": t.iteration, "From": t.from_strategy, "To": t.to_strategy,
             "HV at transition": round(t.hv_at_transition, 4),
             "Trigger": t.trigger.replace("_", " ")}
            for t in insight.phase_transitions
        ])
        st.dataframe(pt_df, use_container_width=True, hide_index=True)
    else:
        st.info("No phase transitions detected.")

    if insight.dominant_strategy_per_phase:
        st.markdown("**Dominant phases**")
        dp_df = pd.DataFrame(insight.dominant_strategy_per_phase)
        dp_df.columns = [c.replace("_", " ").title() for c in dp_df.columns]
        st.dataframe(dp_df, use_container_width=True, hide_index=True)

with col_calib:
    trend_colour = {"improving": "🟢", "degrading": "🔴", "stable": "🟡", "insufficient_data": "⚪"}
    st.markdown(f"**Calibration trend:** {trend_colour.get(insight.calibration_trend, '⚪')} "
                f"{insight.calibration_trend.replace('_', ' ').title()}")

    if insight.calibration_ece_history:
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            y=insight.calibration_ece_history, mode="lines+markers",
            line=dict(color="#9b59b6", width=2), marker=dict(size=5),
            name="Calibration Error",
        ))
        fig_cal.update_layout(
            xaxis_title="Iteration",
            yaxis_title="Calibration Error (ECE)",
            height=200, margin=dict(l=40, r=20, t=20, b=40),
            plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e", font=dict(color="#e0e0e0"),
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    if insight.violation_rate_history:
        st.markdown("**Constraint violation rate**")
        fig_viol = go.Figure()
        fig_viol.add_trace(go.Bar(
            y=insight.violation_rate_history,
            marker_color="#e74c3c", opacity=0.7,
        ))
        fig_viol.update_layout(
            xaxis_title="Iteration", yaxis_title="Fraction violated",
            yaxis_range=[0, 1], height=200,
            margin=dict(l=40, r=20, t=20, b=40),
            plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e", font=dict(color="#e0e0e0"),
        )
        st.plotly_chart(fig_viol, use_container_width=True)

# ── Export ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("**Export**")
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    report_json = json.dumps(asdict(insight), indent=2, default=str)
    st.download_button(
        "⬇ Insight report (JSON)", data=report_json,
        file_name=f"aede_insight_{insight.campaign_hash}.json",
        mime="application/json",
    )

with col_exp2:
    if len(campaign_results["pareto_front"]) > 0:
        import io
        pf = campaign_results["pareto_front"]
        pf_df = pd.DataFrame(pf, columns=[f"obj_{i+1}" for i in range(pf.shape[1])])
        csv_buf = io.StringIO()
        pf_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇ Pareto front (CSV)", data=csv_buf.getvalue(),
            file_name=f"aede_pareto_{insight.campaign_hash}.csv",
            mime="text/csv",
        )