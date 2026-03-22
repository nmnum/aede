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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from aede import run_campaign, ProblemSpec, Objective, BudgetSpec, StoppingCriterion
from aede.modules.problem_framing import HardConstraint, SoftConstraint, load_spec_from_yaml
from aede.modules.trace_logger import DecisionTraceLogger
from aede.modules.insight_engine import InsightEngine
from aede.campaign import _update_pareto_front

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AEDE — Adaptive Experimental Decision Engine",
    page_icon="⚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #1f3a6e;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.2rem;
    }
    .finding-box {
        background: #eef2fb;
        border-left: 4px solid #1f3a6e;
        padding: 0.75rem 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }
    .warning-box {
        background: #fff8e6;
        border-left: 4px solid #e6a817;
        padding: 0.75rem 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f3a6e;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #c5d0e8;
    }
    div[data-testid="stSidebar"] { background: #f0f3fa; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "running": False,
        "results": None,
        "db_path": ":memory:",
        "logger": None,
        "campaign_hash": None,
        "live_hv": [],
        "live_iter": 0,
        "live_weights": [],
        "stop_triggered": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Helpers ───────────────────────────────────────────────────────────────────

ACCENT = "#1f3a6e"
STRATEGY_COLORS = {"bo": "#1f77b4", "nsga": "#e67e22", "diversity": "#27ae60"}

def make_pareto_fig(Y_all, pareto_front, obj_names):
    fig = go.Figure()
    # All observations
    if len(Y_all) > 0:
        fig.add_trace(go.Scatter(
            x=Y_all[:, 0], y=Y_all[:, 1],
            mode="markers",
            marker=dict(color="#aab4c8", size=6, opacity=0.5),
            name="All observations",
        ))
    # Pareto front
    if len(pareto_front) > 0:
        pf = pareto_front[np.argsort(pareto_front[:, 0])]
        fig.add_trace(go.Scatter(
            x=pf[:, 0], y=pf[:, 1],
            mode="markers+lines",
            marker=dict(color=ACCENT, size=10, symbol="diamond"),
            line=dict(color=ACCENT, width=1.5, dash="dot"),
            name="Pareto front",
        ))
    fig.update_layout(
        xaxis_title=obj_names[0],
        yaxis_title=obj_names[1],
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def make_hv_fig(hv_history):
    if not hv_history:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(hv_history))),
        y=hv_history,
        mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor="rgba(31,58,110,0.08)",
        name="Hypervolume",
    ))
    fig.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Pareto Hypervolume",
        height=280,
        margin=dict(l=40, r=20, t=20, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
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
            x=iters, y=vals,
            mode="lines",
            stackgroup="one",
            name=s,
            line=dict(color=STRATEGY_COLORS.get(s, "#888"), width=1),
            fillcolor=STRATEGY_COLORS.get(s, "#888"),
        ))
    fig.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Strategy weight",
        yaxis_range=[0, 1],
        height=250,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def make_eig_fig(hv_history, threshold):
    if not hv_history:
        return go.Figure()
    eig_vals = [h.get("eig_per_cost", 0.0) for h in hv_history]
    iters = [h.get("iteration", i) for i, h in enumerate(hv_history)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=eig_vals,
        mode="lines+markers",
        line=dict(color="#e67e22", width=2),
        marker=dict(size=5),
        name="EIG / cost",
    ))
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text=f"Stop threshold ({threshold})",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="Iteration",
        yaxis_title="EIG per cost unit",
        height=250,
        margin=dict(l=40, r=20, t=20, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗ AEDE")
    st.caption("Adaptive Experimental Decision Engine")
    st.divider()

    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "🚀 Run Campaign"

    tab = st.radio(
        "Navigation",
        ["🚀 Run Campaign", "📊 Live Monitor", "🔍 Insight Report"],
        label_visibility="collapsed",
        key="active_tab",
    )
    st.divider()

    if st.session_state.results:
        r = st.session_state.results
        hv = r["hv_history"][-1] if r["hv_history"] else 0
        st.metric("Final HV", f"{hv:.4f}")
        st.metric("Experiments run", r["n_experiments"])
        st.metric("Pareto front size", len(r["pareto_front"]))
        if st.session_state.stop_triggered:
            st.success("✓ Stopped early (criterion met)")


# ── Tab: Run Campaign ─────────────────────────────────────────────────────────
if tab == "🚀 Run Campaign":
    st.title("Configure & Launch Campaign")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-header">Benchmark & Objectives</div>', unsafe_allow_html=True)

        benchmark = st.selectbox(
            "Benchmark landscape",
            ["vlmop2", "molecular", "high_constraint"],
            format_func=lambda x: {
                "vlmop2": "VLMOP2 — Bi-objective analytic (sanity check)",
                "molecular": "Molecular Proxy — Bi-objective with synthesis constraints",
                "high_constraint": "High Constraint — Tri-objective, ~15% feasible space",
            }[x],
        )

        n_dims = st.slider("Input dimensions", 2, 10, 2 if benchmark == "vlmop2" else 5)

        st.markdown('<div class="section-header">Objective Names</div>', unsafe_allow_html=True)
        n_obj = 3 if benchmark == "high_constraint" else 2
        obj_names = []
        obj_dirs = []
        defaults_names = [
            ("binding_affinity", "maximize"),
            ("safety_score", "maximize"),
            ("metabolic_stability", "maximize"),
        ] if benchmark != "vlmop2" else [("objective_1", "maximize"), ("objective_2", "maximize")]

        for i in range(n_obj):
            c1, c2 = st.columns([2, 1])
            name = c1.text_input(f"Obj {i+1} name", defaults_names[i][0], key=f"oname_{i}")
            direction = c2.selectbox(f"Direction {i+1}", ["maximize", "minimize"],
                                     index=0 if defaults_names[i][1]=="maximize" else 1,
                                     key=f"odir_{i}")
            obj_names.append(name)
            obj_dirs.append(direction)

    with col2:
        st.markdown('<div class="section-header">Budget & Stopping</div>', unsafe_allow_html=True)
        n_exp = st.slider("Total experiments", 20, 300, 80)
        batch_size = st.slider("Experiments per iteration", 2, 20, 5)
        pool_size = st.slider("Candidate pool size", 20, 200, 60)

        st.markdown('<div class="section-header">Stopping Criterion</div>', unsafe_allow_html=True)
        eig_thresh = st.number_input(
            "EIG/cost threshold",
            min_value=0.0001, max_value=0.1, value=0.005, step=0.001, format="%.4f",
            help="Stop when EIG per cost unit falls below this for N consecutive iterations"
        )
        consec = st.slider("Consecutive iterations below threshold", 2, 5, 3)
        min_exp = st.slider("Minimum experiments before stopping", 5, 50, 15)

        st.markdown('<div class="section-header">Surrogate</div>', unsafe_allow_html=True)
        n_mc = st.slider("MC samples (EPI estimation)", 16, 256, 64,
                         help="More samples = better EPI estimates, slower iteration")
        seed = st.number_input("Random seed", value=42, step=1)

    st.divider()

    col_go, col_info = st.columns([1, 3])
    with col_go:
        go_btn = st.button(
            "▶ Launch Campaign",
            type="primary",
            disabled=st.session_state.running,
            use_container_width=True,
        )
    with col_info:
        if st.session_state.running:
            st.info("Campaign is running… switch to Live Monitor tab.")

    if go_btn and not st.session_state.running:
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

        st.session_state.running = True
        st.session_state.live_hv = []
        st.session_state.live_iter = 0
        st.session_state.live_weights = []
        st.session_state.stop_triggered = False
        st.session_state.results = None
        db = "/tmp/aede_campaign.db"
        st.session_state.db_path = db
        if os.path.exists(db):
            os.remove(db)

        def _run():
            def on_iter(iteration, manifest, hv):
                st.session_state.live_hv.append(hv)
                st.session_state.live_iter = iteration
                weights = manifest.strategy_weights.copy()
                weights["iteration"] = iteration
                st.session_state.live_weights.append(weights)
                if manifest.should_stop:
                    st.session_state.stop_triggered = True

            results = run_campaign(
                spec, db_path=db, verbose=False, n_mc=n_mc, on_iteration=on_iter
            )
            st.session_state.results = results
            st.session_state.campaign_hash = results["campaign_hash"]
            st.session_state.logger = results["logger"]
            st.session_state.running = False

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        st.rerun()

# Show live progress on Run Campaign tab while campaign is running
if st.session_state.get("running") and tab == "🚀 Run Campaign":
    st.divider()
    st.info("⏳ Campaign is running — switch to **Live Monitor** tab to watch progress, or wait here.")
    live_hv = st.session_state.get("live_hv", [])
    live_iter = st.session_state.get("live_iter", 0)
    if live_hv:
        st.metric("Current HV", f"{live_hv[-1]:.4f}", f"Iteration {live_iter}")
    time.sleep(0.5)
    st.rerun()


# ── Tab: Live Monitor ─────────────────────────────────────────────────────────
elif tab == "📊 Live Monitor":
    st.title("Live Campaign Monitor")

    if not st.session_state.results and not st.session_state.running:
        st.info("No campaign running. Go to **Run Campaign** to start one.")
        st.stop()

    # Live refresh while running
    if st.session_state.running:
        time.sleep(0.3)
        st.rerun()

    results = st.session_state.results
    if results is None:
        st.warning("Campaign starting…")
        time.sleep(0.5)
        st.rerun()

    hv_hist = results["hv_history"]
    pareto = results["pareto_front"]
    Y_all = results["Y_all"]
    n_obj = pareto.shape[1] if len(pareto) > 0 else 2

    # ── Top metrics ──
    m1, m2, m3, m4 = st.columns(4)
    final_hv = hv_hist[-1] if hv_hist else 0
    m1.metric("Final HV", f"{final_hv:.4f}", delta=f"+{final_hv - hv_hist[0]:.4f}" if len(hv_hist) > 1 else None)
    m2.metric("Iterations", len(hv_hist))
    m3.metric("Experiments", results["n_experiments"])
    m4.metric("Pareto front", len(pareto))

    if st.session_state.stop_triggered:
        st.success("✓ Campaign converged — stopping criterion met before budget exhaustion.")

    st.divider()

    # ── Pareto front ──
    obj_names = [o.name for o in results["insight_report"].cost_efficiency.__class__.__mro__] if False else None
    # Recover obj names from insight or default
    if st.session_state.logger:
        camps = st.session_state.logger.get_all_campaigns()
        if camps:
            try:
                spec_data = json.loads(camps[0].get("spec_json", "{}") if "spec_json" in camps[0] else "{}")
                obj_names = [o["name"] for o in spec_data.get("objectives", [])]
            except Exception:
                obj_names = None

    if obj_names is None:
        obj_names = [f"Objective {i+1}" for i in range(n_obj)]

    col_pf, col_hv = st.columns([1.3, 1])

    with col_pf:
        st.markdown('<div class="section-header">Pareto Front</div>', unsafe_allow_html=True)
        if n_obj == 2:
            fig_pf = make_pareto_fig(Y_all, pareto, obj_names[:2])
            st.plotly_chart(fig_pf, use_container_width=True)
        else:
            # 3D scatter for tri-objective
            fig_3d = go.Figure(data=[
                go.Scatter3d(
                    x=Y_all[:, 0], y=Y_all[:, 1], z=Y_all[:, 2],
                    mode="markers",
                    marker=dict(size=4, color="#aab4c8", opacity=0.4),
                    name="All",
                ),
                go.Scatter3d(
                    x=pareto[:, 0], y=pareto[:, 1], z=pareto[:, 2],
                    mode="markers",
                    marker=dict(size=8, color=ACCENT, symbol="diamond"),
                    name="Pareto front",
                ) if len(pareto) > 0 else go.Scatter3d(x=[], y=[], z=[]),
            ])
            fig_3d.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_3d, use_container_width=True)

    with col_hv:
        st.markdown('<div class="section-header">Hypervolume over iterations</div>', unsafe_allow_html=True)
        st.plotly_chart(make_hv_fig(hv_hist), use_container_width=True)

        st.markdown('<div class="section-header">EIG/cost vs stopping threshold</div>', unsafe_allow_html=True)
        if st.session_state.logger:
            hv_data = st.session_state.logger.get_hv_history(st.session_state.campaign_hash or "")
            # Use threshold from spec if available — default 0.005
            st.plotly_chart(make_eig_fig(hv_data, 0.005), use_container_width=True)

    # ── Strategy weights ──
    st.markdown('<div class="section-header">Portfolio strategy weights over time</div>', unsafe_allow_html=True)
    weight_hist = st.session_state.live_weights
    if not weight_hist and st.session_state.logger:
        weight_hist = st.session_state.logger.get_strategy_weight_history(
            st.session_state.campaign_hash or ""
        )
    if weight_hist:
        st.plotly_chart(make_weights_fig(weight_hist), use_container_width=True)

    # ── Recent decisions ──
    st.markdown('<div class="section-header">Recent selected experiments</div>', unsafe_allow_html=True)
    if st.session_state.logger and st.session_state.campaign_hash:
        decisions = st.session_state.logger.get_recent_decisions(
            st.session_state.campaign_hash, n=15
        )
        if decisions:
            import pandas as pd
            df = pd.DataFrame(decisions)
            display_cols = ["iteration", "candidate_id", "provenance", "decision_score",
                            "expected_pareto_improvement", "feasibility_probability",
                            "experiment_cost", "rank"]
            df_disp = df[[c for c in display_cols if c in df.columns]].copy()
            df_disp["decision_score"] = df_disp["decision_score"].round(4)
            df_disp["expected_pareto_improvement"] = df_disp["expected_pareto_improvement"].round(4)
            df_disp["feasibility_probability"] = df_disp["feasibility_probability"].round(3)
            df_disp["experiment_cost"] = df_disp["experiment_cost"].round(3)
            st.dataframe(df_disp, use_container_width=True, hide_index=True)


# ── Tab: Insight Report ───────────────────────────────────────────────────────
elif tab == "🔍 Insight Report":
    st.title("Campaign Insight Report")

    if not st.session_state.results:
        st.info("Run a campaign first to see the insight report.")
        st.stop()

    results = st.session_state.results
    insight = results["insight_report"]
    cost_eff = insight.cost_efficiency

    # ── Key findings ──
    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
    for finding in insight.key_findings:
        box_class = "warning-box" if "WARNING" in finding else "finding-box"
        st.markdown(f'<div class="{box_class}">{finding}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Cost efficiency ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("HV yield / experiment", f"{cost_eff.hv_per_experiment:.4f}")
    col2.metric("Informative fraction", f"{cost_eff.informative_fraction:.0%}")
    col3.metric("Budget efficiency", f"{cost_eff.budget_efficiency:.0%}")
    col4.metric("Cost / HV unit", f"{cost_eff.cost_per_hv_unit:.2f}")

    st.divider()

    # ── Phase transitions ──
    col_phase, col_calib = st.columns(2)

    with col_phase:
        st.markdown('<div class="section-header">Strategy Phase Transitions</div>', unsafe_allow_html=True)
        if insight.phase_transitions:
            import pandas as pd
            pt_df = pd.DataFrame([
                {
                    "Iteration": t.iteration,
                    "From": t.from_strategy,
                    "To": t.to_strategy,
                    "HV at transition": round(t.hv_at_transition, 4),
                    "Trigger": t.trigger.replace("_", " "),
                }
                for t in insight.phase_transitions
            ])
            st.dataframe(pt_df, use_container_width=True, hide_index=True)
        else:
            st.info("No strategy phase transitions detected.")

        if insight.dominant_strategy_per_phase:
            st.markdown('<div class="section-header">Dominant phases</div>', unsafe_allow_html=True)
            import pandas as pd
            dp_df = pd.DataFrame(insight.dominant_strategy_per_phase)
            dp_df.columns = [c.replace("_", " ").title() for c in dp_df.columns]
            st.dataframe(dp_df, use_container_width=True, hide_index=True)

    with col_calib:
        st.markdown('<div class="section-header">Calibration (EIG/cost stability)</div>', unsafe_allow_html=True)
        trend_colour = {"improving": "🟢", "degrading": "🔴", "stable": "🟡", "insufficient_data": "⚪"}
        st.write(f"Trend: {trend_colour.get(insight.calibration_trend, '⚪')} **{insight.calibration_trend.replace('_', ' ').title()}**")

        if insight.calibration_ece_history:
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                y=insight.calibration_ece_history,
                mode="lines+markers",
                line=dict(color="#9b59b6", width=2),
                marker=dict(size=5),
                name="EIG/cost",
            ))
            fig_cal.update_layout(
                xaxis_title="Iteration",
                yaxis_title="EIG/cost",
                height=220,
                margin=dict(l=40, r=20, t=20, b=40),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_cal, use_container_width=True)

        st.markdown('<div class="section-header">Constraint violation rate</div>', unsafe_allow_html=True)
        if insight.violation_rate_history:
            fig_viol = go.Figure()
            fig_viol.add_trace(go.Bar(
                y=insight.violation_rate_history,
                marker_color="#e74c3c",
                opacity=0.7,
                name="Violation rate",
            ))
            fig_viol.update_layout(
                xaxis_title="Iteration",
                yaxis_title="Fraction violated",
                yaxis_range=[0, 1],
                height=220,
                margin=dict(l=40, r=20, t=20, b=40),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_viol, use_container_width=True)
        else:
            st.info("No constraint violations recorded.")

    # ── Export ──
    st.divider()
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        import json
        from dataclasses import asdict
        report_json = json.dumps(asdict(insight), indent=2, default=str)
        st.download_button(
            "⬇ Download insight report (JSON)",
            data=report_json,
            file_name=f"aede_insight_{insight.campaign_hash}.json",
            mime="application/json",
        )

    with col_exp2:
        if results.get("pareto_front") is not None and len(results["pareto_front"]) > 0:
            import io
            import pandas as pd
            pf = results["pareto_front"]
            pf_df = pd.DataFrame(pf, columns=[f"obj_{i+1}" for i in range(pf.shape[1])])
            csv_buf = io.StringIO()
            pf_df.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇ Download Pareto front (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"aede_pareto_{insight.campaign_hash}.csv",
                mime="text/csv",
            )
