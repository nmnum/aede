"""
Campaign Runner
================
Orchestrates all AEDE modules for a complete campaign.
This is the main entry point — both CLI and Streamlit UI call this.

Usage:
    from aede.campaign import run_campaign
    results = run_campaign(spec, db_path="campaign.db", verbose=True)
"""

from __future__ import annotations
import numpy as np
import time
from typing import Optional
from pathlib import Path

from .modules.problem_framing import ProblemSpec, load_spec_from_yaml, load_spec_from_dict
from .modules.ivo_layer import create_ivo, BaseIVO
from .modules.surrogate import GPSurrogate
from .modules.candidate_generation import CandidateGenerator
from .modules.constraint_layer import ConstraintLayer
from .modules.decision_engine import DecisionEngine
from .modules.trace_logger import DecisionTraceLogger
from .modules.insight_engine import InsightEngine
from .modules.communication import CommunicationLayer


def run_campaign(
    spec: ProblemSpec,
    db_path: str = ":memory:",
    verbose: bool = True,
    n_mc: int = 64,
    on_iteration: Optional[callable] = None,
) -> dict:
    """
    Run a complete AEDE campaign.

    Args:
        spec:         Validated ProblemSpec.
        db_path:      SQLite path for Decision Trace. Use ':memory:' for ephemeral.
        verbose:      Print iteration summaries to stdout.
        n_mc:         Monte Carlo samples for EPI estimation.
        on_iteration: Optional callback(iteration, manifest, hv) for live UI updates.

    Returns:
        dict with keys: campaign_hash, hv_history, n_experiments, insight_report
    """
    rng = np.random.default_rng(spec.random_seed)
    campaign_hash = spec.config_hash()
    if verbose:
        print(f"\n{'='*60}")
        print(f"AEDE Campaign: {spec.name}")
        print(f"Hash: {campaign_hash}")
        print(f"Benchmark: {spec.benchmark}  |  Objectives: {spec.n_objectives()}")
        print(f"Budget: {spec.budget.experiments}  |  Seed: {spec.random_seed}")
        print(f"{'='*60}")

    # ── Initialise modules ────────────────────────────────────────────────────
    ivo = create_ivo(
        spec.benchmark,
        n_var=spec.n_input_dims,
        seed=spec.random_seed,
    )

    surrogate = GPSurrogate(
        n_objectives=spec.n_objectives(),
        n_restarts=2,
        seed=spec.random_seed,
    )

    # Bounds: use IVO-appropriate defaults
    bounds = np.column_stack([
        np.full(spec.n_input_dims, -3.0),
        np.full(spec.n_input_dims, 3.0),
    ])

    generator = CandidateGenerator(
        n_var=spec.n_input_dims,
        bounds=bounds,
        seed=spec.random_seed,
    )

    constraint_layer = ConstraintLayer(spec, seed=spec.random_seed)
    decision_engine = DecisionEngine(spec)
    logger = DecisionTraceLogger(db_path=db_path)
    comms = CommunicationLayer()

    # Register campaign
    spec_dict = spec.model_dump() if hasattr(spec, "model_dump") else spec.dict()
    logger.register_campaign(campaign_hash, spec.name, spec_dict)

    # ── Initial random observations for warm-start ────────────────────────────
    n_init = min(max(spec.budget.per_iteration_cap * 2, 10), spec.budget.experiments // 5)
    X_init = rng.uniform(-3, 3, size=(n_init, spec.n_input_dims))
    init_obs = ivo.observe_batch(X_init, ids=[f"init_{i:03d}" for i in range(n_init)])

    X_all = X_init.copy()
    Y_all = np.array([o.objectives for o in init_obs])
    feasible_flags = np.array([int(o.constraint_valid) for o in init_obs])

    # Update constraint layer with initial observations
    constraint_layer.update(X_all, feasible_flags)
    constraint_layer.update_cost_model(
        X_all, np.array([o.experiment_cost for o in init_obs])
    )

    # Fit initial surrogate (only on feasible observations)
    feasible_mask = feasible_flags.astype(bool)
    if feasible_mask.sum() >= 3:
        surrogate.fit(X_all[feasible_mask], Y_all[feasible_mask])

    # Build initial Pareto front
    pareto_front = _update_pareto_front(Y_all[feasible_mask] if feasible_mask.sum() > 0 else Y_all)

    budget_remaining = spec.budget.experiments - n_init
    iteration = 0

    # ── Main campaign loop ────────────────────────────────────────────────────
    while budget_remaining > 0:
        iteration += 1

        # 1. Generate candidates
        candidates = generator.generate(
            pool_size=spec.candidate_pool_size,
            surrogate=surrogate if surrogate._is_fit else None,
            observations_X=X_all,
            pareto_front=pareto_front,
        )

        # 2. Filter through constraint layer
        filtered, constraint_results = constraint_layer.filter(candidates)

        if len(filtered) == 0:
            if verbose:
                print(f"  [iter {iteration}] All candidates failed constraints — sampling random fallback")
            fallback_X = rng.uniform(-3, 3, size=(spec.budget.per_iteration_cap, spec.n_input_dims))
            from .modules.candidate_generation import Candidate
            filtered = [Candidate(id=f"fallback_{i}", X=x, provenance="diversity") for i, x in enumerate(fallback_X)]
            constraint_results = []

        # 3. Decision Engine
        manifest = decision_engine.decide(
            candidates=filtered,
            constraint_results=constraint_results,
            pareto_front=pareto_front,
            surrogate=surrogate if surrogate._is_fit else None,
            iteration=iteration,
            budget_remaining=budget_remaining,
            strategy_weights=generator.current_weights(),
            n_mc=n_mc,
        )

        # 4. Annotate with rationales
        manifest = comms.annotate_manifest(manifest)

        # 5. Run selected experiments through IVO
        selected_X = np.array([r.X for r in manifest.selected])
        selected_ids = [r.candidate_id for r in manifest.selected]
        new_obs = ivo.observe_batch(selected_X, ids=selected_ids)
        ivo.register_surrogate(surrogate.predict_fn if surrogate._is_fit else None)

        # 6. Update data
        new_Y = np.array([o.objectives for o in new_obs])
        new_feasible = np.array([int(o.constraint_valid) for o in new_obs])

        X_all = np.vstack([X_all, selected_X])
        Y_all = np.vstack([Y_all, new_Y])
        feasible_flags = np.concatenate([feasible_flags, new_feasible])

        # 7. Update constraint layer
        constraint_layer.update(selected_X, new_feasible)
        constraint_layer.update_cost_model(
            selected_X, np.array([o.experiment_cost for o in new_obs])
        )

        # 8. Update surrogate (feasible observations only)
        feasible_mask = feasible_flags.astype(bool)
        if feasible_mask.sum() >= 5:
            surrogate.fit(X_all[feasible_mask], Y_all[feasible_mask])

        # 9. Update Pareto front
        pareto_front = _update_pareto_front(Y_all[feasible_mask] if feasible_mask.sum() > 0 else Y_all)

        # 10. Update strategy weights
        # Compute per-strategy HV improvement from this iteration
        by_provenance = {}
        for rec, obs in zip(manifest.selected, new_obs):
            p = rec.provenance
            by_provenance.setdefault(p, []).append(obs.objectives)

        hv_improvements = {}
        for prov, obs_list in by_provenance.items():
            pf_before = _update_pareto_front(Y_all[feasible_mask][:-len(manifest.selected)] if feasible_mask.sum() > len(manifest.selected) else np.empty((0, spec.n_objectives())))
            pf_after = pareto_front
            ref = spec.reference_point()
            from .modules.candidate_generation import compute_hypervolume
            hv_before = compute_hypervolume(pf_before, ref) if len(pf_before) > 0 else 0.0
            hv_after = compute_hypervolume(pf_after, ref) if len(pf_after) > 0 else 0.0
            hv_improvements[prov] = max(0.0, hv_after - hv_before) * len(obs_list) / len(manifest.selected)

        generator.update_weights(hv_improvements)

        # 11. Log everything
        surrogate_snap = surrogate.state_snapshot() if surrogate._is_fit else {}
        logger.log_iteration(campaign_hash, manifest, surrogate_snap)
        logger.log_observations(campaign_hash, iteration, new_obs)

        budget_remaining -= len(manifest.selected)

        if verbose:
            print(comms.format_iteration_summary(manifest))

        if on_iteration:
            on_iteration(iteration, manifest, float(compute_hypervolume(pareto_front, spec.reference_point())))

        # 12. Check stopping
        if manifest.should_stop:
            if verbose:
                print(comms.format_stop_notice(manifest))
            break

    # ── Post-campaign ─────────────────────────────────────────────────────────
    logger.mark_campaign_done(campaign_hash, "completed")

    # Generate insight report
    insight_engine = InsightEngine(logger)
    insight = insight_engine.generate_report(campaign_hash)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Campaign complete: {spec.name}")
        print(f"Iterations: {iteration}  |  Experiments: {len(X_all) - n_init + n_init}")
        print(f"Final Pareto HV: {compute_hypervolume(pareto_front, spec.reference_point()):.4f}")
        print(f"\nKey findings:")
        for finding in insight.key_findings:
            print(f"  • {finding}")
        print(f"{'='*60}\n")

    return {
        "campaign_hash": campaign_hash,
        "hv_history": decision_engine.hv_history,
        "n_experiments": len(X_all),
        "pareto_front": pareto_front,
        "X_all": X_all,
        "Y_all": Y_all,
        "insight_report": insight,
        "logger": logger,
    }


def _update_pareto_front(Y: np.ndarray) -> np.ndarray:
    """Compute the non-dominated (Pareto) front from observed objectives."""
    if len(Y) == 0:
        return np.empty((0, Y.shape[1] if len(Y.shape) > 1 else 1))
    n, m = Y.shape
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j >= i in all objectives and j > i in at least one
            if np.all(Y[j] >= Y[i]) and np.any(Y[j] > Y[i]):
                is_dominated[i] = True
                break
    return Y[~is_dominated]
