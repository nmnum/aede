"""
Module 3: Decision Engine
==========================
The core of AEDE. Transforms a filtered candidate pool into a ranked
set of decisions, each with a score, rationale, and cost attribution.

Primary novel contributions:
  1. Stopping criterion: EIG-per-cost threshold over consecutive iterations.
  2. Decision scoring: EPI × uncertainty × feasibility / cost.
  3. Gittins-index extension (optional, activated via config).

Decision score formula:
    score = (expected_pareto_improvement × uncertainty_discount × feasibility_prob) / cost

Stopping criterion:
    If mean(EIG/cost) for top-K candidates < threshold for N consecutive
    iterations AND budget > min_experiments, terminate campaign.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from .candidate_generation import Candidate, compute_hypervolume
from .constraint_layer import ConstraintResult
from .problem_framing import ProblemSpec


@dataclass
class DecisionRecord:
    candidate_id: str
    X: np.ndarray
    provenance: str
    decision_score: float
    expected_pareto_improvement: float
    uncertainty_discount: float
    feasibility_probability: float
    experiment_cost: float
    rationale: str
    rank: int
    stopping_flag: bool = False
    gittins_index: Optional[float] = None


@dataclass
class DecisionManifest:
    iteration: int
    selected: List[DecisionRecord]
    all_scored: List[DecisionRecord]
    should_stop: bool
    stop_reason: str
    eig_per_cost: float
    pareto_hypervolume: float
    budget_remaining: int
    strategy_weights: Dict[str, float] = field(default_factory=dict)


class DecisionEngine:

    def __init__(
        self,
        spec: ProblemSpec,
        use_gittins: bool = False,
        gittins_discount: float = 0.95,
    ):
        self.spec = spec
        self.use_gittins = use_gittins
        self.gittins_discount = gittins_discount

        # Stopping criterion state
        self._consecutive_below = 0
        self._eig_history: List[float] = []
        self._hv_history: List[float] = []
        self._experiments_run = 0

        # For uncertainty normalisation
        self._sigma_max_running = 1.0

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _expected_pareto_improvement(
        self,
        candidate: Candidate,
        pareto_front: np.ndarray,
        ref: np.ndarray,
        surrogate,
        n_mc: int = 64,
    ) -> float:
        """MC estimate of expected Pareto HV improvement."""
        if surrogate is None or not surrogate._is_fit:
            if candidate.predicted_objectives is not None:
                val = candidate.predicted_objectives
                aug = np.vstack([pareto_front, val]) if len(pareto_front) > 0 else val.reshape(1, -1)
                hv_base = compute_hypervolume(pareto_front, ref) if len(pareto_front) > 0 else 0.0
                hv_new = compute_hypervolume(aug, ref)
                return max(0.0, hv_new - hv_base)
            return 0.0

        hv_base = compute_hypervolume(pareto_front, ref) if len(pareto_front) > 0 else 0.0
        try:
            samples = surrogate.sample_posterior(candidate.X.reshape(1, -1), n_samples=n_mc)
            # samples: (n_mc, 1, n_obj) -> squeeze to (n_mc, n_obj)
            samples = samples[:, 0, :]
            improvements = []
            for s in samples:
                aug = np.vstack([pareto_front, s]) if len(pareto_front) > 0 else s.reshape(1, -1)
                hv_new = compute_hypervolume(aug, ref)
                improvements.append(max(0.0, hv_new - hv_base))
            return float(np.mean(improvements))
        except Exception:
            return 0.0

    def _uncertainty_discount(self, candidate: Candidate) -> float:
        """Normalised uncertainty — higher uncertainty = higher exploration value."""
        if candidate.predicted_uncertainty is None:
            return 0.5
        sigma = float(np.mean(candidate.predicted_uncertainty))
        # Update running max for normalisation
        self._sigma_max_running = max(self._sigma_max_running, sigma + 1e-8)
        return float(np.clip(sigma / self._sigma_max_running, 0.0, 1.0))

    def _gittins_index(
        self,
        epi: float,
        uncertainty: float,
        cost: float,
        iteration: int,
    ) -> float:
        """
        Simplified Gittins-style index: accounts for opportunity cost over
        remaining campaign horizon with a discount factor.
        Extends the base scoring with a time-value-of-information correction.
        """
        remaining = max(
            self.spec.budget.experiments - self._experiments_run - 1, 1
        )
        discount = self.gittins_discount ** (1.0 / remaining)
        # Gittins correction: scale EPI by how much of the campaign remains
        horizon_correction = 1.0 - discount ** remaining
        return (epi * uncertainty * horizon_correction) / max(cost, 0.01)

    def score_candidates(
        self,
        candidates: List[Candidate],
        constraint_results: List[ConstraintResult],
        pareto_front: np.ndarray,
        surrogate,
        n_mc: int = 64,
    ) -> List[DecisionRecord]:
        ref = self.spec.reference_point()
        cr_map = {cr.candidate_id: cr for cr in constraint_results}
        scored = []

        for cand in candidates:
            cr = cr_map.get(cand.id)
            f_prob = cr.feasibility_probability if cr else 0.7
            cost = cr.estimated_cost if cr else 1.0

            epi = self._expected_pareto_improvement(cand, pareto_front, ref, surrogate, n_mc)
            unc = self._uncertainty_discount(cand)
            raw_score = (epi * unc * f_prob) / max(cost, 0.01)

            if self.use_gittins:
                g_idx = self._gittins_index(epi, unc, cost, len(self._hv_history))
                score = g_idx
            else:
                score = raw_score

            record = DecisionRecord(
                candidate_id=cand.id,
                X=cand.X.copy(),
                provenance=cand.provenance,
                decision_score=float(score),
                expected_pareto_improvement=float(epi),
                uncertainty_discount=float(unc),
                feasibility_probability=float(f_prob),
                experiment_cost=float(cost),
                rationale="",  # filled by Communication Layer
                rank=0,
                gittins_index=g_idx if self.use_gittins else None,
            )
            scored.append(record)

        # Sort descending by score and assign ranks
        scored.sort(key=lambda r: r.decision_score, reverse=True)
        for i, r in enumerate(scored):
            r.rank = i + 1

        return scored

    # ── Stopping criterion ─────────────────────────────────────────────────────

    def check_stopping(
        self, scored: List[DecisionRecord], budget_remaining: int
    ) -> Tuple[bool, str]:
        """
        Returns (should_stop, reason_string).
        Stopping fires if:
        - Experiments run >= min_experiments
        - Top-K EIG/cost < threshold for N consecutive iterations
        - OR budget exhausted
        """
        sc = self.spec.stopping_criterion

        if budget_remaining <= 0:
            return True, "Budget exhausted"

        if self._experiments_run < sc.min_experiments:
            return False, ""

        if not scored:
            return False, ""

        top_k = min(len(scored), self.spec.budget.per_iteration_cap)
        top_scores = [r.decision_score for r in scored[:top_k]]
        mean_eig_per_cost = float(np.mean(top_scores))
        self._eig_history.append(mean_eig_per_cost)

        if mean_eig_per_cost < sc.eig_per_cost_threshold:
            self._consecutive_below += 1
        else:
            self._consecutive_below = 0

        if self._consecutive_below >= sc.consecutive_iterations:
            return True, (
                f"EIG/cost ({mean_eig_per_cost:.4f}) below threshold "
                f"({sc.eig_per_cost_threshold}) for {sc.consecutive_iterations} "
                f"consecutive iterations. Campaign converged."
            )

        return False, ""

    # ── Main decision step ────────────────────────────────────────────────────

    def decide(
        self,
        candidates: List[Candidate],
        constraint_results: List[ConstraintResult],
        pareto_front: np.ndarray,
        surrogate,
        iteration: int,
        budget_remaining: int,
        strategy_weights: Optional[Dict] = None,
        n_mc: int = 64,
    ) -> DecisionManifest:
        scored = self.score_candidates(
            candidates, constraint_results, pareto_front, surrogate, n_mc
        )

        should_stop, stop_reason = self.check_stopping(scored, budget_remaining)

        n_select = min(self.spec.budget.per_iteration_cap, len(scored), budget_remaining)
        selected = scored[:n_select]

        hv = compute_hypervolume(pareto_front, self.spec.reference_point()) if len(pareto_front) > 0 else 0.0
        self._hv_history.append(hv)

        eig = float(np.mean([r.decision_score for r in selected])) if selected else 0.0
        top_k = min(len(scored), self.spec.budget.per_iteration_cap)
        eig_per_cost = float(np.mean([r.decision_score for r in scored[:top_k]])) if scored else 0.0

        if should_stop:
            for r in selected:
                r.stopping_flag = True

        manifest = DecisionManifest(
            iteration=iteration,
            selected=selected,
            all_scored=scored,
            should_stop=should_stop,
            stop_reason=stop_reason,
            eig_per_cost=eig_per_cost,
            pareto_hypervolume=hv,
            budget_remaining=budget_remaining - len(selected),
            strategy_weights=strategy_weights or {},
        )

        self._experiments_run += len(selected)
        return manifest

    # ── Budget tracking ───────────────────────────────────────────────────────

    @property
    def eig_history(self) -> List[float]:
        return self._eig_history.copy()

    @property
    def hv_history(self) -> List[float]:
        return self._hv_history.copy()
