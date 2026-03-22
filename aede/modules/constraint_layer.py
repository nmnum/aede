"""
Module 2: Constraint Intelligence Layer
=========================================
Two-tier constraint handling:
  1. Hard constraints — binary filter, no learning.
  2. Soft constraints — learnable GP classifier, updated each iteration.

The layer also maintains a feasibility model trained on observations,
used by the Decision Engine for the feasibility_probability term.

FMEA-style attribution: the layer tracks which input features are
associated with constraint violations, consumed by the Insight Engine.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from .problem_framing import ProblemSpec, HardConstraint, SoftConstraint
from .candidate_generation import Candidate


@dataclass
class ConstraintResult:
    candidate_id: str
    passes_hard: bool
    feasibility_probability: float        # P(all soft constraints satisfied)
    soft_violation_probs: Dict[str, float] = field(default_factory=dict)
    estimated_cost: float = 1.0
    repaired: bool = False
    repair_distance: float = 0.0


class ConstraintLayer:

    def __init__(self, spec: ProblemSpec, seed: Optional[int] = None):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self._X_scaler = StandardScaler()
        self._is_fit = False
        self._n_violations = 0
        self._n_total = 0
        self._obs_X: List[np.ndarray] = []
        self._obs_feasible: List[int] = []

        # GP classifier for feasibility
        kernel = RBF(length_scale_bounds=(1e-2, 10.0))
        self._feasibility_model = GaussianProcessClassifier(
            kernel=kernel,
            random_state=seed,
            warm_start=True,
            max_iter_predict=200,
        )

        # Cost model: simple linear regression updated online
        self._cost_weights: Optional[np.ndarray] = None
        self._cost_bias: float = 1.0
        self._cost_X: List[np.ndarray] = []
        self._cost_y: List[float] = []

        # Feature violation tracking for FMEA attribution
        self._feature_violation_counts: Optional[np.ndarray] = None
        self._feature_total_counts: Optional[np.ndarray] = None

    # ── Hard constraint evaluation ────────────────────────────────────────────

    def _check_hard(self, X: np.ndarray, properties: Optional[Dict] = None) -> bool:
        """
        Evaluate hard constraints.
        If the IVO produces property values, use them.
        Otherwise use X features as proxies for pre-filtering.
        """
        for c in self.spec.hard_constraints:
            val = self._get_property_value(X, c.property, properties)
            if val is None:
                continue
            if c.min is not None and val < c.min:
                return False
            if c.max is not None and val > c.max:
                return False
            if c.equals is not None and bool(val) != c.equals:
                return False
        return True

    def _get_property_value(
        self, X: np.ndarray, prop_name: str, properties: Optional[Dict] = None
    ) -> Optional[float]:
        if properties and prop_name in properties:
            return float(properties[prop_name])
        # Fall back to feature-based proxies
        if prop_name == "molecular_weight":
            return float(np.sum(np.abs(X)))
        if prop_name == "synthesisability":
            return float(1.0 - np.clip(np.std(X) * 0.5, 0, 1))
        if prop_name == "within_sphere":
            return float(np.sum(X ** 2))
        return None

    # ── Soft constraint estimation ─────────────────────────────────────────────

    def _estimate_soft_risks(self, X: np.ndarray) -> Dict[str, float]:
        risks = {}
        for c in self.spec.soft_constraints:
            if c.property == "synthesis_cost":
                # Linear cost proxy: complexity of feature vector
                risk = float(np.clip(np.std(X) * np.mean(np.abs(X)) / 3.0, 0.0, 0.95))
            elif c.property == "reagent_scarcity":
                # Static prior until we have observations
                risk = 0.1
            else:
                risk = 0.15  # conservative unknown prior
            risks[c.property] = risk
        return risks

    # ── GP feasibility model ───────────────────────────────────────────────────

    def update(self, X_obs: np.ndarray, feasible_flags: np.ndarray):
        """
        Update the GP feasibility model with new observations.
        X_obs: (n, n_features)
        feasible_flags: (n,) binary — 1=feasible, 0=infeasible
        """
        for x, f in zip(X_obs, feasible_flags):
            self._obs_X.append(x.copy())
            self._obs_feasible.append(int(f))
            self._n_violations += 1 - int(f)
            self._n_total += 1

        if self._feature_violation_counts is None:
            n_feat = X_obs.shape[1]
            self._feature_violation_counts = np.zeros(n_feat)
            self._feature_total_counts = np.zeros(n_feat)

        # Track feature-level violations for FMEA
        for x, f in zip(X_obs, feasible_flags):
            if not f:
                self._feature_violation_counts += np.abs(x)
            self._feature_total_counts += np.abs(x)

        # Refit GP classifier if enough data and class balance
        if (
            len(self._obs_X) >= 10
            and len(set(self._obs_feasible)) > 1
        ):
            X_arr = np.array(self._obs_X)
            Xs = self._X_scaler.fit_transform(X_arr)
            y_arr = np.array(self._obs_feasible)
            try:
                self._feasibility_model.fit(Xs, y_arr)
                self._is_fit = True
            except Exception:
                pass

    def predict_feasibility(self, X: np.ndarray) -> np.ndarray:
        """
        Returns P(feasible) for each row of X.
        Falls back to a conservative prior if not enough data.
        """
        if not self._is_fit or len(self._obs_X) < 10:
            # Conservative prior: 0.7 feasibility assumed
            return np.full(len(X), 0.7)
        Xs = self._X_scaler.transform(X)
        try:
            probs = self._feasibility_model.predict_proba(Xs)
            # Column for class 1 (feasible)
            classes = self._feasibility_model.classes_
            feasible_col = np.where(classes == 1)[0][0] if 1 in classes else 0
            return probs[:, feasible_col]
        except Exception:
            return np.full(len(X), 0.7)

    # ── Candidate repair ──────────────────────────────────────────────────────

    def _repair(self, candidate: Candidate) -> Tuple[Candidate, float]:
        """
        Minimal perturbation to push a violating candidate into the feasible region.
        Only attempts repair if GP classifier is fit.
        Max 20% feature modification limit enforced.
        """
        if not self._is_fit:
            return candidate, 0.0

        from scipy.optimize import minimize

        x0 = candidate.X.copy()
        max_delta = 0.2 * np.linalg.norm(x0 + 1e-8)

        def neg_feasibility(x):
            p = self.predict_feasibility(x.reshape(1, -1))
            return -float(p[0])

        def constraint_distance(x):
            return max_delta - np.linalg.norm(x - x0)

        try:
            result = minimize(
                neg_feasibility,
                x0,
                method="SLSQP",
                constraints={"type": "ineq", "fun": constraint_distance},
                options={"maxiter": 30, "ftol": 1e-4},
            )
            if result.success and -result.fun > 0.6:
                repaired_X = result.x
                repair_dist = float(np.linalg.norm(repaired_X - x0))
                repaired = Candidate(
                    id=candidate.id + "_repaired",
                    X=repaired_X,
                    provenance=candidate.provenance,
                    predicted_objectives=candidate.predicted_objectives,
                    predicted_uncertainty=candidate.predicted_uncertainty,
                    acquisition_value=candidate.acquisition_value * 0.9,
                )
                repaired.repaired = True
                return repaired, repair_dist
        except Exception:
            pass
        return candidate, 0.0

    # ── Main filtering pipeline ───────────────────────────────────────────────

    def filter(
        self,
        candidates: List[Candidate],
        attempt_repair: bool = True,
    ) -> Tuple[List[Candidate], List[ConstraintResult]]:
        """
        Filter and optionally repair candidates.
        Returns (filtered_candidates, constraint_results).
        """
        filtered, results = [], []

        all_X = np.array([c.X for c in candidates])
        feas_probs = self.predict_feasibility(all_X)

        for i, (candidate, f_prob) in enumerate(zip(candidates, feas_probs)):
            passes_hard = self._check_hard(candidate.X)
            soft_risks = self._estimate_soft_risks(candidate.X)
            cost = self._estimate_cost(candidate.X)

            # Check soft constraint thresholds
            soft_ok = all(
                soft_risks.get(c.property, 0.0) < c.max_violation_probability
                for c in self.spec.soft_constraints
            )

            result = ConstraintResult(
                candidate_id=candidate.id,
                passes_hard=passes_hard,
                feasibility_probability=float(f_prob),
                soft_violation_probs=soft_risks,
                estimated_cost=cost,
            )

            if not passes_hard:
                results.append(result)
                continue  # Discard — hard constraint violation

            if not soft_ok and attempt_repair and f_prob < 0.5:
                repaired, dist = self._repair(candidate)
                if dist > 0:
                    result.repaired = True
                    result.repair_distance = dist
                    new_probs = self.predict_feasibility(repaired.X.reshape(1, -1))
                    result.feasibility_probability = float(new_probs[0])
                    filtered.append(repaired)
                    results.append(result)
                    continue

            filtered.append(candidate)
            results.append(result)

        return filtered, results

    def _estimate_cost(self, X: np.ndarray) -> float:
        """Linear cost estimate — updated via update_cost_model."""
        if self._cost_weights is not None:
            return float(
                np.clip(
                    self._cost_bias + self._cost_weights @ np.abs(X),
                    0.5,
                    20.0,
                )
            )
        return 1.0 + 0.3 * float(np.std(X))

    def update_cost_model(self, X_obs: np.ndarray, costs: np.ndarray):
        """Update linear cost model with observed costs."""
        for x, c in zip(X_obs, costs):
            self._cost_X.append(np.abs(x))
            self._cost_y.append(float(c))
        if len(self._cost_y) >= 5:
            X_arr = np.array(self._cost_X)
            y_arr = np.array(self._cost_y)
            # Least squares fit
            A = np.column_stack([np.ones(len(X_arr)), X_arr])
            sol, _, _, _ = np.linalg.lstsq(A, y_arr, rcond=None)
            self._cost_bias = float(sol[0])
            self._cost_weights = sol[1:]

    # ── FMEA attribution ──────────────────────────────────────────────────────

    def feature_violation_attribution(self) -> Optional[np.ndarray]:
        """
        Returns per-feature violation risk score (normalised [0,1]).
        Used by the Insight Engine for FMEA-style attribution.
        """
        if self._feature_violation_counts is None or self._n_total == 0:
            return None
        denom = np.maximum(self._feature_total_counts, 1.0)
        return self._feature_violation_counts / denom

    @property
    def violation_rate(self) -> float:
        if self._n_total == 0:
            return 0.0
        return self._n_violations / self._n_total

    @property
    def n_observations(self) -> int:
        return self._n_total
