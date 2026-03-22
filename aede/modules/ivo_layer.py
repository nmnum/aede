"""
Module 4: In-Vitro Oracle (IVO) Layer
=======================================
Simulates experimental outcomes with calibrated noise.
Three benchmark landscapes of increasing difficulty.

All IVO classes share the same interface:
    observe(X: np.ndarray) -> ObservationRecord

The oracle tracks prediction errors against the surrogate model
when one is registered, enabling calibration analysis in the Insight Engine.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict


@dataclass
class ObservationRecord:
    candidate_id: str
    X: np.ndarray                    # Input features
    objectives: np.ndarray           # Observed objective values (maximisation convention)
    constraint_valid: bool           # True if all hard constraints satisfied
    soft_constraint_risks: Dict[str, float] = field(default_factory=dict)
    experiment_cost: float = 1.0
    noise_realisation: np.ndarray = field(default_factory=lambda: np.array([]))
    true_values: np.ndarray = field(default_factory=lambda: np.array([]))
    surrogate_prediction: Optional[np.ndarray] = None
    surrogate_uncertainty: Optional[np.ndarray] = None
    prediction_error: Optional[np.ndarray] = None


class BaseIVO:
    """Abstract base for all IVO benchmarks."""

    def __init__(self, noise_std: float = 0.02, seed: Optional[int] = None):
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self._obs_count = 0
        self._registered_surrogate = None

    def _true_values(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _constraint_check(self, X: np.ndarray) -> tuple[bool, Dict[str, float]]:
        """Returns (hard_valid, soft_risk_dict)."""
        return True, {}

    def _cost(self, X: np.ndarray) -> float:
        return 1.0

    def observe(self, X: np.ndarray, candidate_id: Optional[str] = None) -> ObservationRecord:
        X = np.atleast_1d(X).flatten()
        self._obs_count += 1
        cid = candidate_id or f"obs_{self._obs_count:04d}"

        true_vals = self._true_values(X)
        noise = self.rng.normal(0, self.noise_std, size=true_vals.shape)
        observed = true_vals + noise
        hard_valid, soft_risks = self._constraint_check(X)
        cost = self._cost(X)

        rec = ObservationRecord(
            candidate_id=cid,
            X=X.copy(),
            objectives=observed,
            constraint_valid=hard_valid,
            soft_constraint_risks=soft_risks,
            experiment_cost=cost,
            noise_realisation=noise,
            true_values=true_vals,
        )

        if self._registered_surrogate is not None:
            try:
                pred, unc = self._registered_surrogate(X.reshape(1, -1))
                rec.surrogate_prediction = pred.flatten()
                rec.surrogate_uncertainty = unc.flatten()
                rec.prediction_error = np.abs(pred.flatten() - true_vals)
            except Exception:
                pass

        return rec

    def observe_batch(self, Xs: np.ndarray, ids: Optional[List[str]] = None) -> List[ObservationRecord]:
        ids = ids or [None] * len(Xs)
        return [self.observe(x, cid) for x, cid in zip(Xs, ids)]

    def register_surrogate(self, predict_fn: Callable):
        """Register a surrogate predictor for calibration tracking."""
        self._registered_surrogate = predict_fn

    @property
    def n_objectives(self) -> int:
        raise NotImplementedError

    @property
    def input_dim(self) -> int:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class VLMOP2Benchmark(BaseIVO):
    """
    VLMOP2: classic bi-objective test function.
    Analytic Pareto front, smooth landscape, low noise.
    Used as the sanity-check baseline.

    Objectives (both maximised after negation):
        f1(x) = 1 - exp(-sum((x - 1/sqrt(n))^2))
        f2(x) = 1 - exp(-sum((x + 1/sqrt(n))^2))
    """

    def __init__(self, n_var: int = 2, noise_std: float = 0.01, seed: Optional[int] = None):
        super().__init__(noise_std=noise_std, seed=seed)
        self._n_var = n_var

    @property
    def n_objectives(self) -> int:
        return 2

    @property
    def input_dim(self) -> int:
        return self._n_var

    def _true_values(self, X: np.ndarray) -> np.ndarray:
        n = self._n_var
        x = X[:n]
        f1 = 1.0 - np.exp(-np.sum((x - 1.0 / np.sqrt(n)) ** 2))
        f2 = 1.0 - np.exp(-np.sum((x + 1.0 / np.sqrt(n)) ** 2))
        # Negate for maximisation convention (both originally minimised)
        return np.array([1.0 - f1, 1.0 - f2])

    def pareto_front_samples(self, n: int = 100) -> np.ndarray:
        """True Pareto front samples for evaluation."""
        t = np.linspace(-2, 2, n)
        x = np.column_stack([t, t])
        vals = np.array([self._true_values(xi) for xi in x])
        return vals


class MolecularProxyBenchmark(BaseIVO):
    """
    Synthetic molecular property landscape.
    Uses RDKit-style descriptor topology without requiring RDKit.
    Bi-objective: binding affinity proxy vs. toxicity penalty.

    The landscape is constructed from a random Fourier feature GP prior,
    giving realistic multi-modal structure over a bounded input space.
    """

    def __init__(
        self,
        n_var: int = 5,
        noise_std: float = 0.03,
        seed: Optional[int] = None,
    ):
        super().__init__(noise_std=noise_std, seed=seed)
        self._n_var = n_var
        # Construct fixed random landscape (same for all instances with same seed)
        rng = np.random.default_rng(seed if seed is not None else 0)
        self._W1 = rng.standard_normal((32, n_var)) * 2.0
        self._b1 = rng.uniform(0, 2 * np.pi, 32)
        self._w_affinity = rng.standard_normal(32)
        self._W2 = rng.standard_normal((32, n_var)) * 1.5
        self._b2 = rng.uniform(0, 2 * np.pi, 32)
        self._w_tox = rng.standard_normal(32)

    @property
    def n_objectives(self) -> int:
        return 2

    @property
    def input_dim(self) -> int:
        return self._n_var

    def _rff_value(self, X, W, b, w) -> float:
        z = np.sqrt(2.0 / len(b)) * np.cos(W @ X + b)
        return float(w @ z)

    def _true_values(self, X: np.ndarray) -> np.ndarray:
        x = X[: self._n_var]
        affinity = self._rff_value(x, self._W1, self._b1, self._w_affinity)
        tox_raw = self._rff_value(x, self._W2, self._b2, self._w_tox)
        # Normalise to [0, 1] approximately
        affinity_norm = 0.5 + 0.4 * np.tanh(affinity)
        safety_norm = 0.5 - 0.4 * np.tanh(tox_raw)  # high = safe
        return np.array([affinity_norm, safety_norm])

    def _constraint_check(self, X: np.ndarray) -> tuple[bool, Dict[str, float]]:
        x = X[: self._n_var]
        # Hard: molecular weight proxy (sum of abs features)
        mw_proxy = float(np.sum(np.abs(x)))
        hard_valid = mw_proxy < 8.0  # Lipinski-style cutoff
        # Soft: synthesis cost proxy
        complexity = float(np.std(x) * np.mean(np.abs(x)))
        synthesis_risk = float(np.clip(complexity / 3.0, 0.0, 1.0))
        return hard_valid, {"synthesis_cost": synthesis_risk}

    def _cost(self, X: np.ndarray) -> float:
        x = X[: self._n_var]
        base_cost = 1.0 + 0.5 * float(np.std(x))
        return round(max(0.5, base_cost), 3)


class HighConstraintBenchmark(BaseIVO):
    """
    Tri-objective benchmark with a small feasible region (~15% of space).
    Stress-tests the Constraint Intelligence Layer.

    Objectives: potency, selectivity, metabolic stability (all maximised).
    Hard constraints: 3 independent structural filters.
    Soft constraints: reagent scarcity (Poisson depletion model).
    """

    def __init__(
        self,
        n_var: int = 6,
        noise_std: float = 0.04,
        seed: Optional[int] = None,
        reagent_stock: float = 100.0,
    ):
        super().__init__(noise_std=noise_std, seed=seed)
        self._n_var = n_var
        self._reagent_stock = reagent_stock
        self._reagent_consumed = 0.0
        rng = np.random.default_rng(seed if seed is not None else 1)
        self._centres = rng.uniform(-1, 1, (3, n_var))
        self._scales = rng.uniform(0.5, 1.5, 3)

    @property
    def n_objectives(self) -> int:
        return 3

    @property
    def input_dim(self) -> int:
        return self._n_var

    def _true_values(self, X: np.ndarray) -> np.ndarray:
        x = X[: self._n_var]
        vals = []
        for c, s in zip(self._centres, self._scales):
            v = np.exp(-np.sum((x - c) ** 2) / (2 * s ** 2))
            vals.append(float(v))
        return np.array(vals)

    def _constraint_check(self, X: np.ndarray) -> tuple[bool, Dict[str, float]]:
        x = X[: self._n_var]
        # Hard: 3 independent structural filters
        c1 = float(np.sum(x ** 2)) < 4.0          # within sphere
        c2 = float(x[0]) > -1.5                    # fragment constraint
        c3 = float(np.max(np.abs(x))) < 2.5        # bounding box
        hard_valid = bool(c1 and c2 and c3)

        # Soft: reagent scarcity — risk increases as stock depletes
        depletion_ratio = self._reagent_consumed / max(self._reagent_stock, 1.0)
        reagent_risk = float(np.clip(depletion_ratio ** 2, 0.0, 0.95))

        return hard_valid, {"reagent_scarcity": reagent_risk}

    def _cost(self, X: np.ndarray) -> float:
        x = X[: self._n_var]
        cost = 1.5 + float(np.linalg.norm(x)) * 0.3
        # Deplete reagent
        self._reagent_consumed += self.rng.poisson(lam=cost * 0.5)
        return round(cost, 3)


IVO_REGISTRY = {
    "vlmop2": VLMOP2Benchmark,
    "molecular": MolecularProxyBenchmark,
    "high_constraint": HighConstraintBenchmark,
}


def create_ivo(benchmark: str, n_var: int = 5, seed: Optional[int] = None, **kwargs) -> BaseIVO:
    if benchmark not in IVO_REGISTRY:
        raise ValueError(f"Unknown benchmark '{benchmark}'. Choose from: {list(IVO_REGISTRY)}")
    return IVO_REGISTRY[benchmark](n_var=n_var, seed=seed, **kwargs)
