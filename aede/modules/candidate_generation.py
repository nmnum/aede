"""
Module 1: Candidate Generation Layer
======================================
Three acquisition strategies producing a diverse candidate pool each iteration.
Portfolio weights adapt based on each strategy's recent Pareto contribution.

Strategies:
    BOStrategy      — qNEHVI approximation via MC sampling on GP posterior
    NSGAStrategy    — NSGA-II evolutionary search (pure numpy)
    DiversityStrategy — MaxMin distance sampling

Each candidate carries a provenance tag — critical for the Insight Engine.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod


@dataclass
class Candidate:
    id: str
    X: np.ndarray
    provenance: str          # "bo", "nsga", "diversity"
    predicted_objectives: Optional[np.ndarray] = None
    predicted_uncertainty: Optional[np.ndarray] = None
    acquisition_value: float = 0.0


# ── Hypervolume utilities ──────────────────────────────────────────────────────

def _hv_2d(points: np.ndarray, ref: np.ndarray) -> float:
    pts = np.array([p for p in points if np.all(p >= ref)])
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(-pts[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for p in pts:
        if p[1] > prev_y:
            hv += (p[0] - ref[0]) * (p[1] - prev_y)
            prev_y = p[1]
    return hv


def compute_hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
    """Exact HV for 2 objectives; approximate sweep for 3."""
    if len(points) == 0:
        return 0.0
    pts = np.array(points)
    n_obj = pts.shape[1]
    if n_obj == 2:
        return _hv_2d(pts, ref)
    elif n_obj == 3:
        # Slice algorithm for 3D
        valid = pts[np.all(pts >= ref, axis=1)]
        if len(valid) == 0:
            return 0.0
        valid = valid[np.argsort(-valid[:, 0])]
        hv = 0.0
        for i, p in enumerate(valid):
            slice_pts = valid[:i][:, 1:]
            slice_hv = _hv_2d(slice_pts, ref[1:]) if len(slice_pts) > 0 else 0.0
            full_pts = np.vstack([valid[:i], p])[:, 1:]
            full_hv = _hv_2d(full_pts, ref[1:])
            x_width = p[0] - (valid[i + 1][0] if i + 1 < len(valid) else ref[0])
            hv += max(0, x_width) * (full_hv - slice_hv)
        return hv
    else:
        raise ValueError(f"HV computation not supported for {n_obj} objectives. Use 2 or 3.")


def fast_non_dominated_sort(costs: np.ndarray) -> List[List[int]]:
    """NSGA-style non-dominated sort (minimisation convention)."""
    n = len(costs)
    dominated_by = [[] for _ in range(n)]
    n_dominating = np.zeros(n, dtype=int)
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if np.all(costs[p] <= costs[q]) and np.any(costs[p] < costs[q]):
                dominated_by[p].append(q)
            elif np.all(costs[q] <= costs[p]) and np.any(costs[q] < costs[p]):
                n_dominating[p] += 1
        if n_dominating[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                n_dominating[q] -= 1
                if n_dominating[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return [f for f in fronts if f]


def crowding_distance(costs: np.ndarray, front: List[int]) -> np.ndarray:
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)
    dists = np.zeros(n)
    for obj in range(costs.shape[1]):
        order = np.argsort(costs[front, obj])
        obj_range = costs[front[order[-1]], obj] - costs[front[order[0]], obj]
        if obj_range == 0:
            continue
        dists[order[0]] = dists[order[-1]] = np.inf
        for k in range(1, n - 1):
            dists[order[k]] += (
                costs[front[order[k + 1]], obj] - costs[front[order[k - 1]], obj]
            ) / obj_range
    return dists


# ── Acquisition Strategies ────────────────────────────────────────────────────

class BaseStrategy(ABC):
    def __init__(self, bounds: np.ndarray, rng: np.random.Generator, n_var: int):
        self.bounds = bounds  # (n_var, 2) — [low, high] per dimension
        self.rng = rng
        self.n_var = n_var
        self.total_contribution = 0.0
        self.recent_contribution = 0.0  # EMA of HV contribution

    @abstractmethod
    def propose(self, n: int, surrogate, observations: np.ndarray, pareto_front: np.ndarray) -> List[Candidate]:
        ...

    def update_contribution(self, hv_improvement: float, ema_alpha: float = 0.3):
        self.recent_contribution = (
            ema_alpha * hv_improvement + (1 - ema_alpha) * self.recent_contribution
        )
        self.total_contribution += hv_improvement

    def _random_in_bounds(self, n: int) -> np.ndarray:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return self.rng.uniform(lo, hi, size=(n, self.n_var))


class BOStrategy(BaseStrategy):
    """
    qNEHVI approximation: MC sampling from GP posterior,
    computing expected hypervolume improvement per candidate.
    """

    def __init__(self, bounds, rng, n_var, n_mc: int = 128, n_candidates: int = 512):
        super().__init__(bounds, rng, n_var)
        self.n_mc = n_mc
        self.n_candidates = n_candidates

    def propose(self, n: int, surrogate, observations: np.ndarray, pareto_front: np.ndarray) -> List[Candidate]:
        if surrogate is None or not surrogate._is_fit:
            return self._random_fallback(n, surrogate)

        ref = np.zeros(surrogate.n_objectives)
        hv_base = compute_hypervolume(pareto_front, ref) if len(pareto_front) > 0 else 0.0

        # Generate random candidates and score via MC
        candidates_X = self._random_in_bounds(self.n_candidates)
        # Get posterior samples: (n_mc, n_candidates, n_obj)
        try:
            samples = surrogate.sample_posterior(candidates_X, n_samples=self.n_mc)
        except Exception:
            return self._random_fallback(n, surrogate)

        scores = np.zeros(self.n_candidates)
        for j in range(self.n_candidates):
            ei_sum = 0.0
            for s in range(self.n_mc):
                candidate_val = samples[s, j, :]
                aug = (
                    np.vstack([pareto_front, candidate_val])
                    if len(pareto_front) > 0
                    else candidate_val.reshape(1, -1)
                )
                hv_new = compute_hypervolume(aug, ref)
                ei_sum += max(0.0, hv_new - hv_base)
            scores[j] = ei_sum / self.n_mc

        top_idx = np.argsort(-scores)[:n]
        mu, sigma = surrogate.predict(candidates_X[top_idx])

        return [
            Candidate(
                id=f"bo_{i:04d}",
                X=candidates_X[idx],
                provenance="bo",
                predicted_objectives=mu[k],
                predicted_uncertainty=sigma[k],
                acquisition_value=float(scores[idx]),
            )
            for k, (i, idx) in enumerate(enumerate(top_idx))
        ]

    def _random_fallback(self, n: int, surrogate) -> List[Candidate]:
        Xs = self._random_in_bounds(n)
        if surrogate is not None and surrogate._is_fit:
            mu, sigma = surrogate.predict(Xs)
        else:
            mu = sigma = None
        return [
            Candidate(
                id=f"bo_rand_{i:04d}",
                X=Xs[i],
                provenance="bo",
                predicted_objectives=mu[i] if mu is not None else None,
                predicted_uncertainty=sigma[i] if sigma is not None else None,
                acquisition_value=0.0,
            )
            for i in range(n)
        ]


class NSGAStrategy(BaseStrategy):
    """
    NSGA-II evolutionary search in pure numpy.
    Maintains a population across iterations for warm-starting.
    """

    def __init__(self, bounds, rng, n_var, pop_size: int = 100, n_gen: int = 20):
        super().__init__(bounds, rng, n_var)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self._population: Optional[np.ndarray] = None

    def propose(self, n: int, surrogate, observations: np.ndarray, pareto_front: np.ndarray) -> List[Candidate]:
        if surrogate is None or not surrogate._is_fit:
            Xs = self._random_in_bounds(n)
            return [Candidate(id=f"nsga_{i:04d}", X=x, provenance="nsga") for i, x in enumerate(Xs)]

        # Initialise or warm-start population
        if self._population is None or len(self._population) < self.pop_size:
            self._population = self._random_in_bounds(self.pop_size)

        for _ in range(self.n_gen):
            self._population = self._evolve(self._population, surrogate)

        # Return best non-dominated from population
        mu, sigma = surrogate.predict(self._population)
        costs = -mu  # convert to minimisation for NSGA
        fronts = fast_non_dominated_sort(costs)
        top = fronts[0][:n] if len(fronts[0]) >= n else fronts[0]
        # Fill with second front if needed
        if len(top) < n and len(fronts) > 1:
            top = top + fronts[1][: n - len(top)]

        result = []
        for k, idx in enumerate(top[:n]):
            result.append(Candidate(
                id=f"nsga_{k:04d}",
                X=self._population[idx].copy(),
                provenance="nsga",
                predicted_objectives=mu[idx],
                predicted_uncertainty=sigma[idx],
                acquisition_value=float(-np.mean(costs[idx])),
            ))
        return result

    def _evolve(self, pop: np.ndarray, surrogate) -> np.ndarray:
        mu, _ = surrogate.predict(pop)
        costs = -mu
        fronts = fast_non_dominated_sort(costs)

        # Tournament selection
        parents = self._tournament_select(pop, costs, fronts)

        # SBX crossover + polynomial mutation
        offspring = self._sbx_crossover(parents)
        offspring = self._poly_mutation(offspring)
        offspring = np.clip(offspring, self.bounds[:, 0], self.bounds[:, 1])

        # Combine and select next generation
        combined = np.vstack([pop, offspring])
        mu_all, _ = surrogate.predict(combined)
        costs_all = -mu_all
        fronts_all = fast_non_dominated_sort(costs_all)

        next_pop = []
        for front in fronts_all:
            if len(next_pop) + len(front) <= self.pop_size:
                next_pop.extend(front)
            else:
                needed = self.pop_size - len(next_pop)
                dists = crowding_distance(costs_all, front)
                selected = np.argsort(-dists)[:needed]
                next_pop.extend([front[s] for s in selected])
                break
        return combined[next_pop]

    def _tournament_select(self, pop, costs, fronts) -> np.ndarray:
        rank = np.zeros(len(pop), dtype=int)
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r
        parents = []
        for _ in range(len(pop)):
            a, b = self.rng.integers(0, len(pop), size=2)
            parents.append(a if rank[a] < rank[b] else b)
        return pop[parents]

    def _sbx_crossover(self, parents: np.ndarray, eta: float = 15.0) -> np.ndarray:
        n = len(parents)
        offspring = parents.copy()
        for i in range(0, n - 1, 2):
            if self.rng.random() < 0.9:
                for d in range(self.n_var):
                    u = self.rng.random()
                    beta = (2 * u) ** (1 / (eta + 1)) if u < 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    offspring[i, d] = 0.5 * ((1 + beta) * parents[i, d] + (1 - beta) * parents[i + 1, d])
                    offspring[i + 1, d] = 0.5 * ((1 - beta) * parents[i, d] + (1 + beta) * parents[i + 1, d])
        return offspring

    def _poly_mutation(self, pop: np.ndarray, eta: float = 20.0, prob: float = 0.1) -> np.ndarray:
        result = pop.copy()
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        for i in range(len(pop)):
            for d in range(self.n_var):
                if self.rng.random() < prob:
                    u = self.rng.random()
                    delta = min(result[i, d] - lo[d], hi[d] - result[i, d]) / (hi[d] - lo[d] + 1e-9)
                    if u < 0.5:
                        val = (2 * u + (1 - 2 * u) * (1 - delta) ** (eta + 1)) ** (1 / (eta + 1)) - 1
                    else:
                        val = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta) ** (eta + 1)) ** (1 / (eta + 1))
                    result[i, d] = np.clip(result[i, d] + val * (hi[d] - lo[d]), lo[d], hi[d])
        return result


class DiversityStrategy(BaseStrategy):
    """
    MaxMin distance sampling — always active to prevent clustering.
    Maximises minimum distance from all existing observations.
    """

    def propose(self, n: int, surrogate, observations: np.ndarray, pareto_front: np.ndarray) -> List[Candidate]:
        n_candidates = max(n * 10, 200)
        candidates_X = self._random_in_bounds(n_candidates)

        if observations is None or len(observations) == 0:
            selected_X = candidates_X[:n]
        else:
            obs_X = observations
            selected_X = self._maxmin_select(candidates_X, obs_X, n)

        result = []
        if surrogate is not None and surrogate._is_fit:
            mu, sigma = surrogate.predict(selected_X)
        else:
            mu = sigma = None

        for i, x in enumerate(selected_X):
            result.append(Candidate(
                id=f"div_{i:04d}",
                X=x,
                provenance="diversity",
                predicted_objectives=mu[i] if mu is not None else None,
                predicted_uncertainty=sigma[i] if sigma is not None else None,
                acquisition_value=0.0,
            ))
        return result

    def _maxmin_select(self, candidates: np.ndarray, existing: np.ndarray, n: int) -> np.ndarray:
        selected = []
        remaining = list(range(len(candidates)))
        ref_points = existing.tolist()

        for _ in range(n):
            if not remaining:
                break
            ref_arr = np.array(ref_points)
            min_dists = np.array([
                np.min(np.linalg.norm(candidates[j] - ref_arr, axis=1))
                for j in remaining
            ])
            best = remaining[np.argmax(min_dists)]
            selected.append(best)
            ref_points.append(candidates[best].tolist())
            remaining.remove(best)

        return candidates[selected]


# ── Portfolio Manager ─────────────────────────────────────────────────────────

class CandidateGenerator:
    """
    Orchestrates all strategies with adaptive portfolio weighting.
    Weight updates are based on each strategy's EMA Pareto contribution.
    """

    def __init__(
        self,
        n_var: int,
        bounds: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        bo_fraction: float = 0.5,
        nsga_fraction: float = 0.3,
        diversity_fraction: float = 0.2,
    ):
        self.n_var = n_var
        self.rng = np.random.default_rng(seed)
        if bounds is None:
            bounds = np.column_stack([np.full(n_var, -3.0), np.full(n_var, 3.0)])
        self.bounds = bounds
        self._fractions = np.array([bo_fraction, nsga_fraction, diversity_fraction])
        self._fractions /= self._fractions.sum()

        self.strategies = {
            "bo": BOStrategy(bounds, self.rng, n_var),
            "nsga": NSGAStrategy(bounds, self.rng, n_var),
            "diversity": DiversityStrategy(bounds, self.rng, n_var),
        }
        self.weight_history: List[dict] = []

    def generate(
        self,
        pool_size: int,
        surrogate,
        observations_X: Optional[np.ndarray],
        pareto_front: np.ndarray,
    ) -> List[Candidate]:
        counts = self._allocate(pool_size)
        all_candidates = []
        for (name, strategy), n in zip(self.strategies.items(), counts):
            if n > 0:
                try:
                    cands = strategy.propose(n, surrogate, observations_X, pareto_front)
                    all_candidates.extend(cands)
                except Exception as e:
                    # Graceful degradation: fall back to random for this strategy
                    Xs = strategy._random_in_bounds(n)
                    all_candidates.extend([
                        Candidate(id=f"{name}_fallback_{i}", X=x, provenance=name)
                        for i, x in enumerate(Xs)
                    ])

        self.weight_history.append(dict(zip(self.strategies.keys(), self._fractions.tolist())))
        return all_candidates

    def update_weights(self, hv_improvements: dict, ema_alpha: float = 0.3):
        """Update portfolio weights based on per-strategy HV contribution."""
        for name, strategy in self.strategies.items():
            improvement = hv_improvements.get(name, 0.0)
            strategy.update_contribution(improvement, ema_alpha)

        contributions = np.array([
            max(s.recent_contribution, 1e-6) for s in self.strategies.values()
        ])
        # Smooth towards uniform with a floor to prevent any strategy dying
        new_fracs = contributions / contributions.sum()
        floor = 0.1
        new_fracs = np.maximum(new_fracs, floor)
        self._fractions = new_fracs / new_fracs.sum()

    def _allocate(self, total: int) -> List[int]:
        counts = [int(np.floor(f * total)) for f in self._fractions]
        remainder = total - sum(counts)
        counts[0] += remainder
        return counts

    def current_weights(self) -> dict:
        return dict(zip(self.strategies.keys(), self._fractions.tolist()))
