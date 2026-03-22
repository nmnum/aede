"""
AEDE Test Suite
================
Run with: python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from aede.modules.problem_framing import (
    ProblemSpec, Objective, BudgetSpec, StoppingCriterion,
    HardConstraint, SoftConstraint, load_spec_from_dict,
)
from aede.modules.ivo_layer import VLMOP2Benchmark, MolecularProxyBenchmark, HighConstraintBenchmark, create_ivo
from aede.modules.surrogate import GPSurrogate
from aede.modules.candidate_generation import (
    CandidateGenerator, compute_hypervolume, fast_non_dominated_sort,
    BOStrategy, NSGAStrategy, DiversityStrategy,
)
from aede.modules.constraint_layer import ConstraintLayer
from aede.modules.decision_engine import DecisionEngine
from aede.modules.trace_logger import DecisionTraceLogger
from aede.modules.insight_engine import InsightEngine
from aede import run_campaign


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def basic_spec():
    return ProblemSpec(
        name="test",
        objectives=[Objective("obj1", "maximize"), Objective("obj2", "maximize")],
        budget=BudgetSpec(experiments=30, per_iteration_cap=5),
        stopping_criterion=StoppingCriterion(eig_per_cost_threshold=0.001, min_experiments=5),
        benchmark="vlmop2",
        n_input_dims=2,
        candidate_pool_size=20,
        random_seed=0,
    )


@pytest.fixture
def fitted_surrogate():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (30, 3))
    Y = np.column_stack([np.sin(X[:, 0]), np.cos(X[:, 1])])
    surrogate = GPSurrogate(n_objectives=2, n_restarts=1, seed=0)
    surrogate.fit(X, Y)
    return surrogate, X, Y


# ─── Module 0: Problem Framing ────────────────────────────────────────────────

class TestProblemFraming:

    def test_valid_spec_creates(self, basic_spec):
        assert basic_spec.n_objectives() == 2
        assert len(basic_spec.config_hash()) == 16

    def test_requires_two_objectives(self):
        with pytest.raises(AssertionError):
            ProblemSpec(objectives=[Objective("only_one", "maximize")])

    def test_constraint_overlap_rejected(self):
        with pytest.raises(AssertionError):
            ProblemSpec(
                objectives=[Objective("a", "maximize"), Objective("b", "maximize")],
                hard_constraints=[HardConstraint(property="synthesis_cost")],
                soft_constraints=[SoftConstraint(property="synthesis_cost")],
            )

    def test_config_hash_deterministic(self, basic_spec):
        assert basic_spec.config_hash() == basic_spec.config_hash()

    def test_load_from_dict(self):
        data = {
            "name": "from_dict",
            "objectives": [
                {"name": "f1", "direction": "maximize"},
                {"name": "f2", "direction": "minimize"},
            ],
            "budget": {"experiments": 50, "per_iteration_cap": 5},
            "benchmark": "vlmop2",
            "n_input_dims": 2,
        }
        spec = load_spec_from_dict(data)
        assert spec.n_objectives() == 2
        assert spec.budget.experiments == 50


# ─── Module 4: IVO Layer ──────────────────────────────────────────────────────

class TestIVOLayer:

    def test_vlmop2_shape(self):
        ivo = VLMOP2Benchmark(n_var=2, seed=0)
        obs = ivo.observe(np.array([0.5, 0.3]))
        assert obs.objectives.shape == (2,)
        assert obs.constraint_valid is True

    def test_vlmop2_batch(self):
        ivo = VLMOP2Benchmark(n_var=2, seed=0)
        X = np.random.randn(5, 2)
        obs_list = ivo.observe_batch(X)
        assert len(obs_list) == 5

    def test_molecular_constraints(self):
        ivo = MolecularProxyBenchmark(n_var=5, seed=0)
        # Very large features should violate molecular weight constraint
        X_large = np.ones(5) * 3.0  # sum(|x|) = 15 > 8
        obs = ivo.observe(X_large)
        assert obs.constraint_valid is False

    def test_high_constraint_triobj(self):
        ivo = HighConstraintBenchmark(n_var=6, seed=0)
        obs = ivo.observe(np.zeros(6))
        assert obs.objectives.shape == (3,)

    def test_create_ivo_registry(self):
        for name in ["vlmop2", "molecular", "high_constraint"]:
            ivo = create_ivo(name, n_var=3, seed=0)
            assert ivo is not None

    def test_ivo_noise_nonzero(self):
        ivo = VLMOP2Benchmark(n_var=2, noise_std=0.1, seed=42)
        x = np.array([0.1, 0.2])
        obs1 = ivo.observe(x)
        obs2 = ivo.observe(x)
        # With noise, two observations at same point should differ
        assert not np.allclose(obs1.objectives, obs2.objectives)


# ─── Hypervolume ──────────────────────────────────────────────────────────────

class TestHypervolume:

    def test_hv_2d_known_value(self):
        pf = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
        ref = np.zeros(2)
        hv = compute_hypervolume(pf, ref)
        assert abs(hv - 0.33) < 0.01

    def test_hv_empty_returns_zero(self):
        assert compute_hypervolume(np.empty((0, 2)), np.zeros(2)) == 0.0

    def test_hv_single_point(self):
        pf = np.array([[0.5, 0.5]])
        ref = np.zeros(2)
        hv = compute_hypervolume(pf, ref)
        assert abs(hv - 0.25) < 0.01

    def test_pareto_sort(self):
        costs = np.array([[1, 4], [2, 3], [3, 2], [4, 1], [2, 2], [3, 3]])
        fronts = fast_non_dominated_sort(costs)
        assert len(fronts) >= 2
        # [2,2] should not be on first front — it's dominated by [1,4]? No.
        # Actually [2,2] dominates [3,3] and [2,3] dominates nothing except [3,3]
        # First front should include [1,4],[2,3],[3,2],[4,1] — all non-dominated
        first_front_costs = costs[fronts[0]]
        # Verify none in first front is dominated by another
        for i, p in enumerate(first_front_costs):
            for j, q in enumerate(first_front_costs):
                if i != j:
                    assert not (np.all(q <= p) and np.any(q < p)), f"{q} dominates {p}"


# ─── Module 1: Candidate Generation ──────────────────────────────────────────

class TestCandidateGeneration:

    def test_generator_produces_correct_count(self, fitted_surrogate):
        surrogate, X, Y = fitted_surrogate
        bounds = np.column_stack([np.full(3, -3.0), np.full(3, 3.0)])
        gen = CandidateGenerator(n_var=3, bounds=bounds, seed=0)
        candidates = gen.generate(20, surrogate, X, np.array([[0.5, 0.3]]))
        assert len(candidates) == 20

    def test_provenance_tags_present(self, fitted_surrogate):
        surrogate, X, Y = fitted_surrogate
        bounds = np.column_stack([np.full(3, -3.0), np.full(3, 3.0)])
        gen = CandidateGenerator(n_var=3, bounds=bounds, seed=0)
        candidates = gen.generate(15, surrogate, X, np.array([[0.5, 0.3]]))
        provenances = {c.provenance for c in candidates}
        assert provenances.issubset({"bo", "nsga", "diversity"})

    def test_weight_update_shifts_fractions(self, fitted_surrogate):
        surrogate, X, Y = fitted_surrogate
        bounds = np.column_stack([np.full(3, -3.0), np.full(3, 3.0)])
        gen = CandidateGenerator(n_var=3, bounds=bounds, seed=0)
        # Give all improvement to BO
        gen.update_weights({"bo": 1.0, "nsga": 0.0, "diversity": 0.0})
        w = gen.current_weights()
        assert w["bo"] > w["nsga"]
        assert w["bo"] > w["diversity"]


# ─── Module 2: Constraint Layer ───────────────────────────────────────────────

class TestConstraintLayer:

    def test_hard_constraint_filters(self, basic_spec):
        spec = ProblemSpec(
            name="test",
            objectives=[Objective("a", "maximize"), Objective("b", "maximize")],
            hard_constraints=[HardConstraint(property="molecular_weight", max=5.0)],
            budget=BudgetSpec(experiments=30, per_iteration_cap=5),
            benchmark="vlmop2",
            n_input_dims=2,
        )
        from aede.modules.candidate_generation import Candidate
        layer = ConstraintLayer(spec)
        # Large features -> MW proxy = sum(|x|) = 10 > 5 -> filtered
        bad_cand = Candidate("bad", np.array([3.0, 2.0, 1.0, 1.0, 1.0, 1.0]), "bo")
        filtered, results = layer.filter([bad_cand], attempt_repair=False)
        assert len(filtered) == 0
        assert results[0].passes_hard is False

    def test_feasibility_model_updates(self, basic_spec):
        layer = ConstraintLayer(basic_spec)
        X = np.random.randn(20, 2)
        flags = np.array([1] * 12 + [0] * 8)
        layer.update(X, flags)
        assert layer.n_observations == 20
        assert 0 < layer.violation_rate < 1

    def test_gp_classifier_predicts_after_warmup(self, basic_spec):
        layer = ConstraintLayer(basic_spec)
        X = np.random.randn(15, 2)
        flags = np.array([1] * 8 + [0] * 7)
        layer.update(X, flags)
        probs = layer.predict_feasibility(np.random.randn(5, 2))
        assert probs.shape == (5,)
        assert np.all((probs >= 0) & (probs <= 1))


# ─── Module 3: Decision Engine ────────────────────────────────────────────────

class TestDecisionEngine:

    def test_scoring_produces_records(self, basic_spec, fitted_surrogate):
        surrogate, X, Y = fitted_surrogate
        from aede.modules.candidate_generation import Candidate
        engine = DecisionEngine(basic_spec)
        candidates = [
            Candidate(f"c{i}", X[i], "bo",
                      predicted_objectives=Y[i],
                      predicted_uncertainty=np.array([0.1, 0.1]))
            for i in range(5)
        ]
        pf = np.array([[0.3, 0.4], [0.5, 0.2]])
        scored = engine.score_candidates(candidates, [], pf, surrogate, n_mc=16)
        assert len(scored) == 5
        assert all(r.rank >= 1 for r in scored)
        # Scores should be sorted descending
        scores = [r.decision_score for r in scored]
        assert scores == sorted(scores, reverse=True)

    def test_stopping_criterion_fires(self, basic_spec):
        engine = DecisionEngine(basic_spec)
        engine._experiments_run = 20
        from aede.modules.decision_engine import DecisionRecord
        dummy = DecisionRecord("x", np.zeros(2), "bo", 0.00001, 0, 0, 0.5, 1.0, "", 1)
        for _ in range(3):
            should_stop, _ = engine.check_stopping([dummy], budget_remaining=50)
        assert should_stop is True

    def test_stopping_respects_min_experiments(self, basic_spec):
        engine = DecisionEngine(basic_spec)
        engine._experiments_run = 2  # below min_experiments=5
        from aede.modules.decision_engine import DecisionRecord
        dummy = DecisionRecord("x", np.zeros(2), "bo", 0.00001, 0, 0, 0.5, 1.0, "", 1)
        should_stop, _ = engine.check_stopping([dummy], budget_remaining=50)
        assert should_stop is False


# ─── Module 5: Trace Logger ───────────────────────────────────────────────────

class TestTraceLogger:

    def test_register_and_retrieve_campaign(self):
        logger = DecisionTraceLogger(":memory:")
        logger.register_campaign("abc123", "test_campaign", {"key": "val"})
        camps = logger.get_all_campaigns()
        assert len(camps) == 1
        assert camps[0]["name"] == "test_campaign"

    def test_log_iteration_retrievable(self, basic_spec):
        from aede.modules.decision_engine import DecisionManifest, DecisionRecord
        logger = DecisionTraceLogger(":memory:")
        logger.register_campaign("hash1", "test", {})
        selected = [DecisionRecord("c1", np.zeros(2), "bo", 0.5, 0.1, 0.8, 0.9, 1.0, "reason", 1)]
        manifest = DecisionManifest(
            iteration=1,
            selected=selected,
            all_scored=selected,
            should_stop=False,
            stop_reason="",
            eig_per_cost=0.05,
            pareto_hypervolume=0.3,
            budget_remaining=90,
        )
        logger.log_iteration("hash1", manifest)
        hv = logger.get_hv_history("hash1")
        assert len(hv) == 1
        assert abs(hv[0]["hv"] - 0.3) < 1e-6

    def test_append_only_immutability(self):
        logger = DecisionTraceLogger(":memory:")
        logger.register_campaign("h1", "c1", {})
        logger.register_campaign("h1", "c1", {})  # Duplicate — should not raise, just skip
        camps = logger.get_all_campaigns()
        assert len(camps) == 1


# ─── Full Pipeline Integration ────────────────────────────────────────────────

class TestFullPipeline:

    def test_vlmop2_end_to_end(self):
        spec = ProblemSpec(
            name="integration_test",
            objectives=[Objective("f1", "maximize"), Objective("f2", "maximize")],
            budget=BudgetSpec(experiments=25, per_iteration_cap=4),
            stopping_criterion=StoppingCriterion(eig_per_cost_threshold=0.0001, min_experiments=5),
            benchmark="vlmop2",
            n_input_dims=2,
            candidate_pool_size=20,
            random_seed=1,
        )
        results = run_campaign(spec, db_path=":memory:", verbose=False, n_mc=16)
        assert results["n_experiments"] > 0
        assert len(results["pareto_front"]) > 0
        assert len(results["hv_history"]) > 0
        assert results["hv_history"][-1] > 0

    def test_molecular_end_to_end(self):
        spec = ProblemSpec(
            name="molecular_test",
            objectives=[Objective("affinity", "maximize"), Objective("safety", "maximize")],
            hard_constraints=[HardConstraint(property="molecular_weight", max=8.0)],
            budget=BudgetSpec(experiments=20, per_iteration_cap=4),
            stopping_criterion=StoppingCriterion(eig_per_cost_threshold=0.0001, min_experiments=5),
            benchmark="molecular",
            n_input_dims=5,
            candidate_pool_size=20,
            random_seed=2,
        )
        results = run_campaign(spec, db_path=":memory:", verbose=False, n_mc=16)
        assert results["n_experiments"] > 0

    def test_insight_report_generated(self):
        spec = ProblemSpec(
            name="insight_test",
            objectives=[Objective("f1", "maximize"), Objective("f2", "maximize")],
            budget=BudgetSpec(experiments=20, per_iteration_cap=4),
            stopping_criterion=StoppingCriterion(eig_per_cost_threshold=0.0001, min_experiments=5),
            benchmark="vlmop2",
            n_input_dims=2,
            candidate_pool_size=15,
            random_seed=3,
        )
        results = run_campaign(spec, db_path=":memory:", verbose=False, n_mc=8)
        insight = results["insight_report"]
        assert insight is not None
        assert len(insight.key_findings) > 0
        assert insight.cost_efficiency.total_experiments > 0

    def test_hv_monotonically_nondecreasing(self):
        """Pareto hypervolume should never decrease."""
        spec = ProblemSpec(
            name="hv_test",
            objectives=[Objective("f1", "maximize"), Objective("f2", "maximize")],
            budget=BudgetSpec(experiments=30, per_iteration_cap=4),
            stopping_criterion=StoppingCriterion(eig_per_cost_threshold=0.0001, min_experiments=5),
            benchmark="vlmop2",
            n_input_dims=2,
            candidate_pool_size=20,
            random_seed=5,
        )
        results = run_campaign(spec, db_path=":memory:", verbose=False, n_mc=8)
        hv = results["hv_history"]
        for i in range(1, len(hv)):
            assert hv[i] >= hv[i - 1] - 1e-9, f"HV decreased at iteration {i}: {hv[i-1]} -> {hv[i]}"

    def test_stopping_before_budget_exhaustion(self):
        """With a high threshold, the campaign should stop early."""
        spec = ProblemSpec(
            name="stop_test",
            objectives=[Objective("f1", "maximize"), Objective("f2", "maximize")],
            budget=BudgetSpec(experiments=200, per_iteration_cap=5),
            stopping_criterion=StoppingCriterion(
                eig_per_cost_threshold=0.1,  # Very high — will trigger early
                consecutive_iterations=2,
                min_experiments=10,
            ),
            benchmark="vlmop2",
            n_input_dims=2,
            candidate_pool_size=20,
            random_seed=6,
        )
        results = run_campaign(spec, db_path=":memory:", verbose=False, n_mc=8)
        assert results["n_experiments"] < 200, "Expected early stopping but budget was exhausted"
