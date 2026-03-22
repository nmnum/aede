"""
AEDE Test Runner
=================
Works without pytest. Run: python tests/run_tests.py

When pytest is installed (it will be on GitHub CI): python -m pytest tests/test_aede.py -v
"""

import sys
import os
import warnings
import traceback

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

passed = []
failed = []


def ok(name):
    passed.append(name)
    print(f"  PASS  {name}")


def fail(name, e):
    failed.append(name)
    print(f"  FAIL  {name}: {e}")


def run(name, fn):
    try:
        fn()
        ok(name)
    except Exception as e:
        fail(name, e)


# ── Imports ───────────────────────────────────────────────────────────────────
from aede.modules.problem_framing import (
    ProblemSpec, Objective, BudgetSpec, StoppingCriterion,
    HardConstraint, SoftConstraint, load_spec_from_dict,
)
from aede.modules.ivo_layer import (
    VLMOP2Benchmark, MolecularProxyBenchmark, HighConstraintBenchmark, create_ivo,
)
from aede.modules.surrogate import GPSurrogate
from aede.modules.candidate_generation import (
    CandidateGenerator, compute_hypervolume, fast_non_dominated_sort, Candidate,
)
from aede.modules.constraint_layer import ConstraintLayer
from aede.modules.decision_engine import DecisionEngine, DecisionRecord, DecisionManifest
from aede.modules.trace_logger import DecisionTraceLogger
from aede import run_campaign

print("=" * 60)
print("AEDE Test Suite")
print("=" * 60)

# ── Problem Framing ───────────────────────────────────────────────────────────
print("\n── Module 0: Problem Framing ──")

spec = ProblemSpec(
    name="test",
    objectives=[Objective("obj1", "maximize"), Objective("obj2", "maximize")],
    budget=BudgetSpec(experiments=30, per_iteration_cap=5),
    stopping_criterion=StoppingCriterion(eig_per_cost_threshold=0.001, min_experiments=5),
    benchmark="vlmop2",
    n_input_dims=2,
    candidate_pool_size=20,
    random_seed=0,
)


def test_framing_valid():
    assert spec.n_objectives() == 2
    h = spec.config_hash()
    assert len(h) == 16
    assert h == spec.config_hash()  # deterministic


def test_framing_one_obj():
    try:
        ProblemSpec(objectives=[Objective("only_one", "maximize")])
        raise AssertionError("Should have raised for single objective")
    except AssertionError as e:
        if "Should have raised" in str(e):
            raise


def test_framing_overlap():
    try:
        ProblemSpec(
            objectives=[Objective("a", "maximize"), Objective("b", "maximize")],
            hard_constraints=[HardConstraint(property="synthesis_cost")],
            soft_constraints=[SoftConstraint(property="synthesis_cost")],
        )
        raise AssertionError("Should have raised for overlap")
    except AssertionError as e:
        if "Should have raised" in str(e):
            raise


def test_framing_from_dict():
    s = load_spec_from_dict({
        "name": "from_dict",
        "objectives": [{"name": "f1", "direction": "maximize"}, {"name": "f2", "direction": "maximize"}],
        "budget": {"experiments": 50, "per_iteration_cap": 5},
        "benchmark": "vlmop2",
        "n_input_dims": 2,
    })
    assert s.n_objectives() == 2
    assert s.budget.experiments == 50


run("framing_valid", test_framing_valid)
run("framing_one_obj", test_framing_one_obj)
run("framing_overlap", test_framing_overlap)
run("framing_from_dict", test_framing_from_dict)

# ── IVO Layer ─────────────────────────────────────────────────────────────────
print("\n── Module 4: IVO Layer ──")


def test_vlmop2():
    ivo = VLMOP2Benchmark(n_var=2, seed=0)
    obs = ivo.observe(np.array([0.5, 0.3]))
    assert obs.objectives.shape == (2,)
    assert obs.constraint_valid is True


def test_batch():
    ivo = VLMOP2Benchmark(n_var=2, seed=0)
    obs_list = ivo.observe_batch(np.random.randn(5, 2))
    assert len(obs_list) == 5
    assert all(o.objectives.shape == (2,) for o in obs_list)


def test_mol_constraint():
    ivo = MolecularProxyBenchmark(n_var=5, seed=0)
    obs = ivo.observe(np.ones(5) * 3.0)   # sum(|x|) = 15 > 8 threshold
    assert obs.constraint_valid is False


def test_triobj():
    ivo = HighConstraintBenchmark(n_var=6, seed=0)
    obs = ivo.observe(np.zeros(6))
    assert obs.objectives.shape == (3,)


def test_registry():
    for name in ["vlmop2", "molecular", "high_constraint"]:
        assert create_ivo(name, n_var=3, seed=0) is not None


def test_noise():
    ivo = VLMOP2Benchmark(n_var=2, noise_std=0.1, seed=42)
    x = np.array([0.1, 0.2])
    obs1 = ivo.observe(x)
    obs2 = ivo.observe(x)
    assert not np.allclose(obs1.objectives, obs2.objectives)


run("ivo_vlmop2", test_vlmop2)
run("ivo_batch", test_batch)
run("ivo_mol_constraint", test_mol_constraint)
run("ivo_triobj", test_triobj)
run("ivo_registry", test_registry)
run("ivo_noise", test_noise)

# ── Hypervolume ───────────────────────────────────────────────────────────────
print("\n── Hypervolume ──")


def test_hv_known():
    pf = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
    ref = np.zeros(2)
    hv = compute_hypervolume(pf, ref)
    assert abs(hv - 0.33) < 0.01, f"Expected ~0.33, got {hv}"


def test_hv_empty():
    assert compute_hypervolume(np.empty((0, 2)), np.zeros(2)) == 0.0


def test_hv_single():
    hv = compute_hypervolume(np.array([[0.5, 0.5]]), np.zeros(2))
    assert abs(hv - 0.25) < 0.01, f"Expected 0.25, got {hv}"


def test_pareto_sort():
    costs = np.array([[1, 4], [2, 3], [3, 2], [4, 1], [2, 2], [3, 3]])
    fronts = fast_non_dominated_sort(costs)
    assert len(fronts) >= 2
    first = fronts[0]
    first_costs = costs[first]
    for i, p in enumerate(first_costs):
        for j, q in enumerate(first_costs):
            if i != j:
                dominated = np.all(q <= p) and np.any(q < p)
                assert not dominated, f"{q} dominates {p} in first front"


run("hv_known", test_hv_known)
run("hv_empty", test_hv_empty)
run("hv_single", test_hv_single)
run("hv_pareto_sort", test_pareto_sort)

# ── Surrogate ─────────────────────────────────────────────────────────────────
print("\n── Surrogate ──")

rng = np.random.default_rng(0)
X_fit = rng.uniform(-3, 3, (30, 3))
Y_fit = np.column_stack([np.sin(X_fit[:, 0]), np.cos(X_fit[:, 1])])
surr = GPSurrogate(n_objectives=2, n_restarts=1, seed=0)
surr.fit(X_fit, Y_fit)


def test_surr_fit():
    assert surr._is_fit


def test_surr_predict():
    mu, sigma = surr.predict(X_fit[:5])
    assert mu.shape == (5, 2)
    assert sigma.shape == (5, 2)
    assert np.all(sigma >= 0)


def test_surr_sample():
    s = surr.sample_posterior(X_fit[:3], n_samples=8)
    assert s.shape == (8, 3, 2)


run("surrogate_fit", test_surr_fit)
run("surrogate_predict", test_surr_predict)
run("surrogate_sample", test_surr_sample)

# ── Candidate Generation ──────────────────────────────────────────────────────
print("\n── Module 1: Candidate Generation ──")

bounds = np.column_stack([np.full(3, -3.0), np.full(3, 3.0)])
gen = CandidateGenerator(n_var=3, bounds=bounds, seed=0)


def test_cand_count():
    cands = gen.generate(20, surr, X_fit, np.array([[0.5, 0.3]]))
    assert len(cands) == 20


def test_provenance():
    cands = gen.generate(15, surr, X_fit, np.array([[0.5, 0.3]]))
    provenances = {c.provenance for c in cands}
    assert provenances.issubset({"bo", "nsga", "diversity"})


def test_weight_shift():
    gen2 = CandidateGenerator(n_var=3, bounds=bounds, seed=0)
    gen2.update_weights({"bo": 1.0, "nsga": 0.0, "diversity": 0.0})
    w = gen2.current_weights()
    assert w["bo"] > w["nsga"]
    assert w["bo"] > w["diversity"]


run("cand_count", test_cand_count)
run("cand_provenance", test_provenance)
run("cand_weights", test_weight_shift)

# ── Constraint Layer ──────────────────────────────────────────────────────────
print("\n── Module 2: Constraint Layer ──")


def test_cl_update():
    cl = ConstraintLayer(spec)
    Xc = np.random.randn(20, 2)
    flags = np.array([1] * 12 + [0] * 8)
    cl.update(Xc, flags)
    assert cl.n_observations == 20
    assert 0 < cl.violation_rate < 1


def test_cl_gp():
    cl = ConstraintLayer(spec)
    Xc = np.random.randn(15, 2)
    flags = np.array([1] * 8 + [0] * 7)
    cl.update(Xc, flags)
    probs = cl.predict_feasibility(np.random.randn(5, 2))
    assert probs.shape == (5,)
    assert np.all((probs >= 0) & (probs <= 1))


def test_hard_filter():
    spec_hc = ProblemSpec(
        name="hc",
        objectives=[Objective("a", "maximize"), Objective("b", "maximize")],
        hard_constraints=[HardConstraint(property="molecular_weight", max=5.0)],
        budget=BudgetSpec(experiments=30, per_iteration_cap=5),
        benchmark="vlmop2",
        n_input_dims=5,
    )
    cl = ConstraintLayer(spec_hc)
    bad = Candidate("bad", np.array([3.0, 2.0, 1.0, 1.0, 1.0]), "bo")
    filtered, results = cl.filter([bad], attempt_repair=False)
    assert len(filtered) == 0
    assert results[0].passes_hard is False


run("constraint_update", test_cl_update)
run("constraint_gp", test_cl_gp)
run("constraint_hard_filter", test_hard_filter)

# ── Decision Engine ───────────────────────────────────────────────────────────
print("\n── Module 3: Decision Engine ──")


def test_scoring():
    engine = DecisionEngine(spec)
    cands = [
        Candidate(f"c{i}", X_fit[i], "bo",
                  predicted_objectives=Y_fit[i],
                  predicted_uncertainty=np.array([0.1, 0.1]))
        for i in range(5)
    ]
    pf = np.array([[0.3, 0.4], [0.5, 0.2]])
    scored = engine.score_candidates(cands, [], pf, surr, n_mc=8)
    assert len(scored) == 5
    assert all(r.rank >= 1 for r in scored)
    scores = [r.decision_score for r in scored]
    assert scores == sorted(scores, reverse=True)


def test_stopping():
    engine = DecisionEngine(spec)
    engine._experiments_run = 20
    dummy = DecisionRecord("x", np.zeros(2), "bo", 0.00001, 0, 0, 0.5, 1.0, "", 1)
    stop = False
    for _ in range(3):
        stop, _ = engine.check_stopping([dummy], budget_remaining=50)
    assert stop is True


def test_min_exp_guard():
    engine = DecisionEngine(spec)
    engine._experiments_run = 2   # below min_experiments=5
    dummy = DecisionRecord("x", np.zeros(2), "bo", 0.00001, 0, 0, 0.5, 1.0, "", 1)
    stop, _ = engine.check_stopping([dummy], budget_remaining=50)
    assert stop is False


run("decision_scoring", test_scoring)
run("decision_stopping", test_stopping)
run("decision_min_exp_guard", test_min_exp_guard)

# ── Trace Logger ──────────────────────────────────────────────────────────────
print("\n── Module 5: Trace Logger ──")


def test_logger_register():
    logger = DecisionTraceLogger(":memory:")
    logger.register_campaign("abc123", "test_campaign", {"key": "val"})
    camps = logger.get_all_campaigns()
    assert len(camps) == 1
    assert camps[0]["name"] == "test_campaign"


def test_logger_log():
    logger = DecisionTraceLogger(":memory:")
    logger.register_campaign("h1", "test", {})
    sel = [DecisionRecord("c1", np.zeros(2), "bo", 0.5, 0.1, 0.8, 0.9, 1.0, "reason", 1)]
    mf = DecisionManifest(
        iteration=1, selected=sel, all_scored=sel,
        should_stop=False, stop_reason="",
        eig_per_cost=0.05, pareto_hypervolume=0.3, budget_remaining=90,
    )
    logger.log_iteration("h1", mf)
    hv = logger.get_hv_history("h1")
    assert len(hv) == 1
    assert abs(hv[0]["hv"] - 0.3) < 1e-6


def test_logger_duplicate():
    logger = DecisionTraceLogger(":memory:")
    logger.register_campaign("h1", "c1", {})
    logger.register_campaign("h1", "c1", {})
    assert len(logger.get_all_campaigns()) == 1


run("logger_register", test_logger_register)
run("logger_log", test_logger_log)
run("logger_no_duplicate", test_logger_duplicate)

# ── Integration ───────────────────────────────────────────────────────────────
print("\n── Integration Tests (may take ~30s) ──")


def make_spec(name, benchmark, n_exp=25, seed=1, threshold=0.0001):
    return ProblemSpec(
        name=name,
        objectives=[Objective("f1", "maximize"), Objective("f2", "maximize")],
        budget=BudgetSpec(experiments=n_exp, per_iteration_cap=4),
        stopping_criterion=StoppingCriterion(
            eig_per_cost_threshold=threshold, min_experiments=5
        ),
        benchmark=benchmark,
        n_input_dims=2 if benchmark == "vlmop2" else 5,
        candidate_pool_size=20,
        random_seed=seed,
    )


def test_vlmop2_e2e():
    r = run_campaign(make_spec("t1", "vlmop2"), db_path=":memory:", verbose=False, n_mc=8)
    assert r["n_experiments"] > 0
    assert len(r["pareto_front"]) > 0
    assert r["hv_history"][-1] > 0


def test_molecular_e2e():
    r = run_campaign(make_spec("t2", "molecular", seed=2), db_path=":memory:", verbose=False, n_mc=8)
    assert r["n_experiments"] > 0


def test_insight_report():
    r = run_campaign(make_spec("t3", "vlmop2", seed=3), db_path=":memory:", verbose=False, n_mc=8)
    insight = r["insight_report"]
    assert insight is not None
    assert len(insight.key_findings) > 0
    assert insight.cost_efficiency.total_experiments > 0


def test_hv_nondecreasing():
    r = run_campaign(make_spec("t4", "vlmop2", n_exp=30, seed=5), db_path=":memory:", verbose=False, n_mc=8)
    hv = r["hv_history"]
    for i in range(1, len(hv)):
        assert hv[i] >= hv[i - 1] - 1e-9, f"HV decreased at iteration {i}: {hv[i-1]:.4f} -> {hv[i]:.4f}"


def test_early_stop():
    r = run_campaign(
        ProblemSpec(
            name="stop",
            objectives=[Objective("f1", "maximize"), Objective("f2", "maximize")],
            budget=BudgetSpec(experiments=200, per_iteration_cap=5),
            stopping_criterion=StoppingCriterion(
                eig_per_cost_threshold=0.1, consecutive_iterations=2, min_experiments=10
            ),
            benchmark="vlmop2",
            n_input_dims=2,
            candidate_pool_size=20,
            random_seed=6,
        ),
        db_path=":memory:", verbose=False, n_mc=8,
    )
    assert r["n_experiments"] < 200, f"Expected early stop, used {r['n_experiments']} experiments"


run("pipeline_vlmop2_e2e", test_vlmop2_e2e)
run("pipeline_molecular_e2e", test_molecular_e2e)
run("pipeline_insight_report", test_insight_report)
run("pipeline_hv_nondecreasing", test_hv_nondecreasing)
run("pipeline_early_stop", test_early_stop)

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
total = len(passed) + len(failed)
print(f"Results: {len(passed)}/{total} passed")
if failed:
    print(f"\nFailed tests:")
    for f in failed:
        print(f"  • {f}")
    sys.exit(1)
else:
    print("All tests passed.")
    sys.exit(0)
