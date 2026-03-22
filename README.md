# ⚗ AEDE: Adaptive Experimental Decision Engine

A decision-theoretic framework for self-driving labs that integrates principled stopping criteria, cost-aware experiment selection, and causal post-campaign attribution.

**Live demo:** [aede-app.streamlit.app](https://aede-app.streamlit.app)

---

## What it does

AEDE runs adaptive optimisation campaigns over multi-objective experimental spaces. At each iteration it:

1. Generates candidate experiments using a portfolio of three strategies (Bayesian optimisation, NSGA-II, diversity sampling)
2. Scores each candidate by Expected Pareto Improvement × uncertainty × feasibility ÷ cost
3. Selects the best batch to run
4. Fits a Gaussian Process surrogate to all observations so far
5. Stops automatically when information gain per cost falls below a threshold for N consecutive iterations

After the campaign, an insight report summarises cost efficiency, strategy phase transitions, surrogate calibration, and constraint violation patterns.

---

## Quickstart
```bash
git clone https://github.com/YOUR_USERNAME/aede.git
cd aede
pip install -r requirements.txt
streamlit run app.py
```

For the production BoTorch backend (GPU-accelerated):
```bash
pip install -r requirements-production.txt
```

---

## Using the Streamlit app

### Benchmarks (built-in test problems)

| Benchmark | Objectives | Description |
|---|---|---|
| VLMOP2 | 2 (maximize both) | Standard bi-objective analytic function — use for sanity checking |
| Molecular Proxy | 2 (maximize both) | Drug-discovery proxy with ADMET synthesis constraints |
| High Constraint | 3 (maximize all) | Tri-objective with ~40% infeasible space |

> **Objective directions:** VLMOP2 and Molecular both require **maximize** for both objectives. High Constraint requires **maximize** for all three. Setting a direction to **minimize** is supported — AEDE negates the objective internally so the hypervolume calculation remains correct.

### Recommended settings for a first run

| Setting | Value |
|---|---|
| Benchmark | VLMOP2 |
| Input dimensions | 2 |
| Total experiments | 60 |
| Per iteration | 5 |
| MC samples | 64–128 |
| EIG/cost threshold | 0.005 |
| Consecutive iters | 3 |
| Min experiments | 15 |

Expected outcome: HV ≈ 0.06–0.09, 6 Pareto points, campaign converges in 6–8 iterations.

### Healthy result indicators

| Metric | Healthy range |
|---|---|
| Final HV | > 0.05 (VLMOP2) |
| Pareto front size | ≥ 4 points |
| HV yield / experiment | > 0.001 |
| Budget efficiency | > 50% |
| Calibration trend | Stable or Improving |

---

## Using your own experimental data

The app accepts a CSV upload to warm-start the surrogate from real measurements. This skips the random initialisation phase and fits the GP directly to your data.

### CSV format
```
x_0,x_1,yield,selectivity,feasible
0.12,0.55,0.81,0.63,1
0.43,0.21,0.74,0.71,1
0.88,0.90,0.51,0.82,0
```

**Columns:**
- `x_0, x_1, ..., x_N` — input variables (one per dimension). Normalise to approximately [-3, 3] for best GP performance.
- Objective columns — must **exactly match** the objective names you typed in the sidebar (e.g. if you named them `yield` and `selectivity`, the CSV columns must be `yield` and `selectivity`).
- `feasible` (optional) — 1 if the experiment produced a valid result, 0 if it violated a constraint. Omit this column if all experiments were valid.

**Steps:**
1. Set your objective names in the sidebar to match your CSV column names
2. Set input dimensions to match the number of `x_*` columns in your CSV
3. Expand the upload panel and upload your CSV
4. Click Launch Campaign — AEDE will fit to your data and recommend next experiments

The recommended experiments appear in the **Recent selected experiments** table. Export the Pareto front CSV and insight report JSON using the buttons at the bottom.

---

## Repository structure
```
aede/
├── app.py                          ← Streamlit dashboard
├── requirements.txt                ← Core dependencies
├── requirements-production.txt     ← Adds torch + BoTorch
├── setup.py
├── aede/
│   ├── __init__.py
│   ├── campaign.py                 ← Main orchestrator
│   ├── cli.py                      ← python -m aede.cli
│   └── modules/
│       ├── problem_framing.py      ← ProblemSpec, Objective, BudgetSpec
│       ├── candidate_generation.py ← BO, NSGA-II, diversity strategies
│       ├── constraint_layer.py     ← Hard filter + learnable GP classifier
│       ├── decision_engine.py      ← EPI scoring + stopping criterion
│       ├── ivo_layer.py            ← Benchmark functions
│       ├── trace_logger.py         ← Append-only SQLite
│       ├── insight_engine.py       ← Post-campaign attribution
│       ├── communication.py        ← Rationale generation
│       ├── surrogate.py            ← Default sklearn GP
│       └── surrogate_botorch.py    ← Production BoTorch GP
├── configs/
│   ├── vlmop2_quick.yaml
│   ├── molecular_admet.yaml
│   └── high_constraint.yaml
└── aede/tests/
    ├── run_tests.py                ← Standalone (no pytest needed)
    └── test_aede.py                ← pytest-compatible
```

---

## Running tests
```bash
# Standalone (no pytest required)
python aede/tests/run_tests.py

# With pytest
python -m pytest aede/tests/test_aede.py -v
```

34 tests. CI runs on Python 3.9, 3.10, 3.11 via GitHub Actions.

---

## CLI usage
```bash
python -m aede.cli --config configs/vlmop2_quick.yaml
python -m aede.cli --config configs/molecular_admet.yaml --verbose
```

---

## Python API
```python
from aede import run_campaign, ProblemSpec, Objective, BudgetSpec, StoppingCriterion

spec = ProblemSpec(
    name="my_campaign",
    objectives=[
        Objective("yield", "maximize"),
        Objective("cost",  "minimize"),
    ],
    budget=BudgetSpec(experiments=60, per_iteration_cap=5),
    stopping_criterion=StoppingCriterion(
        eig_per_cost_threshold=0.005,
        consecutive_iterations=3,
        min_experiments=15,
    ),
    benchmark="vlmop2",
    n_input_dims=2,
    candidate_pool_size=60,
    random_seed=42,
)

results = run_campaign(spec, db_path="campaign.db", verbose=True)

print(f"HV: {results['hv_history'][-1]:.4f}")
print(f"Pareto front: {len(results['pareto_front'])} points")
for finding in results["insight_report"].key_findings:
    print(" •", finding)
```

### Warm-starting from real data
```python
import numpy as np

X_real = np.array([[0.12, 0.55], [0.43, 0.21], [0.88, 0.90]])  # inputs
Y_real = np.array([[0.81, 0.63], [0.74, 0.71], [0.51, 0.82]])  # objectives
F_real = np.array([1, 1, 0])                                    # feasibility

results = run_campaign(
    spec,
    db_path=":memory:",
    seed_X=X_real,
    seed_Y=Y_real,
    seed_feasible=F_real,
)
```

---

## Next steps for production use

1. **Manual iteration loop** — after each campaign run, download the recommended `x_*` values from the table, run them in your lab, add the results to your CSV, and re-upload for the next iteration.
2. **Automate the loop** — connect AEDE to your LIMS, ELN, or database so observations flow back automatically.
3. **Replace the benchmark** — subclass `BaseIVO` in `aede/modules/ivo_layer.py` to call your real measurement system or simulation API instead of the synthetic benchmarks.
4. **Production surrogate** — install `requirements-production.txt` to activate the BoTorch backend with proper qNEHVI and GPU support.
5. **Authentication** — add `streamlit-authenticator` to restrict access to the Streamlit app if sharing externally.

---

## Dependencies

**Core:** numpy, scipy, scikit-learn, pandas, pyyaml, streamlit, plotly

**Production:** torch ≥ 2.1, gpytorch ≥ 1.11, botorch ≥ 0.9.4