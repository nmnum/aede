# AEDE — Adaptive Experimental Decision Engine

> A decision-theoretic framework for self-driving labs that integrates principled stopping criteria, cost-aware experiment selection, and causal post-campaign attribution — connecting optimisation to campaign-level budget intelligence.

[![CI](https://github.com/YOUR_USERNAME/aede/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/aede/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## One-command demo

```bash
git clone https://github.com/YOUR_USERNAME/aede.git
cd aede
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` — configure a campaign, launch, watch the Pareto front build live.

To run headless from the CLI:

```bash
python -m aede.cli --config configs/vlmop2_quick.yaml --db results/run1.db
# Then inspect results in the dashboard:
streamlit run app.py
```

---

## Why AEDE

Most AI-driven optimisation systems stop at: *here are good candidates.*

AEDE goes further: **here is the best decision given budget, risk, and expected value — and here is whether we should be running experiments at all.**

The framing comes from chemistry: a reaction earns credibility not by running, but by its yield, selectivity, and energetic cost. AI-driven experimental campaigns should be held to the same standard.

| Chemistry concept | SDL translation | AEDE module |
|---|---|---|
| Theoretical yield | Max Pareto improvement achievable | Surrogate upper confidence bound |
| Observed yield | Actual HV improvement per campaign | Decision Trace Logger |
| Selectivity | Fraction of experiments that were informative | Insight Engine |
| Energetic cost | Budget consumed per unit of Pareto advance | Decision Engine scoring |

---

## Architecture

```
ProblemSpec (config + validation)
        ↓
Candidate Generation (BO · NSGA-II · Diversity)
        ↓
Constraint Intelligence Layer (hard filter + soft GP classifier)
        ↓
Decision Engine ← THE CORE
  • EPI × uncertainty × feasibility / cost scoring
  • Principled stopping criterion (EIG/cost threshold)
  • Optional Gittins index extension
        ↓
In-Vitro Oracle (3 benchmark landscapes)
        ↓
Decision Trace Logger (append-only SQLite)
        ↓
Insight Engine (phase transitions · calibration · attribution · yield report)
        ↓
Streamlit Dashboard + CLI
```

### Module map

| # | Module | Responsibility | Primary output |
|---|--------|---------------|----------------|
| 0 | Problem Framing | Validates config before any computation | `ProblemSpec` hash |
| 1 | Candidate Generation | Multi-strategy pool with portfolio weighting | Candidates + provenance tags |
| 2 | Constraint Intelligence | Hard filter + learnable soft constraint GP | Feasibility probabilities |
| 3 | Decision Engine | Scores, ranks, stops | Decision manifest + rationales |
| 4 | IVO Layer | Simulates experimental outcomes | Observation records |
| 5 | Trace Logger | Append-only decision audit trail | SQLite database |
| 6 | Insight Engine | Post-hoc causal attribution | Insight report |
| 7 | Communication | Human-readable rationales + dashboard | Streamlit UI |

---

## What this does not do

- It is **not** a molecular property predictor — it consumes predictions, does not generate them.
- It is **not** a generative chemistry model — it selects from candidate pools.
- It is **not** a wet-lab automation system — the IVO layer is a calibrated simulator.
- It is **not** claiming biological realism — the benchmarks demonstrate decision-theoretic rigor on a realistic substrate.

---

## Honest comparison with prior work

| System | What it does | What AEDE adds |
|--------|-------------|----------------|
| **BATCHIE** (Stanton et al., 2025, *Nature Comms*) | Batch adaptive design maximising mutual information | Stopping criteria, cost accounting, causal attribution |
| **EGBO** (Deshwal et al., 2024, *npj Comp. Mat.*) | Evolution-guided Bayesian optimisation for multi-objective | Decision-level intelligence on top of EGBO backbone |
| **Atlas** (Aspuru-Guzik group) | Modular BO library | Explicit stopping + budget yield reporting |

AEDE does not claim to outperform these at optimisation. It adds **campaign economics** — when to stop, what each experiment cost, and why the campaign behaved the way it did.

---

## Installation

### Lightweight (default — no GPU required)

```bash
pip install -r requirements.txt
```

Required packages: `numpy scipy scikit-learn pandas pyyaml streamlit plotly`

### Production (BoTorch + GPyTorch backend)

```bash
pip install -r requirements.txt -r requirements-production.txt
```

The production backend (`aede/modules/surrogate_botorch.py`) is a drop-in replacement that activates automatically when `torch` is importable. It provides:

- Proper gradient-based qNEHVI optimisation (not MC sampling)
- GPU acceleration (CUDA + Apple Silicon MPS)
- Automatic relevance determination kernels
- Sobol QMC sampling for better posterior approximation

```python
# To explicitly use BoTorch backend:
from aede.modules.surrogate_botorch import BoTorchSurrogate, get_surrogate

surrogate = get_surrogate(n_objectives=2, seed=42)
# Falls back to sklearn GP if torch not installed
```

---

## Usage

### Python API

```python
from aede import run_campaign, ProblemSpec, Objective, BudgetSpec, StoppingCriterion

spec = ProblemSpec(
    name="my_campaign",
    objectives=[
        Objective(name="binding_affinity", direction="maximize"),
        Objective(name="toxicity_score",   direction="minimize"),
    ],
    budget=BudgetSpec(experiments=200, per_iteration_cap=10),
    stopping_criterion=StoppingCriterion(
        eig_per_cost_threshold=0.005,
        consecutive_iterations=3,
        min_experiments=30,
    ),
    benchmark="molecular",
    n_input_dims=5,
    random_seed=42,
)

results = run_campaign(spec, db_path="results/campaign.db", verbose=True)

print(f"Final Pareto HV:  {results['hv_history'][-1]:.4f}")
print(f"Experiments used: {results['n_experiments']}")
print(f"\nKey findings:")
for finding in results['insight_report'].key_findings:
    print(f"  • {finding}")
```

### YAML config + CLI

```bash
python -m aede.cli --config configs/molecular_admet.yaml --db results/run1.db
```

### Streamlit dashboard

```bash
streamlit run app.py
```

---

## Benchmarks

Three IVO landscapes of increasing difficulty:

| Benchmark | Objectives | Feasible space | Primary use |
|-----------|-----------|---------------|-------------|
| `vlmop2` | 2 (analytic) | 100% | Sanity check, fast iteration |
| `molecular` | 2 (RFF proxy) | ~70% (MW constraint) | Realistic molecular topology |
| `high_constraint` | 3 | ~15% | Stress test for constraint intelligence |

---

## Evaluation metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Hypervolume Indicator (HVI) | Volume dominated by Pareto front | Maximise |
| Budget efficiency | HVI(stop) / HVI(budget exhausted) | > 90% |
| Informative fraction | Experiments that improved HVI / total | > 60% |
| Constraint precision | Feasible candidates selected / total selected | Maximise |
| Stopping accuracy | Convergence before budget exhaustion | Campaign-dependent |

---

## Running tests

```bash
# Without pytest:
python tests/run_tests.py

# With pytest:
pip install pytest
python -m pytest tests/ -v
```

---

## Project structure

```
aede/
├── aede/
│   ├── __init__.py
│   ├── campaign.py              # Main orchestrator
│   ├── cli.py                   # CLI entry point
│   └── modules/
│       ├── problem_framing.py   # Module 0: Config + validation
│       ├── candidate_generation.py  # Module 1: BO, NSGA-II, diversity
│       ├── constraint_layer.py  # Module 2: Hard + soft constraints
│       ├── decision_engine.py   # Module 3: Scoring + stopping
│       ├── ivo_layer.py         # Module 4: Benchmark simulators
│       ├── trace_logger.py      # Module 5: Append-only SQLite log
│       ├── insight_engine.py    # Module 6: Post-hoc attribution
│       ├── communication.py     # Module 7a: Rationale generation
│       ├── surrogate.py         # Default: sklearn GP
│       └── surrogate_botorch.py # Production: BoTorch + GPyTorch
├── app.py                       # Streamlit dashboard
├── configs/                     # Example YAML configs
│   ├── vlmop2_quick.yaml
│   ├── molecular_admet.yaml
│   └── high_constraint.yaml
├── tests/
│   ├── test_aede.py             # pytest-compatible tests
│   └── run_tests.py             # Standalone test runner
├── requirements.txt             # Lightweight (no GPU)
├── requirements-production.txt  # BoTorch stack
└── setup.py
```

---

## Key references

- Stanton et al. (2025). *BATCHIE: Efficient Prediction of Batch Drug Combination Experiments.* Nature Communications.
- Deshwal et al. (2024). *EGBO: Evolution-Guided Bayesian Optimization.* npj Computational Materials.
- Gittins (1979). *Bandit Processes and Dynamic Allocation Indices.* JRSS Series B.
- Balandat et al. (2020). *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization.* NeurIPS.

---

## License

MIT — see [LICENSE](LICENSE).
