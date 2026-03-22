# AEDE User Guide - Getting Optimal Results

This guide walks you through every parameter in the app, explains what it does,
and shows you how to tune it for the best results.

---

## The mental model

AEDE runs your experiment campaign in **batches**. Each batch it asks: *"Which experiments would teach me the most about the Pareto front, given what I already know?"* It scores every candidate experiment by Expected Pareto Improvement × uncertainty × feasibility ÷ cost, runs the top-scoring ones, fits a Gaussian Process to all results so far, and repeats. It stops automatically when the expected information gain per experiment drops below a threshold you set - meaning the campaign has converged and further experiments would not meaningfully improve the Pareto front. Your job is to set the parameters that define the search space, the budget, and how aggressively to stop.

---

## The sidebar - parameter by parameter

### Benchmark

The synthetic test problem AEDE optimises against.

| Option | What it is | When to use it |
|---|---|---|
| **VLMOP2 (bi-objective)** | Clean analytic function, 2 objectives, no constraints | First run, sanity checking, learning the tool |
| **Molecular Proxy** | Simulates drug discovery - 2 objectives, synthesis constraints | Testing constraint handling |
| **High Constraint (tri-obj)** | 3 objectives, ~40% of inputs are infeasible | Stress-testing the constraint layer |

> **Start with VLMOP2.** It converges quickly, results are predictable, and you can verify the engine is working before changing anything else.

---

### Input dimensions

**What it is:** How many input variables your problem has. For VLMOP2 this is the dimensionality of the search space - think of it as the number of knobs you can turn in your experiment (temperature, concentration, pH, pressure, etc.).

**Effect:** Higher dimensions = harder problem. The GP surrogate needs more data to map a 10-dimensional space than a 2-dimensional one.

| Value | Effect |
|---|---|
| 2–3 | Fast convergence, good for learning and demos |
| 4–6 | Realistic for most lab problems - may need more experiments |
| 7–10 | Slow, needs large candidate pool and more MC samples |

**Recommendation:** Match this to your actual problem. For benchmarking, stay at 2–3.

---

### Objectives (name + direction)

**What it is:** The quantities you are trying to optimise. Each objective has a name (used in charts and the insight report) and a direction.

**Direction:**
- **Maximize** - higher is better (yield, binding affinity, selectivity)
- **Minimize** - lower is better (cost, toxicity, reaction time)

AEDE internally negates minimised objectives so the hypervolume calculation always works in the "maximise everything" convention. The Pareto front chart shows the original (un-negated) values.

> **Important for benchmarks:** VLMOP2 and Molecular Proxy are both designed with **maximize for all objectives**. Setting either to minimize on a benchmark will give poor results because the reference point assumes maximisation.

**For your own data:** Name the objectives to match your CSV column headers exactly. The upload parser uses exact string matching.

---

### Total experiments

**What it is:** The hard budget cap - the maximum number of experiments the campaign is allowed to run before it is forced to stop.

**Effect:** This is your insurance policy. If the stopping criterion never fires (because EIG stays high), the campaign will run until this number is reached.

| Value | When to use |
|---|---|
| 20–40 | Quick runs, demos, checking the engine works |
| 60–100 | Normal production runs for 2–4 dimensional problems |
| 100–200 | High-dimensional or heavily constrained problems |

**Rule of thumb:** Set this to 3–5× what you expect the campaign to actually need. AEDE will stop early if it converges - the budget is a ceiling, not a target.

---

### Per iteration (batch size)

**What it is:** How many experiments AEDE selects and runs in each iteration. Equivalent to running experiments in parallel - a batch of 5 means 5 experiments run simultaneously each cycle.

**Effect:** Larger batches = fewer iterations = less GP refitting overhead, but each individual experiment is chosen with less information than if you ran them one at a time.

| Value | When to use |
|---|---|
| 2–3 | Sequential lab - you can only run a few at once |
| 4–6 | Good default for most setups |
| 8–10 | High-throughput screening, automated platforms |

**Recommendation:** Match this to your actual lab parallelism. For benchmarking, 5 is a sensible default.

---

### Candidate pool

**What it is:** How many candidate experiments AEDE generates and scores each iteration before selecting the top batch. It generates candidates using three strategies (Bayesian optimisation, NSGA-II, diversity sampling) then scores all of them and picks the best `per_iteration` ones.

**Effect:** Larger pool = better chance of finding high-quality candidates, but slower per iteration.

| Value | Effect |
|---|---|
| 20–40 | Fast, may miss good candidates in sparse regions |
| 60–100 | Good balance - recommended default |
| 150–200 | Thorough search, slower iterations |

**Recommendation:** Keep at 60 unless iterations feel slow, then reduce. If results look poor (Pareto front not growing), increase to 100–150.

---

### EIG/cost threshold *(the most important parameter)*

**What it is:** Expected Information Gain per cost unit. When the best available experiments score below this value for `N consecutive iterations`, AEDE concludes the campaign has converged and stops.

**Think of it as:** "How much new information am I willing to pay for?" A threshold of 0.005 means stop when expected Pareto improvement per unit cost becomes negligible.

**Effect on behaviour:**

| Threshold | Behaviour | Use when |
|---|---|---|
| 0.010 | Stops early, may miss late improvements | Budget is tight, quick answer needed |
| 0.005 | Good balance - stops when gains become marginal | Default for most runs |
| 0.002 | Runs longer, captures more of the front | High-stakes decisions, time allows |
| 0.001 | Very thorough, rarely stops early | Maximum quality needed |

**Real numbers (VLMOP2, 80 experiment budget):**

| Threshold | HV achieved | Experiments used | Budget saved |
|---|---|---|---|
| 0.010 | 0.254 | 40 | 50% |
| 0.005 | 0.254 | 40 | 50% |
| 0.001 | 0.266 | 50 | 38% |

The 0.005 and 0.01 thresholds got the same HV - meaning the extra experiments at 0.001 produced only marginal improvement. **Start at 0.005.**

---

### Consecutive iters below threshold

**What it is:** How many consecutive iterations must score below the EIG/cost threshold before the campaign stops. A single low-scoring iteration might be a fluke; N consecutive low iterations is a reliable signal of convergence.

| Value | Behaviour |
|---|---|
| 2 | Stops quickly once EIG drops - may stop too early on noisy problems |
| 3 | Default - good balance between responsiveness and robustness |
| 4–5 | Conservative - waits for sustained evidence before stopping |

**Recommendation:** Use 3 for most problems. Increase to 4–5 if your experiments have high noise.

---

### Min experiments

**What it is:** The campaign will never stop before this many experiments have run, regardless of EIG. This prevents stopping in the first iteration when the GP is underfit and EIG estimates are unreliable.

| Value | Behaviour |
|---|---|
| 5–10 | Allows very early stopping - only for ultra-fast benchmarks |
| 15–20 | Default - ensures the GP has enough data before stopping is considered |
| 25–40 | Conservative - useful when observations are very noisy |

**Rule of thumb:** Set to at least 3× your `per_iteration` batch size. If batch = 5, set min_experiments ≥ 15.

---

### MC samples

**What it is:** The number of Monte Carlo samples used to estimate Expected Pareto Improvement (EPI) for each candidate. More samples = more accurate scoring = better experiment selection, but slower per iteration.

| Value | Speed | Quality | When to use |
|---|---|---|---|
| 16–32 | Fast (~5s/iter) | Lower accuracy | Quick demos, debugging, 2D problems |
| 64 | Moderate (~15s/iter) | Good - recommended default | Most production runs |
| 128 | Slow (~30s/iter) | High accuracy | Final runs, high-dimensional problems |
| 256 | Very slow | Maximum accuracy | Research-grade, when you have time |

**Recommendation:** Use 64 for normal runs. If your campaign converges to a poor Pareto front, try 128.

> **Note:** On Streamlit Cloud, higher MC samples also increase the chance of hitting the 1GB memory limit. If the app crashes mid-campaign, reduce MC samples to 32.

---

### Random seed

**What it is:** Fixes the random number generator so the same settings produce identical results every time.

**Use it for:**
- Reproducibility (sharing results with colleagues)
- Debugging (isolating whether a problem is stochastic or deterministic)
- A/B testing parameter changes (change one thing, keep seed fixed)

Change the seed to get a different run when you want to test robustness - if results vary wildly across seeds, you may need more experiments or a larger candidate pool.

---

## Recommended starting configurations

### 1. First-ever run - verify the engine works
```
Benchmark:          VLMOP2
Input dimensions:   2
Both objectives:    maximize
Total experiments:  40
Per iteration:      4
Candidate pool:     40
EIG threshold:      0.005
Consecutive iters:  3
Min experiments:    12
MC samples:         16
Seed:               42
```
Expected: HV > 0.05, converges in 5–8 iterations, ~32 experiments used. Takes ~30 seconds.

---

### 2. Standard production run
```
Benchmark:          VLMOP2 (or your problem)
Input dimensions:   2–4
Total experiments:  80
Per iteration:      5
Candidate pool:     60
EIG threshold:      0.005
Consecutive iters:  3
Min experiments:    15
MC samples:         64
Seed:               42
```
Expected: HV > 0.08 (VLMOP2), converges in 6–10 iterations, saves 40–60% of budget. Takes ~2–4 minutes.

---

### 3. High-quality thorough run
```
Benchmark:          VLMOP2 or Molecular
Input dimensions:   3–5
Total experiments:  120
Per iteration:      5
Candidate pool:     100
EIG threshold:      0.003
Consecutive iters:  4
Min experiments:    20
MC samples:         128
Seed:               42
```
Expected: Larger, denser Pareto front. Takes ~8–15 minutes.

---

## Reading the results

### Pareto front chart
Diamond markers connected by a dotted line are your **Pareto front** - the set of non-dominated solutions. Every point represents a trade-off: you cannot improve one objective without worsening another. Grey dots are all observed experiments including dominated ones. A **good result** looks like a smooth convex curve spread across the objective space. A **poor result** looks like a few isolated points with large gaps.

### Hypervolume over iterations
Should be **monotonically increasing**. A flat line for several iterations followed by a jump means the surrogate was exploring before exploiting. A completely flat line suggests the GP is not fitting well - try increasing MC samples or candidate pool size.

### EIG/cost vs stopping threshold
Watch this line approach the dashed red threshold. When it crosses and stays below for `N consecutive iterations`, the campaign stops. If it never crosses, the campaign runs to the total experiment budget.

### Portfolio strategy weights
Shows which of the three strategies dominated each iteration:
- **bo (blue)** - Bayesian optimisation, exploitation-focused
- **nsga (orange)** - NSGA-II, multi-objective evolution
- **diversity (green)** - Space-filling, exploration-focused

A healthy run typically sees diversity and nsga dominate early (exploration), then bo take over later (exploitation). If diversity dominates throughout, the surrogate is struggling to fit - try more MC samples.

### Insight Report

**Budget saved** - "Saved 50% of total budget" means the campaign used 40 of 80 allowed experiments and the stopping criterion fired. This is AEDE's core value: not wasting lab resources on experiments that would not improve the front.

**Informative fraction** - percentage of iterations that produced a Pareto front improvement. Above 65% is excellent. Below 35% suggests too much time spent exploring unpromising regions.

**Calibration trend** - whether the GP's uncertainty estimates are getting better (🟢 Improving) or worse (🔴 Degrading). Degrading calibration means results are still valid but the stopping criterion may fire at the wrong time.

---

## Common problems and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| Campaign stops at iteration 1 | EIG threshold too high, or min_experiments too low | Lower threshold to 0.002, raise min_experiments to 20 |
| Campaign never stops (uses full budget) | EIG threshold too low | Raise threshold to 0.008–0.01 |
| Pareto front has only 1–2 points | Not enough experiments, or dimensions too high | Increase total experiments; reduce dimensions |
| HV stays flat for many iterations | Surrogate not fitting well | Increase MC samples to 128; increase candidate pool to 100 |
| Results vary wildly between seeds | Too few experiments for the dimensionality | Increase total experiments; increase min_experiments |
| App crashes mid-campaign | Memory limit on Streamlit Cloud | Reduce MC samples to 32; reduce candidate pool to 40 |
| Budget efficiency shows 0% | Campaign ran to full budget exhaustion | Raise EIG threshold or reduce total experiments |

---

## Using your own data

Upload a CSV via the **Upload your own experimental data** panel before clicking Launch:

```
x_0,x_1,yield,selectivity,feasible
0.12,0.55,0.81,0.63,1
0.43,0.21,0.74,0.71,1
0.88,0.90,0.51,0.82,0
```

- `x_0, x_1, ...` - input variables, normalised to roughly −3 to +3
- Objective columns - must match the objective names in the sidebar **exactly**
- `feasible` - 1 if the experiment succeeded, 0 if it failed a constraint (optional)

AEDE fits the GP to your data and recommends the next experiments. After running them in your lab, add those results to the CSV and re-upload for the next iteration.

---

## Quick reference card

| Parameter | Conservative | Recommended | Aggressive |
|---|---|---|---|
| Total experiments | 5× expected need | 3× expected need | 1× expected need |
| Per iteration | 2–3 | 4–6 | 8–10 |
| Candidate pool | 40 | 60 | 100–150 |
| EIG threshold | 0.002 | 0.005 | 0.01 |
| Consecutive iters | 4–5 | 3 | 2 |
| Min experiments | 3× batch size | 3× batch size | 2× batch size |
| MC samples | 128 | 64 | 16–32 |

---