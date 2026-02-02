# ISD-CP Test Lab: Protocol, Metrics, Procedures, and Baselines

This document defines a practical, reproducible evaluation lab for the ISD-CP project. It specifies what to measure, how to measure it in this repository, what “ideal” results look like, and how to produce plots and tables. It also lists baseline models to compare against and links recommended tests to existing scripts.

## Scope
- Covers structure learning metrics (SHD, SID, F1/TPR/FDR) and delta prediction metrics (MAE).
- Includes Mixture-of-Experts (MoE) routing health (entropy, gini).
- Emphasizes interventional tests (core to ISD-CP) and cross-difficulty validation via the curriculum.

## Metrics and Ideal Ranges
- SHD: Edit distance between predicted and true adjacency. Lower is better.
  - Ideal: Near 0 on in-distribution validation; monotonic increase with N but sub-linear under good generalization.
- SID: Interventional discrepancy. Lower is better.
  - Ideal: Small and consistently below baselines across sizes; should correlate with SHD but captures intervention correctness.
- F1, TPR, FDR: Edge classification quality.
  - Ideal: High F1/TPR, low FDR. As N grows, maintain reasonable precision.
- MAE: Mean absolute error on predicted deltas.
  - Ideal: Low and stable across intervention scales; increases mildly for extreme interventions but remains below baselines.
- Entropy (experts): Usage entropy; measures balance.
  - Ideal: Near `log(K)` (e.g., K=8 → ~2.08) when tokens are broadly distributed; may decrease as specialization emerges but should not collapse.
- Gini (experts): Inequality of expert usage.
  - Ideal: Near 0 for healthy load balancing; avoid values approaching 1 (collapse).

## Test Suites and Where to Run Them

1) Static Full Evaluation (ISD-CP across sizes)
- Script: `experiments/full_eval_static.py`
- What: Runs ISD-CP on multiple graph sizes and outputs per-intervention and summary tables.
- Metrics: MAE, SHD, F1, SID (if enabled), plus configurable logit threshold.
- Example:
```bash
python experiments/full_eval_static.py \
  --checkpoint checkpoints/checkpoint_epoch_253.pt \
  --output_dir evaluation_results \
  --device cpu \
  --batch_size 32 \
  --num_graphs 5 \
  --sizes 10,20,30,40,50 \
  --logit_threshold 1.33
```
- Outputs: `evaluation_results/full_eval_static.csv`, `full_eval_static_summary.csv`, `full_eval_static.md`.

2) Baseline Comparison (classic causal discovery)
- Script: `experiments/full_eval_baselines.py`
- What: Runs classical baselines across sizes, producing tables for SHD, SID, F1, MAE.
- Baselines to include: NOTEARS-MLP, PC, GES, LiNGAM, GraN-DAG (if implemented/wrapped), random.
- Example:
```bash
python experiments/full_eval_baselines.py \
  --output_dir evaluation_results \
  --batch_size 200 \
  --num_graphs 3 \
  --sizes 10,20,30,40,50 \
  --device cpu
```
- Outputs: `evaluation_results/full_eval_baselines.csv`, `full_eval_baselines_summary.csv`, `full_eval_baselines.md`.

3) Checkpoint Quick Benchmark (ISD-CP vs Random)
- Script: `experiments/evaluate_checkpoint.py`
- What: Sanity-check metrics and generate a markdown report at a specific epoch.
- Example:
```bash
python experiments/evaluate_checkpoint.py
```
- Outputs: `benchmark_report.md`.

4) Curriculum Cross-Difficulty Validation (in-training)
- Script: `main.py` (built-in validation and cross-difficulty benchmarks).
- What: Runs per-epoch validation with a fixed set and cross-difficulty checks (see console + `training_log.csv`).
- Example:
```bash
python main.py --epochs 2 --batch_size 16 --min_vars 20 --max_vars 50 \
  --intervention_prob 0.5 --num_layers 16 --lambda_dag 0.0 \
  --resume --checkpoint_path checkpoints/checkpoint_epoch_253.pt
```
- Outputs: `training_log.csv` plus console summaries.

## Step-by-Step Procedures

Step A: Prepare environment
- Ensure requirements are installed:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- Verify GPU if desired; otherwise use CPU for reproducibility.

Step B: Validate single checkpoint
- Run quick benchmark:
```bash
python experiments/evaluate_checkpoint.py
```
- Inspect `benchmark_report.md` for MAE, SHD, SID, F1.

Step C: Static evaluation across sizes
- Run the static eval:
```bash
python experiments/full_eval_static.py \
  --checkpoint checkpoints/checkpoint_epoch_253.pt \
  --sizes 10,20,30,40,50 \
  --num_graphs 5 --batch_size 32 --device cpu
```
- Confirm outputs in `evaluation_results/`.

Step D: Baseline comparison
- Run baselines:
```bash
python experiments/full_eval_baselines.py --sizes 10,20,30,40,50 --num_graphs 3
```
- Compare `full_eval_baselines_summary.csv` with ISD-CP summary.

Step E: MoE routing health check
- During any training or evaluation pass, capture expert metrics by printing from model:
  - Entropy and Gini appear in `main.py` progress line and epoch tables.
- Ideal trend: entropy near uniform early; gini low; no collapse.

Step F: Cross-difficulty validation in training
- Run a short training resume to trigger built-in benchmarks:
```bash
python main.py --epochs 1 --resume --checkpoint_path checkpoints/checkpoint_epoch_253.pt
```
- Check the console for `--- Cross-Difficulty Benchmarks ---` and summarize.

## Plots and Tables to Produce

Tables
- Per-size summary: `N, SHD, SID, F1, MAE` for ISD-CP and each baseline. (Already generated in `evaluation_results`.)
- Cross-difficulty validation table from training logs (aggregate by level).

Plots
- SHD vs N: line plot for ISD-CP and baselines.
- SID vs N: line plot for ISD-CP and baselines (if SID computed).
- MAE vs intervention scale: for static eval; show robustness.
- ROC/PR curves: if scoring per-edge probabilities available (use logits→sigmoid and vary threshold).
- Expert usage: bar chart of counts per expert; entropy and gini over epochs.
- FDR vs TPR: calibration plot across thresholds.

Quick generation (example with pandas/matplotlib)
```bash
python - <<'PY'
import pandas as pd
import matplotlib.pyplot as plt

sum_isd = pd.read_csv('evaluation_results/full_eval_static_summary.csv')
sum_base = pd.read_csv('evaluation_results/full_eval_baselines_summary.csv')

plt.figure()
plt.plot(sum_isd['N'], sum_isd['SHD'], label='ISD-CP')
for m in sum_base['Model'].unique():
    dfm = sum_base[sum_base['Model']==m]
    plt.plot(dfm['N'], dfm['SHD'], label=m, linestyle='--')
plt.xlabel('N (variables)'); plt.ylabel('SHD'); plt.legend(); plt.grid(True)
plt.savefig('evaluation_results/plot_shd_vs_n.png')
PY
```

## Baselines to Compare Against (common practice)
- NOTEARS-MLP: Continuous optimization for DAGs.
- PC (constraint-based) and GES (score-based): classical approaches; good references for small-to-medium N.
- LiNGAM: for linear, non-Gaussian settings (observational only).
- GraN-DAG / DAG-GNN: gradient-based deep methods; DAG-GNN uses VAE-like approach.
- Random and simple linear baselines: sanity checks.

Note: This repository already wraps or references several baselines via `experiments/full_eval_baselines.py` and `src/models/baselines/`.

## Mimic NOTEARS/PC/GES Experimental Protocols

This section replicates the common evaluation setup used in NOTEARS (and classical PC/GES papers) using this repository’s tooling.

Key ideas from NOTEARS and classical baselines
- Data: Observational-only samples generated from a linear SEM on ER graphs (no interventions during training/testing for those baselines).
- Noise: Typically Gaussian for NOTEARS; PC/GES also assume i.i.d. Gaussian for conditional independence tests with Fisher’s Z.
- Graph families: ER with average degree in {1, 2, 4}; node counts N in {10, 20, 50, 100}; sample sizes n in {100, 500, 1000}.
- Metrics: SHD (primary), plus TPR/FDR (or F1). Thresholding for NOTEARS typically uses `w_threshold ≈ 0.3`.
- Repeats: Multiple random graphs per setting (e.g., 10) and report mean ± std.

How to set observational-only mode in this repo
- The baseline runner `experiments/full_eval_baselines.py` builds data pipelines with `intervention_prob=0.2` and `use_twin_world=True`. For canonical NOTEARS/PC/GES tests, switch to observational-only:
  1) In `experiments/full_eval_baselines.py`, inside `run_baselines(...)`, set `intervention_prob=0.0` in the call to `generator.generate_pipeline(...)` and set `use_twin_world=False`.
  2) Alternatively, duplicate the script as `full_eval_obs_baselines.py` with those two changes to keep both modes.

Recommended grid (NOTEARS-style)
- Node counts: `N ∈ {10, 20, 50, 100}` (adjust upper bound if runtime is a concern).
- Average degree: control via `edge_prob` to approximate deg ∈ {1, 2, 4}. For ER graphs, `edge_prob ≈ deg / (N-1)`.
- Sample sizes per graph: `n ∈ {100, 500, 1000}`.
- Repeats: 10 graphs per (N, deg, n) setting.

Concrete steps
1) Pick N and target average degree deg.
2) Compute `edge_prob = deg / (N-1)`.
3) In `run_baselines`, set up the generator as:
   - `generator = SCMGenerator(num_nodes=N, edge_prob=edge_prob, seed=..., intervention_prob=0.0)`
4) In `generate_pipeline`, pass `use_twin_world=False` and `intervention_prob=0.0`.
5) Set per-graph sample size `batch_size = n` so `X` uses n observational samples.
6) Run PC, NOTEARS (and optionally GES) and collect SHD/F1 (and SID if desired) using `compute_*` functions.
7) Repeat for 10 seeds per setting; save raw and summary CSVs (already handled by script).

Example: N=20, deg=2, n=500
```bash
python experiments/full_eval_baselines.py \
  --output_dir evaluation_results/obs_notears \
  --batch_size 500 \
  --num_graphs 10 \
  --sizes 20 \
  --device cpu
```
Then, set `edge_prob ≈ 2/19 ≈ 0.105` in the script before running (or expose a CLI flag if you prefer).

Notes on fairness and thresholds
- NOTEARS threshold: Our `NotearsLinear` applies `w_threshold=0.3` before binarization, matching common practice.
- PC conditioning: Limit `max_k` (conditioning set size) for tractability (e.g., 2–3). Increase if compute allows.
- Timeouts: For N≥50, PC may be slow; skip or cap `max_k`.

Outputs to compile (NOTEARS-style)
- For each (N, deg, n): mean ± std of SHD, TPR, FDR (or F1), and runtime.
- Plots: SHD vs n for fixed N,deg; SHD vs N for fixed deg,n; TPR/FDR vs n.

Optional: GES and LiNGAM
- `GESWrapper` is implemented; insert similar calls in the baseline runner if desired (requires `causal-learn`).
- LiNGAM: Install `lingam` package and add a wrapper for linear non-Gaussian SEMs; test with non-Gaussian noises (e.g., Laplace) to match LiNGAM’s assumptions.

## Logging entropy/gini during training
- `training_log.csv` now includes `Expert_Entropy` and `Expert_Gini` per epoch. Use these columns to monitor MoE load balance and correlate routing health with SHD/MAE/F1.

## Ideal Result Profiles
- In-distribution validation:
  - SHD: near 0–5 for N≤20; scales gently with N; consistently below baselines.
  - SID: low and tracking SHD improvements.
  - F1: ≥0.8 for smaller N; maintain ≥0.7 as N grows.
  - MAE: small (project-specific scale); stable under moderate intervention ranges.
  - Entropy/Gini: high entropy (~log(K)); low gini (<0.2); no expert collapse.
- Cross-difficulty benchmarks: ISD-CP should degrade gracefully as difficulty rises and recover post-curriculum level-ups.

## Advanced / OOD Tests (recommended)
- Low-data regime: reduce `samples_per_graph`; observe SHD/SID robustness.
- Extreme interventions: widen `intervention_scale_range` and assess MAE/SHD.
- Sparsity stress: vary `edge_prob` to dense/sparse extremes.
- Threshold sweeps: vary `--logit_threshold` in static eval to produce FDR/TPR curves.

## Practical Tips and Gotchas
- SHD thresholding uses raw logits in `src/training/metrics.py` at threshold `0.0`; static eval uses a configurable threshold (default 1.33). Align thresholds when comparing.
- Acyclicity `h`-loss (`lambda_h`) is off by default; enabling it may reduce cycles but increases compute.
- SID can be expensive; consider computing on subsets for large N.

## References (for methodology alignment)
- NOTEARS: Zheng et al., 2018. Continuous optimization for DAG structure learning.
- DAG-GNN: Yu et al., 2019. Deep generative models for causal discovery.
- PC / GES: Spirtes et al.; Chickering. Classical constraint/score-based methods.
- LiNGAM: Shimizu et al., 2006. Linear non-Gaussian acyclic model.

## Reproducibility Checklist
- Fix random seeds across generators if comparing runs.
- Log all hyperparameters (`training_log.csv` is already created by `main.py`).
- Save plots and tables in `evaluation_results/` with versioned filenames.

## CI Suggestion (optional)
- Add a lightweight job that runs `full_eval_baselines.py` and `full_eval_static.py` on small sizes (N=10,20) and checks regression against stored summaries.
