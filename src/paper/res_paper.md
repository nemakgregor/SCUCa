
What was added and why

1) New experiment/analysis pipeline under src/paper
- experiments.py: Orchestrates the full benchmark suite on UnitCommitment.jl v0.4 MATPOWER cases (case14, case57, case118, case300). Implements all requested modes:
  - RAW (explicit N-1, no ML)
  - WARM (warm starts only)
  - WARM+HINTS (warm + branching hints)
  - WARM+LAZY (warm + hints + lazy N-1)
  - WARM+PRUNE (warm + hints + conservative screening), with τ ∈ {0.3, 0.5, 0.7}
  Splits instances deterministically (70/15/15) via the same _hash01 as the warm-start code. Solves missing TRAIN outputs with RAW to build ML indexes, pretrains warm-start and redundancy indexes, and runs TEST across modes. Logs per-instance metrics to results/raw_logs/*.csv, including runtime, node count, objective, ppm delta placeholder (computed later), root LP size, peak memory, and verification statistics.

- analysis.py: Aggregates all raw log CSVs, computes:
  - Per-instance objective delta in ppm vs RAW baseline
  - Per-case/per-mode medians, IQRs, and 95% bootstrap CIs
  - Wilcoxon signed-rank tests (paired) for WARM+LAZY vs RAW (runtime and nodes)
  Writes results/merged_results.csv (per-instance) and results/summary.csv (aggregated).

- plots.py: Generates all requested figures as both .pdf and .png under results/figures:
  - runtime_cdf_<case>.{pdf,png}
  - nodes_boxplot_ablation_matpower_case118.{pdf,png}
  - pareto_constraints_vs_runtime_matpower_case118.{pdf,png}
  - heatmap_pruned_ratio.{pdf,png}
  - scaling_runtime_vs_size.{pdf,png}
  - memory_distribution.{pdf,png}
  - residual_distribution_ood.{pdf,png}

- tables.py: Writes LaTeX-ready tables under results/tables:
  - topline.tex (RAW vs WARM+LAZY with CI)
  - ablation_case118.tex (incremental methods)
  - screening_case118.tex (constraint ratio and runtime gain vs τ)
  - memory.tex (median + IQR)
  - wilcoxon.tex (p-values for runtime and node counts)

2) Paper integration
- Updated paper.tex to replace figure placeholders with generated figures from results/figures and to \input the generated tables from results/tables. The scientific content and formatting remain unchanged; only the includes for figures and tables were updated.

3) Metrics fidelity and verification
- Peak memory is sampled via a background thread using psutil during each solve.
- Root LP size is recorded before optimize (num_constrs_root, num_vars_root).
- Verification uses the existing verify_solution() to log:
  - max residual across C-101..C-121
  - objective consistency O-301
  - human-readable violation set “OK” or space-separated IDs
- For WARM+PRUNE, the constraint pruning ratio is estimated consistently with the builder tolerances, without rebuilding the baseline model.

How to run

Prereqs:
- Licensed Gurobi and Python 3.10+.
- pip install -r requirements you already use; additionally: psutil, seaborn, scipy, pandas, matplotlib.

1) Run experiments (all cases, all modes)
  python -m src.paper.experiments --save-human-logs

Options:
  --limit-test N            run at most N TEST instances per case (debug)
  --modes ...               subset of modes, default all
  --taus 0.3 0.5 0.7        thresholds for WARM+PRUNE
  --time-limit 600          time limit per run
  --mip-gap 0.05            relative MIP gap

2) Aggregate and compute statistics
  python -m src.paper.analysis

Produces:
  results/merged_results.csv
  results/summary.csv

3) Generate figures
  python -m src.paper.plots

Figures saved to results/figures and referenced directly in paper.tex.

4) Generate LaTeX tables
  python -m src.paper.tables

Tables saved to results/tables and included via \input in paper.tex.

What each metric means

Per-instance logs (results/raw_logs/exp_*.csv) now include:
- runtime_sec, nodes, obj_val, obj_bound
- obj_ppm_vs_raw (added during analysis)
- num_constrs_root/num_vars_root (root LP size)
- num_constrs_final/num_vars_final (post-opt size)
- peak_memory_gb (RSS peak sampled during the solve)
- feasible_ok, max_constraint_residual, objective_inconsistency (verification)
- constr_total_cont, constr_kept_cont, constr_ratio_cont (for WARM+PRUNE)

Design choices and assumptions

- Deterministic split: same hashing utility as warm-start ensures reproducibility and disjoint TRAIN/VAL/TEST.
- Objective ppm delta is defined only when RAW baseline objective is finite and nonzero; otherwise NaN.
- Constraint pruning ratio is computed with the same tolerances and structure as the builder (line-outage via LODF, generator-outage via ISF) and the inferred ML predicate; this avoids overhead of building a second model to count constraints.
- Scalability plot uses root constraints as a robust proxy for system size (number of lines × time horizon) across cases.

Where your existing code is reused

- Model building (exact SCUC with PTDF/LODF)
- WarmStartProvider (training and application)
- RedundancyProvider (screening predicate)
- Lazy contingency callback (WARM+LAZY)
- verify_solution (C-101..C-121 and O-301 checks)
- Branching hints utility

After reproducing

- The updated paper.tex will compile and include generated figures/tables from results/. If desired, tweak the target case in ablation/screening tables or add more \inputs for additional cases.

Troubleshooting

- No instances listed: ensure network access to https://axavier.org/UnitCommitment.jl/0.4/instances or pre-populate src/data/input/ with .json.gz.
- Gurobi license: verify grbgetkey and license environment variables are set.
- Memory sampling: psutil is needed; install with pip install psutil.
- Missing results/summary.csv: run analysis.py after experiments.py finishes.


Notes

- You can rerun experiments with a reduced --limit-test for quick iteration.
- The pipeline is idempotent: TRAIN outputs are reused; TEST logs append to new CSVs without overwriting prior runs.
- To regenerate plots/tables after any new run, re-run analysis.py, plots.py, and tables.py.


