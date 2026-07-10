# SCUC

This repository provides a minimal, modular implementation to load standard UnitCommitment.jl benchmark instances and solve a simplified Security Constrained Unit Commitment variant: a segmented Economic Dispatch with unit commitment, reserves, and base-case transmission constraints using PTDF (ISF). Network PTDF/LODF matrices are computed from the input data.

Highlights
- Data: automatic download and parsing of UnitCommitment.jl JSON instances
- PTDF/LODF: robust construction that handles parallel lines and explicit reference-bus mapping
- Model: segmented costs, binary commitment, and system-wide power balance
- Reserves with shortfall penalty; transmission constraints with overflow slacks
- N-1 security constraints for contingencies
  - Line-outage constraints using LODF against emergency limits
  - Generator-outage constraints using ISF (PTDF) against emergency limits
- Batch solver to download, solve, and save JSON solutions for multiple instances
- ML warm-starts: k-NN warm start based on historical solutions per case; robust “fixed/repair” mode to make starts solver-friendly

## Quick start

1) Create and activate a virtual environment (optional but recommended)
- Python 3.9+ is required.

2) Install dependencies
- Gurobi requires a valid license. If you do not have one, install and license Gurobi first.
- Then install the Python packages:

```bash
pip install -r requirements.txt
```

3) Solve selected instances in batch and save JSON outputs
- Example: solve all instances that contain case57 in their path.
- Solutions are saved under src/data/output mirroring the input path and without .gz extension. For example:
  - input:  src/data/input/matpower/case300/2017-06-24.json.gz
  - output: src/data/output/matpower/case300/2017-06-24.json

```bash
python -m src.optimization_model.SCUC_solver.solve_instances --include case57 --time-limit 600 --mip-gap 0.05
```

Useful flags:
- --include ...         substring tokens to match dataset names (default: case57 case30 case14)
- --roots ...           top-level folders to search remotely (default: matpower test)
- --limit N             solve at most N instances (0 => unlimited)
- --time-limit 600      Gurobi time limit [s]
- --mip-gap 0.05        Gurobi MIP gap
- --option TAG          technique tag placed into the CSV log filename
- --skip-existing / --no-skip-existing   skip instances that already have an output JSON (ON by default)
- --dry-run             list instances and exit

4) (Optional) Single known instance
- The `--include` filter is a substring match, so a single instance can be targeted by its full name:

```bash
python -m src.optimization_model.SCUC_solver.solve_instances \
  --include case57/2017-06-24 \
  --time-limit 600 --mip-gap 0.05
```

## General pipeline: compare raw Gurobi vs warm-start

We provide a convenience script to run the full pipeline for a case folder (e.g., matpower/case57):
- Stage 1: solve TRAIN split with raw Gurobi (no warm start) and save outputs
- Stage 2: pretrain warm-start model from those outputs
- Stage 3: compare on TEST split raw vs warm-start, recording speed and accuracy, and verifying feasibility of warm-start runs.

Run:

```bash
# Compare on case57 with default splits (70/15/15) and common limits
python -m src.optimization_model.SCUC_solver.compare_ml_raw \
  --case matpower/case57 \
  --time-limit 600 --mip-gap 0.05 \
  --train-ratio 0.70 --val-ratio 0.15 \
  --save-logs
```

Notes:
- The script downloads instances as needed, saves raw JSON solutions to `src/data/output`, builds a warm-start index from TRAIN outputs, and then evaluates warm vs raw on TEST.
- It saves CSV logs under:
  `src/data/logs/compare_logs_<case>_<timestamp>.csv`
- Feasibility for warm-start runs is verified using the built-in checker. If slacks are enabled, the checker validates the slack-augmented model rather than a strict no-slack formulation.
- “Feasibility” for warm-start runs is verified using the built-in checker (no strict feasibility if slacks are used; the checker ensures constraints are respected given slacks).

## Paper experiments pipeline

For the publication-oriented benchmark suite under `src/paper`:

```bash
python -m src.paper.experiments --run-id my_run
python -m src.paper.analysis
python -m src.paper.plots
python -m src.paper.tables
```

Notes:
- Canonical training outputs remain under `src/data/output`.
- Experiment-run solutions are isolated under `results/solutions/<mode>/...` to avoid overwriting the training database.
- Aggregated paper metrics use end-to-end per-run time (`runtime_report_sec`, falling back to `wall_sec`).
- `plots.py` writes figures to `results/figures/` and `tables.py` writes LaTeX tables to `results/tables/`.

Method semantics used in paper artifacts:
- `RAW`, `WARM`, `WARM+LAZY`, `LAZY`, `ACTIVESET`, and `ACTIVESET+LAZY` enforce the modeled slack-augmented SCUC constraints through the explicit model, full lazy checking, or outer-loop cut generation.
- Screening/reduction modes such as `PRUNE`, `LPSCREEN`, `PRUNE+LAZY`, `LPSCREEN+LAZY`, and screening-only `STREDUCE` solve reduced models and should be interpreted with post-solve verification; `STREDUCE+LAZY` uses unrestricted lazy contingency checking in the current implementation.
- `SHRINK+LAZY` is a shrinking-horizon approximation and is marked as heuristic (`^\dagger`) in tables/plots.
- If ST-reduction actually uses GRU warm start at run time, the logged mode label is suffixed with `+GRU` (for transparent comparisons).
- The paper experiment catalog includes extra lazy/ML modes for stress testing diverse strategies:
  `LAZY_COMMIT_HINTS`, `LAZY_GNN_T060`, `LAZY_GNN_T080`, `LAZY_COMMIT_GNN_T060`,
  `WARM_LAZY_BRANCH_HINTS`, `WARM_LAZY_GNN_T060`, `WARM_LAZY_COMMIT_GNN_T060`,
  `WARM_LAZY_COMMIT_GRU`, and `WARM_LAZY_ML_STACK`.
- Literature-inspired decomposition/reduction families are also exposed as explicit modes:
  active-set variants (`ACTIVESET_B500`, `ACTIVESET_B5000`, `ACTIVESET_NOCLEAN`,
  `WARM_ACTIVESET`, `ACTIVESET_LAZY_TOPK128`, `WARM_ACTIVESET_LAZY`) and
  spatio-temporal reduction variants (`STREDUCE_CONSERVATIVE`, `STREDUCE_AGGRESSIVE`,
  `STREDUCE_LAZY_CONSERVATIVE`, `STREDUCE_LAZY_AGGRESSIVE`).
- `RAW` is included even in the medium/large start catalog so time-limit runs can show when RAW fails to produce any feasible incumbent. `analysis.py` writes this check to `results/raw_failure_summary.csv`.

### Targeted and partial runs

The experiment runner supports targeted subsets and reruns. When `--resume` is used, completed result keys are skipped unless `--force-rerun` is supplied. Downstream analysis keeps the latest row for each `(case, instance, mode)` tuple, so forced reruns can be appended to the same CSV.

Add `--dry-run` to any command below to print the resolved cases, dates, modes, and whether training artifacts are needed.

Run only selected modes for all small cases:

```bash
python -m src.paper.experiments \
  --run-id my_run \
  --resume \
  --profile small \
  --only-modes ACTIVESET ACTIVESET_B500 ACTIVESET_B5000 ACTIVESET_NOCLEAN \
    WARM_ACTIVESET ACTIVESET_LAZY ACTIVESET_LAZY_TOPK128 WARM_ACTIVESET_LAZY
```

Force a fresh run of one mode on one case:

```bash
python -m src.paper.experiments \
  --run-id my_run \
  --resume \
  --only-case matpower/case89pegase \
  --only-modes RAW_GNN \
  --force-rerun
```

Smoke-test selected exact/heuristic modes on large cases up to case6515 using only two scenarios per case:

```bash
python -m src.paper.experiments \
  --run-id my_run_large_smoke \
  --profile large6800 \
  --only-modes LAZY_ALL ACTIVESET_LAZY SHRINK_LAZY \
  --test-dates 2017-01-15 2017-07-15 \
  --test-time-limit 7200
```

For a fuller large-case panel, remove the smoke-test date restriction or replace it with six representative dates:

```bash
python -m src.paper.experiments \
  --run-id my_run_large \
  --profile large6800 \
  --only-modes LAZY_ALL ACTIVESET_LAZY SHRINK_LAZY \
  --test-dates 2017-01-15 2017-03-15 2017-05-15 2017-07-15 2017-09-15 2017-11-15
```

## Warm-start tools

- Pretrain warm-start indexes from your saved outputs:
```bash
# Pretrain a specific case
python -m src.ml_models.pretrain_warm_start --cases matpower/case57

# Auto-discover cases with outputs and pretrain
python -m src.ml_models.pretrain_warm_start --auto-cases
```

- Generate warm-start files per split:
```bash
python -m src.ml_models.warm_start --case matpower/case57 --pretrain --report
python -m src.ml_models.warm_start --case matpower/case57 --generate-for train
```

- Evaluate warm-start vs raw Gurobi on a case (TRAIN solve, pretrain, TEST comparison):
```bash
python -m src.optimization_model.SCUC_solver.compare_ml_raw \
  --case matpower/case57 \
  --warm-mode repair \
  --time-limit 600 --mip-gap 0.05
```

## CLI options reference

The following sections list the main CLI options for the two primary entry points (see each module's `--help` for the full list).

### 1) Batch solver CLI: src.optimization_model.SCUC_solver.solve_instances

Usage:
- python -m src.optimization_model.SCUC_solver.solve_instances [options]

Options:
- What to solve
  - --include TOKENS...          Substring tokens to match dataset names [default: case57 case30 case14]
  - --roots FOLDERS...           Top-level folders to search under the instance base URL [default: matpower test]
  - --max-depth N                Maximum recursion depth for remote listing [default: 4]
  - --limit N                    Solve at most N instances (0 => unlimited)

- Solver controls
  - --time-limit SECONDS         Gurobi time limit [default: 600]
  - --mip-gap FLOAT              Gurobi MIP gap [default: 0.05]
  - --option TAG                 Technique tag placed into the CSV log filename [default: basic]
  - --skip-existing              Skip instances that already have a JSON solution (ON by default)
  - --no-skip-existing           Re-solve even if an output JSON exists
  - --dry-run                    List instances and exit (do not solve)

Output:
- JSON solutions under src/data/output mirroring input hierarchy
- General CSV logs under src/data/logs

### 2) Comparison pipeline: src.optimization_model.SCUC_solver.compare_ml_raw

Usage:
- python -m src.optimization_model.SCUC_solver.compare_ml_raw [options]

Options:
- Case and split
  - --case CASEFOLDER            Required. e.g., matpower/case118
  - --train-ratio FLOAT          Train ratio [default: 0.70]
  - --val-ratio FLOAT            Validation ratio; remainder is test [default: 0.15]
  - --seed INT                   Split seed [default: 42]
  - --limit-train N              Limit number of TRAIN instances [default: 0 = no limit]
  - --limit-test N               Limit number of TEST instances [default: 0 = no limit]

- Solver controls (applies to RAW and WARM runs)
  - --time-limit SECONDS         Time limit [default: 600]
  - --mip-gap FLOAT              Relative MIP gap [default: 0.05]
  - --skip-existing              TRAIN ONLY: skip instances that already have a JSON solution
  - --download-attempts N        Download retry attempts [default: 3]
  - --download-timeout SECONDS   Per-attempt HTTP timeout [default: 60]

- Warm-start evaluation
  - --warm-mode {repair,commit-only,as-is}
                                Warm-start application mode [default: repair]

- Redundancy pruning (optional; experimental)
  - --rc-enable                  Enable ML-based redundant contingency pruning
  - --rc-thr-rel FLOAT           Relative margin threshold (fraction of F_em) [default: 0.50]
  - --rc-use-train-db            Restrict redundancy k-NN to TRAIN split [default: True]

- Logs
  - --save-logs                  Save human-readable solution and verification logs (.log)
                                to src/data/logs (general CSV logs are always saved)

Outputs:
- Per-case results CSV at src/data/output/<case>/compare_<tag>_<timestamp>.csv
  - Includes a "violations" column: "OK" if no constraint violated, otherwise space-separated constraint IDs (e.g., "C105 C109")
- General logs CSV at src/data/logs/compare_logs_<tag>_<timestamp>.csv
  - Also includes the "violations" column

## What gets downloaded and where

Instances are fetched on demand from:
- https://axavier.org/UnitCommitment.jl/0.4/instances

They are cached under:
- src/data/input

Solutions are saved as JSON under:
- src/data/output

Warm starts are saved under:
- src/data/intermediate/warm_start

Logs are stored under:
- src/data/logs

Example mapping:
- running with name "matpower/case300/2017-06-24" stores
  - src/data/input/matpower/case300/2017-06-24.json.gz (downloaded)
  - src/data/output/matpower/case300/2017-06-24.json (solution for ML)
  - src/data/intermediate/warm_start/warm_matpower_case300_2017_06_24.json
  - src/data/intermediate/warm_start/warm_fixed_matpower_case300_2017_06_24.json (if fixed)

## Repository layout

- src/data_preparation
  - download_data.py / read_data.py: fetch and parse UnitCommitment.jl instances
  - ptdf_lodf.py: PTDF/LODF matrix construction with parallel-line and reference-bus handling
  - data_structure.py, params.py, utils.py: shared data model and parameters
- src/optimization_model/SCUC_solver
  - scuc_model_builder.py: builds a segmented SCUC by composing modular components
  - solve_instances.py: batch solver CLI (remote listing, download, solve, JSON save)
  - compare_ml_raw.py: pipeline to evaluate raw vs warm and export CSVs (main CLI options documented above)
  - fix_warm_start.py (in ml_models): repair/generate robust warm-start JSONs
- src/paper
  - experiments.py: publication benchmark driver across RAW/WARM/HINTS/LAZY/PRUNE/LPSCREEN/SR+LAZY/ACTIVESET/SHRINK/STREDUCE families
  - analysis.py: deduplicates raw logs and builds summary/effect-size CSVs
  - plots.py: renders figures from merged results
  - tables.py: writes LaTeX-ready tables from aggregated outputs
- src/ml_models
  - warm_start.py: k-NN warm-start provider (pretrain, generate warm files, apply to model)
  - pretrain_warm_start.py: convenience tool to prebuild per-case indexes
- src/optimization_model/helpers
  - save_json_solution.py: serialize solutions as JSON under src/data/output
  - verify_solution.py: verify model solution (in-memory) and write a report if requested

## Reproducing figures and tables

- `analysis.py` aggregates all `results/<run-id>/results.csv` ledgers (the repository ships one); `results/raw_logs/*.csv` is used as a fallback when no run ledger is present. Rebuild all aggregated artifacts with:
  - `python -m src.paper.analysis`
  - `python -m src.paper.plots`
  - `python -m src.paper.tables`
- Generated outputs land in `results/` (CSV summaries), `results/figures/` (PNG), and `results/tables/` (LaTeX).
- Exact and heuristic methods are labeled separately in tables and plots (heuristic modes are marked with a dagger).
