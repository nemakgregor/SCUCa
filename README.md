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
pip install gurobipy numpy scipy requests tqdm scikit-learn torch torch-geometric
```

3) Solve selected instances in batch and save JSON outputs
- Example: solve all instances that contain case57 in their path.
- Solutions are saved under src/data/output mirroring the input path and without .gz extension. For example:
  - input:  src/data/input/matpower/case300/2017-06-24.json.gz
  - output: src/data/output/matpower/case300/2017-06-24.json

```bash
python -m src.optimization_model.SCUC_solver.optimizer --include case57 --time-limit 600 --mip-gap 0.05
```

Useful flags (defaults keep everything OFF):
- --instances ...       explicit dataset names
- --include ...         tokens to match dataset names
- --limit N             solve at most N instances
- --time-limit 600      Gurobi time limit [s]
- --mip-gap 0.05        Gurobi MIP gap
- --skip-existing       do not re-solve if a JSON already exists
- --dry-run             list instances and exit
- --warm                enable warm start (OFF by default)
- --require-pretrained  only use pre-trained warm-start index
- --warm-mode fixed     warm-start mode: fixed (recommended), commit-only, or as-is
- --warm-use-train-db   restrict neighbor DB to the training split
- --save-logs           write human-readable solution and verification logs to src/data/logs (OFF by default)

4) (Optional) Single known instance
- You can also solve a single instance by name:

```bash
python -m src.optimization_model.SCUC_solver.optimizer \
  --instances matpower/case57/2017-06-24 \
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
python -m src.optimization_model.SCUC_solver.compare_warm_start \
  --case matpower/case57 \
  --time-limit 600 --mip-gap 0.05 \
  --train-ratio 0.70 --val-ratio 0.15 \
  --save-logs
```

Notes:
- The script downloads instances as needed, saves raw JSON solutions to src/data/output, builds a warm-start index from TRAIN outputs, and then evaluates warm vs raw on TEST.
- It saves a CSV with per-instance results under:
  src/data/output/<case_folder>/compare_matpower_case57_<timestamp>.csv
- “Feasibility” for warm-start runs is verified using the built-in checker (no strict feasibility if slacks are used; the checker ensures constraints are respected given slacks).

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

- Solve with warm start (using a pretrained model), restricting neighbors to the training DB:
```bash
python -m src.optimization_model.SCUC_solver.optimizer \
  --include case57 \
  --warm --require-pretrained --warm-use-train-db \
  --warm-mode fixed \
  --time-limit 600 --mip-gap 0.05
```

## CLI options reference

The following sections list all CLI options for the two main entry points.

### 1) Solver CLI: src.optimization_model.SCUC_solver.optimizer

Usage:
- python -m src.optimization_model.SCUC_solver.optimizer [options]

Options:
- What to solve
  - --instances NAMES...         Explicit dataset names, e.g., matpower/case14/2017-01-01
  - --include TOKENS...          Tokens to include in dataset name (e.g., case57)
  - --limit N                    Solve at most N instances (0 => unlimited)

- Solver controls
  - --time-limit SECONDS         Gurobi time limit [default: 600]
  - --mip-gap FLOAT              Gurobi MIP gap [default: 0.05]
  - --skip-existing              Do not re-solve instances that already have a JSON solution
  - --dry-run                    List instances and exit (do not solve)

- Warm start (all OFF by default)
  - --warm                       Enable warm start (generate from pre-trained DB if available)
  - --require-pretrained         Only use pre-trained warm-start index (do not auto-build)
  - --warm-mode {fixed,commit-only,as-is}
                                Warm-start application mode [default: fixed]
                                Note: fixed maps to "repair" (safer)
  - --warm-use-train-db          Restrict neighbor search to the training split of the index

- Logs
  - --save-logs                  Also save human-readable solution and verification logs (.log)
                                to src/data/logs (general CSV logs are always saved)

Output:
- JSON solutions under src/data/output mirroring input hierarchy
- General CSV logs at src/data/logs/optimizer_logs_<timestamp>.csv

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
  - --warm-mode {fixed,commit-only,as-is}
                                Warm-start application mode [default: fixed]
                                Note: fixed maps to "repair" (safer)

- Redundancy pruning (optional; experimental)
  - --rc-enable                  Enable ML-based redundant contingency pruning
  - --rc-thr-rel FLOAT           Relative margin threshold (fraction of F_em) [default: 0.50]
  - --rc-thr-abs FLOAT           Deprecated; ignored (was absolute MW margin)
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
  - ... (unchanged)
- src/optimization_model/SCUC_solver
  - scuc_model_builder.py: builds a segmented SCUC by composing modular components
  - optimizer.py: simple solver CLI (all options OFF by default, logs OFF by default)
  - compare_warm_start.py: pipeline to benchmark warm-start vs raw on train/test splits
  - compare_ml_raw.py: pipeline to evaluate raw vs warm and export CSVs (this doc lists all its CLI options)
  - solve_instances.py: batch solver (remote listing, download, solve, JSON save)
  - fix_warm_start.py (in ml_models): repair/generate robust warm-start JSONs
- src/ml_models
  - warm_start.py: k-NN warm-start provider (pretrain, generate warm files, apply to model)
  - pretrain_warm_start.py: convenience tool to prebuild per-case indexes
- src/optimization_model/helpers
  - save_json_solution.py: serialize solutions as JSON under src/data/output
  - verify_solution.py: verify model solution (in-memory) and write a report if requested