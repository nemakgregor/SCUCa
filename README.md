# SCUC

This repository provides a minimal, modular implementation to load standard UnitCommitment.jl benchmark instances and solve a simplified Security Constrained Unit Commitment variant: a segmented Economic Dispatch with unit commitment, reserves, and base-case transmission constraints using PTDF (ISF). Network PTDF/LODF matrices are computed from the input data.

Highlights
- Data: automatic download and parsing of UnitCommitment.jl JSON instances
- PTDF/LODF: robust construction that handles parallel lines and explicit reference-bus mapping
- Model: segmented costs, binary commitment, and system-wide power balance
- Reserves with shortfall penalty; transmission constraints with overflow slacks
- NEW: N-1 security constraints for contingencies
  - Line-outage constraints using LODF against emergency limits
  - Generator-outage constraints using ISF (PTDF) against emergency limits
- Batch solver to download, solve, and save JSON solutions for multiple instances

## Quick start

1) Create and activate a virtual environment (optional but recommended)
- Python 3.9+ is required.

2) Install dependencies
- Gurobi requires a valid license. If you do not have one, install and license Gurobi first.
- Then install the Python packages:

```bash
pip install gurobipy numpy scipy requests tqdm
```

3) Solve selected instances in batch and save JSON outputs
- Example: solve all instances that contain case57, case30, or case14 in their path.
- Solutions are saved under src/data/output mirroring the input path and without .gz extension. For example:
  - input:  src/data/input/matpower/case300/2017-06-24.json.gz
  - output: src/data/output/matpower/case300/2017-06-24.json

```bash
python -m src.optimization_model.SCUC_solver.solve_instances --include case57 case30 case14
```

Useful flags:
- --limit 10        solve at most 10 instances
- --time-limit 600  Gurobi time limit in seconds (default 600)
- --mip-gap 0.05    Gurobi MIP gap (default 5%)
- --dry-run         list instances to solve without solving

4) (Optional) Single sample run (legacy entry point)
- You can still run the original single-instance optimizer:

```bash
python -m src.optimization_model.SCUC_solver.optimizer
```

## What gets downloaded and where

Instances are fetched on demand from:
- https://axavier.org/UnitCommitment.jl/0.4/instances

They are cached under:
- src/data/input

Solutions are saved as JSON under:
- src/data/output

Logs are stored under:
- src/data/logs

Example mapping:
- running with name "matpower/case300/2017-06-24" stores
  - src/data/input/matpower/case300/2017-06-24.json.gz (downloaded)
  - src/data/output/matpower/case300/2017-06-24.json (solution for ML)

## Repository layout

- src/data_preparation
  - ... (unchanged)
- src/optimization_model/SCUC_solver
  - scuc_model_builder.py: builds a segmented SCUC by composing modular components
  - solve_instances.py: NEW batch solver (remote listing, download, solve, JSON save)
  - optimizer.py: legacy single-instance entry point
- src/optimization_model/helpers
  - save_json_solution.py: NEW helper to serialize solutions as JSON under src/data/output

## Notes and scope

- The saved JSON is derived from a structured solution with:
  - meta (instance/scenario/time info), objective, status,
  - system totals, per-generator outputs (commitment, segment power, total),
  - reserves (requirement/provided/shortfall),
  - network base-case flows and overflow slacks.

## Troubleshooting

- Gurobi license: ensure a valid Gurobi license is installed and accessible to gurobipy.
- Remote listing: if the website blocks directory listing, the script falls back to locally cached inputs under src/data/input.
- Instance filters: --include accepts tokens that should appear in the dataset name (e.g. 'case57', 'case30').