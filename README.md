```
# SCUCa

This repository provides a minimal, modular implementation to load standard UnitCommitment.jl benchmark instances and solve a simplified Security Constrained Unit Commitment variant: a segmented Economic Dispatch with unit commitment variables and power balance. Network PTDF/LODF matrices are computed from the input data for future use, but the current optimizer does not yet impose transmission constraints.

Highlights
- Data: automatic download and parsing of UnitCommitment.jl JSON instances
- Model: segmented costs, binary commitment, and system-wide power balance
- Modular solver components for variables, constraints, and objective

## Quick start

1) Create and activate a virtual environment (optional but recommended)
- Python 3.9+ is required.

2) Install dependencies
- Gurobi requires a valid license. If you do not have one, install and license Gurobi first.
- Then install the Python packages:

```bash
pip install gurobipy numpy scipy networkx requests tqdm
```

3) Run the sample
- This will download the sample instance to a cache directory and solve a segmented ED with commitment:

```bash
python -m src.optimization_model.SCUC_solver.optimizer
```

By default, the script loads the instance "test/case14". You can change the instance by editing the SAMPLE variable at the top of src/optimization_model/SCUC_solver/optimizer.py. For example, try:
- "matpower/case300/2017-06-24"

## What gets downloaded and where

Instances are fetched on demand from:
- https://axavier.org/UnitCommitment.jl/0.4/instances

They are cached under:
- src/data/input

Example: running with SAMPLE="test/case14" stores src/data/input/test/case14.json.gz.

## Repository layout

- src/data_preparation
  - read_data.py: download/cache and load instances; supports deterministic and stochastic inputs
  - data_structure.py: typed dataclasses for buses, generators, lines, storage, reserves, etc.
  - utils.py: robust JSON-to-objects conversion, legacy schema migration, sanity repairs, PTDF/LODF hook
  - ptdf_lodf.py: builds PTDF and LODF from network susceptances using NetworkX and SciPy
  - params.py: default values and constants used by the loader

- src/optimization_model/SCUC_solver
  - ed_model_builder.py: builds a segmented ED with commitment by composing modular components
  - model_builder.py: simple ED scaffold (kept for reference)
  - optimizer.py: end-to-end script to load an instance, build the model, and optimize with Gurobi

- src/optimization_model/solver/economic_dispatch
  - vars: decision variable factories (commitment, segment power)
  - constraints: commitment fixing, linking, and system-wide power balance
  - objectives: production cost objective with segmented costs
  - data: helper to aggregate load over time

## Programmatic use

You can load and solve from a Python session:

```python
from src.data_preparation.read_data import read_benchmark
from src.optimization_model.SCUC_solver.ed_model_builder import build_model

inst = read_benchmark("test/case14")
sc = inst.deterministic
m = build_model(sc)
m.optimize()
print("Objective:", m.objVal)
```

## Notes and scope

- The current model enforces only system-wide power balance with segmented production costs and commitment. It does not include transmission constraints or reserves in the optimizer yet, although the data layer already computes PTDF/LODF and parses reserves.
- Startup/shutdown costs are parsed and stored where applicable but are not utilized in the current objective.
- This code is intended as a clear, minimal foundation you can extend with ramping, minimum up/down, reserve constraints, network flows, contingencies, etc.

## Troubleshooting

- Gurobi license: ensure a valid Gurobi license is installed and accessible to gurobipy.
- Networkx/SciPy: PTDF/LODF construction requires networkx and SciPy; install both if you see import errors.
- Instance names: if an instance name is not found, double-check it exists at the remote instances URL.

## Dependencies

- gurobipy
- numpy
- scipy
- networkx
- requests
- tqdm

Python 3.9+ is required (uses modern typing and dataclass features).
```