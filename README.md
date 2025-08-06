# SCUCa

This repository provides examples for solving the **Security Constrained Unit Commitment (SCUC)** problem.  It contains notebooks and helper modules used in experiments with both mathematical programming and machine learning approaches.

## Contents

- `gurobi_scuc.ipynb` &ndash; Jupyter notebook that formulates the SCUC model and solves it using the [Gurobi](https://www.gurobi.com/) optimizer.
- `ML_baseline.ipynb` &ndash; baseline notebook illustrating a graph-based model for predicting active transmission line constraints.
- `ml/` &ndash; package with graph utilities and a simple `EdgeModel` used by the notebook.
- `src/` &ndash; modules for preparing data and building the optimization model.
- `instances/` &ndash; sample MATPOWER cases, e.g. `case14`.

For a detailed overview of the GNN formulation, see [this reference](https://arxiv.org/pdf/1706.02216).

## Usage

Open the notebooks in your preferred Jupyter environment.  The `gurobi_scuc.ipynb` notebook requires a working Gurobi installation.  The machine learning example relies on `torch` and `torch_geometric`.

The dataset under `instances/` includes a single `case14` example, which can be expanded with additional MATPOWER cases as needed.

To test the optimization model from the command line run

```bash
python src/optimization_model/SCUC/optimizer.py
```
which loads a bundled `case300` instance and prints a short summary.

### Python dependencies

The notebooks require the following additional packages:

- `gurobipy`
- `torch`
- `torch_geometric`
- `requests`
- `tqdm`


