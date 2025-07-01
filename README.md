# SCUCa

This repository provides examples for solving the **Security Constrained Unit Commitment (SCUC)** problem.  It contains notebooks and helper modules used in experiments with both mathematical programming and machine learning approaches.

## Contents

- `gurobi_scuc.ipynb` &ndash; Jupyter notebook that formulates the SCUC model and solves it using the [Gurobi](https://www.gurobi.com/) optimizer.
- `ML_baseline.ipynb` &ndash; baseline notebook illustrating a graph-based model for predicting active transmission line constraints.
- `ml/src/` &ndash; small Python package with graph utilities and an `EdgeModel` used by the notebook.
- `instances/` &ndash; sample instances in MATPOWER format.

For a detailed overview of the SCUC formulation, see [this reference](https://arxiv.org/pdf/1706.02216).

## Usage

Open the notebooks in your preferred Jupyter environment.  The `gurobi_scuc.ipynb` notebook requires a working Gurobi installation.  The machine learning example relies on `torch` and `torch_geometric`.

The dataset under `instances/` includes a single `case14` example, which can be expanded with additional MATPOWER cases as needed.

