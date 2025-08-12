"""Commitment variables u[g,t] in {0,1}."""

import gurobipy as gp
from gurobipy import GRB
from typing import Sequence


def add_variables(model: gp.Model, generators: Sequence, time_periods: range):
    """Add binary commitment variables to the model.

    Returns
    -------
    Var
        Gurobi Var dict keyed by (gen_name, t)
    """
    indices = [(gen.name, t) for gen in generators for t in time_periods]
    u = model.addVars(indices, vtype=GRB.BINARY, name="u")
    # Expose also as attribute if desired by callers
    model.__dict__["commit"] = u
    return u
