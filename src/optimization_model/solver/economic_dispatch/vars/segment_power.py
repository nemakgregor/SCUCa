"""Segment power variables pseg[g,t,s] >= 0."""

import gurobipy as gp
from typing import Sequence


def add_variables(model: gp.Model, generators: Sequence, time_periods: range):
    """Add segment power variables to the model.

    Returns
    -------
    Var
        Gurobi Var dict keyed by (gen_name, t, s)
    """
    idx = []
    for g in generators:
        Sg = len(g.segments) if g.segments else 0
        for t in time_periods:
            for s in range(Sg):
                idx.append((g.name, t, s))

    pseg = model.addVars(idx, lb=0.0, name="pseg")
    model.__dict__["seg_power"] = pseg
    return pseg
