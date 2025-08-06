"""solver.variables
-------------------
Helper functions for adding *decision variables* to the Gurobi model.

Step‑1 exposes only `add_power_variables`.
"""

import gurobipy as gp
from typing import Sequence


def add_variables(model: gp.Model, units: Sequence, T: range):
    """Create p[g,t] ≥ 0 and return the VarDict.

    Parameters
    ----------
    model : gp.Model
    units : iterable of objects with attribute `.name`
    T     : range of time periods
    """
    gen_power = model.addVars(
        (unit.name for unit in units),
        T,
        lb=0.0,
        name="p[{0},{1}]"
    )
    return gen_power
