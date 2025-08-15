"""ID: V-201 — Commitment variables u[g,t] in {0,1}."""

import logging
import gurobipy as gp
from gurobipy import GRB
from typing import Sequence

logger = logging.getLogger(__name__)


def add_variables(model: gp.Model, generators: Sequence, time_periods: range):
    """Add binary commitment variables to the model.

    Returns
    -------
    Var
        Gurobi Var dict keyed by (gen_name, t)
    """
    indices = [(gen.name, t) for gen in generators for t in time_periods]
    gen_commit = model.addVars(indices, vtype=GRB.BINARY, name="gen_commit")
    model.__dict__["commit"] = gen_commit
    logger.info("Vars(V-201): commitment u[g,t] added=%d", len(indices))
    return gen_commit
