"""ID: V-205/V-206 — Startup/Shutdown binary variables.

- V-205: Startup indicators v[g,t] in {0,1}
- V-206: Shutdown indicators w[g,t] in {0,1}
"""

import logging
from typing import Sequence, Tuple
import gurobipy as gp
from gurobipy import GRB

logger = logging.getLogger(__name__)


def add_variables(
    model: gp.Model, generators: Sequence, time_periods: range
) -> Tuple[gp.tupledict, gp.tupledict]:
    """
    Create binary variables:
      - startup[g,t]  ∈ {0,1}
      - shutdown[g,t] ∈ {0,1}
    """
    idx = [(g.name, t) for g in generators for t in time_periods]
    startup = model.addVars(idx, vtype=GRB.BINARY, name="gen_startup")
    shutdown = model.addVars(idx, vtype=GRB.BINARY, name="gen_shutdown")

    model.__dict__["startup"] = startup
    model.__dict__["shutdown"] = shutdown

    logger.info(
        "Vars(V-205/V-206): startup/shutdown added=%d (pairs=%d)",
        len(idx) * 2,
        len(idx),
    )
    return startup, shutdown
