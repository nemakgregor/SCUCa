"""ID: C-105 — Reserve requirement.

For each reserve product k and time t:
  sum_{g eligible} r[k,g,t] + shortfall[k,t] >= requirement[k,t]
"""

import logging
from typing import Sequence
import gurobipy as gp

from .log_utils import record_constraint_stat

logger = logging.getLogger(__name__)


def add_constraints(
    model: gp.Model,
    reserves: Sequence,
    reserve,
    shortfall,
    time_periods: range,
) -> None:
    """
    For each reserve product k and time t:
        sum_{g eligible} r[k,g,t] + shortfall[k,t] >= requirement[k,t]
    """
    n = 0
    for r in reserves:
        for t in time_periods:
            provided = gp.quicksum(reserve[r.name, g.name, t] for g in r.thermal_units)
            model.addConstr(
                provided + shortfall[r.name, t] >= float(r.amount[t]),
                name=f"reserve_requirement[{r.name},{t}]",
            )
            n += 1
    logger.info("Cons(C-105): reserve requirement added=%d", n)
    record_constraint_stat(model, "C-105_reserve_requirement", n)
