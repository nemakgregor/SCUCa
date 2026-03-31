"""ID: C-103 — System-wide power balance.

Sum over generators of:
  u[gen,t] * min_power[gen,t] + sum_s pseg[gen,t,s]
must equal total system load at time t.
"""

import logging
import gurobipy as gp
from typing import Sequence

from .log_utils import record_constraint_stat

logger = logging.getLogger(__name__)


def add_constraints(
    model: gp.Model,
    total_load: Sequence[float],
    generators: Sequence,
    commit,
    seg_power,
    time_periods: range,
) -> None:
    for t in time_periods:
        production_t = gp.LinExpr()
        for gen in generators:
            production_t += commit[gen.name, t] * float(gen.min_power[t])
            n_segments = len(gen.segments) if gen.segments else 0
            if n_segments > 0:
                production_t += gp.quicksum(
                    seg_power[gen.name, t, s] for s in range(n_segments)
                )

        model.addConstr(
            production_t == float(total_load[t]), name=f"power_balance[{t}]"
        )
    logger.info("Cons(C-103): power balance equalities added=%d", len(time_periods))
    record_constraint_stat(model, "C-103_power_balance", len(time_periods))
