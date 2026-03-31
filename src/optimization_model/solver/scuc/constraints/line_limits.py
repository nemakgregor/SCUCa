"""ID: C-109 — Line flow limits with overflow slacks.

For each line l and time t:
  f[l,t] <= F[l,t] + ov_pos[l,t]
 -f[l,t] <= F[l,t] + ov_neg[l,t]

with ov_pos, ov_neg >= 0 and penalty in the objective.
"""

import logging
from typing import Sequence
import gurobipy as gp

from .log_utils import record_constraint_stat

logger = logging.getLogger(__name__)


def add_constraints(
    model: gp.Model,
    lines: Sequence,
    line_flow,
    over_pos,
    over_neg,
    time_periods: range,
) -> None:
    n = 0
    for line in lines:
        for t in time_periods:
            F = float(line.normal_limit[t])
            model.addConstr(
                line_flow[line.name, t] <= F + over_pos[line.name, t],
                name=f"flow_pos_limit[{line.name},{t}]",
            )
            model.addConstr(
                -line_flow[line.name, t] <= F + over_neg[line.name, t],
                name=f"flow_neg_limit[{line.name},{t}]",
            )
            n += 2
    logger.info(
        "Cons(C-109): line limits added=%d (lines=%d, T=%d)",
        n,
        len(lines),
        len(time_periods),
    )
    record_constraint_stat(model, "C-109_flow_limits", n)
