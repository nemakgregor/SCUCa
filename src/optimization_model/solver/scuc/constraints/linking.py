"""ID: C-102 — Segment capacity linking.

Link segment power to commitment:
  0 <= pseg[gen,t,s] <= amount[gen,s,t] * u[gen,t].
If amount < 0 in data, it is treated as 0.
"""

import logging
import gurobipy as gp
from typing import Sequence

from .log_utils import record_constraint_stat

logger = logging.getLogger(__name__)


def add_constraints(
    model: gp.Model,
    generators: Sequence,
    commit,
    seg_power,
    time_periods: range,
) -> None:
    n = 0
    for gen in generators:
        n_segments = len(gen.segments) if gen.segments else 0
        for t in time_periods:
            u_gen_t = commit[gen.name, t]
            for s in range(n_segments):
                amount = float(gen.segments[s].amount[t])
                if amount < 0:
                    amount = 0.0
                model.addConstr(
                    seg_power[gen.name, t, s] <= amount * u_gen_t,
                    name=f"seg_cap[{gen.name},{t},{s}]",
                )
                n += 1
    logger.info("Cons(C-102): segment capacity linking added=%d", n)
    record_constraint_stat(model, "C-102_segment_cap", n)
