"""ID: C-101 — Commitment fixing constraints.

Fix commitment variables to known statuses when provided by data:
- If status[gen,t] = True  -> u[gen,t] == 1
- If status[gen,t] = False -> u[gen,t] == 0
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
    time_periods: range,
) -> None:
    n_on = 0
    n_off = 0
    for gen in generators:
        for t in time_periods:
            status = gen.commitment_status[t] if gen.commitment_status else None
            if status is True:
                model.addConstr(
                    commit[gen.name, t] == 1, name=f"fix_commit_on[{gen.name},{t}]"
                )
                n_on += 1
            elif status is False:
                model.addConstr(
                    commit[gen.name, t] == 0, name=f"fix_commit_off[{gen.name},{t}]"
                )
                n_off += 1
    logger.info(
        "Cons(C-101): commitment fixing on=%d, off=%d, total=%d",
        n_on,
        n_off,
        n_on + n_off,
    )
    total = n_on + n_off
    record_constraint_stat(model, "C-101_fix_on", n_on)
    record_constraint_stat(model, "C-101_fix_off", n_off)
    record_constraint_stat(model, "C-101_total", total)
