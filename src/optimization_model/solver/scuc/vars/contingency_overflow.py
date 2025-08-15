"""ID: V-212/V-213 — Contingency overflow slacks.

- V-212: cont_overflow_pos[l,t] >= 0
- V-213: cont_overflow_neg[l,t] >= 0

These slacks are shared across all contingencies for a given (line,time).
"""

import logging
from typing import Sequence, Tuple
import gurobipy as gp

logger = logging.getLogger(__name__)


def add_variables(
    model: gp.Model, lines: Sequence, time_periods: range
) -> Tuple[gp.tupledict, gp.tupledict]:
    idx = [(line.name, t) for line in lines for t in time_periods]
    ovp = model.addVars(idx, lb=0.0, name="cont_overflow_pos")
    ovn = model.addVars(idx, lb=0.0, name="cont_overflow_neg")
    model.__dict__["contingency_overflow_pos"] = ovp
    model.__dict__["contingency_overflow_neg"] = ovn
    logger.info(
        "Vars(V-212/V-213): contingency overflow slacks added=%d (lines=%d, T=%d)",
        len(idx) * 2,
        len(lines),
        len(time_periods),
    )
    return ovp, ovn
