"""ID: V-207/V-208/V-209 — Line flow variables and slack.

- V-207: Line flow variables f[l,t] (free continuous)
- V-208: Positive overflow slack ov_pos[l,t] >= 0
- V-209: Negative overflow slack ov_neg[l,t] >= 0
"""

import logging
from typing import Sequence, Tuple
import gurobipy as gp
from gurobipy import GRB

logger = logging.getLogger(__name__)


def add_variables(
    model: gp.Model, lines: Sequence, time_periods: range
) -> Tuple[gp.tupledict, gp.tupledict, gp.tupledict]:
    """
    Create:
      - line_flow[l,t] ∈ ℝ (free)
      - line_overflow_pos[l,t] ≥ 0
      - line_overflow_neg[l,t] ≥ 0
    """
    idx = [(line.name, t) for line in lines for t in time_periods]
    before = model.NumVars

    line_flow = model.addVars(idx, lb=-GRB.INFINITY, name="line_flow")
    over_pos = model.addVars(idx, lb=0.0, name="line_overflow_pos")
    over_neg = model.addVars(idx, lb=0.0, name="line_overflow_neg")

    model.__dict__["line_flow"] = line_flow
    model.__dict__["line_overflow_pos"] = over_pos
    model.__dict__["line_overflow_neg"] = over_neg

    logger.info(
        "Vars(V-207/208/209): line_flow=%d, overflow_pos=%d, overflow_neg=%d (lines=%d, T=%d)",
        len(idx),
        len(idx),
        len(idx),
        len(lines),
        len(time_periods),
    )
    return line_flow, over_pos, over_neg
