import gurobipy as gp
from typing import Sequence


def build_expression(
    model: gp.Model, lines: Sequence, time_periods: range
) -> gp.LinExpr:
    """
    Base-case line overflow penalty:
      sum_{l,t} (ov_pos[l,t] + ov_neg[l,t]) * flow_penalty[l,t]
    """
    expr = gp.LinExpr()
    if not lines:
        return expr
    over_pos = getattr(model, "line_overflow_pos", None)
    over_neg = getattr(model, "line_overflow_neg", None)
    if over_pos is None or over_neg is None:
        return expr
    for line in lines:
        for t in time_periods:
            pen = float(line.flow_penalty[t])
            expr += pen * (over_pos[line.name, t] + over_neg[line.name, t])
    return expr
