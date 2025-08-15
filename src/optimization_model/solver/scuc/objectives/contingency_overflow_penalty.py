import gurobipy as gp
from typing import Sequence


def build_expression(
    model: gp.Model,
    lines: Sequence,
    time_periods: range,
    contingency_penalty_factor: float,
) -> gp.LinExpr:
    """
    Contingency overflow penalty with multiplier:
      sum_{l,t} (cont_ovp[l,t] + cont_ovn[l,t]) * flow_penalty[l,t] * factor
    """
    expr = gp.LinExpr()
    if not lines:
        return expr
    cont_ovp = getattr(model, "contingency_overflow_pos", None)
    cont_ovn = getattr(model, "contingency_overflow_neg", None)
    if cont_ovp is None or cont_ovn is None:
        return expr

    for line in lines:
        for t in time_periods:
            pen = float(line.flow_penalty[t]) * contingency_penalty_factor
            expr += pen * (cont_ovp[line.name, t] + cont_ovn[line.name, t])
    return expr
