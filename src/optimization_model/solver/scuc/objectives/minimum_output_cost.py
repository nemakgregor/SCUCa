import gurobipy as gp
from typing import Sequence


def build_expression(
    model: gp.Model, generators: Sequence, commit, time_periods: range
) -> gp.LinExpr:
    """
    Sum of minimum-output (no-load) cost:
      sum_{g,t} u[g,t] * min_power_cost[g,t]
    """
    expr = gp.LinExpr()
    for g in generators:
        for t in time_periods:
            expr += commit[g.name, t] * float(g.min_power_cost[t])
    return expr
