import gurobipy as gp
from typing import Sequence


def build_expression(
    model: gp.Model, generators: Sequence, seg_power, time_periods: range
) -> gp.LinExpr:
    """
    Sum of energy cost on piecewise-linear segments:
      sum_{g,t,s} pseg[g,t,s] * seg_cost[g,s,t]
    """
    expr = gp.LinExpr()
    for g in generators:
        Sg = len(g.segments) if g.segments else 0
        for t in time_periods:
            for s in range(Sg):
                expr += seg_power[g.name, t, s] * float(g.segments[s].cost[t])
    return expr
