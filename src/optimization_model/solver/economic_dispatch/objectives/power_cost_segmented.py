"""Production cost objective for segmented ED with commitment.

Minimize:
  sum_{g,t} u[g,t] * min_power_cost[g,t] +
  sum_{g,t,s} pseg[g,t,s] * seg_cost[g,s,t]
"""

import gurobipy as gp
from typing import Sequence


def set_objective(
    model: gp.Model,
    generators: Sequence,
    commit,
    seg_power,
    time_periods: range,
) -> None:
    obj = gp.LinExpr()
    for g in generators:
        Sg = len(g.segments) if g.segments else 0
        for t in time_periods:
            # fixed cost part due to minimum output when on
            obj += commit[g.name, t] * float(g.min_power_cost[t])

            # marginal cost on each segment
            for s in range(Sg):
                obj += seg_power[g.name, t, s] * float(g.segments[s].cost[t])

    model.setObjective(obj, gp.GRB.MINIMIZE)
    try:
        model.getObjective().setAttr("ObjName", "TotalProductionCost")
    except Exception:
        pass