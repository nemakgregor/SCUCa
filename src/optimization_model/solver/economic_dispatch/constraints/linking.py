"""Link segment power to commitment: 0 <= pseg[g,t,s] <= amount[g,s,t] * u[g,t]."""

import gurobipy as gp
from typing import Sequence


def add_constraints(
    model: gp.Model,
    generators: Sequence,
    commit,
    seg_power,
    time_periods: range,
) -> None:
    for g in generators:
        Sg = len(g.segments) if g.segments else 0
        for t in time_periods:
            u_gt = commit[g.name, t]
            for s in range(Sg):
                amount = float(g.segments[s].amount[t])
                if amount < 0:
                    amount = 0.0
                model.addConstr(
                    seg_power[g.name, t, s] <= amount * u_gt,
                    name=f"seg_cap[{g.name},{t},{s}]",
                )
