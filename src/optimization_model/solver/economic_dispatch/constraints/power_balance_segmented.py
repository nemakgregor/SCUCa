"""System-wide power balance using commitment and segment power."""

import gurobipy as gp
from typing import Sequence


def add_constraints(
    model: gp.Model,
    total_load: Sequence[float],
    generators: Sequence,
    commit,
    seg_power,
    time_periods: range,
) -> None:
    for t in time_periods:
        production_t = gp.LinExpr()
        for g in generators:
            # Minimum output when committed
            production_t += commit[g.name, t] * float(g.min_power[t])

            # Incremental segments
            Sg = len(g.segments) if g.segments else 0
            if Sg > 0:
                production_t += gp.quicksum(seg_power[g.name, t, s] for s in range(Sg))

        model.addConstr(
            production_t == float(total_load[t]), name=f"power_balance[{t}]"
        )
