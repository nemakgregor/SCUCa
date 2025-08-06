"""solver.constraints
---------------------
Functions that attach *constraints* to the model.

Only system‑wide balance exists for step‑1; future steps will add generator‑
specific, network, reserve, etc.
"""

import gurobipy as gp
from typing import Sequence, Mapping


def add_constraints(
    model: gp.Model,
    buses: Sequence,
    gen_power: Mapping,
    units: Sequence,
    time_periods: range,
) -> None:
    for t in time_periods:
        load = sum(bus.load[t] for bus in buses)
        generation = gp.quicksum(gen_power[u.name, t] for u in units)
        model.addConstr(generation == load, name=f"Balance_{t}")
