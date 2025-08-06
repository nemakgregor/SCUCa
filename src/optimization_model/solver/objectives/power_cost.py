"""solver.objective
-------------------
Build the objective function.

Step‑1 uses a rough linear cost based on the first segment of each unit cost
curve."""

import gurobipy as gp
from typing import Sequence, Mapping


def set_objective(
    model: gp.Model,
    units: Sequence,
    gen_power: Mapping,
    time_periods: range,
) -> None:
    """Attach a simple linear cost objective to *model*.

    Parameters
    ----------
    model : gp.Model
    units : iterable of ThermalUnit / ProfiledUnit objects having .segments
    gen_power : VarDict keyed by (unit_name, t)
    time_periods : range of time indices
    """
    expr = gp.LinExpr()

    for unit in units:
        # Thermal units have at least one cost segment.
        if not hasattr(unit, "segments") or len(unit.segments) == 0:
            raise ValueError(f"Unit {unit.name} has no cost segments; cannot build ED objective.")

        first_segment_cost = unit.segments[0].cost  # list[float] length = T

        for t in time_periods:
            marginal = first_segment_cost[t]
            expr.addTerms(marginal, gen_power[unit.name, t])

    model.setObjective(expr, gp.GRB.MINIMIZE)
    model.getObjective().setAttr("ObjName", "ED_Cost")
