"""ID: C-108 — Line flow definition via DC B-theta formulation.

We replace the dense PTDF/ISF formulation with a sparse DC formulation:

1) Line flow definition (orientation follows scenario line source/target):
   f[l,t] = b_l * (theta[src(l),t] - theta[tgt(l),t])

2) Nodal balance at each bus:
   inj[b,t] = sum_out f[l,t] - sum_in f[l,t]

where inj[b,t] = sum_gen_at_bus p_gen[g,t] - load[b,t].
"""

import logging
from typing import Dict, List

import gurobipy as gp

from .log_utils import record_constraint_stat

logger = logging.getLogger(__name__)


def _total_power_expr(gen, t: int, commit, seg_power) -> gp.LinExpr:
    expr = commit[gen.name, t] * float(gen.min_power[t])
    n_segments = len(gen.segments) if gen.segments else 0
    if n_segments > 0:
        expr += gp.quicksum(seg_power[gen.name, t, s] for s in range(n_segments))
    return expr


def add_constraints(
    model: gp.Model,
    scenario,
    commit,
    seg_power,
    bus_angle,
    line_flow,
    time_periods: range,
) -> None:
    buses = scenario.buses
    lines = scenario.lines

    if not buses or not lines:
        record_constraint_stat(model, "C-108_flow_def", 0)
        return

    # Build incidence lists to keep nodal constraints small and deterministic.
    out_lines: Dict[str, List[str]] = {b.name: [] for b in buses}
    in_lines: Dict[str, List[str]] = {b.name: [] for b in buses}
    for ln in lines:
        out_lines[ln.source.name].append(ln.name)
        in_lines[ln.target.name].append(ln.name)

    # Reference bus: fix angle to 0 for numerical stability.
    ref_1b = getattr(scenario, "ptdf_ref_bus_index", buses[0].index)
    ref_bus = None
    for b in buses:
        if b.index == ref_1b:
            ref_bus = b
            break
    if ref_bus is None:
        ref_bus = buses[0]

    n = 0
    for t in time_periods:
        model.addConstr(bus_angle[ref_bus.name, t] == 0.0, name=f"theta_ref[{ref_bus.name},{t}]")
        n += 1

        # Line flow definition.
        for ln in lines:
            b = float(ln.susceptance)
            model.addConstr(
                line_flow[ln.name, t]
                == b * (bus_angle[ln.source.name, t] - bus_angle[ln.target.name, t]),
                name=f"flow_def[{ln.name},{t}]",
            )
            n += 1

        # Nodal balance.
        for bus in buses:
            inj = gp.LinExpr()
            for gen in bus.thermal_units:
                inj += _total_power_expr(gen, t, commit, seg_power)
            inj += -float(bus.load[t])

            out_sum = gp.quicksum(line_flow[name, t] for name in out_lines[bus.name])
            in_sum = gp.quicksum(line_flow[name, t] for name in in_lines[bus.name])
            model.addConstr(out_sum - in_sum == inj, name=f"nodal_bal[{bus.name},{t}]")
            n += 1

    logger.info("Cons(C-108): DC flow + nodal balance added=%d (lines=%d, buses=%d, T=%d, ref_bus=%s)",
                n, len(lines), len(buses), len(time_periods), ref_bus.index)
    record_constraint_stat(model, "C-108_flow_def", n)

