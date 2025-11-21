"""ID: C-108 — Line flow definition using PTDF.

Base-case DC flows:
  f[line,t] = sum_{b ∈ non_ref_buses} ISF[line, b] * inj[b,t]

where:
  inj[b,t] = sum_{gen at bus b} (u[gen,t]*min[gen,t] + sum_s pseg[gen,t,s]) - load[b,t]

Reference bus used is scenario.ptdf_ref_bus_index (1-based); ISF columns correspond
to all buses except the reference bus, in ascending 1-based index order.
"""

import logging
import gurobipy as gp
from scipy.sparse import csr_matrix

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
    line_flow,
    time_periods: range,
) -> None:
    isf: csr_matrix = scenario.isf.tocsr()
    buses = scenario.buses
    lines = scenario.lines
    ref_1b = getattr(scenario, "ptdf_ref_bus_index", buses[0].index)
    non_ref_bus_indices = sorted([b.index for b in buses if b.index != ref_1b])

    n = 0
    for t in time_periods:
        inj_expr_by_busidx = {}
        for b in buses:
            expr = gp.LinExpr()
            for gen in b.thermal_units:
                expr += _total_power_expr(gen, t, commit, seg_power)
            expr += -float(b.load[t])
            inj_expr_by_busidx[b.index] = expr

        for line in lines:
            l_row = isf.getrow(line.index - 1)
            f_expr = gp.LinExpr()
            for col, coeff in zip(l_row.indices.tolist(), l_row.data.tolist()):
                bus_1b = non_ref_bus_indices[col]
                f_expr += float(coeff) * inj_expr_by_busidx[bus_1b]

            model.addConstr(
                line_flow[line.name, t] == f_expr,
                name=f"flow_def[{line.name},{t}]",
            )
            n += 1
    logger.info(
        "Cons(C-108): line flow PTDF equalities added=%d (ISF shape=%s nnz=%d, ref_bus=%s)",
        n,
        getattr(isf, "shape", None),
        getattr(isf, "nnz", None),
        ref_1b,
    )
    record_constraint_stat(model, "C-108_flow_def", n)
