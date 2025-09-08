"""ID: C-120/C-121 — Contingency constraints with slacks.

We enforce for every time t:

Line outage (C-120):
  For each contingency c that lists one or more outaged lines m, for every monitored
  line l != m:
      f_l,t + LODF[l,m] * f_m,t <= F_emergency[l,t] + cont_ov_pos[l,t]
     -f_l,t - LODF[l,m] * f_m,t <= F_emergency[l,t] + cont_ov_neg[l,t]

Generator outage (C-121):
  For each contingency c that lists one or more outaged generators g at bus b(g), for
  every monitored line l:
      f_l,t - ISF[l,b(g)] * p_g,t <= F_emergency[l,t] + cont_ov_pos[l,t]
     -f_l,t + ISF[l,b(g)] * p_g,t <= F_emergency[l,t] + cont_ov_neg[l,t]

Notes
- LODF matrix has shape (n_lines, n_lines).
- ISF (PTDF) matrix has shape (n_lines, n_buses-1) for all non-reference buses.
- Slack bus convention: the loss of p_g at bus b(g) is balanced by the reference bus.
  If b(g) equals the reference bus, ISF column is absent (treated as zero).
- Slacks are shared per (line,time) across all contingencies.

New:
- Optional filter_predicate(kind, line_l, out_obj, t, coeff, F_em) -> bool.
  If provided and returns False, the corresponding +/- constraints are not added.
  This enables ML-based redundancy pruning.
"""

import logging
import gurobipy as gp
from scipy.sparse import csc_matrix
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Drop extremely small coefficients to reduce constraint count
_LODF_TOL = 1e-4
_ISF_TOL = 1e-8


def _total_power_expr(gen, t: int, commit, seg_power) -> gp.LinExpr:
    """
    Build a linear expression for total power of generator gen at time t:
      p[gen,t] = u[gen,t] * min_power[gen,t] + sum_s pseg[gen,t,s]
    """
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
    cont_over_pos,
    cont_over_neg,
    filter_predicate: Optional[Callable] = None,
) -> None:
    contingencies = scenario.contingencies or []
    lines = scenario.lines or []
    if not contingencies or not lines or line_flow is None:
        return

    lodf_csc: csc_matrix = scenario.lodf.tocsc()  # (n_lines, n_lines)
    isf_csc: csc_matrix = scenario.isf.tocsc()  # (n_lines, n_buses-1)

    # Bus index mapping: non-reference columns in ascending 1-based bus index
    buses = scenario.buses
    ref_1b = getattr(scenario, "ptdf_ref_bus_index", buses[0].index)
    non_ref_bus_indices = sorted([b.index for b in buses if b.index != ref_1b])
    col_by_bus_1b = {bus_1b: col for col, bus_1b in enumerate(non_ref_bus_indices)}

    # Line row index to line object (index in data is 1-based)
    line_by_row = {ln.index - 1: ln for ln in lines}

    n_cons_line = 0
    n_cons_gen = 0
    n_conts_used_line = 0
    n_conts_used_gen = 0
    n_skipped_line = 0
    n_skipped_gen = 0

    for cont in contingencies:
        # Line-outage constraints (C-120)
        if cont.lines:
            n_conts_used_line += 1
            for out_line in cont.lines:
                mcol = out_line.index - 1
                col = lodf_csc.getcol(mcol)
                col_rows = col.indices.tolist()
                col_vals = col.data.tolist()

                for l_row, alpha_lm in zip(col_rows, col_vals):
                    # Skip the outaged line itself; skip tiny coefficients
                    if l_row == mcol:
                        continue
                    if abs(alpha_lm) < _LODF_TOL:
                        continue

                    line_l = line_by_row.get(l_row)
                    if line_l is None:
                        continue

                    for t in time_periods:
                        F_em = float(line_l.emergency_limit[t])
                        coeff = float(alpha_lm)

                        # ML-based pruning hook
                        if filter_predicate is not None:
                            try:
                                keep = bool(
                                    filter_predicate(
                                        "line", line_l, out_line, t, coeff, F_em
                                    )
                                )
                            except Exception:
                                keep = True
                            if not keep:
                                n_skipped_line += 2  # +/- pair
                                continue

                        # + direction
                        model.addConstr(
                            line_flow[line_l.name, t]
                            + coeff * line_flow[out_line.name, t]
                            <= F_em + cont_over_pos[line_l.name, t],
                            name=f"cont_line_pos[{cont.name},{line_l.name},{out_line.name},{t}]",
                        )
                        # - direction
                        model.addConstr(
                            -line_flow[line_l.name, t]
                            - coeff * line_flow[out_line.name, t]
                            <= F_em + cont_over_neg[line_l.name, t],
                            name=f"cont_line_neg[{cont.name},{line_l.name},{out_line.name},{t}]",
                        )
                        n_cons_line += 2

        # Generator-outage constraints (C-121)
        if getattr(cont, "units", None):
            n_conts_used_gen += 1
            for gen in cont.units:
                bus_1b = gen.bus.index
                # If the outaged generator is at the reference bus, ISF effect is zero
                if bus_1b == ref_1b or bus_1b not in col_by_bus_1b:
                    col = None
                else:
                    col = isf_csc.getcol(col_by_bus_1b[bus_1b])
                if col is None:
                    # ISF column absent => coefficient 0 for all lines (constraints reduce to +/- f_l <= F_em + slack)
                    # We still add them unless filtered (coeff=0).
                    for line_l in lines:
                        for t in time_periods:
                            F_em = float(line_l.emergency_limit[t])
                            coeff = 0.0
                            if filter_predicate is not None:
                                try:
                                    keep = bool(
                                        filter_predicate(
                                            "gen", line_l, gen, t, coeff, F_em
                                        )
                                    )
                                except Exception:
                                    keep = True
                                if not keep:
                                    n_skipped_gen += 2
                                    continue
                            model.addConstr(
                                line_flow[line_l.name, t]
                                <= F_em + cont_over_pos[line_l.name, t],
                                name=f"cont_gen_pos[{cont.name},{line_l.name},{gen.name},{t}]",
                            )
                            model.addConstr(
                                -line_flow[line_l.name, t]
                                <= F_em + cont_over_neg[line_l.name, t],
                                name=f"cont_gen_neg[{cont.name},{line_l.name},{gen.name},{t}]",
                            )
                            n_cons_gen += 2
                else:
                    col_rows = col.indices.tolist()
                    col_vals = col.data.tolist()

                    for l_row, isf_lb in zip(col_rows, col_vals):
                        if abs(isf_lb) < _ISF_TOL:
                            continue
                        line_l = line_by_row.get(l_row)
                        if line_l is None:
                            continue

                        for t in time_periods:
                            F_em = float(line_l.emergency_limit[t])
                            coeff = float(isf_lb)
                            if filter_predicate is not None:
                                try:
                                    keep = bool(
                                        filter_predicate(
                                            "gen", line_l, gen, t, coeff, F_em
                                        )
                                    )
                                except Exception:
                                    keep = True
                                if not keep:
                                    n_skipped_gen += 2
                                    continue
                            p_expr = _total_power_expr(gen, t, commit, seg_power)
                            # + direction: f_l - ISF * p_g <= F_em + slack
                            model.addConstr(
                                line_flow[line_l.name, t] - coeff * p_expr
                                <= F_em + cont_over_pos[line_l.name, t],
                                name=f"cont_gen_pos[{cont.name},{line_l.name},{gen.name},{t}]",
                            )
                            # - direction: -(f_l - ISF * p_g) <= F_em + slack
                            model.addConstr(
                                -line_flow[line_l.name, t] + coeff * p_expr
                                <= F_em + cont_over_neg[line_l.name, t],
                                name=f"cont_gen_neg[{cont.name},{line_l.name},{gen.name},{t}]",
                            )
                            n_cons_gen += 2

    logger.info(
        "Cons(C-120/C-121): line-out=%d (conts_used=%d, skipped=%d), gen-out=%d (conts_used=%d, skipped=%d); lines=%d, T=%d",
        n_cons_line,
        n_conts_used_line,
        n_skipped_line,
        n_cons_gen,
        n_conts_used_gen,
        n_skipped_gen,
        len(lines),
        len(time_periods),
    )
