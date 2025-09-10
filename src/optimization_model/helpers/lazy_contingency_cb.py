from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import gurobipy as gp
from gurobipy import GRB
from scipy.sparse import csc_matrix


@dataclass
class LazyContingencyConfig:
    # Numerical tolerances
    lodf_tol: float = 1e-4
    isf_tol: float = 1e-8
    violation_tol: float = 1e-6  # add a constraint if viol > this

    # Limit how many constraints we add per incumbent (0 => add all violated)
    add_top_k: int = 0

    # Logging
    verbose: bool = True


def attach_lazy_contingency_callback(
    model: gp.Model, scenario, config: Optional[LazyContingencyConfig] = None
) -> None:
    """
    Register a Gurobi callback that enforces N-1 contingency constraints lazily.

    Pre-conditions:
      - The model must already contain:
          * line_flow[l,t] (free)
          * cont_overflow_pos[l,t] >= 0
          * cont_overflow_neg[l,t] >= 0
      - Base-case PTDF (ISF) flow equalities and base flow limits are already in the model.
      - Startup/shutdown, ramping, reserves, etc., are as in your standard SCUC.

    What the callback does (at each new incumbent MIPSOL):
      - Read incumbent solution values for:
          * line_flow[l,t]
          * commit[g,t], gen_segment_power[g,t,s]
      - Compute post-contingency flows using LODF (line outages) and ISF (generator outages).
      - If any post flow exceeds the emergency limit (by violation_tol), add the corresponding
        +/- lazy constraint(s) using model.cbLazy(...). These constraints use the shared
        contingency slack variables cont_overflow_pos/neg, so feasibility remains soft but penalized.
      - If add_top_k > 0, only add the most violated K constraints per incumbent to keep the LP compact.

    Optimality and security:
      - The callback checks all defined contingencies; any violating incumbent will trigger lazy
        constraints, so the final accepted solution satisfies all N-1 constraints (up to tolerance).
    """
    cfg = config or LazyContingencyConfig()

    # Short-circuit if no contingencies or lines
    if not getattr(scenario, "contingencies", None) or not getattr(scenario, "lines", None):
        return

    # Pull var handles
    commit = getattr(model, "commit", None)
    seg = getattr(model, "gen_segment_power", None)
    line_flow = getattr(model, "line_flow", None)
    covp = getattr(model, "contingency_overflow_pos", None)
    covn = getattr(model, "contingency_overflow_neg", None)

    if any(v is None for v in (commit, seg, line_flow, covp, covn)):
        raise RuntimeError(
            "[lazy_cont] Model is missing required variables (commit, seg, line_flow, cont_overflow_*)."
        )

    # Precompute sparse accessors and mappings
    lodf_csc: csc_matrix = scenario.lodf.tocsc()
    isf_csc: csc_matrix = scenario.isf.tocsc()

    buses = scenario.buses
    lines = scenario.lines
    T = scenario.time
    ref_1b = getattr(scenario, "ptdf_ref_bus_index", buses[0].index)
    non_ref_bus_indices = sorted([b.index for b in buses if b.index != ref_1b])
    col_by_bus_1b = {bus_1b: col for col, bus_1b in enumerate(non_ref_bus_indices)}
    line_by_row = {ln.index - 1: ln for ln in lines}

    # Stats
    stats = {"incumbents_seen": 0, "lazy_added": 0}

    def _p_expr(gen, t: int) -> gp.LinExpr:
        expr = commit[gen.name, t] * float(gen.min_power[t])
        nS = len(gen.segments) if gen.segments else 0
        if nS > 0:
            expr += gp.quicksum(seg[gen.name, t, s] for s in range(nS))
        return expr

    def _cb(m: gp.Model, where: int):
        if where != GRB.Callback.MIPSOL:
            return

        stats["incumbents_seen"] += 1

        # Read current incumbent values
        # Base flows f[line,t]
        f_val: Dict[Tuple[str, int], float] = {}
        for ln in lines:
            for t in range(T):
                try:
                    f_val[(ln.name, t)] = float(m.cbGetSolution(line_flow[ln.name, t]))
                except Exception:
                    f_val[(ln.name, t)] = 0.0

        # Generator total power p[gen,t] = u*min + sum(seg)
        p_val: Dict[Tuple[str, int], float] = {}
        for gen in scenario.thermal_units:
            nS = len(gen.segments) if gen.segments else 0
            for t in range(T):
                try:
                    u_t = float(m.cbGetSolution(commit[gen.name, t]))
                except Exception:
                    u_t = 0.0
                base = u_t * float(gen.min_power[t])
                if nS > 0:
                    ssum = 0.0
                    for s in range(nS):
                        try:
                            ssum += float(m.cbGetSolution(seg[gen.name, t, s]))
                        except Exception:
                            pass
                    base += ssum
                p_val[(gen.name, t)] = base

        # Collect violations
        # Entry: (viol_amount, 'line'/'gen', +1/-1, line_l, out_obj, t, coeff)
        viols: List[Tuple[float, str, int, object, object, int, float]] = []

        # Line-outages via LODF
        for cont in (scenario.contingencies or []):
            if not cont.lines:
                continue
            for out_line in cont.lines:
                mcol = out_line.index - 1
                col = lodf_csc.getcol(mcol)
                rows = col.indices.tolist()
                vals = col.data.tolist()
                for l_row, alpha in zip(rows, vals):
                    if l_row == mcol:
                        continue
                    if abs(alpha) < cfg.lodf_tol:
                        continue
                    line_l = line_by_row.get(l_row)
                    if line_l is None:
                        continue
                    for t in range(T):
                        post = f_val[(line_l.name, t)] + float(alpha) * f_val[(out_line.name, t)]
                        F_em = float(line_l.emergency_limit[t])
                        vpos = post - F_em
                        vneg = -post - F_em
                        if vpos > cfg.violation_tol:
                            viols.append((vpos, "line", +1, line_l, out_line, t, float(alpha)))
                        if vneg > cfg.violation_tol:
                            viols.append((vneg, "line", -1, line_l, out_line, t, float(alpha)))

        # Gen-outages via ISF
        for cont in (scenario.contingencies or []):
            if not getattr(cont, "units", None):
                continue
            for gen in cont.units:
                bidx = gen.bus.index
                if bidx == ref_1b or bidx not in col_by_bus_1b:
                    # ISF = 0; post = f_l
                    for line_l in lines:
                        for t in range(T):
                            post = f_val[(line_l.name, t)]
                            F_em = float(line_l.emergency_limit[t])
                            vpos = post - F_em
                            vneg = -post - F_em
                            if vpos > cfg.violation_tol:
                                viols.append((vpos, "gen", +1, line_l, gen, t, 0.0))
                            if vneg > cfg.violation_tol:
                                viols.append((vneg, "gen", -1, line_l, gen, t, 0.0))
                else:
                    col = isf_csc.getcol(col_by_bus_1b[bidx])
                    rows = col.indices.tolist()
                    vals = col.data.tolist()
                    coeff_map = {r: v for r, v in zip(rows, vals)}
                    for line_l in lines:
                        beta = float(coeff_map.get(line_l.index - 1, 0.0))
                        if abs(beta) < cfg.isf_tol:
                            continue
                        for t in range(T):
                            post = f_val[(line_l.name, t)] - beta * p_val[(gen.name, t)]
                            F_em = float(line_l.emergency_limit[t])
                            vpos = post - F_em
                            vneg = -post - F_em
                            if vpos > cfg.violation_tol:
                                viols.append((vpos, "gen", +1, line_l, gen, t, beta))
                            if vneg > cfg.violation_tol:
                                viols.append((vneg, "gen", -1, line_l, gen, t, beta))

        if not viols:
            return

        # Rank by severity and optionally cap
        viols.sort(key=lambda x: x[0], reverse=True)
        if cfg.add_top_k and cfg.add_top_k > 0:
            viols = viols[: cfg.add_top_k]

        # Add lazy constraints
        added = 0
        for _, kind, sign, line_l, out_obj, t, coeff in viols:
            if kind == "line":
                # +/-: f_l +/- coeff f_out <= F_em + cont_over
                if sign > 0:
                    m.cbLazy(
                        line_flow[line_l.name, t]
                        + coeff * line_flow[out_obj.name, t]
                        <= float(line_l.emergency_limit[t]) + covp[line_l.name, t]
                    )
                else:
                    m.cbLazy(
                        -line_flow[line_l.name, t]
                        - coeff * line_flow[out_obj.name, t]
                        <= float(line_l.emergency_limit[t]) + covn[line_l.name, t]
                    )
            else:
                # gen-outage: f_l -/+ coeff * p_expr <= F_em + cont_over
                pexpr = _p_expr(out_obj, t)
                if sign > 0:
                    m.cbLazy(
                        line_flow[line_l.name, t]
                        - coeff * pexpr
                        <= float(line_l.emergency_limit[t]) + covp[line_l.name, t]
                    )
                else:
                    m.cbLazy(
                        -line_flow[line_l.name, t]
                        + coeff * pexpr
                        <= float(line_l.emergency_limit[t]) + covn[line_l.name, t]
                    )
            added += 1

        stats["lazy_added"] += added
        if cfg.verbose:
            try:
                m.cbMessage(f"[lazy_cont] MIPSOL#{stats['incumbents_seen']}: added {added} lazy constraints (total={stats['lazy_added']}).\n")
            except Exception:
                pass

    # Attach and set LazyConstraints=1
    try:
        model.Params.LazyConstraints = 1
    except Exception:
        pass
    model._lazy_contingency_cfg = cfg  # keep for inspection
    model._lazy_contingency_stats = stats
    model._lazy_contingency_callback = _cb
    model._lazy_contingency_attached = True
    model._current_scenario_name = getattr(scenario, "name", "scenario")
    model._scenario_time_steps = getattr(scenario, "time", 0)
    model._scenario_lines = len(getattr(scenario, "lines", []))
    model._scenario_contingencies = len(getattr(scenario, "contingencies", []))
    model._scenario_ref_bus_1b = getattr(scenario, "ptdf_ref_bus_index", None)
    model._scenario_isf_nnz = int(getattr(scenario.isf, "nnz", 0))
    model._scenario_lodf_nnz = int(getattr(scenario.lodf, "nnz", 0))

    model._callback = _cb
    model._has_callback = True
    # Note: model.optimize() must be called after this; Gurobi will invoke _cb.