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

    # Optional bandit policy (object with choose_k(context)->int, update(chosen_k,reward,context)->None)
    topk_policy: object = None


def attach_lazy_contingency_callback(
    model: gp.Model, scenario, config: Optional[LazyContingencyConfig] = None
) -> None:
    """
    Register a Gurobi callback that enforces N-1 contingency constraints lazily.
    With optional bandit for adaptive top-k control.
    """
    cfg = config or LazyContingencyConfig()

    if not getattr(scenario, "contingencies", None) or not getattr(
        scenario, "lines", None
    ):
        return

    commit = getattr(model, "commit", None)
    seg = getattr(model, "gen_segment_power", None)
    line_flow = getattr(model, "line_flow", None)
    covp = getattr(model, "contingency_overflow_pos", None)
    covn = getattr(model, "contingency_overflow_neg", None)

    if any(v is None for v in (commit, seg, line_flow, covp, covn)):
        raise RuntimeError(
            "[lazy_cont] Model is missing required variables (commit, seg, line_flow, cont_overflow_*)."
        )

    lodf_csc: csc_matrix = scenario.lodf.tocsc()
    isf_csc: csc_matrix = scenario.isf.tocsc()

    buses = scenario.buses
    lines = scenario.lines
    T = scenario.time
    ref_1b = getattr(scenario, "ptdf_ref_bus_index", buses[0].index)
    non_ref_bus_indices = sorted([b.index for b in buses if b.index != ref_1b])
    col_by_bus_1b = {bus_1b: col for col, bus_1b in enumerate(non_ref_bus_indices)}
    line_by_row = {ln.index - 1: ln for ln in lines}

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

        f_val: Dict[Tuple[str, int], float] = {}
        for ln in lines:
            for t in range(T):
                try:
                    f_val[(ln.name, t)] = float(m.cbGetSolution(line_flow[ln.name, t]))
                except Exception:
                    f_val[(ln.name, t)] = 0.0

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

        viols: List[Tuple[float, str, int, object, object, int, float]] = []

        # Line-outages
        for cont in scenario.contingencies or []:
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
                        post = (
                            f_val[(line_l.name, t)]
                            + float(alpha) * f_val[(out_line.name, t)]
                        )
                        F_em = float(line_l.emergency_limit[t])
                        vpos = post - F_em
                        vneg = -post - F_em
                        if vpos > cfg.violation_tol:
                            viols.append(
                                (vpos, "line", +1, line_l, out_line, t, float(alpha))
                            )
                        if vneg > cfg.violation_tol:
                            viols.append(
                                (vneg, "line", -1, line_l, out_line, t, float(alpha))
                            )

        # Gen-outages
        for cont in scenario.contingencies or []:
            if not getattr(cont, "units", None):
                continue
            for gen in cont.units:
                bidx = gen.bus.index
                if bidx == ref_1b or bidx not in col_by_bus_1b:
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

        # Rank violations by severity
        viols.sort(key=lambda x: x[0], reverse=True)

        # Adaptive top-K via bandit policy (optional)
        eff_top_k = int(cfg.add_top_k) if cfg.add_top_k and cfg.add_top_k > 0 else 0
        if getattr(cfg, "topk_policy", None) is not None:
            try:
                context = {
                    "violations": len(viols),
                    "incumbents": stats["incumbents_seen"],
                }
                chosen = int(cfg.topk_policy.choose_k(context))
                eff_top_k = max(0, chosen)
            except Exception:
                pass

        if eff_top_k > 0:
            viols = viols[:eff_top_k]

        added = 0
        for _, kind, sign, line_l, out_obj, t, coeff in viols:
            if kind == "line":
                if sign > 0:
                    m.cbLazy(
                        line_flow[line_l.name, t] + coeff * line_flow[out_obj.name, t]
                        <= float(line_l.emergency_limit[t]) + covp[line_l.name, t]
                    )
                else:
                    m.cbLazy(
                        -line_flow[line_l.name, t] - coeff * line_flow[out_obj.name, t]
                        <= float(line_l.emergency_limit[t]) + covn[line_l.name, t]
                    )
            else:
                pexpr = _p_expr(out_obj, t)
                if sign > 0:
                    m.cbLazy(
                        line_flow[line_l.name, t] - coeff * pexpr
                        <= float(line_l.emergency_limit[t]) + covp[line_l.name, t]
                    )
                else:
                    m.cbLazy(
                        -line_flow[line_l.name, t] + coeff * pexpr
                        <= float(line_l.emergency_limit[t]) + covn[line_l.name, t]
                    )
            added += 1

        stats["lazy_added"] += added
        # Bandit update with heuristic reward (prefer lower additions)
        if getattr(cfg, "topk_policy", None) is not None:
            try:
                cfg.topk_policy.update(
                    chosen_k=eff_top_k,
                    reward=float(-added),
                    context={"violations": len(viols)},
                )
            except Exception:
                pass

        if cfg.verbose:
            try:
                m.cbMessage(
                    f"[lazy_cont] MIPSOL#{stats['incumbents_seen']}: added {added} lazy constraints (total={stats['lazy_added']}).\n"
                )
            except Exception:
                pass

    try:
        model.Params.LazyConstraints = 1
    except Exception:
        pass
    model._lazy_contingency_cfg = cfg
    model._lazy_contingency_stats = stats
    model._lazy_contingency_callback = _cb
    model._lazy_contingency_attached = True
    model._callback = _cb
    model._has_callback = True
