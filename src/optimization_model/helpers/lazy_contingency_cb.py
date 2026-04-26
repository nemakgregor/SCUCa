from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Iterable

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


@dataclass(frozen=True)
class ContingencyViolation:
    magnitude: float
    kind: str  # "line" | "gen"
    sign: int  # +1 | -1
    line_name: str
    out_name: str
    t: int
    coeff: float


def _violation_key(v: ContingencyViolation) -> Tuple[str, int, str, str, int]:
    return (v.kind, int(v.sign), v.line_name, v.out_name, int(v.t))


def _scenario_index_maps(scenario):
    buses = scenario.buses
    lines = scenario.lines
    ref_1b = getattr(scenario, "ptdf_ref_bus_index", buses[0].index)
    non_ref_bus_indices = sorted([b.index for b in buses if b.index != ref_1b])
    col_by_bus_1b = {bus_1b: col for col, bus_1b in enumerate(non_ref_bus_indices)}
    line_by_row = {ln.index - 1: ln for ln in lines}
    line_by_name = {ln.name: ln for ln in lines}
    gen_by_name = {gen.name: gen for gen in getattr(scenario, "thermal_units", [])}
    return ref_1b, col_by_bus_1b, line_by_row, line_by_name, gen_by_name


def _current_solution_values(
    model: gp.Model, scenario
) -> Tuple[Dict[Tuple[str, int], float], Dict[Tuple[str, int], float]]:
    commit = getattr(model, "commit", None)
    seg = getattr(model, "gen_segment_power", None)
    line_flow = getattr(model, "line_flow", None)
    if any(v is None for v in (commit, seg, line_flow)):
        raise RuntimeError(
            "[lazy_cont] Model is missing required variables (commit, seg, line_flow)."
        )

    T = scenario.time
    f_val: Dict[Tuple[str, int], float] = {}
    for ln in scenario.lines:
        for t in range(T):
            try:
                f_val[(ln.name, t)] = float(line_flow[ln.name, t].X)
            except Exception:
                f_val[(ln.name, t)] = 0.0

    p_val: Dict[Tuple[str, int], float] = {}
    for gen in scenario.thermal_units:
        nS = len(gen.segments) if gen.segments else 0
        for t in range(T):
            try:
                u_t = float(commit[gen.name, t].X)
            except Exception:
                u_t = 0.0
            base = u_t * float(gen.min_power[t])
            if nS > 0:
                ssum = 0.0
                for s in range(nS):
                    try:
                        ssum += float(seg[gen.name, t, s].X)
                    except Exception:
                        pass
                base += ssum
            p_val[(gen.name, t)] = base
    return f_val, p_val


def collect_contingency_violations(
    model: gp.Model,
    scenario,
    config: Optional[LazyContingencyConfig] = None,
    *,
    skip_added: bool = False,
) -> List[ContingencyViolation]:
    cfg = config or LazyContingencyConfig()
    if not getattr(scenario, "contingencies", None) or not getattr(
        scenario, "lines", None
    ):
        return []

    covp = getattr(model, "contingency_overflow_pos", None)
    covn = getattr(model, "contingency_overflow_neg", None)
    if covp is None or covn is None:
        raise RuntimeError(
            "[lazy_cont] Model is missing required variables (cont_overflow_*)."
        )

    lodf_csc: csc_matrix = scenario.lodf.tocsc()
    isf_csc: csc_matrix = scenario.isf.tocsc()
    lines = scenario.lines
    T = scenario.time
    ref_1b, col_by_bus_1b, line_by_row, _, _ = _scenario_index_maps(scenario)
    f_val, p_val = _current_solution_values(model, scenario)
    added_keys = getattr(model, "_lazy_contingency_added_keys", set()) if skip_added else set()

    viols: List[ContingencyViolation] = []

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
                    post = f_val[(line_l.name, t)] + float(alpha) * f_val[
                        (out_line.name, t)
                    ]
                    F_em = float(line_l.emergency_limit[t])
                    ovp_shared = float(covp[line_l.name, t].X)
                    ovn_shared = float(covn[line_l.name, t].X)
                    vpos = post - F_em - ovp_shared
                    vneg = -post - F_em - ovn_shared
                    if vpos > cfg.violation_tol:
                        rec = ContingencyViolation(
                            magnitude=float(vpos),
                            kind="line",
                            sign=+1,
                            line_name=line_l.name,
                            out_name=out_line.name,
                            t=t,
                            coeff=float(alpha),
                        )
                        if _violation_key(rec) not in added_keys:
                            viols.append(rec)
                    if vneg > cfg.violation_tol:
                        rec = ContingencyViolation(
                            magnitude=float(vneg),
                            kind="line",
                            sign=-1,
                            line_name=line_l.name,
                            out_name=out_line.name,
                            t=t,
                            coeff=float(alpha),
                        )
                        if _violation_key(rec) not in added_keys:
                            viols.append(rec)

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
                        ovp_shared = float(covp[line_l.name, t].X)
                        ovn_shared = float(covn[line_l.name, t].X)
                        vpos = post - F_em - ovp_shared
                        vneg = -post - F_em - ovn_shared
                        if vpos > cfg.violation_tol:
                            rec = ContingencyViolation(
                                magnitude=float(vpos),
                                kind="gen",
                                sign=+1,
                                line_name=line_l.name,
                                out_name=gen.name,
                                t=t,
                                coeff=0.0,
                            )
                            if _violation_key(rec) not in added_keys:
                                viols.append(rec)
                        if vneg > cfg.violation_tol:
                            rec = ContingencyViolation(
                                magnitude=float(vneg),
                                kind="gen",
                                sign=-1,
                                line_name=line_l.name,
                                out_name=gen.name,
                                t=t,
                                coeff=0.0,
                            )
                            if _violation_key(rec) not in added_keys:
                                viols.append(rec)
            else:
                col = isf_csc.getcol(col_by_bus_1b[bidx])
                coeff_map = {
                    r: v for r, v in zip(col.indices.tolist(), col.data.tolist())
                }
                for line_l in lines:
                    beta = float(coeff_map.get(line_l.index - 1, 0.0))
                    if abs(beta) < cfg.isf_tol:
                        continue
                    for t in range(T):
                        post = f_val[(line_l.name, t)] - beta * p_val[(gen.name, t)]
                        F_em = float(line_l.emergency_limit[t])
                        ovp_shared = float(covp[line_l.name, t].X)
                        ovn_shared = float(covn[line_l.name, t].X)
                        vpos = post - F_em - ovp_shared
                        vneg = -post - F_em - ovn_shared
                        if vpos > cfg.violation_tol:
                            rec = ContingencyViolation(
                                magnitude=float(vpos),
                                kind="gen",
                                sign=+1,
                                line_name=line_l.name,
                                out_name=gen.name,
                                t=t,
                                coeff=float(beta),
                            )
                            if _violation_key(rec) not in added_keys:
                                viols.append(rec)
                        if vneg > cfg.violation_tol:
                            rec = ContingencyViolation(
                                magnitude=float(vneg),
                                kind="gen",
                                sign=-1,
                                line_name=line_l.name,
                                out_name=gen.name,
                                t=t,
                                coeff=float(beta),
                            )
                            if _violation_key(rec) not in added_keys:
                                viols.append(rec)

    viols.sort(key=lambda x: x.magnitude, reverse=True)
    return viols


def _p_expr(model: gp.Model, gen, t: int) -> gp.LinExpr:
    commit = getattr(model, "commit", None)
    seg = getattr(model, "gen_segment_power", None)
    expr = commit[gen.name, t] * float(gen.min_power[t])
    nS = len(gen.segments) if gen.segments else 0
    if nS > 0:
        expr += gp.quicksum(seg[gen.name, t, s] for s in range(nS))
    return expr


def add_explicit_contingency_constraints(
    model: gp.Model,
    scenario,
    violations: Iterable[ContingencyViolation],
) -> int:
    line_flow = getattr(model, "line_flow", None)
    covp = getattr(model, "contingency_overflow_pos", None)
    covn = getattr(model, "contingency_overflow_neg", None)
    if any(v is None for v in (line_flow, covp, covn)):
        raise RuntimeError(
            "[lazy_cont] Model is missing required variables (line_flow, cont_overflow_*)."
        )

    _, _, _, line_by_name, gen_by_name = _scenario_index_maps(scenario)
    added_keys = getattr(model, "_lazy_contingency_added_keys", None)
    if added_keys is None:
        added_keys = set()
        model._lazy_contingency_added_keys = added_keys

    added = 0
    for v in violations:
        key = _violation_key(v)
        if key in added_keys:
            continue
        line_l = line_by_name[v.line_name]
        if v.kind == "line":
            if v.sign > 0:
                model.addConstr(
                    line_flow[line_l.name, v.t] + v.coeff * line_flow[v.out_name, v.t]
                    <= float(line_l.emergency_limit[v.t]) + covp[line_l.name, v.t],
                    name=f"lazyclose_line_pos[{line_l.name},{v.out_name},{v.t}]",
                )
            else:
                model.addConstr(
                    -line_flow[line_l.name, v.t] - v.coeff * line_flow[v.out_name, v.t]
                    <= float(line_l.emergency_limit[v.t]) + covn[line_l.name, v.t],
                    name=f"lazyclose_line_neg[{line_l.name},{v.out_name},{v.t}]",
                )
        else:
            gen = gen_by_name[v.out_name]
            pexpr = _p_expr(model, gen, v.t)
            if v.sign > 0:
                model.addConstr(
                    line_flow[line_l.name, v.t] - v.coeff * pexpr
                    <= float(line_l.emergency_limit[v.t]) + covp[line_l.name, v.t],
                    name=f"lazyclose_gen_pos[{line_l.name},{v.out_name},{v.t}]",
                )
            else:
                model.addConstr(
                    -line_flow[line_l.name, v.t] + v.coeff * pexpr
                    <= float(line_l.emergency_limit[v.t]) + covn[line_l.name, v.t],
                    name=f"lazyclose_gen_neg[{line_l.name},{v.out_name},{v.t}]",
                )
        added_keys.add(key)
        added += 1
    if added:
        model.update()
    return added


def finalize_lazy_contingency_closure(
    model: gp.Model,
    scenario,
    config: Optional[LazyContingencyConfig] = None,
    *,
    max_rounds: int = 3,
    per_round_time_limit: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    cfg = config or getattr(model, "_lazy_contingency_cfg", None) or LazyContingencyConfig()
    stats = {
        "rounds": 0,
        "constraints_added": 0,
        "remaining_violations": 0,
        "extra_runtime_sec": 0.0,
    }
    if max_rounds <= 0:
        return stats

    status = int(getattr(model, "Status", 0) or 0)
    feasible_statuses = {
        GRB.OPTIMAL,
        GRB.SUBOPTIMAL,
        GRB.TIME_LIMIT,
        GRB.INTERRUPTED,
        GRB.USER_OBJ_LIMIT,
        GRB.SOLUTION_LIMIT,
    }
    if status not in feasible_statuses or getattr(model, "SolCount", 0) <= 0:
        return stats

    old_time_limit = None
    try:
        old_time_limit = float(model.Params.TimeLimit)
    except Exception:
        old_time_limit = None

    cb_cfg = getattr(model, "_lazy_contingency_cfg", None)
    old_top_k = getattr(cb_cfg, "add_top_k", None) if cb_cfg is not None else None
    old_policy = getattr(cb_cfg, "topk_policy", None) if cb_cfg is not None else None

    try:
        if cb_cfg is not None:
            cb_cfg.add_top_k = 0
            cb_cfg.topk_policy = None

        for round_idx in range(1, max_rounds + 1):
            viols = collect_contingency_violations(
                model, scenario, cfg, skip_added=True
            )
            if not viols:
                break
            added = add_explicit_contingency_constraints(model, scenario, viols)
            stats["rounds"] = round_idx
            stats["constraints_added"] += int(added)
            stats["remaining_violations"] = len(viols)
            if added <= 0:
                break

            for var in model.getVars():
                try:
                    var.Start = var.X
                except Exception:
                    pass

            if per_round_time_limit is not None and per_round_time_limit > 0:
                try:
                    model.Params.TimeLimit = float(per_round_time_limit)
                except Exception:
                    pass

            model.optimize()
            try:
                stats["extra_runtime_sec"] += float(getattr(model, "Runtime", 0.0) or 0.0)
            except Exception:
                pass
            if getattr(model, "SolCount", 0) <= 0:
                if verbose:
                    print("[lazy_cont] finalize: re-solve found no feasible solution; stopping.")
                break

        feasible_statuses_final = {
            GRB.OPTIMAL,
            GRB.SUBOPTIMAL,
            GRB.TIME_LIMIT,
            GRB.INTERRUPTED,
            GRB.USER_OBJ_LIMIT,
            GRB.SOLUTION_LIMIT,
        }
        if (
            int(getattr(model, "Status", 0) or 0) in feasible_statuses_final
            and getattr(model, "SolCount", 0) > 0
        ):
            final_viols = collect_contingency_violations(model, scenario, cfg, skip_added=False)
            stats["remaining_violations"] = len(final_viols)
        else:
            stats["remaining_violations"] = -1
        if verbose and stats["constraints_added"] > 0:
            print(
                f"[lazy_cont] final closure: rounds={int(stats['rounds'])}, "
                f"added={int(stats['constraints_added'])}, remaining={int(stats['remaining_violations'])}"
            )
        return stats
    finally:
        if cb_cfg is not None:
            cb_cfg.add_top_k = old_top_k
            cb_cfg.topk_policy = old_policy
        if old_time_limit is not None:
            try:
                model.Params.TimeLimit = old_time_limit
            except Exception:
                pass


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
    ref_1b, col_by_bus_1b, line_by_row, _, _ = _scenario_index_maps(scenario)

    stats = {"incumbents_seen": 0, "lazy_added": 0}
    added_keys = getattr(model, "_lazy_contingency_added_keys", None)
    if added_keys is None:
        added_keys = set()
        model._lazy_contingency_added_keys = added_keys

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

        viols: List[ContingencyViolation] = []

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
                            rec = ContingencyViolation(
                                magnitude=float(vpos),
                                kind="line",
                                sign=+1,
                                line_name=line_l.name,
                                out_name=out_line.name,
                                t=t,
                                coeff=float(alpha),
                            )
                            if _violation_key(rec) not in added_keys:
                                viols.append(rec)
                        if vneg > cfg.violation_tol:
                            rec = ContingencyViolation(
                                magnitude=float(vneg),
                                kind="line",
                                sign=-1,
                                line_name=line_l.name,
                                out_name=out_line.name,
                                t=t,
                                coeff=float(alpha),
                            )
                            if _violation_key(rec) not in added_keys:
                                viols.append(rec)

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
                                rec = ContingencyViolation(
                                    magnitude=float(vpos),
                                    kind="gen",
                                    sign=+1,
                                    line_name=line_l.name,
                                    out_name=gen.name,
                                    t=t,
                                    coeff=0.0,
                                )
                                if _violation_key(rec) not in added_keys:
                                    viols.append(rec)
                            if vneg > cfg.violation_tol:
                                rec = ContingencyViolation(
                                    magnitude=float(vneg),
                                    kind="gen",
                                    sign=-1,
                                    line_name=line_l.name,
                                    out_name=gen.name,
                                    t=t,
                                    coeff=0.0,
                                )
                                if _violation_key(rec) not in added_keys:
                                    viols.append(rec)
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
                                rec = ContingencyViolation(
                                    magnitude=float(vpos),
                                    kind="gen",
                                    sign=+1,
                                    line_name=line_l.name,
                                    out_name=gen.name,
                                    t=t,
                                    coeff=float(beta),
                                )
                                if _violation_key(rec) not in added_keys:
                                    viols.append(rec)
                            if vneg > cfg.violation_tol:
                                rec = ContingencyViolation(
                                    magnitude=float(vneg),
                                    kind="gen",
                                    sign=-1,
                                    line_name=line_l.name,
                                    out_name=gen.name,
                                    t=t,
                                    coeff=float(beta),
                                )
                                if _violation_key(rec) not in added_keys:
                                    viols.append(rec)

        if not viols:
            return

        # Rank violations by severity
        viols.sort(key=lambda x: x.magnitude, reverse=True)

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
        for v in viols:
            key = _violation_key(v)
            if key in added_keys:
                continue
            line_l = next((ln for ln in lines if ln.name == v.line_name), None)
            if line_l is None:
                continue
            if v.kind == "line":
                if v.sign > 0:
                    m.cbLazy(
                        line_flow[line_l.name, v.t]
                        + v.coeff * line_flow[v.out_name, v.t]
                        <= float(line_l.emergency_limit[v.t]) + covp[line_l.name, v.t]
                    )
                else:
                    m.cbLazy(
                        -line_flow[line_l.name, v.t]
                        - v.coeff * line_flow[v.out_name, v.t]
                        <= float(line_l.emergency_limit[v.t]) + covn[line_l.name, v.t]
                    )
            else:
                gen = next((g for g in scenario.thermal_units if g.name == v.out_name), None)
                if gen is None:
                    continue
                pexpr = _p_expr(m, gen, v.t)
                if v.sign > 0:
                    m.cbLazy(
                        line_flow[line_l.name, v.t] - v.coeff * pexpr
                        <= float(line_l.emergency_limit[v.t]) + covp[line_l.name, v.t]
                    )
                else:
                    m.cbLazy(
                        -line_flow[line_l.name, v.t] + v.coeff * pexpr
                        <= float(line_l.emergency_limit[v.t]) + covn[line_l.name, v.t]
                    )
            added_keys.add(key)
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
