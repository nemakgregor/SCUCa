from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

from src.data_preparation.data_structure import UnitCommitmentScenario
from src.optimization_model.SCUC_solver.scuc_model_builder import build_model
from src.optimization_model.helpers.lazy_contingency_cb import (
    LazyContingencyConfig,
    attach_lazy_contingency_callback,
    optimize_with_lazy_callback,
)
from src.scuc_sr.constraint_enumerator import ConstraintEvent, enumerate_potential_constraints


@dataclass
class ActiveSetConfig:
    time_limit: int = 600
    mip_gap: float = 0.05
    lodf_tol: float = 1e-4
    isf_tol: float = 1e-8
    violation_tol: float = 1e-6
    batch_size: int = 2000
    max_rounds: int = 12
    cleanup_inactive: bool = True
    cleanup_tol: float = 1e-5
    cleanup_every: int = 2
    output_flag: int = 1
    threads: int = 0


@dataclass
class ActiveSetReport:
    iterations: int = 0
    added_constraints: int = 0
    dropped_constraints: int = 0
    max_violation: float = 0.0
    enumerated_events: int = 0
    converged: bool = False


@dataclass
class RollingHorizonConfig:
    time_limit: int = 600
    mip_gap: float = 0.05
    window_size: int = 8
    overlap: int = 2
    lodf_tol: float = 1e-4
    isf_tol: float = 1e-8
    violation_tol: float = 1e-6
    lazy_top_k: int = 0
    output_flag: int = 1
    threads: int = 0


@dataclass
class RollingHorizonReport:
    window_count: int = 0
    window_size: int = 0
    overlap: int = 0
    committed_periods: int = 0
    total_runtime: float = 0.0
    total_nodes: float = 0.0
    max_num_vars: int = 0
    max_num_constrs: int = 0
    max_num_bin: int = 0
    max_num_int: int = 0
    max_mip_gap: float = 0.0


class _ValueProxy:
    def __init__(self, value: float):
        self.X = float(value)


class SolutionProxy:
    """
    Lightweight model-like object exposing the attributes used by restore/verify helpers.
    """

    def __init__(
        self,
        *,
        status: int,
        runtime: float,
        mip_gap: float,
        obj_val: float,
        obj_bound: float,
        node_count: float,
        num_vars: int,
        num_constrs: int,
        num_bin: int,
        num_int: int,
        commit: Dict[Tuple[str, int], float],
        startup: Dict[Tuple[str, int], float],
        shutdown: Dict[Tuple[str, int], float],
        gen_segment_power: Dict[Tuple[str, int, int], float],
        line_flow: Dict[Tuple[str, int], float],
        line_overflow_pos: Dict[Tuple[str, int], float],
        line_overflow_neg: Dict[Tuple[str, int], float],
        contingency_overflow_pos: Dict[Tuple[str, int], float],
        contingency_overflow_neg: Dict[Tuple[str, int], float],
        reserve: Optional[Dict[Tuple[str, str, int], float]] = None,
        reserve_shortfall: Optional[Dict[Tuple[str, int], float]] = None,
    ):
        self.Status = int(status)
        self.Runtime = float(runtime)
        self.MIPGap = float(mip_gap)
        self.ObjVal = float(obj_val)
        self.ObjBound = float(obj_bound)
        self.NodeCount = float(node_count)
        self.NumVars = int(num_vars)
        self.NumConstrs = int(num_constrs)
        self.NumBinVars = int(num_bin)
        self.NumIntVars = int(num_int)
        self.commit = {k: _ValueProxy(v) for k, v in commit.items()}
        self.startup = {k: _ValueProxy(v) for k, v in startup.items()}
        self.shutdown = {k: _ValueProxy(v) for k, v in shutdown.items()}
        self.gen_segment_power = {
            k: _ValueProxy(v) for k, v in gen_segment_power.items()
        }
        self.line_flow = {k: _ValueProxy(v) for k, v in line_flow.items()}
        self.line_overflow_pos = {
            k: _ValueProxy(v) for k, v in line_overflow_pos.items()
        }
        self.line_overflow_neg = {
            k: _ValueProxy(v) for k, v in line_overflow_neg.items()
        }
        self.contingency_overflow_pos = {
            k: _ValueProxy(v) for k, v in contingency_overflow_pos.items()
        }
        self.contingency_overflow_neg = {
            k: _ValueProxy(v) for k, v in contingency_overflow_neg.items()
        }
        self.reserve = {k: _ValueProxy(v) for k, v in (reserve or {}).items()}
        self.reserve_shortfall = {
            k: _ValueProxy(v) for k, v in (reserve_shortfall or {}).items()
        }

    def update(self) -> None:
        return None

    def dispose(self) -> None:
        return None


def _set_gurobi_params(
    model: gp.Model,
    *,
    time_limit: float,
    mip_gap: float,
    output_flag: int,
    threads: int = 0,
) -> None:
    try:
        model.Params.OutputFlag = int(output_flag)
    except Exception:
        pass
    model.setParam("TimeLimit", float(time_limit))
    model.setParam("MIPGap", float(mip_gap))
    model.setParam("NumericFocus", 1)
    if threads and threads > 0:
        model.setParam("Threads", int(threads))


def _status_str(status_code: int) -> str:
    return {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.INFEASIBLE: "INFEASIBLE",
    }.get(int(status_code), f"STATUS_{status_code}")


def _current_flows_and_generation(
    scenario: UnitCommitmentScenario, model: gp.Model
) -> Tuple[Dict[Tuple[str, int], float], Dict[Tuple[str, int], float]]:
    T = scenario.time
    lines = scenario.lines or []
    f_val: Dict[Tuple[str, int], float] = {}
    p_total: Dict[Tuple[str, int], float] = {}

    line_flow = getattr(model, "line_flow", None)
    commit = getattr(model, "commit", None)
    seg = getattr(model, "gen_segment_power", None)

    for ln in lines:
        for t in range(T):
            try:
                f_val[(ln.name, t)] = float(line_flow[ln.name, t].X)
            except Exception:
                f_val[(ln.name, t)] = 0.0

    for gen in scenario.thermal_units:
        for t in range(T):
            try:
                u_t = float(commit[gen.name, t].X)
            except Exception:
                u_t = 0.0
            p_t = u_t * float(gen.min_power[t])
            n_seg = len(gen.segments) if gen.segments else 0
            for s in range(n_seg):
                try:
                    p_t += float(seg[gen.name, t, s].X)
                except Exception:
                    pass
            p_total[(gen.name, t)] = p_t
    return f_val, p_total


def _event_worst_violation(
    event: ConstraintEvent,
    flows: Dict[Tuple[str, int], float],
    p_total: Dict[Tuple[str, int], float],
) -> Tuple[float, int]:
    if event.kind == "line":
        post = flows.get((event.line_name, event.t), 0.0) + float(event.coeff) * flows.get(
            (event.out_name, event.t), 0.0
        )
    else:
        post = flows.get((event.line_name, event.t), 0.0) - float(event.coeff) * p_total.get(
            (event.out_name, event.t), 0.0
        )
    v_pos = post - float(event.F_em)
    v_neg = -post - float(event.F_em)
    if v_pos >= v_neg:
        return float(v_pos), +1
    return float(v_neg), -1


def _add_active_set_constraint(
    model: gp.Model,
    scenario: UnitCommitmentScenario,
    event: ConstraintEvent,
    sign: int,
    *,
    prefix: str = "active_set",
):
    line_by_name = scenario.lines_by_name
    gen_by_name = scenario.thermal_units_by_name
    line_flow = getattr(model, "line_flow")
    commit = getattr(model, "commit")
    seg = getattr(model, "gen_segment_power")
    covp = getattr(model, "contingency_overflow_pos")
    covn = getattr(model, "contingency_overflow_neg")

    if event.kind == "line":
        lhs = line_flow[event.line_name, event.t] + float(event.coeff) * line_flow[
            event.out_name, event.t
        ]
        if sign < 0:
            lhs = -lhs
        slack = covp[event.line_name, event.t] if sign > 0 else covn[event.line_name, event.t]
        name = (
            f"{prefix}_line_{'pos' if sign > 0 else 'neg'}"
            f"[{event.line_name},{event.out_name},{event.t}]"
        )
        constr = model.addConstr(lhs <= float(event.F_em) + slack, name=name)
        try:
            model._explicit_total_cont_constraints = (
                int(getattr(model, "_explicit_total_cont_constraints", 0) or 0) + 1
            )
        except Exception:
            pass
        return constr

    gen = gen_by_name[event.out_name]
    n_seg = len(gen.segments) if gen.segments else 0
    p_expr = commit[gen.name, event.t] * float(gen.min_power[event.t])
    if n_seg > 0:
        p_expr += gp.quicksum(seg[gen.name, event.t, s] for s in range(n_seg))
    lhs = line_flow[event.line_name, event.t] - float(event.coeff) * p_expr
    if sign < 0:
        lhs = -lhs
    slack = covp[event.line_name, event.t] if sign > 0 else covn[event.line_name, event.t]
    name = (
        f"{prefix}_gen_{'pos' if sign > 0 else 'neg'}"
        f"[{event.line_name},{event.out_name},{event.t}]"
    )
    constr = model.addConstr(lhs <= float(event.F_em) + slack, name=name)
    try:
        model._explicit_total_cont_constraints = (
            int(getattr(model, "_explicit_total_cont_constraints", 0) or 0) + 1
        )
    except Exception:
        pass
    return constr


def optimize_with_active_set(
    model: gp.Model, scenario: UnitCommitmentScenario, config: ActiveSetConfig
) -> ActiveSetReport:
    """
    Exact outer-loop active-set baseline:
    1. Solve without explicit N-1 constraints.
    2. Evaluate all topology-induced security events on the incumbent.
    3. Add the most violated events as explicit constraints and re-solve.
    """

    events = enumerate_potential_constraints(
        scenario, lodf_tol=float(config.lodf_tol), isf_tol=float(config.isf_tol)
    )
    report = ActiveSetReport(enumerated_events=len(events))
    registry: Dict[Tuple[str, str, str, int, int], gp.Constr] = {}

    t_start = time.time()
    for it in range(1, int(config.max_rounds) + 1):
        elapsed = time.time() - t_start
        remaining = float(config.time_limit) - elapsed
        if remaining <= 0.0:
            break
        _set_gurobi_params(
            model,
            time_limit=max(1e-3, remaining),
            mip_gap=float(config.mip_gap),
            output_flag=int(config.output_flag),
            threads=int(config.threads),
        )
        optimize_with_lazy_callback(model)
        report.iterations = it

        status = int(getattr(model, "Status", -1))
        if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            break

        flows, p_total = _current_flows_and_generation(scenario, model)
        violated: List[Tuple[float, Tuple[str, str, str, int, int], ConstraintEvent, int]] = []
        iter_max_viol = 0.0

        for event in events:
            viol, sign = _event_worst_violation(event, flows, p_total)
            iter_max_viol = max(iter_max_viol, float(viol))
            if viol <= float(config.violation_tol):
                continue
            key = (event.kind, event.line_name, event.out_name, int(event.t), int(sign))
            if key in registry:
                continue
            violated.append((float(viol), key, event, int(sign)))

        report.max_violation = max(report.max_violation, iter_max_viol)
        if not violated:
            report.converged = True
            break

        violated.sort(key=lambda item: item[0], reverse=True)
        if config.batch_size and config.batch_size > 0:
            violated = violated[: int(config.batch_size)]

        added_keys: Set[Tuple[str, str, str, int, int]] = set()
        for _, key, event, sign in violated:
            registry[key] = _add_active_set_constraint(model, scenario, event, sign)
            added_keys.add(key)
        model.update()
        report.added_constraints += len(violated)

        if (
            bool(config.cleanup_inactive)
            and config.cleanup_every > 0
            and it % int(config.cleanup_every) == 0
            and registry
        ):
            removable: List[Tuple[str, str, str, int, int]] = []
            for key, constr in list(registry.items()):
                if key in added_keys:
                    continue
                try:
                    if float(constr.Slack) > float(config.cleanup_tol):
                        removable.append(key)
                except Exception:
                    continue
            if removable:
                for key in removable:
                    try:
                        model.remove(registry[key])
                    except Exception:
                        continue
                    registry.pop(key, None)
                model.update()
                report.dropped_constraints += len(removable)

    return report


def _slice_series(vals: Sequence, start: int, end: int):
    try:
        return list(vals[start:end])
    except Exception:
        return list(vals)


def _slice_scenario_window(
    scenario: UnitCommitmentScenario,
    start: int,
    end: int,
    state_steps: Dict[str, int],
    state_power: Dict[str, float],
) -> UnitCommitmentScenario:
    sc = copy.deepcopy(scenario)
    sc.name = f"{scenario.name}[{start}:{end}]"
    sc.time = int(end - start)
    sc.power_balance_penalty = _slice_series(sc.power_balance_penalty, start, end)

    for bus in sc.buses:
        bus.load = _slice_series(bus.load, start, end)

    for reserve in sc.reserves or []:
        reserve.amount = _slice_series(reserve.amount, start, end)

    for gen in sc.thermal_units:
        gen.max_power = _slice_series(gen.max_power, start, end)
        gen.min_power = _slice_series(gen.min_power, start, end)
        gen.must_run = _slice_series(gen.must_run, start, end)
        gen.min_power_cost = _slice_series(gen.min_power_cost, start, end)
        gen.initial_status = int(state_steps.get(gen.name, gen.initial_status or -1))
        gen.initial_power = float(state_power.get(gen.name, gen.initial_power or 0.0))
        if gen.commitment_status:
            gen.commitment_status = _slice_series(gen.commitment_status, start, end)
        for seg in gen.segments or []:
            seg.amount = _slice_series(seg.amount, start, end)
            seg.cost = _slice_series(seg.cost, start, end)

    for pu in sc.profiled_units or []:
        pu.min_power = _slice_series(pu.min_power, start, end)
        pu.max_power = _slice_series(pu.max_power, start, end)
        pu.cost = _slice_series(pu.cost, start, end)

    for su in sc.storage_units or []:
        su.min_level = _slice_series(su.min_level, start, end)
        su.max_level = _slice_series(su.max_level, start, end)
        su.simultaneous = _slice_series(su.simultaneous, start, end)
        su.charge_cost = _slice_series(su.charge_cost, start, end)
        su.discharge_cost = _slice_series(su.discharge_cost, start, end)
        su.charge_eff = _slice_series(su.charge_eff, start, end)
        su.discharge_eff = _slice_series(su.discharge_eff, start, end)
        su.loss_factor = _slice_series(su.loss_factor, start, end)
        su.min_charge = _slice_series(su.min_charge, start, end)
        su.max_charge = _slice_series(su.max_charge, start, end)
        su.min_discharge = _slice_series(su.min_discharge, start, end)
        su.max_discharge = _slice_series(su.max_discharge, start, end)

    for line in sc.lines:
        line.normal_limit = _slice_series(line.normal_limit, start, end)
        line.emergency_limit = _slice_series(line.emergency_limit, start, end)
        line.flow_penalty = _slice_series(line.flow_penalty, start, end)

    return sc


def _extract_window_values(
    scenario: UnitCommitmentScenario, model: gp.Model
) -> Dict[str, Dict]:
    T = scenario.time
    out: Dict[str, Dict] = {
        "commit": {},
        "startup": {},
        "shutdown": {},
        "gen_segment_power": {},
        "line_flow": {},
        "line_overflow_pos": {},
        "line_overflow_neg": {},
        "contingency_overflow_pos": {},
        "contingency_overflow_neg": {},
        "reserve": {},
        "reserve_shortfall": {},
        "total_power": {},
    }

    commit = getattr(model, "commit", None)
    startup = getattr(model, "startup", None)
    shutdown = getattr(model, "shutdown", None)
    seg = getattr(model, "gen_segment_power", None)
    line_flow = getattr(model, "line_flow", None)
    ovp = getattr(model, "line_overflow_pos", None)
    ovn = getattr(model, "line_overflow_neg", None)
    covp = getattr(model, "contingency_overflow_pos", None)
    covn = getattr(model, "contingency_overflow_neg", None)
    reserve = getattr(model, "reserve", None)
    shortfall = getattr(model, "reserve_shortfall", None)

    for gen in scenario.thermal_units:
        n_seg = len(gen.segments) if gen.segments else 0
        for t in range(T):
            u_t = float(commit[gen.name, t].X) if commit is not None else 0.0
            out["commit"][(gen.name, t)] = u_t
            out["startup"][(gen.name, t)] = (
                float(startup[gen.name, t].X) if startup is not None else 0.0
            )
            out["shutdown"][(gen.name, t)] = (
                float(shutdown[gen.name, t].X) if shutdown is not None else 0.0
            )
            total = u_t * float(gen.min_power[t])
            for s in range(n_seg):
                v = float(seg[gen.name, t, s].X) if seg is not None else 0.0
                out["gen_segment_power"][(gen.name, t, s)] = v
                total += v
            out["total_power"][(gen.name, t)] = total

    for line in scenario.lines or []:
        for t in range(T):
            out["line_flow"][(line.name, t)] = (
                float(line_flow[line.name, t].X) if line_flow is not None else 0.0
            )
            out["line_overflow_pos"][(line.name, t)] = (
                float(ovp[line.name, t].X) if ovp is not None else 0.0
            )
            out["line_overflow_neg"][(line.name, t)] = (
                float(ovn[line.name, t].X) if ovn is not None else 0.0
            )
            out["contingency_overflow_pos"][(line.name, t)] = (
                float(covp[line.name, t].X) if covp is not None else 0.0
            )
            out["contingency_overflow_neg"][(line.name, t)] = (
                float(covn[line.name, t].X) if covn is not None else 0.0
            )

    if reserve is not None:
        for r in scenario.reserves or []:
            for gen in r.thermal_units:
                for t in range(T):
                    out["reserve"][(r.name, gen.name, t)] = float(
                        reserve[r.name, gen.name, t].X
                    )
    if shortfall is not None:
        for r in scenario.reserves or []:
            for t in range(T):
                out["reserve_shortfall"][(r.name, t)] = float(shortfall[r.name, t].X)

    return out


def _advance_state_tracker(
    scenario: UnitCommitmentScenario,
    state_steps: Dict[str, int],
    state_power: Dict[str, float],
    window_vals: Dict[str, Dict],
    committed_local_periods: Iterable[int],
) -> None:
    for t in committed_local_periods:
        for gen in scenario.thermal_units:
            prev = int(state_steps.get(gen.name, gen.initial_status or -1))
            u_t = 1 if window_vals["commit"].get((gen.name, t), 0.0) >= 0.5 else 0
            if u_t == 1:
                state_steps[gen.name] = prev + 1 if prev > 0 else 1
            else:
                state_steps[gen.name] = prev - 1 if prev < 0 else -1
            state_power[gen.name] = float(window_vals["total_power"].get((gen.name, t), 0.0))


def _compute_proxy_objective(
    scenario: UnitCommitmentScenario,
    commit: Dict[Tuple[str, int], float],
    startup: Dict[Tuple[str, int], float],
    gen_segment_power: Dict[Tuple[str, int, int], float],
    line_overflow_pos: Dict[Tuple[str, int], float],
    line_overflow_neg: Dict[Tuple[str, int], float],
    contingency_overflow_pos: Dict[Tuple[str, int], float],
    contingency_overflow_neg: Dict[Tuple[str, int], float],
    reserve_shortfall: Dict[Tuple[str, int], float],
) -> float:
    from src.optimization_model.helpers.verify_solution import _startup_cost_for_gen
    from src.optimization_model.solver.scuc.objectives.power_cost_segmented import (
        _CONTINGENCY_PENALTY_FACTOR,
    )

    T = scenario.time
    obj = 0.0
    for gen in scenario.thermal_units:
        n_seg = len(gen.segments) if gen.segments else 0
        for t in range(T):
            obj += float(commit.get((gen.name, t), 0.0)) * float(gen.min_power_cost[t])
            obj += float(startup.get((gen.name, t), 0.0)) * float(_startup_cost_for_gen(gen))
            for s in range(n_seg):
                obj += float(gen_segment_power.get((gen.name, t, s), 0.0)) * float(
                    gen.segments[s].cost[t]
                )

    for reserve in scenario.reserves or []:
        for t in range(T):
            obj += float(reserve_shortfall.get((reserve.name, t), 0.0)) * float(
                reserve.shortfall_penalty
            )

    for line in scenario.lines or []:
        for t in range(T):
            pen = float(line.flow_penalty[t])
            obj += pen * (
                float(line_overflow_pos.get((line.name, t), 0.0))
                + float(line_overflow_neg.get((line.name, t), 0.0))
            )
            obj += pen * _CONTINGENCY_PENALTY_FACTOR * (
                float(contingency_overflow_pos.get((line.name, t), 0.0))
                + float(contingency_overflow_neg.get((line.name, t), 0.0))
            )
    return float(obj)


def solve_rolling_horizon(
    scenario: UnitCommitmentScenario,
    config: RollingHorizonConfig,
    *,
    env: Optional[gp.Env] = None,
) -> Tuple[SolutionProxy, RollingHorizonReport]:
    """
    Approximate shrinking-horizon competitor inspired by Castelli et al.:
    solve overlapping windows and commit only the non-overlap prefix of each window.
    """

    T = int(scenario.time)
    win = max(2, min(int(config.window_size), T))
    overlap = max(0, min(int(config.overlap), win - 1))
    step = max(1, win - overlap)

    state_steps = {
        gen.name: int(gen.initial_status if gen.initial_status is not None else -1)
        for gen in scenario.thermal_units
    }
    state_power = {
        gen.name: float(gen.initial_power if gen.initial_power is not None else 0.0)
        for gen in scenario.thermal_units
    }

    global_store: Dict[str, Dict] = {
        "commit": {},
        "startup": {},
        "shutdown": {},
        "gen_segment_power": {},
        "line_flow": {},
        "line_overflow_pos": {},
        "line_overflow_neg": {},
        "contingency_overflow_pos": {},
        "contingency_overflow_neg": {},
        "reserve": {},
        "reserve_shortfall": {},
    }
    report = RollingHorizonReport()
    report.window_size = int(win)
    report.overlap = int(overlap)
    worst_status = GRB.OPTIMAL
    t_start = time.time()

    starts = list(range(0, T, step))
    for w_idx, start in enumerate(starts):
        remaining = float(config.time_limit) - (time.time() - t_start)
        if remaining <= 0.0:
            worst_status = GRB.TIME_LIMIT
            break
        end = min(T, start + win)
        if end <= start:
            continue
        sc_win = _slice_scenario_window(scenario, start, end, state_steps, state_power)
        model = build_model(sc_win, use_lazy_contingencies=True, env=env)
        _set_gurobi_params(
            model,
            time_limit=max(1e-3, remaining),
            mip_gap=float(config.mip_gap),
            output_flag=int(config.output_flag),
            threads=int(config.threads),
        )
        lazy_cfg = LazyContingencyConfig(
            lodf_tol=float(config.lodf_tol),
            isf_tol=float(config.isf_tol),
            violation_tol=float(config.violation_tol),
            add_top_k=int(config.lazy_top_k) if config.lazy_top_k and config.lazy_top_k > 0 else 0,
            verbose=False,
        )
        attach_lazy_contingency_callback(model, sc_win, lazy_cfg)
        try:
            model.Params.LazyConstraints = 1
        except Exception:
            pass

        t0 = time.time()
        optimize_with_lazy_callback(model)
        report.total_runtime += time.time() - t0
        report.window_count += 1
        report.total_nodes += float(getattr(model, "NodeCount", 0.0) or 0.0)
        report.max_num_vars = max(report.max_num_vars, int(getattr(model, "NumVars", 0) or 0))
        report.max_num_constrs = max(
            report.max_num_constrs, int(getattr(model, "NumConstrs", 0) or 0)
        )
        report.max_num_bin = max(report.max_num_bin, int(getattr(model, "NumBinVars", 0) or 0))
        report.max_num_int = max(report.max_num_int, int(getattr(model, "NumIntVars", 0) or 0))
        try:
            report.max_mip_gap = max(report.max_mip_gap, float(model.MIPGap))
        except Exception:
            pass

        status = int(getattr(model, "Status", -1))
        if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            worst_status = status
        elif worst_status == GRB.OPTIMAL and status != GRB.OPTIMAL:
            worst_status = status

        window_vals = _extract_window_values(sc_win, model)
        commit_until = end if end >= T else min(end, start + step)
        local_commit_periods = list(range(0, commit_until - start))

        for lt in local_commit_periods:
            gt = start + lt
            report.committed_periods += 1
            for gen in sc_win.thermal_units:
                global_store["commit"][(gen.name, gt)] = window_vals["commit"].get(
                    (gen.name, lt), 0.0
                )
                global_store["startup"][(gen.name, gt)] = window_vals["startup"].get(
                    (gen.name, lt), 0.0
                )
                global_store["shutdown"][(gen.name, gt)] = window_vals["shutdown"].get(
                    (gen.name, lt), 0.0
                )
                n_seg = len(gen.segments) if gen.segments else 0
                for s in range(n_seg):
                    global_store["gen_segment_power"][(gen.name, gt, s)] = window_vals[
                        "gen_segment_power"
                    ].get((gen.name, lt, s), 0.0)
            for line in sc_win.lines or []:
                for key in (
                    "line_flow",
                    "line_overflow_pos",
                    "line_overflow_neg",
                    "contingency_overflow_pos",
                    "contingency_overflow_neg",
                ):
                    global_store[key][(line.name, gt)] = window_vals[key].get((line.name, lt), 0.0)
            for reserve in sc_win.reserves or []:
                global_store["reserve_shortfall"][(reserve.name, gt)] = window_vals[
                    "reserve_shortfall"
                ].get((reserve.name, lt), 0.0)
                for gen in reserve.thermal_units:
                    global_store["reserve"][(reserve.name, gen.name, gt)] = window_vals[
                        "reserve"
                    ].get((reserve.name, gen.name, lt), 0.0)

        _advance_state_tracker(sc_win, state_steps, state_power, window_vals, local_commit_periods)

        try:
            model.dispose()
        except Exception:
            pass

        if end >= T:
            break

    obj_val = _compute_proxy_objective(
        scenario,
        global_store["commit"],
        global_store["startup"],
        global_store["gen_segment_power"],
        global_store["line_overflow_pos"],
        global_store["line_overflow_neg"],
        global_store["contingency_overflow_pos"],
        global_store["contingency_overflow_neg"],
        global_store["reserve_shortfall"],
    )
    proxy = SolutionProxy(
        status=int(worst_status),
        runtime=float(report.total_runtime),
        mip_gap=float(report.max_mip_gap),
        obj_val=float(obj_val),
        obj_bound=float(obj_val),
        node_count=float(report.total_nodes),
        num_vars=int(report.max_num_vars),
        num_constrs=int(report.max_num_constrs),
        num_bin=int(report.max_num_bin),
        num_int=int(report.max_num_int),
        commit=global_store["commit"],
        startup=global_store["startup"],
        shutdown=global_store["shutdown"],
        gen_segment_power=global_store["gen_segment_power"],
        line_flow=global_store["line_flow"],
        line_overflow_pos=global_store["line_overflow_pos"],
        line_overflow_neg=global_store["line_overflow_neg"],
        contingency_overflow_pos=global_store["contingency_overflow_pos"],
        contingency_overflow_neg=global_store["contingency_overflow_neg"],
        reserve=global_store["reserve"],
        reserve_shortfall=global_store["reserve_shortfall"],
    )
    return proxy, report
