from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, List

import math

from src.data_preparation.data_structure import (
    UnitCommitmentScenario,
    ThermalUnit,
    Reserve,
)
from src.data_preparation.params import DataParams

# Reuse tolerances and penalty factor from the same modules used by the model
from src.optimization_model.solver.scuc.constraints.contingencies import (
    _LODF_TOL as CONT_LODF_TOL,
    _ISF_TOL as CONT_ISF_TOL,
)
from src.optimization_model.solver.scuc.objectives.power_cost_segmented import (
    _CONTINGENCY_PENALTY_FACTOR as CONT_PENALTY_FACTOR,
)

# Unique indices for verification items
# Constraints
ID_C_FIX = "C-101"  # Commitment fixing
ID_C_LINK = "C-102"  # Segment capacity linking
ID_C_BAL = "C-103"  # Power balance
ID_C_RES_HEAD = "C-104"  # Reserve headroom linking (shared across products)
ID_C_RES_REQ = "C-105"  # Reserve requirement
ID_C_SU_DEF = "C-106"  # Startup/shutdown definition
ID_C_RAMP = "C-107"  # Ramping with startup/shutdown limits
ID_C_MIN_UP = "C-110"  # Minimum up-time
ID_C_MIN_DOWN = "C-111"  # Minimum down-time
ID_C_MIN_INIT = "C-112"  # Initial condition enforcement (min up/down)
ID_C_FLOW_DEF = "C-108"  # Line flow PTDF equality
ID_C_FLOW_LIM = "C-109"  # Line flow limits with overflow slacks
ID_C_CONT_LINE = "C-120"  # Post-contingency limits (line outages via LODF)
ID_C_CONT_GEN = "C-121"  # Post-contingency limits (generator outages via ISF)

# Variables
ID_V_COMMIT = "V-201"  # Commitment u[gen,t] in {0,1}
ID_V_PSEG = "V-202"  # Segment power pseg[gen,t,s] >= 0
ID_V_R = "V-203"  # Reserve provision r[k,gen,t] >= 0
ID_V_S = "V-204"  # Reserve shortfall s[k,t] >= 0
ID_V_SU = "V-205"  # Startup v[gen,t] in {0,1}
ID_V_SD = "V-206"  # Shutdown w[gen,t] in {0,1}
ID_V_FLOW = "V-207"  # Line flows f[line,t] finite
ID_V_OVP = "V-208"  # Line overflow+ >= 0
ID_V_OVN = "V-209"  # Line overflow- >= 0

# Objective
ID_O_TOTAL = "O-301"  # TotalCostWithReservePenalty consistency

EPS = 1e-6  # numerical tolerance


@dataclass
class CheckItem:
    idx: str
    name: str
    value: float  # 0 means OK; otherwise worst violation


def _get_model_vars(model):
    commit = getattr(model, "commit", None)
    seg = getattr(model, "gen_segment_power", None)
    r = getattr(model, "reserve", None)
    s = getattr(model, "reserve_shortfall", None)
    su = getattr(model, "startup", None)
    sd = getattr(model, "shutdown", None)
    f = getattr(model, "line_flow", None)
    ovp = getattr(model, "line_overflow_pos", None)
    ovn = getattr(model, "line_overflow_neg", None)
    covp = getattr(model, "contingency_overflow_pos", None)
    covn = getattr(model, "contingency_overflow_neg", None)
    return commit, seg, r, s, su, sd, f, ovp, ovn, covp, covn


def _compute_p_of_gen_t(gen: ThermalUnit, t: int, commit, seg) -> float:
    u = float(commit[gen.name, t].X) if commit is not None else 0.0
    p = u * float(gen.min_power[t])
    n_segments = len(gen.segments) if gen.segments else 0
    if n_segments > 0 and seg is not None:
        for s in range(n_segments):
            p += float(seg[gen.name, t, s].X)
    return p


def _compute_production_at_t(
    units: Sequence[ThermalUnit], t: int, commit, seg
) -> float:
    prod = 0.0
    for gen in units:
        prod += _compute_p_of_gen_t(gen, t, commit, seg)
    return prod


def _initial_u(gen: ThermalUnit) -> float:
    try:
        return (
            1.0 if (gen.initial_status is not None and gen.initial_status > 0) else 0.0
        )
    except Exception:
        return 0.0


def _initial_p(gen: ThermalUnit) -> float:
    try:
        return float(gen.initial_power) if gen.initial_power is not None else 0.0
    except Exception:
        return 0.0


def _startup_cost_for_gen(gen: ThermalUnit) -> float:
    """
    Same policy as objective: 'hot start' = min of startup category costs; 0 if none.
    """
    try:
        if gen.startup_categories:
            return float(min(cat.cost for cat in gen.startup_categories))
    except Exception:
        pass
    return 0.0


def _objective_from_vars(
    units: Sequence[ThermalUnit],
    reserves: Sequence[Reserve],
    lines,
    T: int,
    commit,
    seg,
    r_short,
    ovp,
    ovn,
    covp=None,
    covn=None,
    su=None,
    contingency_penalty_factor: float = CONT_PENALTY_FACTOR,
) -> float:
    total = 0.0
    # Production cost
    for gen in units:
        n_segments = len(gen.segments) if gen.segments else 0
        s_cost = _startup_cost_for_gen(gen)
        for t in range(T):
            u = float(commit[gen.name, t].X) if commit is not None else 0.0
            total += u * float(gen.min_power_cost[t])
            for sidx in range(n_segments):
                total += float(seg[gen.name, t, sidx].X) * float(
                    gen.segments[sidx].cost[t]
                )
            if su is not None and s_cost > 0.0:
                total += float(su[gen.name, t].X) * s_cost
    # Reserve shortfall penalty
    if reserves and r_short is not None:
        for reserve_def in reserves:
            pen = float(reserve_def.shortfall_penalty)
            for t in range(T):
                total += float(r_short[reserve_def.name, t].X) * pen
    # Base-case line overflow penalty
    if lines and ovp is not None and ovn is not None:
        for line in lines:
            for t in range(T):
                pen = float(line.flow_penalty[t])
                total += pen * (float(ovp[line.name, t].X) + float(ovn[line.name, t].X))
    # Contingency overflow penalty (shared slacks)
    if lines and covp is not None and covn is not None:
        for line in lines:
            for t in range(T):
                pen = float(line.flow_penalty[t]) * contingency_penalty_factor
                total += pen * (
                    float(covp[line.name, t].X) + float(covn[line.name, t].X)
                )
    return total


def verify_solution_to_log(
    scenario: UnitCommitmentScenario,
    model,
    out_dir: Optional[Path] = None,
    filename: Optional[str] = None,
) -> Path:
    out_dir = out_dir or DataParams._LOGS
    out_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = "SCUC_verification.log"
    out_path = out_dir / filename

    T = scenario.time
    units = scenario.thermal_units
    reserves = scenario.reserves or []
    lines = scenario.lines or []

    total_load = getattr(model, "_total_load", None)
    if total_load is None:
        total_load = [sum(b.load[t] for b in scenario.buses) for t in range(T)]

    (
        commit,
        seg,
        r,
        s,
        su,
        sd,
        f,
        ovp,
        ovn,
        covp,
        covn,
    ) = _get_model_vars(model)

    checks: List[CheckItem] = []

    # Variables: V-201 (commitment domain & integrality)
    worst = 0.0
    if commit is not None:
        for gen in units:
            for t in range(T):
                val = float(commit[gen.name, t].X)
                worst = max(worst, max(0.0, -val))
                worst = max(worst, max(0.0, val - 1.0))
                worst = max(worst, abs(val - round(val)))
    checks.append(CheckItem(ID_V_COMMIT, "Commitment u[gen,t] in {0,1}", worst))

    # Variables: V-202 (segment power >= 0)
    worst = 0.0
    if seg is not None:
        for gen in units:
            n_segments = len(gen.segments) if gen.segments else 0
            for t in range(T):
                for s_idx in range(n_segments):
                    val = float(seg[gen.name, t, s_idx].X)
                    worst = max(worst, max(0.0, -val))
    checks.append(CheckItem(ID_V_PSEG, "Segment power pseg[gen,t,s] >= 0", worst))

    # Variables: V-203 (reserve provision >= 0)
    worst = 0.0
    if reserves and r is not None:
        for reserve_def in reserves:
            for gen in reserve_def.thermal_units:
                for t in range(T):
                    val = float(r[reserve_def.name, gen.name, t].X)
                    worst = max(worst, max(0.0, -val))
    checks.append(CheckItem(ID_V_R, "Reserve provision r[k,gen,t] >= 0", worst))

    # Variables: V-204 (shortfall >= 0)
    worst = 0.0
    if reserves and s is not None:
        for reserve_def in reserves:
            for t in range(T):
                val = float(s[reserve_def.name, t].X)
                worst = max(worst, max(0.0, -val))
    checks.append(CheckItem(ID_V_S, "Reserve shortfall s[k,t] >= 0", worst))

    # Variables: V-205 (startup binary)
    worst = 0.0
    if su is not None:
        for gen in units:
            for t in range(T):
                val = float(su[gen.name, t].X)
                worst = max(worst, max(0.0, -val))
                worst = max(worst, max(0.0, val - 1.0))
                worst = max(worst, abs(val - round(val)))
    checks.append(CheckItem(ID_V_SU, "Startup v[gen,t] in {0,1}", worst))

    # Variables: V-206 (shutdown binary)
    worst = 0.0
    if sd is not None:
        for gen in units:
            for t in range(T):
                val = float(sd[gen.name, t].X)
                worst = max(worst, max(0.0, -val))
                worst = max(worst, max(0.0, val - 1.0))
                worst = max(worst, abs(val - round(val)))
    checks.append(CheckItem(ID_V_SD, "Shutdown w[gen,t] in {0,1}", worst))

    # Variables: V-207 (line flow – finite)
    worst = 0.0
    if f is not None and lines:
        for line in lines:
            for t in range(T):
                val = float(f[line.name, t].X)
                if not math.isfinite(val):
                    worst = float("inf")
                    break
    checks.append(CheckItem(ID_V_FLOW, "Line flow f[line,t] finite", worst))

    # Variables: V-208 (overflow+ >= 0)
    worst = 0.0
    if ovp is not None and lines:
        for line in lines:
            for t in range(T):
                worst = max(worst, max(0.0, -float(ovp[line.name, t].X)))
    checks.append(CheckItem(ID_V_OVP, "Line overflow+ ov_pos[line,t] >= 0", worst))

    # Variables: V-209 (overflow- >= 0)
    worst = 0.0
    if ovn is not None and lines:
        for line in lines:
            for t in range(T):
                worst = max(worst, max(0.0, -float(ovn[line.name, t].X)))
    checks.append(CheckItem(ID_V_OVN, "Line overflow- ov_neg[line,t] >= 0", worst))

    # Constraints: C-101 (commitment fixing)
    worst = 0.0
    for gen in units:
        if not gen.commitment_status:
            continue
        for t in range(T):
            st = gen.commitment_status[t]
            if st is None or commit is None:
                continue
            val = float(commit[gen.name, t].X)
            worst = max(worst, abs(val - (1.0 if st else 0.0)))
    checks.append(CheckItem(ID_C_FIX, "Commitment fixing (fix_commit_on/off)", worst))

    # Constraints: C-102 (linking: seg <= amount * u)
    worst = 0.0
    if seg is not None and commit is not None:
        for gen in units:
            n_segments = len(gen.segments) if gen.segments else 0
            for t in range(T):
                u = float(commit[gen.name, t].X)
                for s_idx in range(n_segments):
                    amount = float(gen.segments[s_idx].amount[t])
                    if amount < 0:
                        amount = 0.0
                    val = float(seg[gen.name, t, s_idx].X)
                    viol = val - amount * u
                    if viol > worst:
                        worst = viol
    checks.append(CheckItem(ID_C_LINK, "Segment capacity linking", max(0.0, worst)))

    # Constraints: C-103 (power balance: equality)
    worst = 0.0
    for t in range(T):
        lhs = _compute_production_at_t(units, t, commit, seg)
        rhs = float(total_load[t])
        worst = max(worst, abs(lhs - rhs))
    checks.append(CheckItem(ID_C_BAL, "System power balance", worst))

    # Constraints: C-104 (reserve headroom (shared across products))
    worst = 0.0
    if reserves and commit is not None and seg is not None and r is not None:
        eligible_by_gen: Dict[str, List] = {}
        for res in reserves:
            for g in res.thermal_units:
                eligible_by_gen.setdefault(g.name, []).append(res)
        for gen in units:
            n_segments = len(gen.segments) if gen.segments else 0
            for t in range(T):
                energy_above_min = 0.0
                for s_idx in range(n_segments):
                    energy_above_min += float(seg[gen.name, t, s_idx].X)
                total_reserve = 0.0
                for res in eligible_by_gen.get(gen.name, []):
                    total_reserve += float(r[res.name, gen.name, t].X)
                headroom = float(gen.max_power[t]) - float(gen.min_power[t])
                rhs = headroom * float(commit[gen.name, t].X)
                viol = energy_above_min + total_reserve - rhs
                if viol > worst:
                    worst = viol
    checks.append(
        CheckItem(ID_C_RES_HEAD, "Reserve headroom linking (shared)", max(0.0, worst))
    )

    # Constraints: C-105 (reserve requirement)
    worst = 0.0
    if reserves and r is not None and s is not None:
        for reserve_def in reserves:
            for t in range(T):
                provided = 0.0
                for gen in reserve_def.thermal_units:
                    provided += float(r[reserve_def.name, gen.name, t].X)
                left = provided + float(s[reserve_def.name, t].X)
                req = float(reserve_def.amount[t])
                viol = req - left
                if viol > worst:
                    worst = viol
    checks.append(CheckItem(ID_C_RES_REQ, "Reserve requirement", max(0.0, worst)))

    # Constraints: C-106 (startup/shutdown definition + exclusivity)
    worst = 0.0
    if commit is not None and su is not None and sd is not None:
        for gen in units:
            u0 = _initial_u(gen)
            for t in range(T):
                gen_commit_t = float(commit[gen.name, t].X)
                gen_startup_t = float(su[gen.name, t].X)
                gen_shutdown_t = float(sd[gen.name, t].X)
                u_prev = u0 if t == 0 else float(commit[gen.name, t - 1].X)
                eq_violation = abs(
                    (gen_commit_t - u_prev) - (gen_startup_t - gen_shutdown_t)
                )
                excl_violation = max(0.0, gen_startup_t + gen_shutdown_t - 1.0)
                worst = max(worst, eq_violation, excl_violation)
    checks.append(
        CheckItem(
            ID_C_SU_DEF,
            "Startup/shutdown definition (u[t]-u_prev == v-w) and exclusivity (v+w<=1)",
            worst,
        )
    )

    # Constraints: C-107 (ramping with startup/shutdown limits + initial power)
    worst = 0.0
    if commit is not None and seg is not None and su is not None and sd is not None:
        for gen in units:
            ru = float(gen.ramp_up)
            rd = float(gen.ramp_down)
            SU = float(gen.startup_limit)
            SD = float(gen.shutdown_limit)

            p0 = _initial_p(gen)
            u0 = _initial_u(gen)

            for t in range(T):
                p_t = _compute_p_of_gen_t(gen, t, commit, seg)
                v_t = float(su[gen.name, t].X)
                w_t = float(sd[gen.name, t].X)

                if t == 0:
                    viol_up = (p_t - p0) - (ru * u0 + SU * v_t)
                    u_t = float(commit[gen.name, t].X)
                    viol_dn = (p0 - p_t) - (rd * u_t + SD * w_t)
                else:
                    p_prev = _compute_p_of_gen_t(gen, t - 1, commit, seg)
                    u_prev = float(commit[gen.name, t - 1].X)
                    u_t = float(commit[gen.name, t].X)
                    viol_up = (p_t - p_prev) - (ru * u_prev + SU * v_t)
                    viol_dn = (p_prev - p_t) - (rd * u_t + SD * w_t)

                worst = max(worst, max(0.0, viol_up))
                worst = max(worst, max(0.0, viol_dn))
    checks.append(
        CheckItem(
            ID_C_RAMP,
            "Ramping with startup/shutdown limits (incl. initial status/power)",
            worst,
        )
    )

    # Constraints: C-110 (min up-time windows)
    worst = 0.0
    if su is not None and commit is not None:
        for gen in units:
            Lu = int(getattr(gen, "min_up", 0) or 0)
            if Lu > 0:
                for t in range(T):
                    start_k = max(0, t - Lu + 1)
                    if start_k <= t:
                        lhs = 0.0
                        for k in range(start_k, t + 1):
                            lhs += float(su[gen.name, k].X)
                        rhs = float(commit[gen.name, t].X)
                        viol = lhs - rhs
                        worst = max(worst, max(0.0, viol))
    checks.append(CheckItem(ID_C_MIN_UP, "Minimum up-time window", worst))

    # Constraints: C-111 (min down-time windows)
    worst = 0.0
    if sd is not None and commit is not None:
        for gen in units:
            Ld = int(getattr(gen, "min_down", 0) or 0)
            if Ld > 0:
                for t in range(T):
                    start_k = max(0, t - Ld + 1)
                    if start_k <= t:
                        lhs = 0.0
                        for k in range(start_k, t + 1):
                            lhs += float(sd[gen.name, k].X)
                        rhs = 1.0 - float(commit[gen.name, t].X)
                        viol = lhs - rhs
                        worst = max(worst, max(0.0, viol))
    checks.append(CheckItem(ID_C_MIN_DOWN, "Minimum down-time window", worst))

    # Constraints: C-112 (initial-condition enforcement at horizon start)
    worst = 0.0
    if commit is not None:
        for gen in units:
            Lu = int(getattr(gen, "min_up", 0) or 0)
            Ld = int(getattr(gen, "min_down", 0) or 0)
            s0 = getattr(gen, "initial_status", None)
            if s0 is None:
                continue

            if s0 > 0 and Lu > 0:
                remaining_on = max(0, Lu - int(s0))
                for t in range(min(remaining_on, T)):
                    val = float(commit[gen.name, t].X)
                    worst = max(worst, abs(val - 1.0))

            if s0 < 0 and Ld > 0:
                s_off = -int(s0)
                remaining_off = max(0, Ld - s_off)
                for t in range(min(remaining_off, T)):
                    val = float(commit[gen.name, t].X)
                    worst = max(worst, abs(val - 0.0))
    checks.append(CheckItem(ID_C_MIN_INIT, "Initial up/down enforcement", worst))

    # Constraints: C-108 (line flow PTDF equality)
    worst = 0.0
    if lines and f is not None:
        isf = scenario.isf.tocsr()
        ref_1b = getattr(scenario, "ptdf_ref_bus_index", scenario.buses[0].index)
        non_ref_bus_indices = sorted(
            [b.index for b in scenario.buses if b.index != ref_1b]
        )

        for t in range(T):
            inj_by_busidx: Dict[int, float] = {}
            for b in scenario.buses:
                gen_at_b = 0.0
                for gen in b.thermal_units:
                    gen_at_b += _compute_p_of_gen_t(gen, t, commit, seg)
                inj_by_busidx[b.index] = gen_at_b - float(b.load[t])

            for line in lines:
                row = isf.getrow(line.index - 1)
                fhat = 0.0
                for col, coeff in zip(row.indices.tolist(), row.data.tolist()):
                    bus_1b = non_ref_bus_indices[col]
                    fhat += float(coeff) * float(inj_by_busidx[bus_1b])
                actual = float(f[line.name, t].X)
                worst = max(worst, abs(fhat - actual))
    checks.append(CheckItem(ID_C_FLOW_DEF, "Line flow PTDF equality", worst))

    # Constraints: C-109 (flow limits with overflow slacks)
    worst = 0.0
    if lines and f is not None and ovp is not None and ovn is not None:
        for line in lines:
            for t in range(T):
                F = float(line.normal_limit[t])
                val = float(f[line.name, t].X)
                vpos = val - F - float(ovp[line.name, t].X)
                vneg = -val - F - float(ovn[line.name, t].X)
                if vpos > worst:
                    worst = vpos
                if vneg > worst:
                    worst = vneg
    checks.append(
        CheckItem(
            ID_C_FLOW_LIM, "Line flow limits with overflow slacks", max(0.0, worst)
        )
    )

    # C-120 — Post-contingency line limits (line outages via LODF), with shared slacks
    worst_line = 0.0
    if lines and f is not None and scenario.contingencies:
        lodf = scenario.lodf.tocsc()
        for cont in scenario.contingencies:
            if not cont.lines:
                continue
            for out_line in cont.lines:
                mcol = out_line.index - 1
                col = lodf.getcol(mcol)
                for l_row, alpha in zip(col.indices.tolist(), col.data.tolist()):
                    if l_row == mcol:
                        continue
                    if abs(alpha) < CONT_LODF_TOL:
                        continue
                    line_l = scenario.lines[l_row]
                    for t in range(T):
                        base_l = float(f[line_l.name, t].X)
                        base_m = float(f[out_line.name, t].X)
                        post = base_l + float(alpha) * base_m
                        ovp_shared = (
                            float(covp[line_l.name, t].X) if covp is not None else 0.0
                        )
                        ovn_shared = (
                            float(covn[line_l.name, t].X) if covn is not None else 0.0
                        )
                        vpos = post - float(line_l.emergency_limit[t]) - ovp_shared
                        vneg = -post - float(line_l.emergency_limit[t]) - ovn_shared
                        worst_line = max(worst_line, vpos, vneg)
    checks.append(
        CheckItem(
            ID_C_CONT_LINE,
            "Post-contingency line limits (line outage via LODF)",
            max(0.0, worst_line),
        )
    )

    # C-121 — Post-contingency line limits (generator outages via ISF), with shared slacks
    worst_gen = 0.0
    if lines and f is not None and scenario.contingencies:
        isf_csc = scenario.isf.tocsc()
        ref_1b = getattr(scenario, "ptdf_ref_bus_index", scenario.buses[0].index)
        non_ref_bus_indices = sorted(
            [b.index for b in scenario.buses if b.index != ref_1b]
        )
        col_by_bus_1b = {bus_1b: col for col, bus_1b in enumerate(non_ref_bus_indices)}

        for cont in scenario.contingencies:
            if not getattr(cont, "units", None):
                continue
            for gen in cont.units:
                bidx = gen.bus.index
                if bidx == ref_1b or bidx not in col_by_bus_1b:
                    continue
                col = isf_csc.getcol(col_by_bus_1b[bidx])
                for l_row, coeff in zip(col.indices.tolist(), col.data.tolist()):
                    if abs(coeff) < CONT_ISF_TOL:
                        continue
                    line_l = scenario.lines[l_row]
                    for t in range(T):
                        post = float(f[line_l.name, t].X) - float(
                            coeff
                        ) * _compute_p_of_gen_t(gen, t, commit, seg)
                        ovp_shared = (
                            float(covp[line_l.name, t].X) if covp is not None else 0.0
                        )
                        ovn_shared = (
                            float(covn[line_l.name, t].X) if covn is not None else 0.0
                        )
                        vpos = post - float(line_l.emergency_limit[t]) - ovp_shared
                        vneg = -post - float(line_l.emergency_limit[t]) - ovn_shared
                        worst_gen = max(worst_gen, vpos, vneg)
    checks.append(
        CheckItem(
            ID_C_CONT_GEN,
            "Post-contingency line limits (generator outage via ISF)",
            max(0.0, worst_gen),
        )
    )

    # Objective: O-301 (consistency)
    worst = 0.0
    try:
        obj_val = float(getattr(model, "ObjVal"))
    except Exception:
        obj_val = math.nan
    calc_val = _objective_from_vars(
        units,
        reserves,
        lines,
        T,
        commit,
        seg,
        s,
        ovp,
        ovn,
        covp,
        covn,
        su=su,
        contingency_penalty_factor=CONT_PENALTY_FACTOR,
    )
    if not (math.isnan(obj_val) or math.isinf(obj_val)):
        worst = abs(calc_val - obj_val)
    else:
        worst = float("nan")
    checks.append(
        CheckItem(ID_O_TOTAL, "Objective consistency (recomputed vs solver)", worst)
    )

    lines_out: List[str] = []
    lines_out.append("===== SCUC Solution Verification =====")
    lines_out.append(f"Scenario      : {scenario.name}")
    lines_out.append(f"Time steps    : {T} (step = {scenario.time_step} min)")
    lines_out.append(
        f"Counts        : {len(units)} units, {len(scenario.buses)} buses, {len(scenario.lines)} lines, {len(reserves)} reserve products"
    )
    lines_out.append("")
    lines_out.append("Index, Name, Result")
    for c in checks:
        if isinstance(c.value, float) and not math.isnan(c.value) and c.value <= EPS:
            res = "OK"
        else:
            res = (
                f"{c.value:.8f}"
                if isinstance(c.value, float) and not math.isnan(c.value)
                else str(c.value)
            )
        lines_out.append(f"{c.idx}, {c.name}, {res}")

    overall_ok = all(
        (isinstance(c.value, float) and c.value <= EPS)
        or (c.idx == ID_O_TOTAL and math.isnan(c.value))
        for c in checks
    )
    lines_out.append("")
    lines_out.append(f"Overall: {'OK' if overall_ok else 'VIOLATIONS DETECTED'}")

    out_path.write_text("\n".join(lines_out), encoding="utf-8")
    return out_path
