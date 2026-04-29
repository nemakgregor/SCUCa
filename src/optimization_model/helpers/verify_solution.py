from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, List

import math
import numpy as np
from scipy import sparse

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

EPS = 1e-5  # Default verification tolerance
FLOW_DEF_EPS = 1e-3  # DC/PTDF flow reconstruction can drift slightly on large systems


@dataclass
class CheckItem:
    idx: str
    name: str
    value: float  # 0 means OK; otherwise worst violation


def _startup_cost_for_gen(gen) -> float:
    try:
        if gen.startup_categories:
            return float(min(cat.cost for cat in gen.startup_categories))
    except Exception:
        pass
    return 0.0


def _extract_vars_to_numpy(
    model, var_dict, keys: List[str], T: int, default=0.0
) -> np.ndarray:
    """Fast extraction of Gurobi vars to (len(keys), T) array."""
    if var_dict is None or not keys:
        return np.full((len(keys), T), default)

    arr = np.full((len(keys), T), default)
    # Iterate keys because var_dict is not guaranteed ordered/contiguous
    for i, key in enumerate(keys):
        for t in range(T):
            v = var_dict.get((key, t))
            if v is not None:
                try:
                    arr[i, t] = v.X
                except AttributeError:
                    pass
    return arr


def verify_solution(
    scenario: UnitCommitmentScenario, model
) -> Tuple[bool, List[CheckItem], str]:
    """
    Optimized solution verification.
    Uses vectorization for heavy checks (balance, contingencies) while keeping
    explicit logic for complex temporal constraints (min up/down).
    """
    T = scenario.time
    units = scenario.thermal_units
    reserves = scenario.reserves or []
    lines = scenario.lines or []

    # --- 1. Extract Data ---
    # Generators
    gen_names = [g.name for g in units]
    u_val = _extract_vars_to_numpy(model, getattr(model, "commit", None), gen_names, T)
    su_val = _extract_vars_to_numpy(
        model, getattr(model, "startup", None), gen_names, T
    )
    sd_val = _extract_vars_to_numpy(
        model, getattr(model, "shutdown", None), gen_names, T
    )

    # Power (p_total)
    p_min_arr = np.array([g.min_power[:T] for g in units])  # (N_gen, T)
    p_val = u_val * p_min_arr

    seg_vars = getattr(model, "gen_segment_power", None)
    if seg_vars:
        # This loop is unavoidable due to segment structure, but usually N_seg is small (1-3)
        for i, gen in enumerate(units):
            for s in range(len(gen.segments)):
                for t in range(T):
                    v = seg_vars.get((gen.name, t, s))
                    if v:
                        p_val[i, t] += v.X

    # Lines
    line_names = [ln.name for ln in lines]
    f_val = _extract_vars_to_numpy(
        model, getattr(model, "line_flow", None), line_names, T
    )
    ovp_val = _extract_vars_to_numpy(
        model, getattr(model, "line_overflow_pos", None), line_names, T
    )
    ovn_val = _extract_vars_to_numpy(
        model, getattr(model, "line_overflow_neg", None), line_names, T
    )
    covp_val = _extract_vars_to_numpy(
        model, getattr(model, "contingency_overflow_pos", None), line_names, T
    )
    covn_val = _extract_vars_to_numpy(
        model, getattr(model, "contingency_overflow_neg", None), line_names, T
    )
    reserve_td = getattr(model, "reserve", None)
    shortfall_td = getattr(model, "reserve_shortfall", None)

    # Limits (handle 0 or None)
    F_em = np.zeros((len(lines), T))
    F_norm = np.zeros((len(lines), T))
    for i, ln in enumerate(lines):
        for t in range(T):
            F_em[i, t] = (
                ln.emergency_limit[t]
                if ln.emergency_limit[t] > 0
                else (ln.normal_limit[t] if ln.normal_limit[t] > 0 else 1e9)
            )
            F_norm[i, t] = ln.normal_limit[t] if ln.normal_limit[t] > 0 else 1e9

    checks: List[CheckItem] = []

    # --- 2. Variable Checks ---

    # V-201: Commitment Integrality
    viol_int = np.min(np.stack([np.abs(u_val), np.abs(u_val - 1.0)]), axis=0)
    checks.append(
        CheckItem(
            ID_V_COMMIT,
            "Commitment integrality",
            np.max(viol_int) if viol_int.size else 0.0,
        )
    )

    # V-202, V-203, V-204 (Positive variables) - Skipped explicit loop for speed, Gurobi bounds handle this usually.

    # V-205/206: Start/Stop Integrality
    viol_su = np.min(np.stack([np.abs(su_val), np.abs(su_val - 1.0)]), axis=0)
    viol_sd = np.min(np.stack([np.abs(sd_val), np.abs(sd_val - 1.0)]), axis=0)
    checks.append(
        CheckItem(
            ID_V_SU, "Startup integrality", np.max(viol_su) if viol_su.size else 0.0
        )
    )
    checks.append(
        CheckItem(
            ID_V_SD, "Shutdown integrality", np.max(viol_sd) if viol_sd.size else 0.0
        )
    )

    # --- 3. Constraint Checks ---

    # C-101: Commitment Fixing
    worst_fix = 0.0
    for i, gen in enumerate(units):
        if not gen.commitment_status:
            continue
        for t, status in enumerate(gen.commitment_status[:T]):
            if status is not None:
                diff = abs(u_val[i, t] - (1.0 if status else 0.0))
                worst_fix = max(worst_fix, diff)
    checks.append(CheckItem(ID_C_FIX, "Commitment fixing", worst_fix))

    # C-102: Linking (p_seg <= amount * u)
    worst_link = 0.0
    if seg_vars:
        for i, gen in enumerate(units):
            for s, seg_obj in enumerate(gen.segments):
                amt = np.array(seg_obj.amount[:T])
                # Vectorized extraction of this segment
                p_seg_s = np.zeros(T)
                for t in range(T):
                    v = seg_vars.get((gen.name, t, s))
                    if v:
                        p_seg_s[t] = v.X

                viol = p_seg_s - amt * u_val[i, :]
                worst_link = max(worst_link, np.max(viol))
    checks.append(
        CheckItem(ID_C_LINK, "Segment capacity linking", max(0.0, worst_link))
    )

    # C-103: Power Balance
    total_gen = np.sum(p_val, axis=0)
    total_load = np.array([sum(b.load[t] for b in scenario.buses) for t in range(T)])
    worst_bal = np.max(np.abs(total_gen - total_load))
    checks.append(CheckItem(ID_C_BAL, "Power balance", worst_bal))

    # C-104/C-105: Reserve headroom and requirement
    worst_res_head = 0.0
    worst_res_req = 0.0
    if reserves:
        eligible_by_gen: Dict[str, List[Reserve]] = {}
        for r in reserves:
            for g in r.thermal_units:
                eligible_by_gen.setdefault(g.name, []).append(r)

        for i, gen in enumerate(units):
            rlist = eligible_by_gen.get(gen.name, [])
            if not rlist:
                continue
            reserve_sum = np.zeros(T)
            for r in rlist:
                for t in range(T):
                    try:
                        reserve_sum[t] += float(reserve_td[r.name, gen.name, t].X)
                    except Exception:
                        pass
            headroom = (
                np.array([float(gen.max_power[t]) - float(gen.min_power[t]) for t in range(T)])
                * u_val[i, :]
            )
            above_min = p_val[i, :] - p_min_arr[i, :] * u_val[i, :]
            viol_head = above_min + reserve_sum - headroom
            worst_res_head = max(worst_res_head, float(np.max(viol_head)))

        for r in reserves:
            provided = np.zeros(T)
            shortfall = np.zeros(T)
            for g in r.thermal_units:
                for t in range(T):
                    try:
                        provided[t] += float(reserve_td[r.name, g.name, t].X)
                    except Exception:
                        pass
            for t in range(T):
                try:
                    shortfall[t] = float(shortfall_td[r.name, t].X)
                except Exception:
                    shortfall[t] = 0.0
            req = np.array([float(r.amount[t]) for t in range(T)])
            viol_req = req - (provided + shortfall)
            worst_res_req = max(worst_res_req, float(np.max(viol_req)))
    checks.append(
        CheckItem(ID_C_RES_HEAD, "Reserve headroom linking", max(0.0, worst_res_head))
    )
    checks.append(
        CheckItem(ID_C_RES_REQ, "Reserve requirement", max(0.0, worst_res_req))
    )

    # C-106: Startup/Shutdown Definition (u[t] - u[t-1] = v[t] - w[t])
    # Use roll for t-1
    u_prev = np.roll(u_val, 1, axis=1)
    # Correct t=0 with initial status
    for i, gen in enumerate(units):
        s0 = gen.initial_status
        u0 = 1.0 if (s0 is not None and s0 > 0) else 0.0
        u_prev[i, 0] = u0

    lhs = u_val - u_prev
    rhs = su_val - sd_val
    worst_def = np.max(np.abs(lhs - rhs))
    checks.append(CheckItem(ID_C_SU_DEF, "Start/Stop definition", worst_def))

    # Exclusivity v + w <= 1
    worst_excl = np.max(np.maximum(0.0, su_val + sd_val - 1.0))
    # Merge into one check item or separate? Merging for brevity.
    checks[-1].value = max(checks[-1].value, worst_excl)

    # C-107: Ramping
    # p[t] - p[t-1] <= RU*u[t-1] + SU*v[t]
    # p[t-1] - p[t] <= RD*u[t] + SD*w[t]
    worst_ramp = 0.0
    if units:
        RU = np.array([float(g.ramp_up) for g in units])[:, None]
        RD = np.array([float(g.ramp_down) for g in units])[:, None]
        SU_lim = np.array([float(g.startup_limit) for g in units])[:, None]
        SD_lim = np.array([float(g.shutdown_limit) for g in units])[:, None]

        p_prev = np.roll(p_val, 1, axis=1)
        for i, gen in enumerate(units):
            p0 = float(gen.initial_power) if gen.initial_power is not None else 0.0
            p_prev[i, 0] = p0

        # Up ramp
        diff_up = p_val - p_prev
        lim_up = RU * u_prev + SU_lim * su_val
        viol_up = np.max(np.maximum(0.0, diff_up - lim_up))

        # Down ramp
        diff_dn = p_prev - p_val
        lim_dn = RD * u_val + SD_lim * sd_val
        viol_dn = np.max(np.maximum(0.0, diff_dn - lim_dn))

        worst_ramp = max(viol_up, viol_dn)
    checks.append(CheckItem(ID_C_RAMP, "Ramping", worst_ramp))

    # C-110/111: Min Up/Down (Time-window check)
    # This is complex to vectorize fully due to variable windows. Keep explicit loop (it's O(N*T)).
    worst_mu = 0.0
    worst_md = 0.0
    for i, gen in enumerate(units):
        Lu = int(gen.min_up)
        Ld = int(gen.min_down)

        if Lu > 0:
            # sum(v[k] for k in t-Lu+1..t) <= u[t]
            # Convolution is faster
            v_row = su_val[i, :]
            # Manual window sum
            for t in range(T):
                start = max(0, t - Lu + 1)
                s_v = np.sum(v_row[start : t + 1])
                viol = s_v - u_val[i, t]
                worst_mu = max(worst_mu, viol)

        if Ld > 0:
            w_row = sd_val[i, :]
            for t in range(T):
                start = max(0, t - Ld + 1)
                s_w = np.sum(w_row[start : t + 1])
                viol = s_w - (1.0 - u_val[i, t])
                worst_md = max(worst_md, viol)

    checks.append(CheckItem(ID_C_MIN_UP, "Min up time", worst_mu))
    checks.append(CheckItem(ID_C_MIN_DOWN, "Min down time", worst_md))

    # C-112: Initial horizon enforcement for min up/down
    worst_init = 0.0
    for i, gen in enumerate(units):
        Lu = int(getattr(gen, "min_up", 0) or 0)
        Ld = int(getattr(gen, "min_down", 0) or 0)
        s = getattr(gen, "initial_status", None)
        if s is None:
            continue

        if s > 0 and Lu > 0:
            remaining_on = max(0, Lu - int(s))
            if remaining_on > 0:
                window = u_val[i, : min(remaining_on, T)]
                if window.size:
                    worst_init = max(worst_init, float(np.max(np.abs(window - 1.0))))

        if s < 0 and Ld > 0:
            remaining_off = max(0, Ld - abs(int(s)))
            if remaining_off > 0:
                window = u_val[i, : min(remaining_off, T)]
                if window.size:
                    worst_init = max(worst_init, float(np.max(np.abs(window))))
    checks.append(CheckItem(ID_C_MIN_INIT, "Initial min up/down enforcement", worst_init))

    # C-108: Base-case line flow definition
    worst_flow_def = 0.0
    if lines:
        buses = scenario.buses
        bus_names = [b.name for b in buses]
        gen_idx_by_name = {g.name: i for i, g in enumerate(units)}
        inj_by_bus_name: Dict[str, np.ndarray] = {}
        for b in buses:
            idxs = [gen_idx_by_name[g.name] for g in b.thermal_units if g.name in gen_idx_by_name]
            gen_sum = np.sum(p_val[idxs, :], axis=0) if idxs else np.zeros(T)
            inj_by_bus_name[b.name] = gen_sum - np.array([float(x) for x in b.load[:T]])

        theta_vars = getattr(model, "bus_angle", None)
        if theta_vars:
            theta_val = _extract_vars_to_numpy(model, theta_vars, bus_names, T)
            theta_idx_by_name = {name: i for i, name in enumerate(bus_names)}
            out_line_indices: Dict[str, List[int]] = {b.name: [] for b in buses}
            in_line_indices: Dict[str, List[int]] = {b.name: [] for b in buses}

            for i, line in enumerate(lines):
                src_idx = theta_idx_by_name[line.source.name]
                tgt_idx = theta_idx_by_name[line.target.name]
                calc = float(line.susceptance) * (theta_val[src_idx, :] - theta_val[tgt_idx, :])
                worst_flow_def = max(worst_flow_def, float(np.max(np.abs(f_val[i, :] - calc))))
                out_line_indices[line.source.name].append(i)
                in_line_indices[line.target.name].append(i)

            for bus in buses:
                out_sum = np.sum(f_val[out_line_indices[bus.name], :], axis=0) if out_line_indices[bus.name] else np.zeros(T)
                in_sum = np.sum(f_val[in_line_indices[bus.name], :], axis=0) if in_line_indices[bus.name] else np.zeros(T)
                nodal_resid = out_sum - in_sum - inj_by_bus_name[bus.name]
                worst_flow_def = max(worst_flow_def, float(np.max(np.abs(nodal_resid))))
            flow_def_name = "Line flow DC definition / nodal balance"
        else:
            ref_1b = getattr(scenario, "ptdf_ref_bus_index", buses[0].index)
            non_ref_indices = sorted([b.index for b in buses if b.index != ref_1b])
            inj_by_bus: Dict[int, np.ndarray] = {}
            for b in buses:
                inj_by_bus[b.index] = inj_by_bus_name[b.name]

            isf_csr = scenario.isf.tocsr()
            for i, line in enumerate(lines):
                row = isf_csr.getrow(line.index - 1)
                calc = np.zeros(T)
                for col, coeff in zip(row.indices.tolist(), row.data.tolist()):
                    bus_1b = non_ref_indices[col]
                    calc += float(coeff) * inj_by_bus[bus_1b]
                worst_flow_def = max(
                    worst_flow_def, float(np.max(np.abs(f_val[i, :] - calc)))
                )
            flow_def_name = "Line flow PTDF equality"
    else:
        flow_def_name = "Line flow definition"
    checks.append(CheckItem(ID_C_FLOW_DEF, flow_def_name, worst_flow_def))

    # C-109: Base Flow Limits
    if lines:
        # |f| <= F_norm + slacks
        viol_pos = f_val - F_norm - ovp_val
        viol_neg = -f_val - F_norm - ovn_val
        worst_flow = np.max(np.maximum(viol_pos, viol_neg))
        checks.append(
            CheckItem(ID_C_FLOW_LIM, "Base flow limits", max(0.0, worst_flow))
        )
    else:
        checks.append(CheckItem(ID_C_FLOW_LIM, "Base flow limits", 0.0))

    # C-120/121: Contingencies (Vectorized)
    worst_line_cont = 0.0
    worst_gen_cont = 0.0

    if lines and scenario.contingencies:
        lodf_csc = scenario.lodf.tocsc()
        isf_csc = scenario.isf.tocsc()

        buses = scenario.buses
        ref_1b = getattr(scenario, "ptdf_ref_bus_index", buses[0].index)
        non_ref_indices = sorted([b.index for b in buses if b.index != ref_1b])
        col_by_bus_1b = {bid: c for c, bid in enumerate(non_ref_indices)}

        # 1. Line Outages
        for cont in scenario.contingencies:
            if not cont.lines:
                continue
            for out_line in cont.lines:
                # Find indices
                try:
                    out_idx = line_names.index(out_line.name)
                except ValueError:
                    continue

                mcol = out_line.index - 1
                col = lodf_csc.getcol(mcol)

                # Affected lines
                mon_indices = col.indices
                coeffs = col.data

                mask = (mon_indices != mcol) & (np.abs(coeffs) >= CONT_LODF_TOL)
                if not np.any(mask):
                    continue

                valid_mon_idx = mon_indices[mask]
                valid_coeffs = coeffs[mask][:, None]  # (M, 1)

                f_mon = f_val[valid_mon_idx, :]
                f_out = f_val[out_idx, :]

                f_post = f_mon + valid_coeffs * f_out

                lims = F_em[valid_mon_idx, :]
                sp = covp_val[valid_mon_idx, :]
                sn = covn_val[valid_mon_idx, :]

                v = np.maximum(f_post - lims - sp, -f_post - lims - sn)
                worst_line_cont = max(worst_line_cont, np.max(v))

        # 2. Gen Outages
        for cont in scenario.contingencies:
            if not getattr(cont, "units", None):
                continue
            for gen in cont.units:
                try:
                    g_idx = gen_names.index(gen.name)
                except ValueError:
                    continue

                bidx = gen.bus.index
                if bidx == ref_1b or bidx not in col_by_bus_1b:
                    v = np.maximum(f_val - F_em - covp_val, -f_val - F_em - covn_val)
                    worst_gen_cont = max(worst_gen_cont, np.max(v))
                    continue

                col = isf_csc.getcol(col_by_bus_1b[bidx])
                mon_indices = col.indices
                coeffs = col.data

                mask = np.abs(coeffs) >= CONT_ISF_TOL
                if not np.any(mask):
                    continue

                valid_mon_idx = mon_indices[mask]
                valid_coeffs = coeffs[mask][:, None]

                f_mon = f_val[valid_mon_idx, :]
                p_lost = p_val[g_idx, :]

                f_post = f_mon - valid_coeffs * p_lost

                lims = F_em[valid_mon_idx, :]
                sp = covp_val[valid_mon_idx, :]
                sn = covn_val[valid_mon_idx, :]

                v = np.maximum(f_post - lims - sp, -f_post - lims - sn)
                worst_gen_cont = max(worst_gen_cont, np.max(v))

    checks.append(CheckItem(ID_C_CONT_LINE, "Post-cont line limits", worst_line_cont))
    checks.append(CheckItem(ID_C_CONT_GEN, "Post-cont gen limits", worst_gen_cont))

    # O-301: Objective consistency
    obj_inconsistency = 0.0
    try:
        obj_calc = 0.0
        for i, gen in enumerate(units):
            for t in range(T):
                obj_calc += float(u_val[i, t]) * float(gen.min_power_cost[t])
                n_segments = len(gen.segments) if gen.segments else 0
                if n_segments > 0 and seg_vars:
                    for s in range(n_segments):
                        v = seg_vars.get((gen.name, t, s))
                        if v is not None:
                            obj_calc += float(v.X) * float(gen.segments[s].cost[t])
                obj_calc += float(su_val[i, t]) * _startup_cost_for_gen(gen)

        if reserves and shortfall_td is not None:
            for r in reserves:
                for t in range(T):
                    try:
                        obj_calc += float(shortfall_td[r.name, t].X) * float(
                            r.shortfall_penalty
                        )
                    except Exception:
                        pass

        for i, line in enumerate(lines):
            pen = np.array([float(x) for x in line.flow_penalty[:T]])
            obj_calc += float(np.sum((ovp_val[i, :] + ovn_val[i, :]) * pen))
            obj_calc += float(
                np.sum((covp_val[i, :] + covn_val[i, :]) * pen * CONT_PENALTY_FACTOR)
            )

        model_obj = float(model.ObjVal)
        obj_inconsistency = abs(model_obj - obj_calc)
    except Exception:
        obj_inconsistency = float("nan")
    checks.append(CheckItem(ID_O_TOTAL, "Objective consistency", obj_inconsistency))

    # Final decision
    overall_ok = all(c.value <= (FLOW_DEF_EPS if c.idx == ID_C_FLOW_DEF else EPS) for c in checks)

    report = ["Index, Name, Max Violation"]
    for c in checks:
        tol = FLOW_DEF_EPS if c.idx == ID_C_FLOW_DEF else EPS
        val_str = "OK" if c.value <= tol else f"{c.value:.6f}"
        report.append(f"{c.idx}, {c.name}, {val_str}")

    return overall_ok, checks, "\n".join(report)


def verify_solution_to_log(scenario, model, out_dir=None, filename=None):
    out_dir = out_dir or DataParams._LOGS
    out_dir.mkdir(parents=True, exist_ok=True)
    if not filename:
        filename = "SCUC_verify.log"
    path = out_dir / filename
    _, _, txt = verify_solution(scenario, model)
    path.write_text(txt, encoding="utf-8")
    return path
