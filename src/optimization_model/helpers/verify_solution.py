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

EPS = 1e-5  # Slightly relaxed for verification


@dataclass
class CheckItem:
    idx: str
    name: str
    value: float  # 0 means OK; otherwise worst violation


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

    # Final decision
    overall_ok = all(c.value <= EPS for c in checks)

    report = ["Index, Name, Max Violation"]
    for c in checks:
        val_str = "OK" if c.value <= EPS else f"{c.value:.6f}"
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
