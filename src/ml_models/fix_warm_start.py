"""
Warm-start "fixer" that repairs a warm JSON without using a solver (no Gurobi).

Goal
- Start from a warm-start JSON produced from a neighbor solution (k-NN).
- Make it solver-friendly and as-feasible-as-possible by:
  • fixing commitment to enforce must-run and min up/down (given initial status),
  • enforcing ramp limits (with startup/shutdown limits) on total generation,
  • clipping segment power to available segment capacity when committed,
  • balancing total production to match total load at each time step exactly
    (this addresses power_balance equalities),
  • setting reserves to a safe feasible pattern (zero provision with shortfall covering requirement),
  • recomputing base-case line flows via PTDF and setting overflow slacks.

Notes
- Power balance (system equality) is enforced in this repair by construction,
  provided that the load lies within the system's feasible envelope
  sum_i p_lb_i(t) <= Load(t) <= sum_i p_ub_i(t) for each t,
  where p_lb/p_ub include min output and ramp/SU/SD limits.
  If the envelope does not contain the load, exact balancing is impossible for that commitment,
  and we fall back to the nearest feasible bound (p_lb or p_ub). We report any residual gap.
- Min up/down and ramp feasibility are enforced by construction (lock-based commitment repair and
  ramp-aware bounds for p_t).
- Reserves are set conservatively: provision r[k,g,t] = 0 for all, shortfall = requirement,
  which always satisfies reserve constraints (feasible; not optimal).
- Line limits are respected by setting overflow slacks (feasible via slacks).
- No Gurobi import is used anywhere in this fixer.

Usage (CLI):
  python -m src.ml_models.fix_warm_start --instance matpower/case57/2017-06-24

Optional flags:
  --warm-file PATH       Use an existing warm JSON. If omitted, try to find the default
                         warm_<instance>.json. If missing, auto-generate via neighbor DB.
  --require-pretrained   Fail if no pre-trained index is found (when auto-generating).
  --use-train-db         Restrict neighbor search to the training split (when auto-generating).

This writes:
  src/data/intermediate/warm_start/warm_fixed_<instance>.json
and prints a short report (balance feasibility, max balance gap, etc.).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math

from src.data_preparation.params import DataParams
from src.data_preparation.read_data import read_benchmark


# ------------------------------ helpers ------------------------------ #


def _sanitize_name(s: str) -> str:
    s = (s or "").strip().strip("/\\").replace("\\", "/")
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    t = "".join(out)
    while "__" in t:
        t = t.replace("__", "_")
    return t.strip("_").lower()


def _warm_path_for_instance(instance_name: str) -> Path:
    tag = _sanitize_name(instance_name)
    return (DataParams._WARM_START / f"warm_{tag}.json").resolve()


def fixed_warm_path_for_instance(instance_name: str) -> Path:
    tag = _sanitize_name(instance_name)
    return (DataParams._WARM_START / f"warm_fixed_{tag}.json").resolve()


def _ensure_list_length(vals, T: int, pad_with_last: bool = True, default=0):
    if vals is None:
        return [default] * T
    vals = list(vals)
    if len(vals) < T:
        pad_val = vals[-1] if (pad_with_last and vals) else default
        vals = vals + [pad_val] * (T - len(vals))
    elif len(vals) > T:
        vals = vals[:T]
    return vals


def _b01(x) -> int:
    try:
        return 1 if float(x) >= 0.5 else 0
    except Exception:
        return 0


def _startup_shutdown_from_commit(
    commit: List[int], initial_u: int
) -> Tuple[List[int], List[int]]:
    T = len(commit)
    v = [0] * T
    w = [0] * T
    prev = int(initial_u)
    for t in range(T):
        u = int(commit[t])
        if u > prev:
            v[t] = 1
        elif u < prev:
            w[t] = 1
        prev = u
    return v, w


def _enforce_min_up_down(
    u_raw: List[int],
    must_run: List[bool],
    min_up: int,
    min_down: int,
    initial_status_steps: Optional[int],
) -> List[int]:
    """
    Return a commitment series that enforces:
      - must_run[t] -> 1
      - min up/down constraints using a simple lock mechanism
      - initial_status boundary condition

    Lock policy:
      - When we change state at t (0->1 or 1->0), lock the next (L-1) periods to that state,
        where L is min_up for ON and min_down for OFF.
      - If initial_status is provided and its absolute value is less than L for the
        corresponding state, start with a lock for the remaining steps.
    """
    T = len(u_raw)
    u = [1 if must_run[t] or _b01(u_raw[t]) else 0 for t in range(T)]

    # Initial state & initial lock from initial_status
    if initial_status_steps is None or initial_status_steps == 0:
        prev_state = u[0]
        lock_remaining = 0
        locked_state = prev_state
    else:
        if initial_status_steps > 0:
            prev_state = 1
            need = max(0, min_up - int(initial_status_steps))
            lock_remaining = need
            locked_state = 1
        else:
            prev_state = 0
            s_off = -int(initial_status_steps)
            need = max(0, min_down - s_off)
            lock_remaining = need
            locked_state = 0

    u_fixed: List[int] = [0] * T

    for t in range(T):
        desired = u[t]

        # Apply lock first
        if lock_remaining > 0:
            desired = locked_state

        # Must-run overrides desired
        if must_run[t]:
            desired = 1
            # starting an ON run implies lock for min_up-1 (next periods)
            lock_remaining = max(lock_remaining, max(0, min_up - 1))
            locked_state = 1

        # Detect change vs previous state
        if desired != prev_state:
            # New state implies starting a new lock
            if desired == 1:
                lock_remaining = max(lock_remaining, max(0, min_up - 1))
                locked_state = 1
            else:
                lock_remaining = max(lock_remaining, max(0, min_down - 1))
                locked_state = 0

        u_fixed[t] = desired
        prev_state = desired
        lock_remaining = max(0, lock_remaining - 1)

    return u_fixed


def _add_to_segments(row: List[float], caps: List[float], add: float) -> float:
    """
    Add energy 'add' to row subject to per-segment caps (row[s] <= caps[s]).
    Returns leftover that could not be added (0 if fully added).
    """
    remaining = float(add)
    nS = len(row)
    for s in range(nS):
        if remaining <= 1e-12:
            break
        cap_s = max(0.0, float(caps[s]))
        cur = max(0.0, float(row[s]))
        free = max(0.0, cap_s - cur)
        if free <= 0.0:
            continue
        take = min(free, remaining)
        row[s] = cur + take
        remaining -= take
    return remaining


def _remove_from_segments(row: List[float], remove: float) -> float:
    """
    Remove energy 'remove' from row (prefer removing from higher segments first).
    Returns leftover that could not be removed (0 if fully removed).
    """
    remaining = float(remove)
    nS = len(row)
    for s in reversed(range(nS)):
        if remaining <= 1e-12:
            break
        cur = max(0.0, float(row[s]))
        if cur <= 0.0:
            continue
        take = min(cur, remaining)
        row[s] = cur - take
        remaining -= take
    return remaining


# --------------------------- core fixing logic --------------------------- #


def _fix_generators_and_reserves(
    scenario, warm: Dict
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """
    Fix generator commitment, segment power with capacity and ramping, balance to exact load,
    and set a safe reserve pattern.

    Returns
    -------
    (gen_out, reserves_out, p_total, p_above_min, balance_info)
      - gen_out:  dict[name] -> {"commit": [...], "segment_power": [[...], ...]}
      - reserves_out: dict[rname] -> provided_by_gen, total_provided, requirement, shortfall
      - p_total: dict[name] -> [p_t]
      - p_above_min: dict[name] -> [above_min_t]
      - balance_info: {"feasible": bool, "max_abs_gap": float, "gaps": List[float]}
    """
    sc = scenario
    T = sc.time

    # System total load per time
    total_load = [sum(float(b.load[t]) for b in sc.buses) for t in range(T)]

    warm_gen = warm.get("generators", {}) or {}
    gen_out: Dict[str, Dict] = {}
    p_total: Dict[str, List[float]] = {}
    p_above_min: Dict[str, List[float]] = {}

    # 1) Fix commitment with must-run + min up/down
    for gen in sc.thermal_units:
        gw = warm_gen.get(gen.name, {}) or {}
        u_raw = _ensure_list_length(gw.get("commit", []), T, True, 0)
        u_raw = [_b01(x) for x in u_raw]
        must = [bool(x) for x in gen.must_run]
        u_fixed = _enforce_min_up_down(
            u_raw=u_raw,
            must_run=must,
            min_up=int(getattr(gen, "min_up", 0) or 0),
            min_down=int(getattr(gen, "min_down", 0) or 0),
            initial_status_steps=getattr(gen, "initial_status", None),
        )
        gen_out[gen.name] = {
            "commit": u_fixed,
            "segment_power": [
                [0.0] * (len(gen.segments) if gen.segments else 0) for _ in range(T)
            ],
        }

    # 2) Compute startup/shutdown from commitment
    start_flags: Dict[str, List[int]] = {}
    stop_flags: Dict[str, List[int]] = {}
    init_u_map: Dict[str, int] = {}
    for gen in sc.thermal_units:
        init_u = 1 if (gen.initial_status is not None and gen.initial_status > 0) else 0
        init_u_map[gen.name] = init_u
        v, w = _startup_shutdown_from_commit(gen_out[gen.name]["commit"], init_u)
        start_flags[gen.name] = v
        stop_flags[gen.name] = w

    # 3) Forward pass per time: ramp-aware feasible ranges and balancing to exact total load
    balance_gaps: List[float] = [0.0] * T
    feasible_all = True

    # Keep p_prev and u_prev per generator
    p_prev_map: Dict[str, float] = {}
    u_prev_map: Dict[str, int] = {}
    for gen in sc.thermal_units:
        p_prev_map[gen.name] = (
            float(gen.initial_power) if (gen.initial_power is not None) else 0.0
        )
        u_prev_map[gen.name] = init_u_map[gen.name]

    for t in range(T):
        # Pre-collect bounds and warm desired rows
        per_gen_data = []
        sum_lb = 0.0
        sum_ub = 0.0

        for gen in sc.thermal_units:
            name = gen.name
            u_t = int(gen_out[name]["commit"][t])
            v_t = int(start_flags[name][t])
            w_t = int(stop_flags[name][t])

            min_t = float(gen.min_power[t]) * u_t
            max_t = float(gen.max_power[t]) * u_t

            # Segment caps for above-min power at this t
            nS = len(gen.segments) if gen.segments else 0
            caps = [
                (float(gen.segments[s].amount[t]) if nS > 0 else 0.0) * u_t
                for s in range(nS)
            ]

            # Warm desired above-min row
            warm_seg = warm_gen.get(name, {}).get("segment_power", None)
            if isinstance(warm_seg, list):
                warm_row = warm_seg[t] if t < len(warm_seg) else []
                warm_row = (warm_row + [0.0] * nS)[:nS]
                warm_row = [max(0.0, float(warm_row[s])) for s in range(nS)]
            else:
                warm_row = [0.0] * nS
            # Clip to caps
            warm_row = [min(warm_row[s], max(0.0, caps[s])) for s in range(nS)]
            desired_above = sum(warm_row)

            # Ramp-aware bounds
            p_prev = float(p_prev_map[name])
            u_prev = int(u_prev_map[name])
            ru = float(gen.ramp_up)
            rd = float(gen.ramp_down)
            SU = float(gen.startup_limit)
            SD = float(gen.shutdown_limit)

            up_ub = p_prev + ru * float(u_prev) + SU * float(v_t)
            dn_lb = p_prev - (rd * float(u_t) + SD * float(w_t))

            p_lb = max(min_t, dn_lb)
            p_ub = min(max_t, up_ub)
            if p_lb > p_ub:
                # Inconsistent; collapse to a point (still feasible from model's viewpoint)
                p_ub = p_lb

            # Initial pick within bounds
            target = min(max(min_t + desired_above, p_lb), p_ub)
            # Convert to row consistent with 'target'
            above_need = max(0.0, target - min_t)
            row = [0.0] * nS
            if above_need > 0.0 and nS > 0:
                leftover = _add_to_segments(row, caps, above_need)
                if leftover > 1e-9:
                    # Should not happen if above_need <= sum(caps), but keep robust
                    pass

            per_gen_data.append(
                {
                    "gen": gen,
                    "name": name,
                    "u_t": u_t,
                    "min_t": min_t,
                    "max_t": max_t,
                    "p_lb": p_lb,
                    "p_ub": p_ub,
                    "row": row,
                    "caps": caps,
                }
            )
            sum_lb += p_lb
            sum_ub += p_ub

        # Balance to exact total load if possible
        L = float(total_load[t])
        if L < sum_lb - 1e-8:
            # Cannot meet load with current commitment/ramp; choose p = p_lb (closest)
            feasible_all = False
            balance_gaps[t] = L - sum_lb
            # finalize rows as per p_lb
            for item in per_gen_data:
                gen = item["gen"]
                min_t = item["min_t"]
                row = item["row"]
                p_lb = item["p_lb"]
                current = min_t + sum(row)
                if current > p_lb + 1e-12:
                    # remove surplus
                    need_remove = current - p_lb
                    _remove_from_segments(row, need_remove)
                elif current < p_lb - 1e-12:
                    # add deficit (limited by caps)
                    need_add = p_lb - current
                    _add_to_segments(row, item["caps"], need_add)
        elif L > sum_ub + 1e-8:
            # Cannot meet load; choose p = p_ub (closest)
            feasible_all = False
            balance_gaps[t] = L - sum_ub
            for item in per_gen_data:
                gen = item["gen"]
                min_t = item["min_t"]
                row = item["row"]
                p_ub = item["p_ub"]
                current = min_t + sum(row)
                if current < p_ub - 1e-12:
                    need_add = p_ub - current
                    _add_to_segments(row, item["caps"], need_add)
                elif current > p_ub + 1e-12:
                    need_remove = current - p_ub
                    _remove_from_segments(row, need_remove)
        else:
            # Feasible to hit exact load
            delta = L - sum(
                min(item["min_t"] + sum(item["row"]), item["p_ub"])
                for item in per_gen_data
            )
            # The above formula already uses our current target; recompute from scratch
            current_sum = sum(item["min_t"] + sum(item["row"]) for item in per_gen_data)
            delta = L - current_sum
            if abs(delta) > 1e-9:
                if delta > 0:
                    # Distribute upward within p_ub bounds
                    remaining = delta
                    # Multi-pass greedy water-filling
                    # First pass in natural order, then repeat if needed
                    for _pass in range(2):
                        for item in per_gen_data:
                            if remaining <= 1e-12:
                                break
                            min_t = item["min_t"]
                            row = item["row"]
                            caps = item["caps"]
                            p_ub = item["p_ub"]
                            current = min_t + sum(row)
                            up_slack = max(0.0, p_ub - current)
                            if up_slack <= 0.0:
                                continue
                            take = min(up_slack, remaining)
                            leftover = _add_to_segments(row, caps, take)
                            # leftover should be ~0; but if not, reduce effective take
                            effective = take - leftover
                            remaining -= effective
                        if remaining <= 1e-12:
                            break
                    # Any remaining small residual can be distributed again (should be ~0)
                    balance_gaps[t] = remaining if remaining != 0 else 0.0
                else:
                    # Distribute downward within p_lb bounds
                    remaining = -delta
                    for _pass in range(2):
                        for item in per_gen_data:
                            if remaining <= 1e-12:
                                break
                            min_t = item["min_t"]
                            row = item["row"]
                            p_lb = item["p_lb"]
                            current = min_t + sum(row)
                            dn_slack = max(0.0, current - p_lb)
                            if dn_slack <= 0.0:
                                continue
                            take = min(dn_slack, remaining)
                            leftover = _remove_from_segments(row, take)
                            effective = take - leftover
                            remaining -= effective
                        if remaining <= 1e-12:
                            break
                    balance_gaps[t] = -remaining if remaining != 0 else 0.0

        # Finalize rows and p_t; update prevs
        for item in per_gen_data:
            gen = item["gen"]
            name = item["name"]
            row = [max(0.0, float(x)) for x in item["row"]]
            gen_out[name]["segment_power"][t] = row
            p_t = float(item["min_t"]) + sum(row)
            p_total.setdefault(name, [])
            p_total[name].append(float(p_t))
            p_above_min.setdefault(name, [])
            p_above_min[name].append(float(sum(row)))
            p_prev_map[name] = p_t
            u_prev_map[name] = int(gen_out[name]["commit"][t])

    max_abs_gap = max(abs(float(g)) for g in balance_gaps) if balance_gaps else 0.0
    balance_info = {
        "feasible": bool(feasible_all and max_abs_gap <= 1e-6),
        "max_abs_gap": float(max_abs_gap),
        "gaps": [float(x) for x in balance_gaps],
    }

    # 4) Reserves — set to minimal safe feasible pattern: all provision zero; shortfall = requirement
    reserves_out: Dict[str, Dict] = {}
    if getattr(sc, "reserves", None):
        for r in sc.reserves:
            requirement = [float(r.amount[t]) for t in range(T)]
            shortfall = [float(r.amount[t]) for t in range(T)]
            provided_by_gen = {g.name: [0.0 for _ in range(T)] for g in r.thermal_units}
            total_provided = [0.0 for _ in range(T)]
            reserves_out[r.name] = {
                "provided_by_gen": provided_by_gen,
                "total_provided": total_provided,
                "requirement": requirement,
                "shortfall": shortfall,
            }

    return gen_out, reserves_out, p_total, p_above_min, balance_info


def _fix_network(sc, p_total: Dict[str, List[float]]) -> Dict:
    """
    Compute flows with PTDF and set overflow slacks (base-case).
    Returns a dict 'network' with { "lines": { line_name: {...} } }
    """
    T = sc.time
    if not sc.lines or sc.isf is None:
        return {}

    isf = sc.isf.tocsr()
    buses = sc.buses
    lines = sc.lines

    ref_1b = getattr(sc, "ptdf_ref_bus_index", buses[0].index)
    non_ref_bus_indices = sorted([b.index for b in buses if b.index != ref_1b])

    # Precompute p_total at buses
    inj: Dict[int, List[float]] = {b.index: [0.0] * T for b in buses}
    for b in buses:
        for gen in b.thermal_units:
            pt = p_total[gen.name]
            for t in range(T):
                inj[b.index][t] += float(pt[t])
        for t in range(T):
            inj[b.index][t] -= float(b.load[t])

    # Compute flows and overflow slacks
    lines_out = {}
    for line in lines:
        flow = [0.0 for _ in range(T)]
        ovp = [0.0 for _ in range(T)]
        ovn = [0.0 for _ in range(T)]
        lim = [float(line.normal_limit[t]) for t in range(T)]
        pen = [float(line.flow_penalty[t]) for t in range(T)]

        row = isf.getrow(line.index - 1)
        cols = row.indices.tolist()
        coeffs = row.data.tolist()

        for t in range(T):
            fval = 0.0
            for col, coeff in zip(cols, coeffs):
                bus_1b = non_ref_bus_indices[col]
                fval += float(coeff) * float(inj[bus_1b][t])
            flow[t] = fval
            # Overflow
            ovp[t] = max(0.0, fval - lim[t])
            ovn[t] = max(0.0, -fval - lim[t])

        lines_out[line.name] = {
            "source": line.source.name,
            "target": line.target.name,
            "flow": list(map(float, flow)),
            "limit": lim,
            "overflow_pos": list(map(float, ovp)),
            "overflow_neg": list(map(float, ovn)),
            "penalty": pen,
        }

    return {"lines": lines_out}


def _power_balance_gap(
    sc, p_total: Dict[str, List[float]]
) -> Tuple[float, List[float]]:
    T = sc.time
    load_t = [sum(b.load[t] for b in sc.buses) for t in range(T)]
    prod_t = [0.0] * T
    for gen in sc.thermal_units:
        pt = p_total[gen.name]
        for t in range(T):
            prod_t[t] += float(pt[t])
    gap = [float(prod_t[t] - load_t[t]) for t in range(T)]
    worst = max(abs(x) for x in gap) if gap else 0.0
    return worst, gap


def fix_warm_payload(instance_name: str, sc, warm: Dict) -> Dict:
    """
    Produce a fixed-warm payload for 'instance_name' and scenario 'sc'.
    """
    gen_out, reserves_out, p_total, _p_above_min, balance_info = (
        _fix_generators_and_reserves(sc, warm)
    )
    network_out = _fix_network(sc, p_total)

    fixed = {
        "instance_name": instance_name,
        "case_folder": "/".join(instance_name.strip().split("/")[:2]),
        "neighbor": warm.get("neighbor", "unknown_neighbor"),
        "distance": float(warm.get("distance", 0.0)),
        "coverage": float(warm.get("coverage", 0.0)),
        "generators": gen_out,
        "reserves": reserves_out,
        "network": network_out,
        "balance": {
            "feasible": bool(balance_info.get("feasible", False)),
            "max_abs_gap": float(balance_info.get("max_abs_gap", 0.0)),
        },
    }
    return fixed


def fix_warm_file(
    instance_name: str,
    warm_file: Optional[Path] = None,
    *,
    require_pretrained: bool = False,
    use_train_db: bool = False,
) -> Optional[Path]:
    """
    Fix an existing warm JSON (or auto-generate it if missing) and write warm_fixed_<instance>.json.

    Returns the path to the fixed warm JSON or None on failure.
    """
    name = instance_name
    inst = read_benchmark(name, quiet=True)
    sc = inst.deterministic

    # Find/generate warm JSON
    warm_path = Path(warm_file) if warm_file else _warm_path_for_instance(name)
    if not warm_path.exists():
        # auto-generate via neighbor DB (no solver)
        try:
            from src.ml_models.warm_start import (
                WarmStartProvider,
            )  # lazy import to avoid cycles

            cf = "/".join(name.strip().split("/")[:2])
            wsp = WarmStartProvider(case_folder=cf)
            trained, cov = wsp.ensure_trained(
                cf, allow_build_if_missing=not require_pretrained
            )
            if not trained:
                print(
                    f"[fix_warm] No warm-start index available for {cf} (coverage={cov:.3f})."
                )
                return None
            warm_path = wsp.generate_and_save_warm_start(
                name,
                use_train_index_only=use_train_db,
                exclude_self=True,
                auto_fix=False,
            )
            if not warm_path or not Path(warm_path).exists():
                print("[fix_warm] Failed to generate warm-start JSON.")
                return None
            print(f"[fix_warm] Generated warm JSON: {warm_path}")
        except Exception as e:
            print(f"[fix_warm] Auto-generation failed: {e}")
            return None

    # Load warm JSON
    try:
        warm = json.loads(Path(warm_path).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[fix_warm] Could not read warm file '{warm_path}': {e}")
        return None

    # Fix payload
    fixed = fix_warm_payload(name, sc, warm)

    # Save fixed
    out_path = fixed_warm_path_for_instance(name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixed, indent=2), encoding="utf-8")

    # Report
    # Recompute power balance gaps from the fixed payload for clarity
    p_total_map = {
        g: [
            fixed["generators"][g]["commit"][t]
            * float(sc.thermal_units_by_name[g].min_power[t])
            + sum(fixed["generators"][g]["segment_power"][t])
            for t in range(sc.time)
        ]
        for g in fixed["generators"]
    }
    worst_gap, gap = _power_balance_gap(sc, p_total_map)

    print(f"[fix_warm] Wrote fixed warm: {out_path}")
    feas_flag = fixed.get("balance", {}).get("feasible", False)
    print(
        f"[fix_warm] Power balance feasibility: {'OK' if feas_flag and worst_gap <= 1e-6 else 'NEAR/NOT OK'}, "
        f"Max |production - load| = {worst_gap:.6f} MW"
    )

    return out_path


# ------------------------------- CLI ------------------------------- #


def main():
    ap = argparse.ArgumentParser(
        description="Fix a warm-start JSON without using Gurobi (min-up/down, ramp, segment caps, reserves, PTDF slacks, and balance to load)."
    )
    ap.add_argument(
        "--instance",
        required=True,
        help="Dataset name, e.g., matpower/case57/2017-06-24",
    )
    ap.add_argument(
        "--warm-file",
        default=None,
        help="Path to an existing warm JSON. If omitted, use default warm_<instance>.json (auto-generate if missing).",
    )
    ap.add_argument(
        "--require-pretrained",
        action="store_true",
        default=False,
        help="If auto-generating the warm JSON, require a pre-trained index.",
    )
    ap.add_argument(
        "--use-train-db",
        action="store_true",
        default=False,
        help="If auto-generating, restrict neighbor DB to the training split.",
    )
    args = ap.parse_args()

    out = fix_warm_file(
        instance_name=args.instance,
        warm_file=Path(args.warm_file) if args.warm_file else None,
        require_pretrained=args.require_pretrained,
        use_train_db=args.use_train_db,
    )
    if out is None:
        print("[fix_warm] Failed.")
    else:
        print(f"[fix_warm] Fixed warm saved to: {out}")


if __name__ == "__main__":
    main()
