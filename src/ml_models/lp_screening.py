"""
LP Relaxation-Guided Contingency Screening (LPSCREEN).

Novel method: solve the LP relaxation of the SCUC model (without contingency
constraints) to obtain instance-specific base-case flows, then use PTDF/LODF
to compute post-contingency margins and prune constraints with large margins.

Advantages over existing methods
---------------------------------
- No training data needed — works on any new grid/instance.
- Instance-specific — uses the current demand/topology, not a neighbor's.
- Exact physics — PTDF/LODF give exact post-contingency flows from any dispatch.
- Fast — LP relaxation solves in seconds even for large grids.
- Combinable — can stack with WARM, HINTS, LAZY for further speedup.

Modes
-----
- LPSCREEN-τ: Fixed threshold LP screening (simplest, no ML).
- LPSCREEN+LAZY: LP screening + lazy callback for guaranteed N-1 security.

CLI:
    python -m src.ml_models.lp_screening --instance matpower/case118/2017-01-01 --tau 0.10
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Dict, List, Optional, Tuple, Set, Callable

import numpy as np
from scipy.sparse import csc_matrix

from src.data_preparation.read_data import read_benchmark
from src.optimization_model.SCUC_solver.scuc_model_builder import build_model

logger = logging.getLogger(__name__)

# Same tolerances as contingencies.py
_LODF_TOL = 1e-4
_ISF_TOL = 1e-8


def _solve_lp_relaxation(
    scenario,
    time_limit: float = 30.0,
    mip_gap: float = 0.0,
) -> Optional[Dict]:
    """
    Build the SCUC model WITHOUT contingency constraints, relax all integers
    to continuous, solve the LP, and extract base-case line flows + generator
    total power.

    Returns dict with:
      - "flows": {line_name: [flow_t for t in T]}
      - "pgen":  {gen_name: [p_t for t in T]}
      - "commit_frac": {gen_name: [u_t_frac for t in T]}  (fractional commitment)
      - "lp_obj": float
      - "lp_time": float (seconds)
    or None on failure.
    """
    import gurobipy as gp
    from gurobipy import GRB

    # Build model without contingencies and without lazy mode
    model = build_model(
        scenario=scenario,
        contingency_filter=None,
        use_lazy_contingencies=True,  # skip adding contingencies entirely
    )

    # Relax all integer/binary variables to continuous
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    for var in model.getVars():
        if var.VType in (GRB.BINARY, GRB.INTEGER):
            var.VType = GRB.CONTINUOUS

    model.update()

    t0 = time.time()
    model.optimize()
    lp_time = time.time() - t0

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        logger.warning("[lpscreen] LP relaxation did not solve optimally (status=%d)", model.Status)
        model.dispose()
        return None

    lp_obj = float(model.ObjVal)

    T = scenario.time
    lines = scenario.lines or []
    line_flow = getattr(model, "line_flow", None)
    commit = getattr(model, "commit", None)
    seg = getattr(model, "gen_segment_power", None)

    # Extract flows
    flows: Dict[str, List[float]] = {}
    for ln in lines:
        fl = []
        for t in range(T):
            try:
                fl.append(float(line_flow[ln.name, t].X))
            except Exception:
                fl.append(0.0)
        flows[ln.name] = fl

    # Extract generator dispatch and fractional commitment
    pgen: Dict[str, List[float]] = {}
    commit_frac: Dict[str, List[float]] = {}
    for gen in scenario.thermal_units:
        pt = []
        uf = []
        nS = len(gen.segments) if gen.segments else 0
        for t in range(T):
            try:
                u_val = float(commit[gen.name, t].X)
            except Exception:
                u_val = 0.0
            p_val = u_val * float(gen.min_power[t])
            if nS > 0:
                for s in range(nS):
                    try:
                        p_val += float(seg[gen.name, t, s].X)
                    except Exception:
                        pass
            pt.append(p_val)
            uf.append(u_val)
        pgen[gen.name] = pt
        commit_frac[gen.name] = uf

    model.dispose()

    return {
        "flows": flows,
        "pgen": pgen,
        "commit_frac": commit_frac,
        "lp_obj": lp_obj,
        "lp_time": lp_time,
    }


def compute_lp_margins(
    scenario,
    lp_result: Dict,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """
    Compute minimum relative emergency margin across time for each
    (monitored_line, outaged_element) pair using LP relaxation flows.

    Returns:
      (line_pairs_min_margin, gen_pairs_min_margin)
      where each dict maps (line_name, outage_name) -> min_t margin(t)
    """
    T = scenario.time
    lines = scenario.lines or []
    flows = lp_result["flows"]
    pgen = lp_result["pgen"]

    lodf_csc: csc_matrix = scenario.lodf.tocsc()
    isf_csc: csc_matrix = scenario.isf.tocsc()

    buses = scenario.buses
    ref_1b = getattr(scenario, "ptdf_ref_bus_index", buses[0].index)
    non_ref_bus_indices = sorted([b.index for b in buses if b.index != ref_1b])
    col_by_bus_1b = {bus_1b: col for col, bus_1b in enumerate(non_ref_bus_indices)}
    line_by_row = {ln.index - 1: ln for ln in lines}

    line_margins: Dict[Tuple[str, str], float] = {}
    gen_margins: Dict[Tuple[str, str], float] = {}

    # Line-outage margins
    for cont in scenario.contingencies or []:
        if not cont.lines:
            continue
        for out_line in cont.lines:
            mcol = out_line.index - 1
            col = lodf_csc.getcol(mcol)
            for l_row, alpha in zip(col.indices.tolist(), col.data.tolist()):
                if l_row == mcol or abs(alpha) < _LODF_TOL:
                    continue
                line_l = line_by_row.get(l_row)
                if line_l is None:
                    continue
                key = (line_l.name, out_line.name)
                s_min = float("inf")
                for t in range(T):
                    f_l = float(flows.get(line_l.name, [0.0] * T)[t])
                    f_m = float(flows.get(out_line.name, [0.0] * T)[t])
                    post = f_l + alpha * f_m
                    F_em = float(line_l.emergency_limit[t])
                    if F_em <= 0:
                        continue
                    s = (F_em - abs(post)) / F_em
                    if s < s_min:
                        s_min = s
                if s_min != float("inf"):
                    line_margins[key] = s_min

    # Gen-outage margins
    for cont in scenario.contingencies or []:
        if not getattr(cont, "units", None):
            continue
        for gen in cont.units:
            bidx = gen.bus.index
            if bidx == ref_1b or bidx not in col_by_bus_1b:
                continue
            col = isf_csc.getcol(col_by_bus_1b[bidx])
            isf_map = {r: v for r, v in zip(col.indices.tolist(), col.data.tolist())}
            for line_l in lines:
                beta = float(isf_map.get(line_l.index - 1, 0.0))
                if abs(beta) < _ISF_TOL:
                    continue
                key = (line_l.name, gen.name)
                s_min = float("inf")
                for t in range(T):
                    f_l = float(flows.get(line_l.name, [0.0] * T)[t])
                    p_g = float(pgen.get(gen.name, [0.0] * T)[t])
                    post = f_l - beta * p_g
                    F_em = float(line_l.emergency_limit[t])
                    if F_em <= 0:
                        continue
                    s = (F_em - abs(post)) / F_em
                    if s < s_min:
                        s_min = s
                if s_min != float("inf"):
                    gen_margins[key] = s_min

    return line_margins, gen_margins


class LPScreener:
    """
    LP Relaxation-Guided Contingency Screener.

    Usage:
        screener = LPScreener()
        pred, stats = screener.screen(scenario, tau=0.10)
        # pred is a filter_predicate for contingencies.add_constraints
    """

    def screen(
        self,
        scenario,
        *,
        tau: float = 0.10,
        lp_time_limit: float = 30.0,
    ) -> Optional[Tuple[Callable, Dict]]:
        """
        Run LP relaxation, compute margins, build a pruning predicate.

        Parameters
        ----------
        scenario : UnitCommitmentScenario
        tau : float
            Prune contingency pairs where LP min-margin >= tau.
        lp_time_limit : float
            Time limit for the LP relaxation solve.

        Returns
        -------
        (predicate, stats) or None on LP failure.
        predicate(kind, line_l, out_obj, t, coeff, F_em) -> bool (True=keep)
        """
        t0 = time.time()
        lp_result = _solve_lp_relaxation(scenario, time_limit=lp_time_limit)
        if lp_result is None:
            return None

        line_margins, gen_margins = compute_lp_margins(scenario, lp_result)
        screen_time = time.time() - t0

        stats = {
            "lp_time": lp_result["lp_time"],
            "screen_time": screen_time,
            "lp_obj": lp_result["lp_obj"],
            "total_line_pairs": len(line_margins),
            "total_gen_pairs": len(gen_margins),
            "skipped_line": 0,
            "skipped_gen": 0,
        }

        thr = float(tau)

        def _predicate(
            kind: str, line_l, out_obj, t: int, coeff: float, F_em: float
        ) -> bool:
            if kind == "line":
                key = (line_l.name, out_obj.name)
                srel = line_margins.get(key, None)
                if srel is None:
                    return True  # unknown pair → keep
                if srel >= thr:
                    stats["skipped_line"] += 1
                    return False
                return True
            else:
                key = (line_l.name, out_obj.name)
                srel = gen_margins.get(key, None)
                if srel is None:
                    return True
                if srel >= thr:
                    stats["skipped_gen"] += 1
                    return False
                return True

        return _predicate, stats

    def screen_masks(
        self,
        scenario,
        *,
        tau: float = 0.10,
        lp_time_limit: float = 30.0,
    ) -> Optional[Tuple[Tuple[set, set], Dict]]:
        """
        Alternative interface: return keep-masks (sets of pairs to KEEP).
        """
        t0 = time.time()
        lp_result = _solve_lp_relaxation(scenario, time_limit=lp_time_limit)
        if lp_result is None:
            return None

        line_margins, gen_margins = compute_lp_margins(scenario, lp_result)
        screen_time = time.time() - t0

        thr = float(tau)
        keep_line = set()
        keep_gen = set()

        for key, srel in line_margins.items():
            if srel < thr:
                keep_line.add(key)

        for key, srel in gen_margins.items():
            if srel < thr:
                keep_gen.add(key)

        stats = {
            "lp_time": lp_result["lp_time"],
            "screen_time": screen_time,
            "lp_obj": lp_result["lp_obj"],
            "total_line_pairs": len(line_margins),
            "total_gen_pairs": len(gen_margins),
            "kept_line_pairs": len(keep_line),
            "kept_gen_pairs": len(keep_gen),
            "pruned_line_pairs": len(line_margins) - len(keep_line),
            "pruned_gen_pairs": len(gen_margins) - len(keep_gen),
        }

        return (keep_line, keep_gen), stats


def main():
    ap = argparse.ArgumentParser(
        description="LP Relaxation-Guided Contingency Screening demo."
    )
    ap.add_argument("--instance", required=True, help="e.g., matpower/case118/2017-01-01")
    ap.add_argument("--tau", type=float, default=0.10, help="Pruning threshold")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    inst = read_benchmark(args.instance, quiet=True)
    sc = inst.deterministic

    screener = LPScreener()
    result = screener.screen(sc, tau=args.tau)

    if result is None:
        print("[lpscreen] LP relaxation failed.")
        return

    pred, stats = result
    print(f"[lpscreen] LP solved in {stats['lp_time']:.2f}s")
    print(f"[lpscreen] Total screen time: {stats['screen_time']:.2f}s")
    print(f"[lpscreen] Line pairs: {stats['total_line_pairs']}")
    print(f"[lpscreen] Gen pairs: {stats['total_gen_pairs']}")
    print(f"[lpscreen] LP objective: {stats['lp_obj']:.2f}")


if __name__ == "__main__":
    main()
