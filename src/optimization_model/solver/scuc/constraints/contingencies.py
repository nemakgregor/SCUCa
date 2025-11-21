import logging
import gurobipy as gp
from scipy.sparse import csc_matrix
from typing import Optional, Callable, Dict, Tuple, Set, Iterable

from .log_utils import record_constraint_stat

logger = logging.getLogger(__name__)

# Drop extremely small coefficients to reduce constraint count
_LODF_TOL = 1e-4
_ISF_TOL = 1e-8


def _total_power_expr(gen, t: int, commit, seg_power) -> gp.LinExpr:
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
    keep_masks: Optional[Dict[str, Set[Tuple[str, str]]]] = None,
    filter_predicate: Optional[Callable] = None,
    name_constraints: bool = True,
    monitored_line_whitelist: Optional[Set[str]] = None,
) -> None:
    """
    Add explicit post-contingency constraints (C-120/C-121) using batched addConstrs.

    Pruning:
      - monitored_line_whitelist: if provided, only these monitored lines (L) are considered.
      - keep_masks: fast pair-level masks of (L|M) and (L|G).
      - filter_predicate: per-time predicate(kind, L, out_obj, t, coeff, F_em) -> bool.

    Instrumentation recorded on model:
      - _explicit_total_cont_constraints
      - _explicit_added_line_by_line: line_name -> inequalities
      - _explicit_pair_line_counts: "L|M" -> inequalities
      - _explicit_pair_gen_counts:  "L|G" -> inequalities
    """
    contingencies = scenario.contingencies or []
    lines = scenario.lines or []
    if not contingencies or not lines or line_flow is None:
        return

    lodf_csc: csc_matrix = scenario.lodf.tocsc()  # (n_lines, n_lines)
    isf_csc: csc_matrix = scenario.isf.tocsc()  # (n_lines, n_buses-1)

    buses = scenario.buses
    ref_1b = getattr(scenario, "ptdf_ref_bus_index", buses[0].index)
    non_ref_bus_indices = sorted([b.index for b in buses if b.index != ref_1b])
    col_by_bus_1b = {bus_1b: col for col, bus_1b in enumerate(non_ref_bus_indices)}

    line_by_row = {ln.index - 1: ln for ln in lines}

    keep_line_pairs = set()
    keep_gen_pairs = set()
    if keep_masks:
        keep_line_pairs = set(keep_masks.get("line", set()))
        keep_gen_pairs = set(keep_masks.get("gen", set()))

    def _mon_allowed(line_nm: str) -> bool:
        if monitored_line_whitelist is None:
            return True
        return line_nm in monitored_line_whitelist

    T_list = list(time_periods)

    explicit_total = int(getattr(model, "_explicit_total_cont_constraints", 0))
    by_line = getattr(model, "_explicit_added_line_by_line", None)
    if by_line is None:
        by_line = {}
        try:
            model._explicit_added_line_by_line = by_line
        except Exception:
            pass
    pair_line = getattr(model, "_explicit_pair_line_counts", None)
    if pair_line is None:
        pair_line = {}
        try:
            model._explicit_pair_line_counts = pair_line
        except Exception:
            pass
    pair_gen = getattr(model, "_explicit_pair_gen_counts", None)
    if pair_gen is None:
        pair_gen = {}
        try:
            model._explicit_pair_gen_counts = pair_gen
        except Exception:
            pass

    def _incr_line(line_name: str, k: int = 1):
        if not line_name:
            return
        by_line[line_name] = by_line.get(line_name, 0) + int(k)

    def _incr_pair_line(lname: str, oname: str, k: int = 1):
        key = f"{lname}|{oname}"
        pair_line[key] = pair_line.get(key, 0) + int(k)

    def _incr_pair_gen(lname: str, gname: str, k: int = 1):
        key = f"{lname}|{gname}"
        pair_gen[key] = pair_gen.get(key, 0) + int(k)

    def _kept_ts_line(line_l, out_line, coeff: float) -> Iterable[int]:
        if filter_predicate is None:
            return T_list
        kept = []
        for t in T_list:
            F_em = float(line_l.emergency_limit[t])
            try:
                keep = bool(filter_predicate("line", line_l, out_line, t, coeff, F_em))
            except Exception:
                keep = True
            if keep:
                kept.append(t)
        return kept

    def _kept_ts_gen(line_l, gen, coeff: float) -> Iterable[int]:
        if filter_predicate is None:
            return T_list
        kept = []
        for t in T_list:
            F_em = float(line_l.emergency_limit[t])
            try:
                keep = bool(filter_predicate("gen", line_l, gen, t, coeff, F_em))
            except Exception:
                keep = True
            if keep:
                kept.append(t)
        return kept

    n_cons_line = 0
    n_cons_gen = 0
    n_conts_used_line = 0
    n_conts_used_gen = 0
    n_skipped_line = 0
    n_skipped_gen = 0
    n_skipped_gen_zero = 0

    allowed_lines_count = (
        len(monitored_line_whitelist) if monitored_line_whitelist else len(lines)
    )

    for cont in contingencies:
        if cont.lines:
            n_conts_used_line += 1
            for out_line in cont.lines:
                mcol = out_line.index - 1
                col = lodf_csc.getcol(mcol)
                col_rows = col.indices.tolist()
                col_vals = col.data.tolist()

                for l_row, alpha_lm in zip(col_rows, col_vals):
                    if l_row == mcol or abs(alpha_lm) < _LODF_TOL:
                        continue
                    line_l = line_by_row.get(l_row)
                    if line_l is None:
                        continue
                    if not _mon_allowed(line_l.name):
                        n_skipped_line += 2 * len(T_list)
                        continue
                    if (
                        keep_line_pairs
                        and (line_l.name, out_line.name) not in keep_line_pairs
                    ):
                        n_skipped_line += 2 * len(T_list)
                        continue

                    coeff = float(alpha_lm)
                    kept_ts = list(_kept_ts_line(line_l, out_line, coeff))
                    if not kept_ts:
                        continue

                    model.addConstrs(
                        (
                            line_flow[line_l.name, t]
                            + coeff * line_flow[out_line.name, t]
                            <= float(line_l.emergency_limit[t])
                            + cont_over_pos[line_l.name, t]
                            for t in kept_ts
                        )
                    )
                    model.addConstrs(
                        (
                            -line_flow[line_l.name, t]
                            - coeff * line_flow[out_line.name, t]
                            <= float(line_l.emergency_limit[t])
                            + cont_over_neg[line_l.name, t]
                            for t in kept_ts
                        )
                    )
                    added = 2 * len(kept_ts)
                    _incr_line(line_l.name, added)
                    _incr_pair_line(line_l.name, out_line.name, added)
                    explicit_total += added
                    n_cons_line += added

        if getattr(cont, "units", None):
            n_conts_used_gen += 1
            for gen in cont.units:
                bus_1b = gen.bus.index
                if bus_1b == ref_1b or bus_1b not in col_by_bus_1b:
                    n_skipped_gen_zero += 2 * allowed_lines_count * len(T_list)
                    continue

                col = isf_csc.getcol(col_by_bus_1b[bus_1b])
                col_rows = col.indices.tolist()
                col_vals = col.data.tolist()
                isf_map = {r: v for r, v in zip(col_rows, col_vals)}

                for l_row, isf_lb in zip(col_rows, col_vals):
                    if abs(isf_lb) < _ISF_TOL:
                        continue
                    line_l = line_by_row.get(l_row)
                    if line_l is None:
                        continue
                    if not _mon_allowed(line_l.name):
                        n_skipped_gen += 2 * len(T_list)
                        continue

                    if keep_gen_pairs and (line_l.name, gen.name) not in keep_gen_pairs:
                        n_skipped_gen += 2 * len(T_list)
                        continue

                    coeff = float(isf_map.get(line_l.index - 1, 0.0))
                    if abs(coeff) < _ISF_TOL:
                        continue
                    kept_ts = list(_kept_ts_gen(line_l, gen, coeff))
                    if not kept_ts:
                        continue

                    model.addConstrs(
                        (
                            line_flow[line_l.name, t]
                            - coeff
                            * (
                                commit[gen.name, t] * float(gen.min_power[t])
                                + gp.quicksum(
                                    seg_power[gen.name, t, s]
                                    for s in range(
                                        len(gen.segments) if gen.segments else 0
                                    )
                                )
                            )
                            <= float(line_l.emergency_limit[t])
                            + cont_over_pos[line_l.name, t]
                            for t in kept_ts
                        )
                    )
                    model.addConstrs(
                        (
                            -line_flow[line_l.name, t]
                            + coeff
                            * (
                                commit[gen.name, t] * float(gen.min_power[t])
                                + gp.quicksum(
                                    seg_power[gen.name, t, s]
                                    for s in range(
                                        len(gen.segments) if gen.segments else 0
                                    )
                                )
                            )
                            <= float(line_l.emergency_limit[t])
                            + cont_over_neg[line_l.name, t]
                            for t in kept_ts
                        )
                    )
                    added = 2 * len(kept_ts)
                    _incr_pair_gen(line_l.name, gen.name, added)
                    explicit_total += added
                    n_cons_gen += added

    try:
        model._explicit_total_cont_constraints = explicit_total
    except Exception:
        pass

    logger.info(
        "Cons(C-120/C-121): line-out=%d (conts_used=%d, skipped=%d), gen-out=%d (conts_used=%d, skipped=%d, skipped_gen_zero=%d); lines=%d, T=%d",
        n_cons_line,
        n_conts_used_line,
        n_skipped_line,
        n_cons_gen,
        n_conts_used_gen,
        n_skipped_gen,
        n_skipped_gen_zero,
        len(lines),
        len(time_periods),
    )
    record_constraint_stat(model, "C-120_line_constraints", n_cons_line)
    record_constraint_stat(model, "C-121_gen_constraints", n_cons_gen)
    record_constraint_stat(model, "C-120_121_total", n_cons_line + n_cons_gen)
