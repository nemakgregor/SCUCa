import argparse
import csv
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Callable

import glob
import psutil
from gurobipy import GRB
from tqdm import tqdm

from src.data_preparation.params import DataParams
from src.data_preparation.read_data import read_benchmark
from src.ml_models.warm_start import WarmStartProvider, _hash01
from src.ml_models.redundant_constraints import RedundancyProvider as RCProvider
from src.optimization_model.SCUC_solver.scuc_model_builder import build_model
from src.optimization_model.SCUC_solver.solve_instances import (
    list_remote_instances,
    list_local_cached_instances,
)
from src.optimization_model.helpers.branching_hints import (
    apply_branching_hints_from_starts,
)
from src.optimization_model.helpers.lazy_contingency_cb import (
    attach_lazy_contingency_callback,
    optimize_with_lazy_callback,
    LazyContingencyConfig,
)
from src.optimization_model.helpers.save_json_solution import (
    save_solution_as_json,
    compute_output_path,
)
from src.optimization_model.helpers.verify_solution import (
    verify_solution,
    verify_solution_to_log,
)
from src.optimization_model.helpers.save_solution import save_solution_to_log
from src.optimization_model.helpers.run_utils import (
    allocate_run_id,
    make_log_filename,
)

# ML modules
from src.ml_models.commitment_hints import CommitmentHints
from src.ml_models.bandits import EpsilonGreedyTopK
from src.ml_models.lp_screening import LPScreener
from src.ml_models.st_reduction import STReductionProvider
from src.scuc_sr.analysis import radii_for_scenario
from src.scuc_sr.novel_methods import (
    ActiveSetConfig,
    RollingHorizonConfig,
    optimize_with_active_set,
    solve_rolling_horizon,
)

try:
    from src.ml_models.gnn_screening import GNNLineScreener
except Exception:
    GNNLineScreener = None

try:
    from src.ml_models.gru_warmstart import GRUDispatchWarmStart
except Exception:
    GRUDispatchWarmStart = None

logger = logging.getLogger(__name__)


def _configure_logging():
    """
    Make logs readable and avoid duplicate gurobipy messages.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s:%(name)s:%(message)s",
        )
    logging.getLogger("gurobipy").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def _case_tag(case_folder: str) -> str:
    cf = case_folder.strip().strip("/\\").replace("\\", "/")
    return "".join(ch if ch.isalnum() else "_" for ch in cf).strip("_").lower()


def _experiment_solution_base_dir(mode_label: str) -> Path:
    return Path("results") / "solutions" / _case_tag(mode_label)


def _list_case_instances(case_folder: str) -> List[str]:
    """
    List instances for the exact case_folder (avoid prefix collisions).
    A match is:
      - exactly case_folder
      - or starts with case_folder + '/'
    """
    cf = case_folder.rstrip("/")

    def _belongs(path: str) -> bool:
        return path == cf or path.startswith(cf + "/")

    items = list_remote_instances(
        include_filters=[cf], roots=["matpower", "test"], max_depth=4
    )
    items = [x for x in items if _belongs(x)]
    if not items:
        items = list_local_cached_instances(include_filters=[cf])
        items = [x for x in items if _belongs(x)]
    return sorted(set(items))


def _discover_cases_from_outputs() -> List[str]:
    """
    Discover case folders from existing JSON solutions under src/data/output.
    Returns a list like ["matpower/case14", "matpower/case118"].
    """
    base = DataParams._OUTPUT.resolve()
    cases: set = set()
    if base.exists():
        for p in base.rglob("*.json"):
            try:
                rel = p.resolve().relative_to(base).as_posix()
            except Exception:
                continue
            parts = rel.split("/")
            if len(parts) >= 3:
                cases.add("/".join(parts[:2]))
    return sorted(cases)


def _split_instances(
    names: List[str], train_ratio: float, val_ratio: float, seed: int
) -> Tuple[List[str], List[str], List[str]]:
    tr: List[str] = []
    va: List[str] = []
    te: List[str] = []
    for nm in sorted(names):
        r = _hash01(nm, seed)
        if r < train_ratio:
            tr.append(nm)
        elif r < train_ratio + val_ratio:
            va.append(nm)
        else:
            te.append(nm)
    return tr, va, te


def _is_json_output_present(instance_name: str) -> bool:
    return compute_output_path(instance_name).is_file()


class PeakMemSampler:
    """Background sampler to record process peak RSS during a block."""

    def __init__(self, interval_sec: float = 0.2):
        self.interval = interval_sec
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak_rss = 0

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread:
            try:
                self._thread.join(timeout=2.0)
            except Exception:
                pass

    def _run(self):
        proc = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss
                if rss > self._peak_rss:
                    self._peak_rss = rss
            except Exception:
                pass
            self._stop.wait(self.interval)

    @property
    def peak_gb(self) -> float:
        return float(self._peak_rss) / (1024.0**3)


@dataclass
class RunConfig:
    # Base mode
    mode: str  # RAW | WARM | WARM+HINTS | WARM+LAZY | WARM+PRUNE | WARM+LPSCREEN | WARM+PRUNE+LAZY | WARM+LPSCREEN+LAZY | WARM+SR+LAZY | ACTIVESET | ACTIVESET+LAZY | SHRINK+LAZY | STREDUCE | STREDUCE+LAZY
    tau: Optional[float] = None  # for WARM+PRUNE / WARM+LPSCREEN and lazy hybrids

    # Solver controls
    time_limit: int = 600
    mip_gap: float = 0.05

    # Lazy settings
    lazy_top_k: int = 0
    lazy_lodf_tol: float = 1e-4
    lazy_isf_tol: float = 1e-8
    lazy_viol_tol: float = 1e-6

    # ML feature toggles for this specific run
    use_commit_hints: bool = False
    commit_mode: str = "hint"
    commit_thr: float = 0.98
    commit_on_raw: bool = False

    use_gnn_screen: bool = False
    gnn_thr: float = 0.60

    use_gru_warm: bool = False

    use_adaptive_topk: bool = False
    topk_candidates: Optional[List[int]] = None
    topk_epsilon: float = 0.1

    # Monitored-set / SR settings
    sr_sigma: float = 0.3
    sr_l2_thr: float = 400.0
    sr_sigma_thr: float = 5.0

    # Castelli-inspired active-set settings
    active_set_batch: int = 2000
    active_set_max_rounds: int = 12
    active_set_cleanup: bool = True
    active_set_cleanup_tol: float = 1e-5

    # Castelli-inspired shrinking horizon
    shrink_window: int = 8
    shrink_overlap: int = 2

    # ST-reduction settings
    st_commit_fix_thr: float = 0.98
    st_line_keep_thr: float = 0.60

    # Logs
    save_human_logs: bool = False
    save_json_solution: bool = True
    save_json_to_canonical_output: bool = False


def _save_logs_if_requested(sc, model, enabled: bool) -> None:
    if not enabled:
        return
    run_id = allocate_run_id(sc.name or "scenario")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        sol_fname = make_log_filename("solution", sc.name, run_id, ts)
        save_solution_to_log(sc, model, filename=sol_fname)
    except Exception:
        pass
    try:
        ver_fname = make_log_filename("verify", sc.name, run_id, ts)
        verify_solution_to_log(sc, model, filename=ver_fname)
    except Exception:
        pass


def _build_contingency_filter(
    rc: Optional[RCProvider],
    sc,
    instance_name: str,
    enable: bool,
    thr_rel: float,
    use_train_db: bool,
) -> Optional[callable]:
    if not enable or rc is None:
        return None
    try:
        res = rc.make_filter_for_instance(
            sc,
            instance_name,
            thr_rel=thr_rel,
            use_train_index_only=use_train_db,
            exclude_self=True,
        )
        if res is None:
            return None
        predicate, stats = res
        tqdm.write(
            f"[prune] Neighbor {stats.get('neighbor')} (dist={stats.get('distance'):.3f}); pruning active."
        )
        return predicate
    except Exception:
        return None


def _allowed_lines_by_radius(
    sc,
    l2_thr: float = 400.0,
    sigma_thr: float = 5.0,
    balance: str = "agc",
    sigma_sr: float = 0.3,
) -> Set[str]:
    allowed: Set[str] = set()
    try:
        rd = radii_for_scenario(sc, t=0, balance=balance, sigma_sr=float(sigma_sr))
        l2_map = rd.get("l2", {})
        sig_map = rd.get("sigma", {})

        for ln in sc.lines or []:
            a, b = sorted((ln.source.name, ln.target.name))
            key = f"{a}-{b}"
            r_l2 = l2_map.get(key)
            r_sig = sig_map.get(key)

            keep = False
            if r_l2 is None and r_sig is None:
                keep = True
            else:
                if r_l2 is not None and r_l2 <= l2_thr:
                    keep = True
                if r_sig is not None and r_sig <= sigma_thr:
                    keep = True

            if keep:
                allowed.add(ln.name)
    except Exception:
        for ln in sc.lines or []:
            allowed.add(ln.name)
    return allowed


def _estimate_cont_counts(
    sc,
    filter_predicate: Optional[Callable],
    keep_masks: Optional[Dict[str, Set[Tuple[str, str]]]] = None,
    monitored_line_whitelist: Optional[Set[str]] = None,
) -> Tuple[int, int, int, int]:
    """
    Estimate total and kept contingency constraints count for (line outages, gen outages).
    Counts +/- pairs. Uses same tolerances as builder.
    """
    from src.optimization_model.solver.scuc.constraints.contingencies import (
        _LODF_TOL,
        _ISF_TOL,
    )

    lines = sc.lines or []
    contingencies = sc.contingencies or []
    lodf = sc.lodf.tocsc()
    isf = sc.isf.tocsc()

    T = sc.time
    buses = sc.buses
    ref_1b = getattr(sc, "ptdf_ref_bus_index", buses[0].index)
    non_ref_bus_indices = sorted([b.index for b in buses if b.index != ref_1b])
    col_by_bus_1b = {bus_1b: col for col, bus_1b in enumerate(non_ref_bus_indices)}
    line_by_row = {ln.index - 1: ln for ln in lines}

    total_line = 0
    kept_line = 0
    total_gen = 0
    kept_gen = 0

    keep_line_pairs = (keep_masks or {}).get("line")
    keep_gen_pairs = (keep_masks or {}).get("gen")

    def _mon_allowed(line_name: str) -> bool:
        if monitored_line_whitelist is None:
            return True
        return line_name in monitored_line_whitelist

    # Line outage
    for cont in contingencies:
        if not cont.lines:
            continue
        for out_line in cont.lines:
            mcol = out_line.index - 1
            col = lodf.getcol(mcol)
            for l_row, alpha in zip(col.indices.tolist(), col.data.tolist()):
                if l_row == mcol:
                    continue
                if abs(alpha) < _LODF_TOL:
                    continue
                line_l = line_by_row.get(l_row)
                if line_l is None:
                    continue
                total_line += 2 * T
                if not _mon_allowed(line_l.name):
                    continue
                if keep_line_pairs is not None and (
                    line_l.name,
                    out_line.name,
                ) not in keep_line_pairs:
                    continue
                if filter_predicate is None:
                    kept_line += 2 * T
                    continue
                for t in range(T):
                    F_em = float(line_l.emergency_limit[t])
                    keep = True
                    try:
                        keep = bool(
                            filter_predicate(
                                "line", line_l, out_line, t, float(alpha), F_em
                            )
                        )
                    except Exception:
                        keep = True
                    kept_line += 2 if keep else 0

    # Gen outage
    for cont in contingencies:
        if not getattr(cont, "units", None):
            continue
        for gen in cont.units:
            bidx = gen.bus.index
            if bidx == ref_1b or bidx not in col_by_bus_1b:
                # coeff zero => constraints still exist (coeff=0), +/- per line
                for line_l in lines:
                    total_gen += 2 * T
                    if not _mon_allowed(line_l.name):
                        continue
                    if keep_gen_pairs is not None and (
                        line_l.name,
                        gen.name,
                    ) not in keep_gen_pairs:
                        continue
                    if filter_predicate is None:
                        kept_gen += 2 * T
                        continue
                    for t in range(T):
                        F_em = float(line_l.emergency_limit[t])
                        keep = True
                        try:
                            keep = bool(
                                filter_predicate("gen", line_l, gen, t, 0.0, F_em)
                            )
                        except Exception:
                            keep = True
                        kept_gen += 2 if keep else 0
            else:
                col = isf.getcol(col_by_bus_1b[bidx])
                coeff_map = {
                    r: v for r, v in zip(col.indices.tolist(), col.data.tolist())
                }
                for line_l in lines:
                    coeff = float(coeff_map.get(line_l.index - 1, 0.0))
                    if abs(coeff) < _ISF_TOL:
                        continue
                    total_gen += 2 * T
                    if not _mon_allowed(line_l.name):
                        continue
                    if keep_gen_pairs is not None and (
                        line_l.name,
                        gen.name,
                    ) not in keep_gen_pairs:
                        continue
                    if filter_predicate is None:
                        kept_gen += 2 * T
                        continue
                    for t in range(T):
                        F_em = float(line_l.emergency_limit[t])
                        keep = True
                        try:
                            keep = bool(
                                filter_predicate("gen", line_l, gen, t, coeff, F_em)
                            )
                        except Exception:
                            keep = True
                        kept_gen += 2 if keep else 0

    return total_line + total_gen, kept_line + kept_gen, total_line, kept_line


def _status_str(code: int) -> str:
    return DataParams.SOLVER_STATUS_STR.get(code, f"STATUS_{code}")


def _build_mode_label(base: str, cfg: RunConfig) -> str:
    """
    Build a human-readable mode label with feature suffixes, e.g.:
      - RAW+GNN
      - WARM+HINTS+COMMIT
      - WARM+LAZY+K128
      - WARM+LAZY+BANDIT+COMMIT
      - WARM+PRUNE-0.50+GNN
      - WARM+PRUNE-0.50+LAZY
    """
    parts = [base]
    suppress_feature_suffix = base in {
        "ACTIVESET",
        "ACTIVESET+LAZY",
        "SHRINK+LAZY",
        "STREDUCE",
        "STREDUCE+LAZY",
    }
    # Tau for PRUNE / LPSCREEN
    if cfg.tau is not None:
        if base == "WARM+PRUNE":
            parts[-1] = f"WARM+PRUNE-{cfg.tau:.2f}"
        elif base == "WARM+LPSCREEN":
            parts[-1] = f"WARM+LPSCREEN-{cfg.tau:.2f}"
        elif base == "WARM+PRUNE+LAZY":
            parts[-1] = f"WARM+PRUNE-{cfg.tau:.2f}+LAZY"
        elif base == "WARM+LPSCREEN+LAZY":
            parts[-1] = f"WARM+LPSCREEN-{cfg.tau:.2f}+LAZY"
    # Feature suffixes
    if cfg.use_gnn_screen and "LAZY" not in base and not suppress_feature_suffix:
        parts.append("GNN")
    if cfg.use_commit_hints and not suppress_feature_suffix:
        parts.append("COMMIT")
    if cfg.use_gru_warm and (not suppress_feature_suffix or base.startswith("STREDUCE")):
        parts.append("GRU")
    if base == "WARM+LAZY":
        if cfg.use_adaptive_topk:
            parts.append("BANDIT")
        elif cfg.lazy_top_k and cfg.lazy_top_k > 0:
            parts.append(f"K{int(cfg.lazy_top_k)}")
    return "+".join(parts)


def _finalize_result_row(
    *,
    scenario,
    model,
    instance_name: str,
    case_folder: str,
    mode_label: str,
    mode_cfg: RunConfig,
    started_at: float,
    peak_mem_gb: float,
    screen_setup_sec: float,
    applied_starts: int,
    hints_applied: int,
    num_vars_root: int,
    num_constrs_root: int,
    constr_total,
    constr_kept,
    constr_ratio,
    monitored_line_count,
    active_set_report=None,
    rolling_report=None,
    st_profile=None,
    st_gru_applied: bool = False,
) -> Dict:
    saved_solution_path = None
    saved_output_scope = ""
    try:
        if bool(mode_cfg.save_json_solution):
            out_base_dir = (
                None
                if bool(mode_cfg.save_json_to_canonical_output)
                else _experiment_solution_base_dir(mode_label)
            )
            saved_solution_path = save_solution_as_json(
                scenario,
                model,
                instance_name=instance_name,
                out_base_dir=out_base_dir,
                extra_meta={
                    "mode": mode_label,
                    "case_folder": case_folder,
                    "artifact_scope": (
                        "canonical_train"
                        if bool(mode_cfg.save_json_to_canonical_output)
                        else "experiment_results"
                    ),
                },
            )
            saved_output_scope = (
                "canonical_train"
                if bool(mode_cfg.save_json_to_canonical_output)
                else "experiment_results"
            )
    except Exception:
        pass

    feasible_ok = None
    max_constraint_residual = None
    obj_inconsistency = None
    violations_str = None
    try:
        feasible_ok, checks, _ = verify_solution(scenario, model)
        vals = []
        for c in checks:
            try:
                if isinstance(c.idx, str) and c.idx.startswith("C-"):
                    vals.append(float(c.value))
            except Exception:
                pass
        max_constraint_residual = max(vals) if vals else 0.0
        for c in checks:
            if c.idx == "O-301":
                try:
                    obj_inconsistency = float(c.value)
                except Exception:
                    obj_inconsistency = None
                break
        bad_ids = []
        for c in checks:
            try:
                if isinstance(c.idx, str) and c.idx.startswith("C-"):
                    if float(c.value) > 1e-6:
                        bad_ids.append(c.idx.replace("-", ""))
            except Exception:
                pass
        violations_str = "OK" if not bad_ids else " ".join(sorted(set(bad_ids)))
    except Exception:
        feasible_ok = None
        max_constraint_residual = None
        obj_inconsistency = None
        violations_str = None

    _save_logs_if_requested(scenario, model, bool(mode_cfg.save_human_logs))

    status_code = int(getattr(model, "Status", -1))
    runtime = float(getattr(model, "Runtime", 0.0) or 0.0)
    try:
        mip_gap = float(model.MIPGap)
    except Exception:
        mip_gap = None
    try:
        obj_val = float(model.ObjVal)
    except Exception:
        obj_val = None
    try:
        obj_bound = float(model.ObjBound)
    except Exception:
        obj_bound = None
    try:
        nodes = float(getattr(model, "NodeCount", 0.0))
    except Exception:
        nodes = None

    try:
        model.update()
    except Exception:
        pass
    num_vars_final = int(getattr(model, "NumVars", 0))
    num_constrs_final = int(getattr(model, "NumConstrs", 0))

    explicit_added_cont = int(
        getattr(model, "_explicit_total_cont_constraints", 0) or 0
    )
    lazy_stats = getattr(model, "_lazy_contingency_stats", {}) or {}
    lazy_added_cont = int(lazy_stats.get("lazy_added", 0) or 0)
    method_exactness = "heuristic" if mode_cfg.mode == "SHRINK+LAZY" else "exact"
    constr_ratio_explicit = constr_ratio
    constr_realized_cont = None
    constr_ratio_realized = None
    try:
        if constr_total is not None:
            tot = int(constr_total)
            if tot > 0:
                kept_explicit = int(constr_kept or 0)
                realized = int(kept_explicit + lazy_added_cont)
                constr_realized_cont = realized
                constr_ratio_realized = float(realized) / float(tot)
    except Exception:
        constr_realized_cont = None
        constr_ratio_realized = None

    finished_at = time.time()

    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "instance_name": instance_name,
        "case_folder": case_folder,
        "mode": mode_label,
        "tau": "" if mode_cfg.tau is None else f"{mode_cfg.tau:.2f}",
        "status": _status_str(status_code),
        "status_code": status_code,
        "runtime_sec": runtime,
        "wall_sec": finished_at - started_at,
        "runtime_report_sec": finished_at - started_at,
        "mip_gap": "" if mip_gap is None else f"{mip_gap:.8f}",
        "obj_val": "" if obj_val is None else f"{obj_val:.6f}",
        "obj_bound": "" if obj_bound is None else f"{obj_bound:.6f}",
        "nodes": "" if nodes is None else f"{nodes:.0f}",
        "feasible_ok": "" if feasible_ok is None else ("OK" if feasible_ok else "FAIL"),
        "max_constraint_residual": ""
        if max_constraint_residual is None
        else f"{max_constraint_residual:.8e}",
        "objective_inconsistency": ""
        if obj_inconsistency is None
        else f"{obj_inconsistency:.8e}",
        "violations": "" if violations_str is None else violations_str,
        "num_vars_root": num_vars_root,
        "num_constrs_root": num_constrs_root,
        "num_vars_final": num_vars_final,
        "num_constrs_final": num_constrs_final,
        "peak_memory_gb": f"{peak_mem_gb:.3f}",
        "screen_setup_sec": f"{screen_setup_sec:.6f}",
        "warm_start_applied_vars": applied_starts,
        "branch_hints_applied": hints_applied,
        "constr_total_cont": "" if constr_total is None else int(constr_total),
        "constr_kept_cont": "" if constr_kept is None else int(constr_kept),
        # Backward-compatible alias: explicit screening ratio only.
        "constr_ratio_cont": ""
        if constr_ratio_explicit is None
        else f"{constr_ratio_explicit:.4f}",
        "constr_ratio_cont_explicit": ""
        if constr_ratio_explicit is None
        else f"{constr_ratio_explicit:.4f}",
        "constr_realized_cont": ""
        if constr_realized_cont is None
        else int(constr_realized_cont),
        "constr_ratio_cont_realized": ""
        if constr_ratio_realized is None
        else f"{constr_ratio_realized:.4f}",
        "screen_monitored_lines": ""
        if monitored_line_count is None
        else int(monitored_line_count),
        "explicit_added_cont": explicit_added_cont,
        "lazy_added_cont": lazy_added_cont,
        "active_set_iters": int(getattr(active_set_report, "iterations", 0) or 0),
        "active_set_added": int(
            getattr(active_set_report, "added_constraints", 0) or 0
        ),
        "active_set_dropped": int(
            getattr(active_set_report, "dropped_constraints", 0) or 0
        ),
        "shrink_window_count": int(getattr(rolling_report, "window_count", 0) or 0),
        "shrink_window_size": int(
            getattr(rolling_report, "window_size", mode_cfg.shrink_window)
            if rolling_report is not None
            else mode_cfg.shrink_window
        ),
        "shrink_overlap": int(
            getattr(rolling_report, "overlap", mode_cfg.shrink_overlap)
            if rolling_report is not None
            else mode_cfg.shrink_overlap
        ),
        "fixed_commit_vars": int(getattr(st_profile, "fixed_commit_vars", 0) or 0),
        "fixed_commit_on": int(getattr(st_profile, "fixed_commit_on", 0) or 0),
        "fixed_commit_off": int(getattr(st_profile, "fixed_commit_off", 0) or 0),
        "st_kept_line_pairs": int(getattr(st_profile, "kept_line_pairs", 0) or 0),
        "st_kept_gen_pairs": int(getattr(st_profile, "kept_gen_pairs", 0) or 0),
        "st_used_commit_model": int(
            1 if bool(getattr(st_profile, "used_commit_model", False)) else 0
        ),
        "st_used_gnn_model": int(
            1 if bool(getattr(st_profile, "used_gnn_model", False)) else 0
        ),
        "st_used_gru_model": int(1 if bool(st_gru_applied) else 0),
        "method_exactness": method_exactness,
        "saved_output_scope": saved_output_scope,
        "solution_json_path": "" if saved_solution_path is None else str(saved_solution_path),
    }

    try:
        model.dispose()
    except Exception:
        pass
    return row


def solve_one(
    instance_name: str,
    mode_cfg: RunConfig,
    wsp: Optional[WarmStartProvider],
    rc: Optional[RCProvider],
    use_train_db_for_rc: bool,
    args: argparse.Namespace,
) -> Dict:
    started_at = time.time()
    inst = read_benchmark(instance_name, quiet=True)
    sc = inst.deterministic
    case_folder = "/".join(instance_name.split("/")[:2])

    # Build final label now for logging
    mode_label = _build_mode_label(mode_cfg.mode, mode_cfg)
    tqdm.write(f"[run] {instance_name} | {mode_label}")
    logger.info("Solving %s with %s", instance_name, mode_label)

    pure_lazy_mode = mode_cfg.mode == "WARM+LAZY"
    prune_lazy_mode = mode_cfg.mode == "WARM+PRUNE+LAZY"
    lpscreen_lazy_mode = mode_cfg.mode == "WARM+LPSCREEN+LAZY"
    sr_lazy_mode = mode_cfg.mode == "WARM+SR+LAZY"
    active_set_mode = mode_cfg.mode == "ACTIVESET"
    active_set_lazy_mode = mode_cfg.mode == "ACTIVESET+LAZY"
    shrink_lazy_mode = mode_cfg.mode == "SHRINK+LAZY"
    st_mode = mode_cfg.mode == "STREDUCE"
    st_lazy_mode = mode_cfg.mode == "STREDUCE+LAZY"
    use_lazy = (
        pure_lazy_mode
        or prune_lazy_mode
        or lpscreen_lazy_mode
        or sr_lazy_mode
        or active_set_lazy_mode
        or shrink_lazy_mode
        or st_lazy_mode
    )
    omit_explicit_contingencies = (
        pure_lazy_mode
        or prune_lazy_mode
        or lpscreen_lazy_mode
        or active_set_mode
        or active_set_lazy_mode
        or shrink_lazy_mode
    )

    rc_keep_masks = None
    lp_keep_masks = None
    lp_stats = None
    st_keep_masks = None
    monitored_line_whitelist = None
    screen_monitored_count = None
    screen_setup_sec = 0.0
    active_set_report = None
    rolling_report = None
    st_profile = None
    st_gru_applied = False
    sc_model = sc

    # Redundancy pruning predicate / keep masks (PRUNE family)
    rc_pred = None
    if mode_cfg.mode.startswith("WARM+PRUNE"):
        tau = 0.5 if mode_cfg.tau is None else float(mode_cfg.tau)
        if prune_lazy_mode and rc is not None:
            t_rc = time.time()
            try:
                res = rc.make_masks_for_instance(
                    sc,
                    instance_name,
                    thr_rel=tau,
                    use_train_index_only=use_train_db_for_rc,
                    exclude_self=True,
                )
                if res:
                    rc_keep_masks = {"line": res[0][0], "gen": res[0][1]}
                    stats = res[1]
                    dist = stats.get("distance")
                    dist_str = f"{dist:.3f}" if dist is not None else "n/a"
                    tqdm.write(
                        f"[prune+lazy] Neighbor {stats.get('neighbor')} (dist={dist_str}); "
                        f"line_pairs={len(rc_keep_masks['line'])}, gen_pairs={len(rc_keep_masks['gen'])}"
                    )
            except Exception as e:
                logger.warning("PRUNE+LAZY mask build failed for %s: %s", instance_name, e)
                rc_keep_masks = None
            screen_setup_sec += time.time() - t_rc
        else:
            t_rc = time.time()
            rc_pred = _build_contingency_filter(
                rc,
                sc,
                instance_name,
                enable=True,
                thr_rel=tau,
                use_train_db=use_train_db_for_rc,
            )
            screen_setup_sec += time.time() - t_rc

    # LP relaxation screening (LPSCREEN; no training data needed)
    lp_pred = None
    if mode_cfg.mode.startswith("WARM+LPSCREEN"):
        tau_lp = 0.10 if mode_cfg.tau is None else float(mode_cfg.tau)
        try:
            lp_screener = LPScreener()
            if lpscreen_lazy_mode:
                lp_res = lp_screener.screen_masks(sc, tau=tau_lp, lp_time_limit=30.0)
                if lp_res is not None:
                    (keep_line, keep_gen), lp_stats = lp_res
                    lp_keep_masks = {"line": keep_line, "gen": keep_gen}
                    tqdm.write(
                        f"[lpscreen+lazy] LP solved in {lp_stats['lp_time']:.2f}s; "
                        f"kept_line_pairs={lp_stats['kept_line_pairs']}, kept_gen_pairs={lp_stats['kept_gen_pairs']}"
                    )
                else:
                    logger.warning("LP screening masks failed for %s", instance_name)
            else:
                lp_res = lp_screener.screen(sc, tau=tau_lp, lp_time_limit=30.0)
                if lp_res is not None:
                    lp_pred, lp_stats = lp_res
                    tqdm.write(
                        f"[lpscreen] LP solved in {lp_stats['lp_time']:.2f}s; "
                        f"line_pairs={lp_stats['total_line_pairs']}, gen_pairs={lp_stats['total_gen_pairs']}"
                    )
                else:
                    logger.warning("LP screening failed for %s", instance_name)
            if lp_stats is not None:
                screen_setup_sec += float(lp_stats.get("screen_time", 0.0) or 0.0)
        except Exception as e:
            logger.warning("LP screening error for %s: %s", instance_name, e)

    if sr_lazy_mode:
        t_sr = time.time()
        monitored_line_whitelist = _allowed_lines_by_radius(
            sc,
            l2_thr=float(mode_cfg.sr_l2_thr),
            sigma_thr=float(mode_cfg.sr_sigma_thr),
            sigma_sr=float(mode_cfg.sr_sigma),
        )
        screen_setup_sec += time.time() - t_sr
        screen_monitored_count = len(monitored_line_whitelist)
        tqdm.write(
            f"[sr+lazy] explicit monitored lines={len(monitored_line_whitelist)}/{len(sc.lines or [])}"
        )

    if st_mode or st_lazy_mode:
        stp = STReductionProvider(case_folder=case_folder)
        st_profile = stp.build_profile(
            sc,
            instance_name,
            commit_fix_thr=float(mode_cfg.st_commit_fix_thr),
            line_keep_thr=float(mode_cfg.st_line_keep_thr),
        )
        sc_model = st_profile.scenario
        screen_setup_sec += float(st_profile.setup_sec)
        screen_monitored_count = int(len(st_profile.monitored_lines))
        if (
            st_profile.keep_line_pairs is not None
            or st_profile.keep_gen_pairs is not None
        ):
            st_keep_masks = {
                "line": st_profile.keep_line_pairs or set(),
                "gen": st_profile.keep_gen_pairs or set(),
            }
        tqdm.write(
            f"[streduce] fixed_commit={st_profile.fixed_commit_vars} "
            f"critical_lines={len(st_profile.monitored_lines)}/{len(sc.lines or [])} "
            f"kept_pairs=({st_profile.kept_line_pairs},{st_profile.kept_gen_pairs})"
        )

    if shrink_lazy_mode:
        rh_cfg = RollingHorizonConfig(
            time_limit=int(mode_cfg.time_limit),
            mip_gap=float(mode_cfg.mip_gap),
            window_size=int(mode_cfg.shrink_window),
            overlap=int(mode_cfg.shrink_overlap),
            lodf_tol=float(mode_cfg.lazy_lodf_tol),
            isf_tol=float(mode_cfg.lazy_isf_tol),
            violation_tol=float(mode_cfg.lazy_viol_tol),
            lazy_top_k=int(mode_cfg.lazy_top_k),
            output_flag=1,
        )
        with PeakMemSampler(interval_sec=0.2) as mems:
            proxy_model, rolling_report = solve_rolling_horizon(sc, rh_cfg)
            peak_mem_gb = mems.peak_gb
        try:
            rolling_report.window_size = int(mode_cfg.shrink_window)
            rolling_report.overlap = int(mode_cfg.shrink_overlap)
        except Exception:
            pass
        return _finalize_result_row(
            scenario=sc,
            model=proxy_model,
            instance_name=instance_name,
            case_folder=case_folder,
            mode_label=mode_label,
            mode_cfg=mode_cfg,
            started_at=started_at,
            peak_mem_gb=peak_mem_gb,
            screen_setup_sec=screen_setup_sec,
            applied_starts=0,
            hints_applied=0,
            num_vars_root=int(getattr(rolling_report, "max_num_vars", 0) or 0),
            num_constrs_root=int(getattr(rolling_report, "max_num_constrs", 0) or 0),
            constr_total=None,
            constr_kept=None,
            constr_ratio=None,
            monitored_line_count=screen_monitored_count,
            active_set_report=None,
            rolling_report=rolling_report,
            st_profile=None,
            st_gru_applied=False,
        )

    # GNN screening — explicit models only
    gnn_pred = None
    if not use_lazy and not (st_mode or st_lazy_mode) and bool(mode_cfg.use_gnn_screen):
        if GNNLineScreener is None:
            logger.warning("GNN screening requested for %s but torch/gnn deps are unavailable.", instance_name)
        else:
            try:
                case_folder = "/".join(instance_name.split("/")[:2])
                gnn = GNNLineScreener(case_folder=case_folder)
                if gnn.ensure_trained():
                    gnn_pred = gnn.make_pruning_predicate(
                        sc, thr_pred=float(mode_cfg.gnn_thr)
                    )
            except Exception:
                gnn_pred = None

    # Combine predicates if multiple available (AND logic = safer)
    def _combine_pred(kind, line_l, out_obj, t, coeff, F_em):
        ok_rc = (
            True
            if rc_pred is None
            else bool(rc_pred(kind, line_l, out_obj, t, coeff, F_em))
        )
        ok_lp = (
            True
            if lp_pred is None
            else bool(lp_pred(kind, line_l, out_obj, t, coeff, F_em))
        )
        ok_gn = (
            True
            if gnn_pred is None
            else bool(gnn_pred(kind, line_l, out_obj, t, coeff, F_em))
        )
        return ok_rc and ok_lp and ok_gn

    # Final predicate for explicit runs
    cont_filter = None
    if mode_cfg.mode == "RAW":
        cont_filter = gnn_pred
    elif mode_cfg.mode == "WARM":
        cont_filter = gnn_pred
    elif mode_cfg.mode == "WARM+HINTS":
        cont_filter = gnn_pred
    elif mode_cfg.mode == "WARM+PRUNE":
        cont_filter = (
            _combine_pred if (rc_pred is not None or gnn_pred is not None) else None
        )
    elif mode_cfg.mode == "WARM+LPSCREEN":
        cont_filter = (
            _combine_pred if (lp_pred is not None or gnn_pred is not None) else None
        )

    # Build model with explicit contingencies, lazy-only contingencies, or a monitored-set hybrid.
    model = build_model(
        scenario=sc_model,
        contingency_filter=None
        if (omit_explicit_contingencies or sr_lazy_mode)
        else cont_filter,
        contingency_keep_masks=st_keep_masks,
        use_lazy_contingencies=bool(omit_explicit_contingencies),
        radius_line_whitelist=monitored_line_whitelist,
    )
    try:
        model.update()
    except Exception:
        pass

    # Root size
    num_vars_root = int(getattr(model, "NumVars", 0))
    num_constrs_root = int(getattr(model, "NumConstrs", 0))

    # Apply warm starts (WARM-family + STREDUCE)
    applied_starts = 0
    if mode_cfg.mode in (
        "WARM",
        "WARM+HINTS",
        "WARM+LAZY",
        "WARM+PRUNE",
        "WARM+LPSCREEN",
        "WARM+PRUNE+LAZY",
        "WARM+LPSCREEN+LAZY",
        "WARM+SR+LAZY",
        "STREDUCE",
        "STREDUCE+LAZY",
    ):
        if wsp is not None:
            try:
                wsp.generate_and_save_warm_start(
                    instance_name,
                    use_train_index_only=True,
                    exclude_self=True,
                    auto_fix=True,
                )
            except Exception as e:
                logger.warning("Warm-start generation failed for %s: %s", instance_name, e)
            try:
                applied_starts = wsp.apply_warm_start_to_model(
                    model, sc_model, instance_name, mode="repair"
                )
            except Exception as e:
                logger.warning("Warm-start apply failed for %s: %s", instance_name, e)
                applied_starts = 0

    # Commitment hints (if requested)
    if bool(mode_cfg.use_commit_hints) and (
        mode_cfg.mode
        in (
            "WARM",
            "WARM+HINTS",
            "WARM+LAZY",
            "WARM+PRUNE",
            "WARM+LPSCREEN",
            "WARM+PRUNE+LAZY",
            "WARM+LPSCREEN+LAZY",
            "WARM+SR+LAZY",
        )
        or (mode_cfg.mode == "RAW" and bool(mode_cfg.commit_on_raw))
    ):
        try:
            case_folder = "/".join(instance_name.split("/")[:2])
            ch = CommitmentHints(case_folder=case_folder)
            if ch.ensure_trained():
                _ = ch.apply_to_model(
                    model,
                    sc,
                    instance_name,
                    thr=float(mode_cfg.commit_thr),
                    mode=str(mode_cfg.commit_mode),
                )
        except Exception as e:
            logger.warning("Commitment hints failed for %s: %s", instance_name, e)

    # GRU-based dispatch warm start (if requested, and also for STREDUCE if available)
    if (bool(mode_cfg.use_gru_warm) or st_mode or st_lazy_mode) and mode_cfg.mode in (
        "WARM",
        "WARM+HINTS",
        "WARM+LAZY",
        "WARM+PRUNE",
        "WARM+LPSCREEN",
        "WARM+PRUNE+LAZY",
        "WARM+LPSCREEN+LAZY",
        "WARM+SR+LAZY",
        "STREDUCE",
        "STREDUCE+LAZY",
    ):
        if GRUDispatchWarmStart is None:
            if bool(mode_cfg.use_gru_warm):
                logger.warning(
                    "GRU warm-start requested for %s but torch deps are unavailable.",
                    instance_name,
                )
        else:
            try:
                case_folder = "/".join(instance_name.split("/")[:2])
                gw = GRUDispatchWarmStart(case_folder=case_folder)
                if gw.ensure_trained():
                    _ = gw.apply_to_model(model, sc_model)
                    if st_mode or st_lazy_mode:
                        st_gru_applied = True
            except Exception as e:
                logger.warning("GRU warm-start failed for %s: %s", instance_name, e)
    if (st_mode or st_lazy_mode) and st_gru_applied and ("+GRU" not in mode_label):
        mode_label = f"{mode_label}+GRU"

    # Branching hints from Start values (for HINTS/LAZY/PRUNE modes)
    hints_applied = 0
    if mode_cfg.mode in (
        "WARM+HINTS",
        "WARM+LAZY",
        "WARM+PRUNE",
        "WARM+LPSCREEN",
        "WARM+PRUNE+LAZY",
        "WARM+LPSCREEN+LAZY",
        "WARM+SR+LAZY",
        "STREDUCE",
        "STREDUCE+LAZY",
    ):
        try:
            hints_applied = apply_branching_hints_from_starts(model)
        except Exception as e:
            logger.warning("Branching hints failed for %s: %s", instance_name, e)
            hints_applied = 0

    # Attach lazy callback for LAZY; optionally bandit for adaptive top-k
    if use_lazy and sc_model.contingencies:
        lazy_keep_masks = None
        if rc_keep_masks is not None:
            lazy_keep_masks = rc_keep_masks
        elif lp_keep_masks is not None:
            lazy_keep_masks = lp_keep_masks
        cfg = LazyContingencyConfig(
            lodf_tol=mode_cfg.lazy_lodf_tol,
            isf_tol=mode_cfg.lazy_isf_tol,
            violation_tol=mode_cfg.lazy_viol_tol,
            add_top_k=int(mode_cfg.lazy_top_k)
            if mode_cfg.lazy_top_k and mode_cfg.lazy_top_k > 0
            else 0,
            keep_line_pairs=(
                lazy_keep_masks["line"]
                if lazy_keep_masks is not None and not st_lazy_mode
                else None
            ),
            keep_gen_pairs=(
                lazy_keep_masks["gen"]
                if lazy_keep_masks is not None and not st_lazy_mode
                else None
            ),
            verbose=True,
        )
        if bool(mode_cfg.use_adaptive_topk):
            try:
                pol = EpsilonGreedyTopK(
                    K_list=(mode_cfg.topk_candidates or [64, 128, 256]),
                    epsilon=float(mode_cfg.topk_epsilon),
                )
                cfg.topk_policy = pol
            except Exception:
                pass
        attach_lazy_contingency_callback(model, sc_model, cfg)

    # Solver params
    try:
        model.Params.OutputFlag = 1
        model.Params.MIPGap = mode_cfg.mip_gap
        model.Params.TimeLimit = mode_cfg.time_limit
        model.Params.NumericFocus = 1
        if use_lazy:
            model.Params.LazyConstraints = 1
    except Exception:
        pass

    # Optimize with memory sampler
    with PeakMemSampler(interval_sec=0.2) as mems:
        if active_set_mode or active_set_lazy_mode:
            as_cfg = ActiveSetConfig(
                time_limit=int(mode_cfg.time_limit),
                mip_gap=float(mode_cfg.mip_gap),
                lodf_tol=float(mode_cfg.lazy_lodf_tol),
                isf_tol=float(mode_cfg.lazy_isf_tol),
                violation_tol=float(mode_cfg.lazy_viol_tol),
                batch_size=int(mode_cfg.active_set_batch),
                max_rounds=int(mode_cfg.active_set_max_rounds),
                cleanup_inactive=bool(mode_cfg.active_set_cleanup),
                cleanup_tol=float(mode_cfg.active_set_cleanup_tol),
                output_flag=1,
            )
            active_set_report = optimize_with_active_set(model, sc_model, as_cfg)
        else:
            optimize_with_lazy_callback(model)
        peak_mem_gb = mems.peak_gb

    # Constraint ratio for screened / monitored-set explicit contingency burden.
    constr_total = None
    constr_kept = None
    constr_ratio = None
    if mode_cfg.mode in (
        "WARM+PRUNE",
        "WARM+LPSCREEN",
        "WARM+PRUNE+LAZY",
        "WARM+LPSCREEN+LAZY",
        "WARM+SR+LAZY",
        "STREDUCE",
        "STREDUCE+LAZY",
    ):
        estimate_pred = (
            cont_filter if mode_cfg.mode in ("WARM+PRUNE", "WARM+LPSCREEN") else None
        )
        if prune_lazy_mode:
            estimate_masks = rc_keep_masks
        elif lpscreen_lazy_mode:
            estimate_masks = lp_keep_masks
        elif st_mode or st_lazy_mode:
            estimate_masks = st_keep_masks
        else:
            estimate_masks = None
        tot, kept, _, _ = _estimate_cont_counts(
            sc,
            estimate_pred,
            keep_masks=estimate_masks,
            monitored_line_whitelist=monitored_line_whitelist,
        )
        constr_total = int(tot)
        constr_kept = int(kept)
        constr_ratio = float(kept) / float(tot) if tot > 0 else 1.0

    return _finalize_result_row(
        scenario=sc,
        model=model,
        instance_name=instance_name,
        case_folder=case_folder,
        mode_label=mode_label,
        mode_cfg=mode_cfg,
        started_at=started_at,
        peak_mem_gb=peak_mem_gb,
        screen_setup_sec=screen_setup_sec,
        applied_starts=applied_starts,
        hints_applied=hints_applied,
        num_vars_root=num_vars_root,
        num_constrs_root=num_constrs_root,
        constr_total=constr_total,
        constr_kept=constr_kept,
        constr_ratio=constr_ratio,
        monitored_line_count=screen_monitored_count,
        active_set_report=active_set_report,
        rolling_report=rolling_report,
        st_profile=st_profile,
        st_gru_applied=st_gru_applied,
    )


def _prepare_csv_log(case_folder: str) -> Path:
    out_dir = Path("results") / "raw_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"exp_{_case_tag(case_folder)}_{ts}.csv"
    header = [
        "timestamp",
        "instance_name",
        "case_folder",
        "mode",
        "tau",
        "status",
        "status_code",
        "runtime_sec",
        "wall_sec",
        "runtime_report_sec",
        "mip_gap",
        "obj_val",
        "obj_bound",
        "nodes",
        "feasible_ok",
        "max_constraint_residual",
        "objective_inconsistency",
        "violations",
        "num_vars_root",
        "num_constrs_root",
        "num_vars_final",
        "num_constrs_final",
        "peak_memory_gb",
        "screen_setup_sec",
        "warm_start_applied_vars",
        "branch_hints_applied",
        "constr_total_cont",
        "constr_kept_cont",
        "constr_ratio_cont",
        "constr_ratio_cont_explicit",
        "constr_realized_cont",
        "constr_ratio_cont_realized",
        "screen_monitored_lines",
        "explicit_added_cont",
        "lazy_added_cont",
        "active_set_iters",
        "active_set_added",
        "active_set_dropped",
        "shrink_window_count",
        "shrink_window_size",
        "shrink_overlap",
        "fixed_commit_vars",
        "fixed_commit_on",
        "fixed_commit_off",
        "st_kept_line_pairs",
        "st_kept_gen_pairs",
        "st_used_commit_model",
        "st_used_gnn_model",
        "st_used_gru_model",
        "method_exactness",
        "saved_output_scope",
        "solution_json_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(header)
    return path


def _append_csv(path: Path, row: Dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                row.get(k, "")
                for k in [
                    "timestamp",
                    "instance_name",
                    "case_folder",
                    "mode",
                    "tau",
                    "status",
                    "status_code",
                    "runtime_sec",
                    "wall_sec",
                    "runtime_report_sec",
                    "mip_gap",
                    "obj_val",
                    "obj_bound",
                    "nodes",
                    "feasible_ok",
                    "max_constraint_residual",
                    "objective_inconsistency",
                    "violations",
                    "num_vars_root",
                    "num_constrs_root",
                    "num_vars_final",
                    "num_constrs_final",
                    "peak_memory_gb",
                    "screen_setup_sec",
                    "warm_start_applied_vars",
                    "branch_hints_applied",
                    "constr_total_cont",
                    "constr_kept_cont",
                    "constr_ratio_cont",
                    "constr_ratio_cont_explicit",
                    "constr_realized_cont",
                    "constr_ratio_cont_realized",
                    "screen_monitored_lines",
                    "explicit_added_cont",
                    "lazy_added_cont",
                    "active_set_iters",
                    "active_set_added",
                    "active_set_dropped",
                    "shrink_window_count",
                    "shrink_window_size",
                    "shrink_overlap",
                    "fixed_commit_vars",
                    "fixed_commit_on",
                    "fixed_commit_off",
                    "st_kept_line_pairs",
                    "st_kept_gen_pairs",
                    "st_used_commit_model",
                    "st_used_gnn_model",
                    "st_used_gru_model",
                    "method_exactness",
                    "saved_output_scope",
                    "solution_json_path",
                ]
            ]
        )


_SMALL_CASES = {"case14", "case30", "case57", "case89pegase"}


def _time_limit_for_case(case_folder: str, default_tl: int) -> int:
    """
    Per the paper: 60s for small cases (14, 30, 57, 89pegase), 600s for larger.
    If a user explicitly overrides --time-limit, use that instead.
    """
    tag = case_folder.strip("/").split("/")[-1].lower()
    if tag in _SMALL_CASES:
        return 60
    return 600


def _auto_mode_list(args: argparse.Namespace, case_folder: str = "") -> List[RunConfig]:
    """
    Systematic factorial experiment design.

    Three orthogonal axes:
      1. Constraint management: EXPLICIT | PRUNE-τ | LPSCREEN-τ | LAZY | LAZY+BANDIT
      2. Search guidance:       none | WARM | WARM+HINTS | WARM+COMMIT | WARM+GRU
      3. Screening overlay:     none | GNN

    We test key combinations (not full factorial — would be ~200 runs per instance).
    """
    TL = _time_limit_for_case(case_folder, args.time_limit)
    MG = args.mip_gap
    Kcands = args.adv_topk_candidates or [64, 128, 256]
    taus = args.taus or [0.1, 0.2, 0.3, 0.5, 0.8]
    modes: List[RunConfig] = []

    # ── Group 1: Baselines (no constraint reduction) ──
    # RAW = full explicit N-1, no ML
    modes.append(RunConfig(mode="RAW", time_limit=TL, mip_gap=MG))
    # RAW + ML search guidance (no constraint reduction)
    modes.append(RunConfig(mode="RAW", time_limit=TL, mip_gap=MG,
                           use_commit_hints=True, commit_on_raw=True))  # RAW+COMMIT
    modes.append(RunConfig(mode="RAW", time_limit=TL, mip_gap=MG,
                           use_gnn_screen=True))  # RAW+GNN

    # ── Group 2: Warm start only (no constraint reduction) ──
    modes.append(RunConfig(mode="WARM", time_limit=TL, mip_gap=MG))
    modes.append(RunConfig(mode="WARM+HINTS", time_limit=TL, mip_gap=MG))
    modes.append(RunConfig(mode="WARM+HINTS", time_limit=TL, mip_gap=MG,
                           use_commit_hints=True))  # +COMMIT
    modes.append(RunConfig(mode="WARM+HINTS", time_limit=TL, mip_gap=MG,
                           use_gru_warm=True))  # +GRU
    modes.append(RunConfig(mode="WARM+HINTS", time_limit=TL, mip_gap=MG,
                           use_commit_hints=True, use_gru_warm=True))  # +COMMIT+GRU

    # ── Group 3: Lazy N-1 enforcement ──
    modes.append(RunConfig(mode="WARM+LAZY", time_limit=TL, mip_gap=MG,
                           lazy_top_k=0,
                           lazy_lodf_tol=float(args.lazy_lodf_tol),
                           lazy_isf_tol=float(args.lazy_isf_tol),
                           lazy_viol_tol=float(args.lazy_viol_tol)))  # WARM+LAZY
    modes.append(RunConfig(mode="WARM+LAZY", time_limit=TL, mip_gap=MG,
                           use_adaptive_topk=True, topk_candidates=Kcands,
                           topk_epsilon=args.adv_topk_epsilon,
                           lazy_lodf_tol=float(args.lazy_lodf_tol),
                           lazy_isf_tol=float(args.lazy_isf_tol),
                           lazy_viol_tol=float(args.lazy_viol_tol)))  # +BANDIT
    modes.append(RunConfig(mode="WARM+LAZY", time_limit=TL, mip_gap=MG,
                           lazy_top_k=0, use_commit_hints=True,
                           lazy_lodf_tol=float(args.lazy_lodf_tol),
                           lazy_isf_tol=float(args.lazy_isf_tol),
                           lazy_viol_tol=float(args.lazy_viol_tol)))  # +COMMIT
    modes.append(RunConfig(mode="WARM+LAZY", time_limit=TL, mip_gap=MG,
                           lazy_top_k=0, use_gru_warm=True,
                           lazy_lodf_tol=float(args.lazy_lodf_tol),
                           lazy_isf_tol=float(args.lazy_isf_tol),
                           lazy_viol_tol=float(args.lazy_viol_tol)))  # +GRU

    # ── Group 4: kNN constraint screening (PRUNE) ──
    for tau in taus:
        modes.append(RunConfig(mode="WARM+PRUNE", tau=float(tau),
                               time_limit=TL, mip_gap=MG))
        modes.append(RunConfig(mode="WARM+PRUNE", tau=float(tau),
                               time_limit=TL, mip_gap=MG,
                               use_gnn_screen=True))  # +GNN

    # ── Group 5: LP Relaxation screening (LPSCREEN) — novel, no training data ──
    for tau in taus:
        modes.append(RunConfig(mode="WARM+LPSCREEN", tau=float(tau),
                               time_limit=TL, mip_gap=MG))
        modes.append(RunConfig(mode="WARM+LPSCREEN+LAZY", tau=float(tau),
                               time_limit=TL, mip_gap=MG,
                               lazy_top_k=0,
                               lazy_lodf_tol=float(args.lazy_lodf_tol),
                               lazy_isf_tol=float(args.lazy_isf_tol),
                               lazy_viol_tol=float(args.lazy_viol_tol)))

    for tau in taus:
        modes.append(RunConfig(mode="WARM+PRUNE+LAZY", tau=float(tau),
                               time_limit=TL, mip_gap=MG,
                               lazy_top_k=0,
                               lazy_lodf_tol=float(args.lazy_lodf_tol),
                               lazy_isf_tol=float(args.lazy_isf_tol),
                               lazy_viol_tol=float(args.lazy_viol_tol)))

    modes.append(RunConfig(mode="WARM+SR+LAZY",
                           time_limit=TL, mip_gap=MG,
                           lazy_top_k=0,
                           lazy_lodf_tol=float(args.lazy_lodf_tol),
                           lazy_isf_tol=float(args.lazy_isf_tol),
                           lazy_viol_tol=float(args.lazy_viol_tol),
                           sr_sigma=float(args.sr_sigma),
                           sr_l2_thr=float(args.sr_l2_thr),
                           sr_sigma_thr=float(args.sr_sigma_thr)))

    modes.append(RunConfig(mode="ACTIVESET",
                           time_limit=TL, mip_gap=MG,
                           active_set_batch=int(args.active_set_batch),
                           active_set_max_rounds=int(args.active_set_max_rounds),
                           active_set_cleanup_tol=float(args.active_set_cleanup_tol)))
    modes.append(RunConfig(mode="ACTIVESET+LAZY",
                           time_limit=TL, mip_gap=MG,
                           lazy_lodf_tol=float(args.lazy_lodf_tol),
                           lazy_isf_tol=float(args.lazy_isf_tol),
                           lazy_viol_tol=float(args.lazy_viol_tol),
                           active_set_batch=int(args.active_set_batch),
                           active_set_max_rounds=int(args.active_set_max_rounds),
                           active_set_cleanup_tol=float(args.active_set_cleanup_tol)))
    modes.append(RunConfig(mode="SHRINK+LAZY",
                           time_limit=TL, mip_gap=MG,
                           lazy_top_k=0,
                           lazy_lodf_tol=float(args.lazy_lodf_tol),
                           lazy_isf_tol=float(args.lazy_isf_tol),
                           lazy_viol_tol=float(args.lazy_viol_tol),
                           shrink_window=int(args.shrink_window),
                           shrink_overlap=int(args.shrink_overlap)))
    modes.append(RunConfig(mode="STREDUCE",
                           time_limit=TL, mip_gap=MG,
                           st_commit_fix_thr=float(args.st_commit_fix_thr),
                           st_line_keep_thr=float(args.st_line_keep_thr)))
    modes.append(RunConfig(mode="STREDUCE+LAZY",
                           time_limit=TL, mip_gap=MG,
                           lazy_top_k=0,
                           lazy_lodf_tol=float(args.lazy_lodf_tol),
                           lazy_isf_tol=float(args.lazy_isf_tol),
                           lazy_viol_tol=float(args.lazy_viol_tol),
                           st_commit_fix_thr=float(args.st_commit_fix_thr),
                           st_line_keep_thr=float(args.st_line_keep_thr)))

    return modes


def _load_skip_solved_keys(
    logs: List[Path],
    require_ok: bool = False,
    strict_status: bool = False,
) -> Tuple[Set[Tuple[str, str]], Set[str]]:
    """
    Read (instance_name, mode) pairs from one or more CSV logs to skip in future runs.
    Returns:
      - keys_mode: set of (instance_name, mode_label)
      - keys_inst: set of instance_name (for per-instance skipping)
    A row is considered 'solved' if:
      - strict_status=True  -> status == OPTIMAL
      - strict_status=False -> status in {OPTIMAL, SUBOPTIMAL, TIME_LIMIT}
    Additionally, if require_ok=True we also require violations == 'OK' (or feasible_ok == 'OK' if violations missing).
    """
    ok_status = {"OPTIMAL"} if strict_status else {"OPTIMAL", "SUBOPTIMAL", "TIME_LIMIT"}
    keys_mode: Set[Tuple[str, str]] = set()
    keys_inst: Set[str] = set()
    for p in logs:
        if not p or not Path(p).is_file():
            continue
        try:
            with open(p, "r", encoding="utf-8") as fh:
                rd = csv.DictReader(fh)
                for row in rd:
                    inst = (row.get("instance_name") or "").strip()
                    mode = (row.get("mode") or "").strip()
                    status = (row.get("status") or "").strip().upper()
                    if not inst or not mode:
                        continue
                    if status not in ok_status:
                        continue
                    if require_ok:
                        vio = (row.get("violations") or "").strip()
                        feas = (row.get("feasible_ok") or "").strip()
                        if not (vio == "OK" or feas == "OK"):
                            continue
                    keys_mode.add((inst, mode))
                    keys_inst.add(inst)
        except Exception:
            continue
    return keys_mode, keys_inst


def _collect_skip_logs(paths_from_cli: List[str]) -> List[Path]:
    if paths_from_cli:
        return [Path(p) for p in paths_from_cli]
    # default: scan results/raw_logs/*.csv
    return [Path(p) for p in glob.glob(str(Path("results") / "raw_logs" / "*.csv"))]


def main():
    _configure_logging()

    ap = argparse.ArgumentParser(
        description="Run SCUC+ML experiments across modes and cases."
    )
    ap.add_argument(
        "--cases",
        nargs="*",
        default=[
            "matpower/case14",
            "matpower/case30",
            "matpower/case57",
            "matpower/case89pegase",
            "matpower/case118",
            "matpower/case300",
        ],
        help="Case folders",
    )
    ap.add_argument(
        "--time-limit", type=int, default=400, help="Time limit per run [s]"
    )
    ap.add_argument("--mip-gap", type=float, default=0.05, help="Relative MIP gap")
    ap.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio")
    ap.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio")
    ap.add_argument("--seed", type=int, default=42, help="Split seed")
    ap.add_argument(
        "--limit-test", type=int, default=0, help="Limit number of TEST instances"
    )
    ap.add_argument(
        "--skip-train-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip solving TRAIN if JSON exists (use --no-skip-train-existing to force recompute).",
    )
    ap.add_argument(
        "--save-human-logs",
        action="store_true",
        default=False,
        help="Save human-readable logs (solution/verify)",
    )
    ap.add_argument(
        "--modes",
        nargs="*",
        default=[
            "RAW",
            "WARM",
            "WARM+HINTS",
            "WARM+LAZY",
            "WARM+PRUNE",
            "WARM+LPSCREEN",
            "WARM+PRUNE+LAZY",
            "WARM+LPSCREEN+LAZY",
            "WARM+SR+LAZY",
            "ACTIVESET",
            "ACTIVESET+LAZY",
            "SHRINK+LAZY",
            "STREDUCE",
            "STREDUCE+LAZY",
        ],
        help="Which modes to run",
    )
    ap.add_argument(
        "--taus",
        nargs="*",
        type=float,
        default=[0.1, 0.2, 0.3, 0.5, 0.8],
        help="Tau sweep for PRUNE/LPSCREEN families (including lazy hybrids)",
    )
    ap.add_argument(
        "--lazy-top-k",
        type=int,
        default=0,
        help="Lazy: add top-K violated constraints (0=all)",
    )
    ap.add_argument(
        "--lazy-viol-tol",
        type=float,
        default=1e-6,
        help="Lazy: violation tolerance [MW]",
    )
    ap.add_argument(
        "--lazy-lodf-tol",
        type=float,
        default=1e-4,
        help="Lazy: |LODF| ignore threshold",
    )
    ap.add_argument(
        "--lazy-isf-tol", type=float, default=1e-8, help="Lazy: |ISF| ignore threshold"
    )
    ap.add_argument(
        "--sr-sigma",
        type=float,
        default=0.3,
        help="SR+LAZY: sigma parameter for scenario radii",
    )
    ap.add_argument(
        "--sr-l2-thr",
        type=float,
        default=400.0,
        help="SR+LAZY: L2 radius threshold for explicit monitored lines",
    )
    ap.add_argument(
        "--sr-sigma-thr",
        type=float,
        default=5.0,
        help="SR+LAZY: sigma radius threshold for explicit monitored lines",
    )
    ap.add_argument(
        "--active-set-batch",
        type=int,
        default=2000,
        help="ACTIVESET: max violated contingency inequalities added per outer iteration",
    )
    ap.add_argument(
        "--active-set-max-rounds",
        type=int,
        default=12,
        help="ACTIVESET: max outer iterations",
    )
    ap.add_argument(
        "--active-set-cleanup-tol",
        type=float,
        default=1e-5,
        help="ACTIVESET: drop inactive explicit cuts if slack exceeds this tolerance",
    )
    ap.add_argument(
        "--shrink-window",
        type=int,
        default=8,
        help="SHRINK+LAZY: rolling window size in time periods",
    )
    ap.add_argument(
        "--shrink-overlap",
        type=int,
        default=2,
        help="SHRINK+LAZY: overlap between consecutive windows",
    )
    ap.add_argument(
        "--st-commit-fix-thr",
        type=float,
        default=0.98,
        help="STREDUCE: commitment fixing confidence threshold",
    )
    ap.add_argument(
        "--st-line-keep-thr",
        type=float,
        default=0.60,
        help="STREDUCE: line criticality threshold for GNN screening",
    )
    ap.add_argument(
        "--rc-max-items",
        type=int,
        default=0,
        help="Redundancy index: limit number of outputs when building (0=all)",
    )
    ap.add_argument(
        "--rc-restrict-train",
        action="store_true",
        default=True,
        help="Redundancy index: restrict to TRAIN split",
    )

    # Train resolving strategy
    ap.add_argument(
        "--train-use-existing-only",
        action="store_true",
        default=False,
        help="Do NOT solve TRAIN stage; use existing outputs only.",
    )
    ap.add_argument(
        "--train-resolve-missing",
        action="store_true",
        default=False,
        help="Solve missing TRAIN outputs with RAW (overrides above).",
    )

    # ML accelerators (manual mode control; kept for backward compatibility)
    ap.add_argument(
        "--adv-commit-hints",
        action="store_true",
        default=False,
        help="Apply commitment hints",
    )
    ap.add_argument(
        "--adv-commit-mode",
        choices=["hint", "start", "fix"],
        default="hint",
        help="Commit hints apply mode",
    )
    ap.add_argument(
        "--adv-commit-thr",
        type=float,
        default=0.8,
        help="Fixing threshold (if mode=fix)",
    )
    ap.add_argument(
        "--adv-commit-on-raw",
        action="store_true",
        default=False,
        help="Also apply commitment hints to RAW",
    )
    ap.add_argument(
        "--adv-gnn-screen",
        action="store_true",
        default=False,
        help="Use GNN screening for explicit models",
    )
    ap.add_argument(
        "--adv-gnn-thr", type=float, default=0.60, help="GNN keep threshold"
    )
    ap.add_argument(
        "--adv-gru-warm",
        action="store_true",
        default=False,
        help="Apply GRU-based dispatch warm start",
    )
    ap.add_argument(
        "--adv-adaptive-topk",
        action="store_true",
        default=False,
        help="Lazy: use bandit for adaptive top-k",
    )
    ap.add_argument(
        "--adv-topk-candidates",
        nargs="*",
        type=int,
        default=[64, 128, 256],
        help="Candidate K values for bandit",
    )
    ap.add_argument(
        "--adv-topk-epsilon", type=float, default=0.1, help="Bandit epsilon"
    )

    # One-click autopilot
    ap.add_argument(
        "--run-experiments",
        dest="run_experiments",
        action="store_true",
        help="Run a comprehensive set of ML combinations automatically",
    )
    ap.add_argument(
        "--from-scratch",
        action="store_true",
        default=False,
        help="Force full rerun: recompute TRAIN RAW outputs even if canonical JSON already exists.",
    )

    # NEW: skip previously solved runs from existing log CSVs
    ap.add_argument(
        "--skip-solved",
        action="store_true",
        default=False,
        help="Skip TEST runs for (instance,mode) pairs already present in results logs. "
             "Use --skip-solved-logs to point at specific CSV(s). If omitted, scans results/raw_logs/*.csv.",
    )
    ap.add_argument(
        "--skip-solved-logs",
        nargs="*",
        default=[],
        help="Paths to existing results CSV logs to consider for skipping (default: scan results/raw_logs/*.csv).",
    )
    ap.add_argument(
        "--skip-solved-require-ok",
        action="store_true",
        default=False,
        help="Consider 'solved' only if verification reported violations == 'OK' (or feasible_ok == 'OK').",
    )
    ap.add_argument(
        "--skip-solved-strict-status",
        action="store_true",
        default=False,
        help="Consider 'solved' only if status == OPTIMAL (default: OPTIMAL/SUBOPTIMAL/TIME_LIMIT).",
    )
    ap.add_argument(
        "--skip-solved-per-instance",
        action="store_true",
        default=False,
        help="Skip all modes for an instance if any row for that instance matches the solved criteria in the logs.",
    )

    args = ap.parse_args()

    # Autopilot: choose cases automatically and enable all ML modules, solve missing TRAIN
    auto_mode = bool(args.run_experiments)
    if auto_mode:
        # Respect explicitly supplied/default --cases; auto-discover only when empty.
        if not args.cases:
            auto_cases = _discover_cases_from_outputs()
            if not auto_cases:
                auto_cases = ["matpower/case300"]
            args.cases = auto_cases
            tqdm.write(f"[auto] Cases discovered from outputs: {', '.join(args.cases)}")
        else:
            tqdm.write(f"[auto] Using configured cases: {', '.join(args.cases)}")
        args.train_resolve_missing = True
        if not args.from_scratch:
            args.skip_train_existing = True
        args.adv_commit_hints = True
        args.adv_gru_warm = True
        args.adv_gnn_screen = True
        args.adv_adaptive_topk = True

    if args.from_scratch:
        args.skip_train_existing = False
        tqdm.write("[from-scratch] TRAIN RAW recompute is enabled (existing canonical JSON will be overwritten).")

    # Resolve TRAIN strategy selection
    resolve_train = True
    if args.train_use_existing_only and not args.train_resolve_missing:
        resolve_train = False

    # Prepare skip-solved registry (if requested)
    skip_keys_mode: Set[Tuple[str, str]] = set()
    skip_keys_inst: Set[str] = set()
    if args.skip_solved:
        log_paths = _collect_skip_logs(args.skip_solved_logs)
        skip_keys_mode, skip_keys_inst = _load_skip_solved_keys(
            log_paths,
            require_ok=bool(args.skip_solved_require_ok),
            strict_status=bool(args.skip_solved_strict_status),
        )
        tqdm.write(
            f"[skip-solved] Loaded {len(skip_keys_mode)} (instance,mode) pairs and {len(skip_keys_inst)} instances from {len(log_paths)} log file(s)."
        )

    for case_folder in args.cases:
        all_instances = _list_case_instances(case_folder)
        if not all_instances:
            tqdm.write(f"[{case_folder}] no instances found.")
            continue

        train, val, test = _split_instances(
            all_instances, args.train_ratio, args.val_ratio, args.seed
        )
        if args.limit_test and args.limit_test > 0:
            test = test[: args.limit_test]

        tqdm.write(
            f"[{case_folder}] total={len(all_instances)} | train={len(train)} val={len(val)} test={len(test)}"
        )

        # Stage A: Solve TRAIN (RAW explicit) if needed
        if train and resolve_train:
            tqdm.write(
                f"[{case_folder}] Stage A: solving TRAIN missing outputs with RAW explicit ..."
            )
            train_to_run = [
                nm
                for nm in train
                if not (args.skip_train_existing and _is_json_output_present(nm))
            ]
            if not train_to_run:
                tqdm.write(f"[{case_folder}] TRAIN set already solved (RAW); skipping.")
            else:
                for nm in tqdm(
                    train_to_run, desc=f"{case_folder} TRAIN RAW", unit="inst"
                ):
                    case_tl = _time_limit_for_case(case_folder, args.time_limit) if auto_mode else args.time_limit
                    cfg = RunConfig(
                        mode="RAW",
                        time_limit=case_tl,
                        mip_gap=args.mip_gap,
                        save_human_logs=False,
                        save_json_to_canonical_output=True,
                        lazy_top_k=args.lazy_top_k,
                        lazy_lodf_tol=args.lazy_lodf_tol,
                        lazy_isf_tol=args.lazy_isf_tol,
                        lazy_viol_tol=args.lazy_viol_tol,
                    )
                    _ = solve_one(
                        nm, cfg, wsp=None, rc=None, use_train_db_for_rc=True, args=args
                    )
        else:
            tqdm.write(
                f"[{case_folder}] Stage A: skipping TRAIN solve (use-existing-only={args.train_use_existing_only})."
            )

        # Stage B: Pretrain warm-start index
        wsp = WarmStartProvider(
            case_folder=case_folder,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            split_seed=args.seed,
        )
        wsp.pretrain(force=True)
        trained, cov = wsp.ensure_trained(case_folder, allow_build_if_missing=False)
        tqdm.write(f"[{case_folder}] Warm index trained={trained}, coverage={cov:.3f}")

        # Stage B2: Redundancy index if PRUNE present
        mode_list_for_case: List[RunConfig] = []
        if auto_mode:
            mode_list_for_case = _auto_mode_list(args, case_folder=case_folder)
            prune_requested = any(m.mode.startswith("WARM+PRUNE") for m in mode_list_for_case)
        else:
            # Manual modes list
            prune_requested = any(
                m.strip().upper().startswith("WARM+PRUNE") for m in args.modes
            )
            for m in args.modes:
                m_up = m.strip().upper()
                if m_up == "RAW":
                    mode_list_for_case.append(
                        RunConfig(
                            mode="RAW",
                            time_limit=args.time_limit,
                            mip_gap=args.mip_gap,
                            use_gnn_screen=bool(args.adv_gnn_screen),
                            gnn_thr=float(args.adv_gnn_thr),
                            use_commit_hints=bool(args.adv_commit_hints)
                            and bool(args.adv_commit_on_raw),
                            commit_mode=str(args.adv_commit_mode),
                            commit_thr=float(args.adv_commit_thr),
                            commit_on_raw=bool(args.adv_commit_on_raw),
                            save_human_logs=args.save_human_logs,
                        )
                    )
                elif m_up == "WARM":
                    mode_list_for_case.append(
                        RunConfig(
                            mode="WARM",
                            time_limit=args.time_limit,
                            mip_gap=args.mip_gap,
                            use_gnn_screen=bool(args.adv_gnn_screen),
                            gnn_thr=float(args.adv_gnn_thr),
                            use_commit_hints=bool(args.adv_commit_hints),
                            commit_mode=str(args.adv_commit_mode),
                            commit_thr=float(args.adv_commit_thr),
                            use_gru_warm=bool(args.adv_gru_warm),
                            save_human_logs=args.save_human_logs,
                        )
                    )
                elif m_up == "WARM+HINTS":
                    mode_list_for_case.append(
                        RunConfig(
                            mode="WARM+HINTS",
                            time_limit=args.time_limit,
                            mip_gap=args.mip_gap,
                            use_gnn_screen=bool(args.adv_gnn_screen),
                            gnn_thr=float(args.adv_gnn_thr),
                            use_commit_hints=bool(args.adv_commit_hints),
                            commit_mode=str(args.adv_commit_mode),
                            commit_thr=float(args.adv_commit_thr),
                            use_gru_warm=bool(args.adv_gru_warm),
                            save_human_logs=args.save_human_logs,
                        )
                    )
                elif m_up == "WARM+LAZY":
                    mode_list_for_case.append(
                        RunConfig(
                            mode="WARM+LAZY",
                            time_limit=args.time_limit,
                            mip_gap=args.mip_gap,
                            lazy_top_k=args.lazy_top_k,
                            lazy_lodf_tol=args.lazy_lodf_tol,
                            lazy_isf_tol=args.lazy_isf_tol,
                            lazy_viol_tol=args.lazy_viol_tol,
                            use_adaptive_topk=bool(args.adv_adaptive_topk),
                            topk_candidates=args.adv_topk_candidates,
                            topk_epsilon=float(args.adv_topk_epsilon),
                            use_commit_hints=bool(args.adv_commit_hints),
                            commit_mode=str(args.adv_commit_mode),
                            commit_thr=float(args.adv_commit_thr),
                            use_gru_warm=bool(args.adv_gru_warm),
                            save_human_logs=args.save_human_logs,
                        )
                    )
                elif m_up == "WARM+PRUNE":
                    for tau in args.taus:
                        mode_list_for_case.append(
                            RunConfig(
                                mode="WARM+PRUNE",
                                tau=float(tau),
                                time_limit=args.time_limit,
                                mip_gap=args.mip_gap,
                                use_gnn_screen=bool(args.adv_gnn_screen),
                                gnn_thr=float(args.adv_gnn_thr),
                                use_commit_hints=bool(args.adv_commit_hints),
                                commit_mode=str(args.adv_commit_mode),
                                commit_thr=float(args.adv_commit_thr),
                                use_gru_warm=bool(args.adv_gru_warm),
                                save_human_logs=args.save_human_logs,
                            )
                        )
                elif m_up == "WARM+LPSCREEN":
                    for tau in args.taus:
                        mode_list_for_case.append(
                            RunConfig(
                                mode="WARM+LPSCREEN",
                                tau=float(tau),
                                time_limit=args.time_limit,
                                mip_gap=args.mip_gap,
                                save_human_logs=args.save_human_logs,
                            )
                        )
                elif m_up == "WARM+PRUNE+LAZY":
                    for tau in args.taus:
                        mode_list_for_case.append(
                            RunConfig(
                                mode="WARM+PRUNE+LAZY",
                                tau=float(tau),
                                time_limit=args.time_limit,
                                mip_gap=args.mip_gap,
                                lazy_top_k=args.lazy_top_k,
                                lazy_lodf_tol=args.lazy_lodf_tol,
                                lazy_isf_tol=args.lazy_isf_tol,
                                lazy_viol_tol=args.lazy_viol_tol,
                                use_commit_hints=bool(args.adv_commit_hints),
                                commit_mode=str(args.adv_commit_mode),
                                commit_thr=float(args.adv_commit_thr),
                                use_gru_warm=bool(args.adv_gru_warm),
                                save_human_logs=args.save_human_logs,
                            )
                        )
                elif m_up == "WARM+LPSCREEN+LAZY":
                    for tau in args.taus:
                        mode_list_for_case.append(
                            RunConfig(
                                mode="WARM+LPSCREEN+LAZY",
                                tau=float(tau),
                                time_limit=args.time_limit,
                                mip_gap=args.mip_gap,
                                lazy_top_k=args.lazy_top_k,
                                lazy_lodf_tol=args.lazy_lodf_tol,
                                lazy_isf_tol=args.lazy_isf_tol,
                                lazy_viol_tol=args.lazy_viol_tol,
                                use_commit_hints=bool(args.adv_commit_hints),
                                commit_mode=str(args.adv_commit_mode),
                                commit_thr=float(args.adv_commit_thr),
                                use_gru_warm=bool(args.adv_gru_warm),
                                save_human_logs=args.save_human_logs,
                            )
                        )
                elif m_up == "WARM+SR+LAZY":
                    mode_list_for_case.append(
                        RunConfig(
                            mode="WARM+SR+LAZY",
                            time_limit=args.time_limit,
                            mip_gap=args.mip_gap,
                            lazy_top_k=args.lazy_top_k,
                            lazy_lodf_tol=args.lazy_lodf_tol,
                            lazy_isf_tol=args.lazy_isf_tol,
                            lazy_viol_tol=args.lazy_viol_tol,
                            use_commit_hints=bool(args.adv_commit_hints),
                            commit_mode=str(args.adv_commit_mode),
                            commit_thr=float(args.adv_commit_thr),
                            use_gru_warm=bool(args.adv_gru_warm),
                            sr_sigma=float(args.sr_sigma),
                            sr_l2_thr=float(args.sr_l2_thr),
                            sr_sigma_thr=float(args.sr_sigma_thr),
                            save_human_logs=args.save_human_logs,
                        )
                    )
                elif m_up == "ACTIVESET":
                    mode_list_for_case.append(
                        RunConfig(
                            mode="ACTIVESET",
                            time_limit=args.time_limit,
                            mip_gap=args.mip_gap,
                            active_set_batch=int(args.active_set_batch),
                            active_set_max_rounds=int(args.active_set_max_rounds),
                            active_set_cleanup_tol=float(args.active_set_cleanup_tol),
                            save_human_logs=args.save_human_logs,
                        )
                    )
                elif m_up == "ACTIVESET+LAZY":
                    mode_list_for_case.append(
                        RunConfig(
                            mode="ACTIVESET+LAZY",
                            time_limit=args.time_limit,
                            mip_gap=args.mip_gap,
                            lazy_top_k=args.lazy_top_k,
                            lazy_lodf_tol=args.lazy_lodf_tol,
                            lazy_isf_tol=args.lazy_isf_tol,
                            lazy_viol_tol=args.lazy_viol_tol,
                            active_set_batch=int(args.active_set_batch),
                            active_set_max_rounds=int(args.active_set_max_rounds),
                            active_set_cleanup_tol=float(args.active_set_cleanup_tol),
                            save_human_logs=args.save_human_logs,
                        )
                    )
                elif m_up == "SHRINK+LAZY":
                    mode_list_for_case.append(
                        RunConfig(
                            mode="SHRINK+LAZY",
                            time_limit=args.time_limit,
                            mip_gap=args.mip_gap,
                            lazy_top_k=args.lazy_top_k,
                            lazy_lodf_tol=args.lazy_lodf_tol,
                            lazy_isf_tol=args.lazy_isf_tol,
                            lazy_viol_tol=args.lazy_viol_tol,
                            shrink_window=int(args.shrink_window),
                            shrink_overlap=int(args.shrink_overlap),
                            save_human_logs=args.save_human_logs,
                        )
                    )
                elif m_up == "STREDUCE":
                    mode_list_for_case.append(
                        RunConfig(
                            mode="STREDUCE",
                            time_limit=args.time_limit,
                            mip_gap=args.mip_gap,
                            st_commit_fix_thr=float(args.st_commit_fix_thr),
                            st_line_keep_thr=float(args.st_line_keep_thr),
                            save_human_logs=args.save_human_logs,
                        )
                    )
                elif m_up == "STREDUCE+LAZY":
                    mode_list_for_case.append(
                        RunConfig(
                            mode="STREDUCE+LAZY",
                            time_limit=args.time_limit,
                            mip_gap=args.mip_gap,
                            lazy_top_k=args.lazy_top_k,
                            lazy_lodf_tol=args.lazy_lodf_tol,
                            lazy_isf_tol=args.lazy_isf_tol,
                            lazy_viol_tol=args.lazy_viol_tol,
                            st_commit_fix_thr=float(args.st_commit_fix_thr),
                            st_line_keep_thr=float(args.st_line_keep_thr),
                            save_human_logs=args.save_human_logs,
                        )
                    )
                else:
                    tqdm.write(f"Unknown mode '{m}' - ignored")

        rc = None
        if prune_requested:
            rc = RCProvider(
                case_folder=case_folder,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                split_seed=args.seed,
            )
            rc_items = (
                args.rc_max_items
                if (args.rc_max_items and args.rc_max_items > 0)
                else None
            )
            rc.pretrain(
                force=True,
                max_items=rc_items,
                restrict_to_train=bool(args.rc_restrict_train),
            )
            rc_ok, rc_cov = rc.ensure_trained(case_folder, allow_build_if_missing=False)
            tqdm.write(
                f"[{case_folder}] Redundancy index ready={rc_ok}, coverage={rc_cov:.3f}"
            )
            if not rc_ok:
                rc = None

        # Stage B3: Pretrain extra ML modules if needed
        # Use train split names to avoid data leakage from test/val outputs
        train_names_set = set(train)

        if auto_mode:
            any_commit = any(
                cfg.use_commit_hints
                or (cfg.mode == "RAW" and cfg.commit_on_raw)
                or cfg.mode.startswith("STREDUCE")
                for cfg in mode_list_for_case
            )
            any_gnn = any(
                cfg.use_gnn_screen or cfg.mode.startswith("STREDUCE")
                for cfg in mode_list_for_case
            )
            any_gru = any(
                cfg.use_gru_warm or cfg.mode.startswith("STREDUCE")
                for cfg in mode_list_for_case
            )
            if any_commit:
                try:
                    ch = CommitmentHints(case_folder=case_folder)
                    ch.pretrain(force=True, restrict_to_names=train_names_set)
                    _ = ch.ensure_trained()
                    tqdm.write(f"[{case_folder}] Commitment hints model prepared.")
                except Exception:
                    tqdm.write(
                        f"[{case_folder}] Commitment hints pretrain failed or skipped."
                    )
            if any_gnn:
                if GNNLineScreener is None:
                    tqdm.write(
                        f"[{case_folder}] GNN screening skipped (torch/gnn deps unavailable)."
                    )
                else:
                    try:
                        gnn = GNNLineScreener(case_folder=case_folder)
                        gnn.pretrain(force=True, restrict_to_names=train_names_set)
                        _ = gnn.ensure_trained()
                        tqdm.write(f"[{case_folder}] GNN screening model prepared.")
                    except Exception:
                        tqdm.write(
                            f"[{case_folder}] GNN screening pretrain failed or skipped."
                        )
            if any_gru:
                if GRUDispatchWarmStart is None:
                    tqdm.write(
                        f"[{case_folder}] GRU warm-start skipped (torch deps unavailable)."
                    )
                else:
                    try:
                        gw = GRUDispatchWarmStart(case_folder=case_folder)
                        gw.pretrain(force=True, restrict_to_names=train_names_set)
                        _ = gw.ensure_trained()
                        tqdm.write(f"[{case_folder}] GRU warm-start model prepared.")
                    except Exception:
                        tqdm.write(
                            f"[{case_folder}] GRU warm-start pretrain failed or skipped."
                        )
        else:
            # Manual flags drive pretraining (backward-compatible)
            st_requested = any(m.strip().upper().startswith("STREDUCE") for m in args.modes)
            if bool(args.adv_commit_hints) or bool(args.adv_commit_on_raw) or st_requested:
                try:
                    ch = CommitmentHints(case_folder=case_folder)
                    ch.pretrain(force=True, restrict_to_names=train_names_set)
                    _ = ch.ensure_trained()
                    tqdm.write(f"[{case_folder}] Commitment hints model prepared.")
                except Exception:
                    tqdm.write(
                        f"[{case_folder}] Commitment hints pretrain failed or skipped."
                    )
            if bool(args.adv_gnn_screen) or st_requested:
                if GNNLineScreener is None:
                    tqdm.write(
                        f"[{case_folder}] GNN screening skipped (torch/gnn deps unavailable)."
                    )
                else:
                    try:
                        gnn = GNNLineScreener(case_folder=case_folder)
                        gnn.pretrain(force=True, restrict_to_names=train_names_set)
                        _ = gnn.ensure_trained()
                        tqdm.write(f"[{case_folder}] GNN screening model prepared.")
                    except Exception:
                        tqdm.write(
                            f"[{case_folder}] GNN screening pretrain failed or skipped."
                        )
            if bool(args.adv_gru_warm) or st_requested:
                if GRUDispatchWarmStart is None:
                    tqdm.write(
                        f"[{case_folder}] GRU warm-start skipped (torch deps unavailable)."
                    )
                else:
                    try:
                        gw = GRUDispatchWarmStart(case_folder=case_folder)
                        gw.pretrain(force=True, restrict_to_names=train_names_set)
                        _ = gw.ensure_trained()
                        tqdm.write(f"[{case_folder}] GRU warm-start model prepared.")
                    except Exception:
                        tqdm.write(
                            f"[{case_folder}] GRU warm-start pretrain failed or skipped."
                        )

        # Stage C: Run TEST across modes
        log_path = _prepare_csv_log(case_folder)
        tqdm.write(f"[{case_folder}] logging to {log_path}")

        for nm in tqdm(test, desc=f"{case_folder} TEST", unit="inst"):
            for cfg in mode_list_for_case:
                # Compute label now to evaluate skip logic
                mode_label = _build_mode_label(cfg.mode, cfg)
                if args.skip_solved:
                    # Skip by instance or by (instance,mode)
                    if args.skip_solved_per_instance:
                        if nm in skip_keys_inst:
                            tqdm.write(
                                f"  {nm} | {mode_label:>18} | SKIPPED (instance present in skip logs)"
                            )
                            continue
                    if (nm, mode_label) in skip_keys_mode:
                        tqdm.write(
                            f"  {nm} | {mode_label:>18} | SKIPPED (pair present in skip logs)"
                        )
                        continue
                row = solve_one(
                    nm, cfg, wsp=wsp, rc=rc, use_train_db_for_rc=True, args=args
                )
                _append_csv(log_path, row)
                tqdm.write(
                    f"  {nm} | {row['mode']:>18} | status={row['status']:<12} time={row['runtime_sec']}s nodes={row['nodes']}"
                )
                logger.info(
                    "Done %s | %-18s | status=%s time=%ss",
                    nm,
                    row["mode"],
                    row["status"],
                    row["runtime_sec"],
                )

        tqdm.write(f"[{case_folder}] done. Log: {log_path}")


if __name__ == "__main__":
    main()
