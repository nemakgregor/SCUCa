from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import statistics
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from src.data_preparation.params import DataParams
from src.data_preparation.read_data import read_benchmark
from src.ml_models.bandits import EpsilonGreedyTopK
from src.ml_models.commitment_hints import CommitmentHints
from src.ml_models.gnn_screening import GNNLineScreener
from src.ml_models.gru_warmstart import GRUDispatchWarmStart
from src.ml_models.lp_screening import LPScreener
from src.ml_models.redundant_constraints import RedundancyProvider as RCProvider
from src.ml_models.st_reduction import STReductionProvider
from src.ml_models.warm_start import WarmStartProvider
from src.optimization_model.SCUC_solver.scuc_model_builder import build_model
from src.optimization_model.helpers.branching_hints import apply_branching_hints_from_starts
from src.optimization_model.helpers.lazy_contingency_cb import LazyContingencyConfig, attach_lazy_contingency_callback
from src.optimization_model.helpers.save_json_solution import save_solution_as_json
from src.optimization_model.helpers.verify_solution import verify_solution
from src.paper.experiment_spec import (
    CASES_ALL,
    CASES_MEDLARGE,
    CASES_SMALL,
    DEFAULT_NO_REL_HEUR_TIME_RATIO,
    DROP_MEDIAN_RUNTIME_FRACTION,
    DROP_SUCCESS_RATE_MIN,
    EXPERIMENT_SEED,
    MODE_CATALOG_MEDLARGE_START,
    MODE_CATALOG_MEDLARGE_FULL,
    MODE_CATALOG_SMALL,
    PILOT_N,
    TEST_DATES_6,
    TEST_TL_CAP_SEC_BY_CASE,
    TEST_TL_INIT_SEC_BY_CASE,
    TIME_LIMIT_MULTIPLIER,
    TRAIN_BASE_MODE,
    TRAIN_BASE_MODE_ID,
    TRAIN_DATES_24,
    TRAIN_TL_SEC_BY_CASE,
    ModeSpec,
)
from src.scuc_sr.analysis import radii_for_scenario
from src.scuc_sr.novel_methods import ActiveSetConfig, RollingHorizonConfig, optimize_with_active_set, solve_rolling_horizon

logger = logging.getLogger(__name__)

CSV_FIELDS = [
    "timestamp_utc","run_id","result_key","stage","case_folder","instance_name","mode_id","mode_family","time_limit_sec","mip_gap_target",
    "status","status_code","runtime_sec","wall_sec","mip_gap","obj_val","obj_bound","nodes","has_incumbent","feasible_ok","violations",
    "max_constraint_residual","objective_inconsistency","pass","exact_method","screen_setup_sec","num_vars_root","num_constrs_root",
    "num_vars_final","num_constrs_final","warm_start_applied_vars","branch_hints_applied","constr_total_cont","constr_kept_cont",
    "constr_ratio_cont_explicit","constr_realized_cont","constr_ratio_cont_realized","screen_monitored_lines","explicit_added_cont",
    "lazy_added_cont","active_set_iters","active_set_added","active_set_dropped","shrink_window_count","shrink_window_size","shrink_overlap",
    "fixed_commit_vars","fixed_commit_on","fixed_commit_off","st_kept_line_pairs","st_kept_gen_pairs","candidate_solution_json",
    "train_solution_json","error_message",
]


@dataclass
class RunPaths:
    root: Path
    csv_path: Path
    state_path: Path
    logs_dir: Path
    solutions_dir: Path
    train_output_dir: Path
    artifacts_dir: Path
    warm_dir: Path


@dataclass
class CaseArtifacts:
    case_folder: str
    train_names: Set[str]
    warm_provider: WarmStartProvider
    redundancy_provider: RCProvider
    commitment_hints: CommitmentHints
    gnn_screener: GNNLineScreener
    gru_warmstart: GRUDispatchWarmStart
    streduction: STReductionProvider


@dataclass
class SolvePayload:
    scenario: object
    model: object
    screen_setup_sec: float
    num_vars_root: int
    num_constrs_root: int
    warm_start_applied_vars: int
    branch_hints_applied: int
    constr_total_cont: Optional[int]
    constr_kept_cont: Optional[int]
    constr_ratio_cont_explicit: Optional[float]
    screen_monitored_lines: Optional[int]
    active_set_iters: int
    active_set_added: int
    active_set_dropped: int
    shrink_window_count: int
    shrink_window_size: int
    shrink_overlap: int
    fixed_commit_vars: int
    fixed_commit_on: int
    fixed_commit_off: int
    st_kept_line_pairs: int
    st_kept_gen_pairs: int


def _medlarge_catalog(kind: str) -> List[ModeSpec]:
    return list(MODE_CATALOG_MEDLARGE_FULL if str(kind).strip().lower() == "full" else MODE_CATALOG_MEDLARGE_START)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("gurobipy").setLevel(logging.WARNING)


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(text))


def _gurobi_log_path(paths: RunPaths, stage: str, mode: ModeSpec, instance_name: str) -> Path:
    safe_case = _slug(instance_name)
    safe_mode = _slug(mode.mode_id.lower())
    return paths.logs_dir / "gurobi" / stage.lower() / f"{safe_case}__{safe_mode}.log"


def _set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_no_rel_heur_time(mode: ModeSpec) -> float:
    if mode.no_rel_heur_time is not None:
        return max(0.0, float(mode.no_rel_heur_time))
    return max(0.0, float(mode.time_limit_sec) * float(DEFAULT_NO_REL_HEUR_TIME_RATIO))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * float(attempt + 1))


def _atomic_write_json(path: Path, payload: Dict) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _configure_run_paths(run_id: str, resume: bool) -> RunPaths:
    root = (Path("results") / run_id).resolve()
    csv_path = root / "results.csv"
    state_path = root / "state.json"
    logs_dir = root / "logs"
    solutions_dir = root / "solutions"
    train_output_dir = root / "train_solutions"
    artifacts_dir = root / "artifacts"
    warm_dir = artifacts_dir / "warm_start"
    if root.exists() and not resume and (csv_path.exists() or state_path.exists()):
        raise RuntimeError(f"Run '{run_id}' already exists. Use --resume or choose a new --run-id.")
    for path in (root, logs_dir, solutions_dir, train_output_dir, artifacts_dir, warm_dir):
        path.mkdir(parents=True, exist_ok=True)
    DataParams._OUTPUT = train_output_dir
    DataParams._INTERMEDIATE = artifacts_dir
    DataParams._WARM_START = warm_dir
    DataParams._LOGS = logs_dir
    return RunPaths(root, csv_path, state_path, logs_dir, solutions_dir, train_output_dir, artifacts_dir, warm_dir)


def _default_state(run_id: str, medlarge_mode_ids: Optional[Sequence[str]] = None) -> Dict:
    return {
        "run_id": run_id,
        "seed": EXPERIMENT_SEED,
        "completed_keys": [],
        "artifacts_built_cases": [],
        "case_state": {},
        "alive_mode_ids": list(medlarge_mode_ids or [mode.mode_id for mode in MODE_CATALOG_MEDLARGE_START]),
    }


def _load_completed_keys_from_csv(csv_path: Path) -> Set[str]:
    if not csv_path.exists():
        return set()
    completed: Set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("result_key") or "").strip()
            status = (row.get("status") or "").strip().upper()
            if key and status != "ERROR":
                completed.add(key)
    return completed


def _load_result_lookup(csv_path: Path) -> Dict[str, Dict]:
    if not csv_path.exists():
        return {}
    lookup: Dict[str, Dict] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("result_key") or "").strip()
            if key:
                lookup[key] = row
    return lookup


def _load_state(paths: RunPaths, run_id: str, resume: bool, medlarge_mode_ids: Optional[Sequence[str]] = None) -> Dict:
    state = _default_state(run_id, medlarge_mode_ids=medlarge_mode_ids)
    if resume and paths.state_path.exists():
        state = json.loads(paths.state_path.read_text(encoding="utf-8"))
    state["completed_keys"] = sorted(_load_completed_keys_from_csv(paths.csv_path))
    return state


def _save_state(paths: RunPaths, state: Dict) -> None:
    state["completed_keys"] = sorted(set(state.get("completed_keys", [])))
    state["artifacts_built_cases"] = sorted(set(state.get("artifacts_built_cases", [])))
    persisted_state = dict(state)
    # `completed_keys` is reconstructed from results.csv on load, so keeping the
    # full list in state.json only bloats memory and can fail on large runs.
    persisted_state.pop("completed_keys", None)
    _atomic_write_json(paths.state_path, persisted_state)


def _append_result_row(paths: RunPaths, row: Dict) -> None:
    write_header = not paths.csv_path.exists()
    with paths.csv_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        fh.flush()
        os.fsync(fh.fileno())


def _status_str(code: int) -> str:
    return DataParams.SOLVER_STATUS_STR.get(code, f"STATUS_{code}")


def _result_key(stage: str, mode_id: str, instance_name: str, time_limit_sec: int) -> str:
    return f"{stage}::{mode_id}::{instance_name}::{int(time_limit_sec)}"


def _case_instances(case_folder: str) -> Tuple[List[str], List[str]]:
    return [f"{case_folder}/{date}" for date in TRAIN_DATES_24], [f"{case_folder}/{date}" for date in TEST_DATES_6]


def _has_incumbent(model) -> bool:
    return bool(int(getattr(model, "SolCount", 0) or 0))


def _allowed_lines_by_radius(scenario, l2_thr: float, sigma_thr: float, sigma_sr: float) -> Set[str]:
    radius_data = radii_for_scenario(scenario, t=0, balance="agc", sigma_sr=float(sigma_sr))
    l2_map = radius_data.get("l2", {})
    sigma_map = radius_data.get("sigma", {})
    allowed: Set[str] = set()
    for line in scenario.lines or []:
        source, target = sorted((line.source.name, line.target.name))
        key = f"{source}-{target}"
        keep = False
        if l2_map.get(key) is None and sigma_map.get(key) is None:
            keep = True
        elif l2_map.get(key) is not None and l2_map[key] <= float(l2_thr):
            keep = True
        elif sigma_map.get(key) is not None and sigma_map[key] <= float(sigma_thr):
            keep = True
        if keep:
            allowed.add(line.name)
    return allowed


def _estimate_cont_counts(scenario, filter_predicate, keep_masks, monitored_line_whitelist) -> Tuple[int, int]:
    from src.optimization_model.solver.scuc.constraints.contingencies import _ISF_TOL, _LODF_TOL

    total_constraints = 0
    kept_constraints = 0
    lines = scenario.lines or []
    contingencies = scenario.contingencies or []
    lodf = scenario.lodf.tocsc()
    isf = scenario.isf.tocsc()
    horizon = scenario.time
    line_by_row = {line.index - 1: line for line in lines}
    buses = scenario.buses
    ref_bus = getattr(scenario, "ptdf_ref_bus_index", buses[0].index)
    non_ref = sorted(bus.index for bus in buses if bus.index != ref_bus)
    col_by_bus = {bus_1b: col for col, bus_1b in enumerate(non_ref)}
    keep_line_pairs = None if keep_masks is None else keep_masks.get("line")
    keep_gen_pairs = None if keep_masks is None else keep_masks.get("gen")

    def monitored(name: str) -> bool:
        return monitored_line_whitelist is None or name in monitored_line_whitelist

    for contingency in contingencies:
        for out_line in getattr(contingency, "lines", None) or []:
            col = lodf.getcol(out_line.index - 1)
            for row, alpha in zip(col.indices.tolist(), col.data.tolist()):
                if row == out_line.index - 1 or abs(alpha) < _LODF_TOL:
                    continue
                line_l = line_by_row.get(row)
                if line_l is None:
                    continue
                total_constraints += 2 * horizon
                if not monitored(line_l.name):
                    continue
                if keep_line_pairs is not None and (line_l.name, out_line.name) not in keep_line_pairs:
                    continue
                if filter_predicate is None:
                    kept_constraints += 2 * horizon
                    continue
                for t in range(horizon):
                    if bool(filter_predicate("line", line_l, out_line, t, float(alpha), float(line_l.emergency_limit[t]))):
                        kept_constraints += 2
        for gen in getattr(contingency, "units", None) or []:
            bus_idx = gen.bus.index
            if bus_idx == ref_bus or bus_idx not in col_by_bus:
                coeff_items = [(line_l, 0.0) for line_l in lines]
            else:
                col = isf.getcol(col_by_bus[bus_idx])
                coeff_map = {row: value for row, value in zip(col.indices.tolist(), col.data.tolist())}
                coeff_items = [(line_l, float(coeff_map.get(line_l.index - 1, 0.0))) for line_l in lines]
            for line_l, coeff in coeff_items:
                if bus_idx != ref_bus and bus_idx in col_by_bus and abs(coeff) < _ISF_TOL:
                    continue
                total_constraints += 2 * horizon
                if not monitored(line_l.name):
                    continue
                if keep_gen_pairs is not None and (line_l.name, gen.name) not in keep_gen_pairs:
                    continue
                if filter_predicate is None:
                    kept_constraints += 2 * horizon
                    continue
                for t in range(horizon):
                    if bool(filter_predicate("gen", line_l, gen, t, coeff, float(line_l.emergency_limit[t]))):
                        kept_constraints += 2
    return total_constraints, kept_constraints


def _verify_model(scenario, model) -> Tuple[Optional[bool], Optional[float], Optional[float], Optional[str]]:
    if not _has_incumbent(model):
        return None, None, None, None
    feasible_ok, checks, _ = verify_solution(scenario, model)
    max_constraint_residual = 0.0
    objective_inconsistency = 0.0
    bad_ids: List[str] = []
    for check in checks:
        if isinstance(check.idx, str) and check.idx.startswith("C-"):
            value = float(check.value)
            max_constraint_residual = max(max_constraint_residual, value)
            tol = 1e-3 if check.idx == "C-108" else 1e-5
            if value > tol:
                bad_ids.append(check.idx.replace("-", ""))
        if check.idx == "O-301":
            objective_inconsistency = float(check.value)
    return feasible_ok, max_constraint_residual, objective_inconsistency, ("OK" if not bad_ids else " ".join(sorted(set(bad_ids))))


def _make_error_row(*, run_id: str, stage: str, case_folder: str, instance_name: str, mode: ModeSpec, started_at: float, error_message: str) -> Dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "result_key": _result_key(stage, mode.mode_id, instance_name, mode.time_limit_sec),
        "stage": stage,
        "case_folder": case_folder,
        "instance_name": instance_name,
        "mode_id": mode.mode_id,
        "mode_family": mode.mode_family,
        "time_limit_sec": int(mode.time_limit_sec),
        "mip_gap_target": float(mode.mip_gap),
        "status": "ERROR",
        "status_code": "",
        "runtime_sec": "",
        "wall_sec": f"{time.time() - started_at:.6f}",
        "mip_gap": "",
        "obj_val": "",
        "obj_bound": "",
        "nodes": "",
        "has_incumbent": 0,
        "feasible_ok": "",
        "violations": "",
        "max_constraint_residual": "",
        "objective_inconsistency": "",
        "pass": 0,
        "exact_method": 1 if mode.exact_method else 0,
        "screen_setup_sec": "",
        "num_vars_root": "",
        "num_constrs_root": "",
        "num_vars_final": "",
        "num_constrs_final": "",
        "warm_start_applied_vars": "",
        "branch_hints_applied": "",
        "constr_total_cont": "",
        "constr_kept_cont": "",
        "constr_ratio_cont_explicit": "",
        "constr_realized_cont": "",
        "constr_ratio_cont_realized": "",
        "screen_monitored_lines": "",
        "explicit_added_cont": "",
        "lazy_added_cont": "",
        "active_set_iters": "",
        "active_set_added": "",
        "active_set_dropped": "",
        "shrink_window_count": "",
        "shrink_window_size": "",
        "shrink_overlap": "",
        "fixed_commit_vars": "",
        "fixed_commit_on": "",
        "fixed_commit_off": "",
        "st_kept_line_pairs": "",
        "st_kept_gen_pairs": "",
        "candidate_solution_json": "",
        "train_solution_json": "",
        "error_message": error_message,
    }


def _build_success_row(*, run_id: str, stage: str, case_folder: str, instance_name: str, mode: ModeSpec, started_at: float, payload: SolvePayload, candidate_solution_json: Optional[Path], train_solution_json: Optional[Path]) -> Dict:
    feasible_ok, max_constraint_residual, objective_inconsistency, violations = _verify_model(payload.scenario, payload.model)
    status_code = int(getattr(payload.model, "Status", -1))
    runtime_sec = float(getattr(payload.model, "Runtime", 0.0) or 0.0)
    mip_gap = getattr(payload.model, "MIPGap", None)
    obj_val = getattr(payload.model, "ObjVal", None)
    obj_bound = getattr(payload.model, "ObjBound", None)
    nodes = getattr(payload.model, "NodeCount", None)
    has_incumbent = _has_incumbent(payload.model)
    num_vars_final = int(getattr(payload.model, "NumVars", 0) or 0)
    num_constrs_final = int(getattr(payload.model, "NumConstrs", 0) or 0)
    explicit_added_cont = int(getattr(payload.model, "_explicit_total_cont_constraints", 0) or 0)
    lazy_stats = getattr(payload.model, "_lazy_contingency_stats", {}) or {}
    lazy_added_cont = int(lazy_stats.get("lazy_added", 0) or 0)
    constr_realized_cont = ""
    constr_ratio_cont_realized = ""
    if payload.constr_total_cont:
        realized = int(payload.constr_kept_cont or 0) + int(lazy_added_cont)
        constr_realized_cont = realized
        constr_ratio_cont_realized = f"{realized / float(payload.constr_total_cont):.6f}"
    pass_flag = int(
        has_incumbent and feasible_ok is True and violations == "OK" and mip_gap is not None and float(mip_gap) <= float(mode.mip_gap) + 1e-12
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "result_key": _result_key(stage, mode.mode_id, instance_name, mode.time_limit_sec),
        "stage": stage,
        "case_folder": case_folder,
        "instance_name": instance_name,
        "mode_id": mode.mode_id,
        "mode_family": mode.mode_family,
        "time_limit_sec": int(mode.time_limit_sec),
        "mip_gap_target": float(mode.mip_gap),
        "status": _status_str(status_code),
        "status_code": status_code,
        "runtime_sec": f"{runtime_sec:.6f}",
        "wall_sec": f"{time.time() - started_at:.6f}",
        "mip_gap": "" if mip_gap is None else f"{float(mip_gap):.8f}",
        "obj_val": "" if obj_val is None else f"{float(obj_val):.6f}",
        "obj_bound": "" if obj_bound is None else f"{float(obj_bound):.6f}",
        "nodes": "" if nodes is None else f"{float(nodes):.0f}",
        "has_incumbent": int(has_incumbent),
        "feasible_ok": "" if feasible_ok is None else ("OK" if feasible_ok else "FAIL"),
        "violations": "" if violations is None else violations,
        "max_constraint_residual": "" if max_constraint_residual is None else f"{float(max_constraint_residual):.8e}",
        "objective_inconsistency": "" if objective_inconsistency is None else f"{float(objective_inconsistency):.8e}",
        "pass": pass_flag,
        "exact_method": 1 if mode.exact_method else 0,
        "screen_setup_sec": f"{payload.screen_setup_sec:.6f}",
        "num_vars_root": payload.num_vars_root,
        "num_constrs_root": payload.num_constrs_root,
        "num_vars_final": num_vars_final,
        "num_constrs_final": num_constrs_final,
        "warm_start_applied_vars": payload.warm_start_applied_vars,
        "branch_hints_applied": payload.branch_hints_applied,
        "constr_total_cont": "" if payload.constr_total_cont is None else int(payload.constr_total_cont),
        "constr_kept_cont": "" if payload.constr_kept_cont is None else int(payload.constr_kept_cont),
        "constr_ratio_cont_explicit": "" if payload.constr_ratio_cont_explicit is None else f"{payload.constr_ratio_cont_explicit:.6f}",
        "constr_realized_cont": constr_realized_cont,
        "constr_ratio_cont_realized": constr_ratio_cont_realized,
        "screen_monitored_lines": "" if payload.screen_monitored_lines is None else int(payload.screen_monitored_lines),
        "explicit_added_cont": explicit_added_cont,
        "lazy_added_cont": lazy_added_cont,
        "active_set_iters": payload.active_set_iters,
        "active_set_added": payload.active_set_added,
        "active_set_dropped": payload.active_set_dropped,
        "shrink_window_count": payload.shrink_window_count,
        "shrink_window_size": payload.shrink_window_size,
        "shrink_overlap": payload.shrink_overlap,
        "fixed_commit_vars": payload.fixed_commit_vars,
        "fixed_commit_on": payload.fixed_commit_on,
        "fixed_commit_off": payload.fixed_commit_off,
        "st_kept_line_pairs": payload.st_kept_line_pairs,
        "st_kept_gen_pairs": payload.st_kept_gen_pairs,
        "candidate_solution_json": "" if candidate_solution_json is None else str(candidate_solution_json),
        "train_solution_json": "" if train_solution_json is None else str(train_solution_json),
        "error_message": "",
    }


def _is_train_artifact_eligible(row: Dict) -> bool:
    return bool(
        str(row.get("stage") or "").strip().upper() == "TRAIN"
        and int(row.get("has_incumbent") or 0) == 1
        and str(row.get("feasible_ok") or "").strip().upper() == "OK"
        and str(row.get("violations") or "").strip().upper() == "OK"
    )


def _prepare_build_components(mode: ModeSpec, instance_name: str, case_artifacts: CaseArtifacts):
    scenario = read_benchmark(instance_name, quiet=True).deterministic
    scenario_model = scenario
    screen_setup_sec = 0.0
    screen_monitored_lines: Optional[int] = None
    st_profile = None
    st_keep_masks = None
    monitored_line_whitelist = None
    rc_keep_masks = None
    lp_keep_masks = None
    rc_pred = None
    lp_pred = None
    gnn_pred = None
    if mode.mode_family in ("WARM_PRUNE", "WARM_PRUNE_LAZY"):
        start = time.time()
        if mode.mode_family == "WARM_PRUNE_LAZY":
            result = case_artifacts.redundancy_provider.make_masks_for_instance(scenario, instance_name, thr_rel=float(mode.tau), use_train_index_only=True, exclude_self=True)
            if result is not None:
                (line_pairs, gen_pairs), _ = result
                rc_keep_masks = {"line": line_pairs, "gen": gen_pairs}
        else:
            result = case_artifacts.redundancy_provider.make_filter_for_instance(scenario, instance_name, thr_rel=float(mode.tau), use_train_index_only=True, exclude_self=True)
            if result is not None:
                rc_pred, _ = result
        screen_setup_sec += time.time() - start
    if mode.mode_family in ("WARM_LPSCREEN", "WARM_LPSCREEN_LAZY"):
        screener = LPScreener()
        if mode.mode_family == "WARM_LPSCREEN_LAZY":
            result = screener.screen_masks(scenario, tau=float(mode.tau), lp_time_limit=30.0)
            if result is not None:
                (keep_line, keep_gen), stats = result
                lp_keep_masks = {"line": keep_line, "gen": keep_gen}
                screen_setup_sec += float(stats.get("screen_time", 0.0) or 0.0)
        else:
            result = screener.screen(scenario, tau=float(mode.tau), lp_time_limit=30.0)
            if result is not None:
                lp_pred, stats = result
                screen_setup_sec += float(stats.get("screen_time", 0.0) or 0.0)
    if mode.mode_family == "WARM_SR_LAZY":
        monitored_line_whitelist = _allowed_lines_by_radius(scenario, l2_thr=float(mode.sr_l2_thr), sigma_thr=float(mode.sr_sigma_thr), sigma_sr=float(mode.sr_sigma))
        screen_monitored_lines = len(monitored_line_whitelist)
    if mode.mode_family in ("STREDUCE", "STREDUCE_LAZY"):
        st_profile = case_artifacts.streduction.build_profile(scenario, instance_name, commit_fix_thr=float(mode.st_commit_fix_thr), line_keep_thr=float(mode.st_line_keep_thr))
        scenario_model = st_profile.scenario
        screen_setup_sec += float(st_profile.setup_sec)
        screen_monitored_lines = len(st_profile.monitored_lines)
        st_keep_masks = {"line": st_profile.keep_line_pairs or set(), "gen": st_profile.keep_gen_pairs or set()}
    if mode.use_gnn_screening:
        gnn_pred = case_artifacts.gnn_screener.make_pruning_predicate(scenario, thr_pred=float(mode.gnn_thr))
    return scenario, scenario_model, st_profile, st_keep_masks, monitored_line_whitelist, screen_setup_sec, screen_monitored_lines, rc_pred, rc_keep_masks, lp_pred, lp_keep_masks, gnn_pred


def _build_model_for_mode(mode: ModeSpec, scenario, scenario_model, st_keep_masks, monitored_line_whitelist, rc_pred, lp_pred, gnn_pred):
    omit_explicit = mode.mode_family in {"LAZY", "WARM_LAZY", "WARM_PRUNE_LAZY", "WARM_LPSCREEN_LAZY", "ACTIVESET", "ACTIVESET_LAZY", "SHRINK_LAZY"}

    def combined(kind, line_l, out_obj, t, coeff, emergency_limit):
        keep_rc = True if rc_pred is None else bool(rc_pred(kind, line_l, out_obj, t, coeff, emergency_limit))
        keep_lp = True if lp_pred is None else bool(lp_pred(kind, line_l, out_obj, t, coeff, emergency_limit))
        keep_gnn = True if gnn_pred is None else bool(gnn_pred(kind, line_l, out_obj, t, coeff, emergency_limit))
        return keep_rc and keep_lp and keep_gnn

    contingency_filter = None
    if mode.mode_family in {"RAW", "WARM"}:
        contingency_filter = gnn_pred
    elif mode.mode_family == "WARM_PRUNE":
        contingency_filter = combined if rc_pred or gnn_pred else None
    elif mode.mode_family == "WARM_LPSCREEN":
        contingency_filter = combined if lp_pred or gnn_pred else None
    model = build_model(
        scenario=scenario_model,
        contingency_filter=None if (omit_explicit or mode.mode_family == "WARM_SR_LAZY") else contingency_filter,
        contingency_keep_masks=st_keep_masks,
        use_lazy_contingencies=bool(omit_explicit),
        radius_line_whitelist=monitored_line_whitelist,
    )
    model.update()
    return model


def _apply_training_artifacts(mode: ModeSpec, model, scenario, scenario_model, instance_name: str, case_artifacts: CaseArtifacts) -> Tuple[int, int]:
    warm_start_applied = 0
    branch_hints_applied = 0
    if mode.use_warm_start:
        case_artifacts.warm_provider.generate_and_save_warm_start(instance_name, use_train_index_only=True, exclude_self=True)
        warm_start_applied = case_artifacts.warm_provider.apply_warm_start_to_model(model, scenario_model, instance_name, mode="repair")
    if mode.use_commit_hints:
        case_artifacts.commitment_hints.apply_to_model(model, scenario, instance_name, thr=float(mode.commit_thr), mode=str(mode.commit_mode))
    if mode.use_gru_warmstart:
        case_artifacts.gru_warmstart.apply_to_model(model, scenario_model)
    if mode.use_branch_hints:
        branch_hints_applied = apply_branching_hints_from_starts(model)
    return warm_start_applied, branch_hints_applied


def _attach_lazy_if_needed(mode: ModeSpec, model, scenario_model, rc_keep_masks, lp_keep_masks) -> None:
    if not mode.use_lazy_callback:
        return
    lazy_keep_masks = rc_keep_masks or lp_keep_masks
    cfg = LazyContingencyConfig(
        lodf_tol=float(mode.lazy_lodf_tol),
        isf_tol=float(mode.lazy_isf_tol),
        violation_tol=float(mode.lazy_viol_tol),
        add_top_k=int(mode.lazy_top_k),
        keep_line_pairs=None if mode.mode_family == "STREDUCE_LAZY" else (None if lazy_keep_masks is None else lazy_keep_masks.get("line")),
        keep_gen_pairs=None if mode.mode_family == "STREDUCE_LAZY" else (None if lazy_keep_masks is None else lazy_keep_masks.get("gen")),
        verbose=False,
    )
    if mode.lazy_bandit:
        cfg.topk_policy = EpsilonGreedyTopK(K_list=list(mode.bandit_k_list), epsilon=float(mode.bandit_epsilon), seed=EXPERIMENT_SEED)
    attach_lazy_contingency_callback(model, scenario_model, cfg)


def _set_solver_params(model, mode: ModeSpec, *, gurobi_log_path: Optional[Path] = None) -> None:
    if gurobi_log_path is not None:
        gurobi_log_path.parent.mkdir(parents=True, exist_ok=True)
        model.Params.LogFile = str(gurobi_log_path)
    model.Params.OutputFlag = 1
    model.Params.NumericFocus = 1
    model.Params.MIPGap = float(mode.mip_gap)
    model.Params.TimeLimit = float(mode.time_limit_sec)
    no_rel_heur_time = _resolve_no_rel_heur_time(mode)
    if no_rel_heur_time > 0.0:
        model.Params.NoRelHeurTime = float(no_rel_heur_time)
    if mode.use_lazy_callback:
        model.Params.LazyConstraints = 1


def _solve_payload(instance_name: str, mode: ModeSpec, case_artifacts: CaseArtifacts, *, paths: Optional[RunPaths] = None, stage: str = "TEST") -> SolvePayload:
    scenario, scenario_model, st_profile, st_keep_masks, monitored_line_whitelist, screen_setup_sec, screen_monitored_lines, rc_pred, rc_keep_masks, lp_pred, lp_keep_masks, gnn_pred = _prepare_build_components(mode, instance_name, case_artifacts)
    if mode.mode_family == "SHRINK_LAZY":
        rh_cfg = RollingHorizonConfig(
            time_limit=int(mode.time_limit_sec), mip_gap=float(mode.mip_gap), window_size=int(mode.shrink_window), overlap=int(mode.shrink_overlap),
            lodf_tol=float(mode.lazy_lodf_tol), isf_tol=float(mode.lazy_isf_tol), violation_tol=float(mode.lazy_viol_tol), lazy_top_k=int(mode.lazy_top_k),
            no_rel_heur_time=float(_resolve_no_rel_heur_time(mode)), output_flag=0,
        )
        proxy_model, report = solve_rolling_horizon(scenario, rh_cfg)
        return SolvePayload(
            scenario=scenario, model=proxy_model, screen_setup_sec=screen_setup_sec,
            num_vars_root=int(getattr(report, "max_num_vars", 0) or 0), num_constrs_root=int(getattr(report, "max_num_constrs", 0) or 0),
            warm_start_applied_vars=0, branch_hints_applied=0, constr_total_cont=None, constr_kept_cont=None, constr_ratio_cont_explicit=None,
            screen_monitored_lines=screen_monitored_lines, active_set_iters=0, active_set_added=0, active_set_dropped=0,
            shrink_window_count=int(getattr(report, "window_count", 0) or 0), shrink_window_size=int(getattr(report, "window_size", mode.shrink_window) or 0),
            shrink_overlap=int(getattr(report, "overlap", mode.shrink_overlap) or 0), fixed_commit_vars=0, fixed_commit_on=0, fixed_commit_off=0, st_kept_line_pairs=0, st_kept_gen_pairs=0,
        )
    model = _build_model_for_mode(mode, scenario, scenario_model, st_keep_masks, monitored_line_whitelist, rc_pred, lp_pred, gnn_pred)
    num_vars_root = int(getattr(model, "NumVars", 0) or 0)
    num_constrs_root = int(getattr(model, "NumConstrs", 0) or 0)
    warm_start_applied, branch_hints_applied = _apply_training_artifacts(mode, model, scenario, scenario_model, instance_name, case_artifacts)
    _attach_lazy_if_needed(mode, model, scenario_model, rc_keep_masks, lp_keep_masks)
    _set_solver_params(model, mode, gurobi_log_path=None if paths is None else _gurobi_log_path(paths, stage, mode, instance_name))
    active_set_iters = 0
    active_set_added = 0
    active_set_dropped = 0
    if mode.mode_family in {"ACTIVESET", "ACTIVESET_LAZY"}:
        report = optimize_with_active_set(model, scenario, ActiveSetConfig(
            time_limit=int(mode.time_limit_sec), mip_gap=float(mode.mip_gap), lodf_tol=float(mode.lazy_lodf_tol), isf_tol=float(mode.lazy_isf_tol),
            violation_tol=float(mode.lazy_viol_tol), batch_size=int(mode.active_set_batch), max_rounds=int(mode.active_set_max_rounds),
            cleanup_inactive=bool(mode.active_set_cleanup), cleanup_tol=float(mode.active_set_cleanup_tol),
            no_rel_heur_time=float(_resolve_no_rel_heur_time(mode)), output_flag=0,
        ))
        active_set_iters = int(getattr(report, "iterations", 0) or 0)
        active_set_added = int(getattr(report, "added_constraints", 0) or 0)
        active_set_dropped = int(getattr(report, "dropped_constraints", 0) or 0)
    else:
        callback = getattr(model, "_lazy_contingency_callback", None)
        model.optimize() if callback is None else model.optimize(callback)
    constr_total_cont = None
    constr_kept_cont = None
    constr_ratio_cont_explicit = None
    if mode.mode_family in {"WARM_PRUNE","WARM_LPSCREEN","WARM_PRUNE_LAZY","WARM_LPSCREEN_LAZY","WARM_SR_LAZY","STREDUCE","STREDUCE_LAZY"}:
        estimate_pred = rc_pred if mode.mode_family == "WARM_PRUNE" else lp_pred if mode.mode_family == "WARM_LPSCREEN" else None
        estimate_masks = rc_keep_masks if mode.mode_family == "WARM_PRUNE_LAZY" else lp_keep_masks if mode.mode_family == "WARM_LPSCREEN_LAZY" else st_keep_masks if mode.mode_family in {"STREDUCE","STREDUCE_LAZY"} else None
        constr_total_cont, constr_kept_cont = _estimate_cont_counts(scenario, estimate_pred, estimate_masks, monitored_line_whitelist)
        constr_ratio_cont_explicit = None if not constr_total_cont else float(constr_kept_cont) / float(constr_total_cont)
    return SolvePayload(
        scenario=scenario, model=model, screen_setup_sec=screen_setup_sec, num_vars_root=num_vars_root, num_constrs_root=num_constrs_root,
        warm_start_applied_vars=warm_start_applied, branch_hints_applied=branch_hints_applied, constr_total_cont=constr_total_cont, constr_kept_cont=constr_kept_cont,
        constr_ratio_cont_explicit=constr_ratio_cont_explicit, screen_monitored_lines=screen_monitored_lines, active_set_iters=active_set_iters,
        active_set_added=active_set_added, active_set_dropped=active_set_dropped, shrink_window_count=0, shrink_window_size=0, shrink_overlap=0,
        fixed_commit_vars=int(getattr(st_profile, "fixed_commit_vars", 0) or 0), fixed_commit_on=int(getattr(st_profile, "fixed_commit_on", 0) or 0),
        fixed_commit_off=int(getattr(st_profile, "fixed_commit_off", 0) or 0), st_kept_line_pairs=int(getattr(st_profile, "kept_line_pairs", 0) or 0),
        st_kept_gen_pairs=int(getattr(st_profile, "kept_gen_pairs", 0) or 0),
    )


def _save_candidate_solution(paths: RunPaths, stage: str, mode: ModeSpec, instance_name: str, payload: SolvePayload) -> Optional[Path]:
    if not _has_incumbent(payload.model):
        return None
    return save_solution_as_json(payload.scenario, payload.model, instance_name=instance_name, out_base_dir=paths.solutions_dir / stage.lower() / mode.mode_id.lower(), extra_meta={"mode_id": mode.mode_id, "mode_family": mode.mode_family, "stage": stage})


def _save_train_solution(paths: RunPaths, instance_name: str, payload: SolvePayload) -> Optional[Path]:
    if not _has_incumbent(payload.model):
        return None
    return save_solution_as_json(payload.scenario, payload.model, instance_name=instance_name, out_base_dir=paths.train_output_dir, extra_meta={"artifact_scope": "train", "stage": "TRAIN", "mode_id": TRAIN_BASE_MODE_ID})


def _run_single_solve(*, run_id: str, stage: str, instance_name: str, mode: ModeSpec, case_artifacts: CaseArtifacts, paths: RunPaths) -> Dict:
    started_at = time.time()
    try:
        payload = _solve_payload(instance_name, mode, case_artifacts, paths=paths, stage=stage)
        candidate_solution_json = _save_candidate_solution(paths, stage, mode, instance_name, payload)
        provisional = _build_success_row(run_id=run_id, stage=stage, case_folder=case_artifacts.case_folder, instance_name=instance_name, mode=mode, started_at=started_at, payload=payload, candidate_solution_json=candidate_solution_json, train_solution_json=None)
        train_solution_json = _save_train_solution(paths, instance_name, payload) if _is_train_artifact_eligible(provisional) else None
        row = _build_success_row(run_id=run_id, stage=stage, case_folder=case_artifacts.case_folder, instance_name=instance_name, mode=mode, started_at=started_at, payload=payload, candidate_solution_json=candidate_solution_json, train_solution_json=train_solution_json)
        payload.model.dispose()
        return row
    except Exception as exc:
        logger.exception("Solve failed: stage=%s case=%s instance=%s mode=%s", stage, case_artifacts.case_folder, instance_name, mode.mode_id)
        return _make_error_row(run_id=run_id, stage=stage, case_folder=case_artifacts.case_folder, instance_name=instance_name, mode=mode, started_at=started_at, error_message=str(exc))


def _record_row(paths: RunPaths, state: Dict, row: Dict) -> None:
    _append_result_row(paths, row)
    completed = set(state.get("completed_keys", []))
    if str(row.get("status") or "").strip().upper() != "ERROR":
        completed.add(row["result_key"])
    state["completed_keys"] = sorted(completed)
    _save_state(paths, state)


def _build_case_artifacts(case_folder: str, train_names: Iterable[str]) -> CaseArtifacts:
    train_name_set = set(train_names)
    if not train_name_set:
        raise RuntimeError(f"No successful train solutions for {case_folder}.")
    warm_provider = WarmStartProvider(case_folder=case_folder, coverage_threshold=0.0, train_ratio=1.0, val_ratio=0.0, split_seed=EXPERIMENT_SEED)
    if warm_provider.pretrain(force=True) is None:
        raise RuntimeError(f"Warm-start index build failed for {case_folder}.")
    redundancy_provider = RCProvider(case_folder=case_folder, train_ratio=1.0, val_ratio=0.0, split_seed=EXPERIMENT_SEED)
    if redundancy_provider.pretrain(force=True, restrict_to_train=True) is None:
        raise RuntimeError(f"Redundancy index build failed for {case_folder}.")
    commitment_hints = CommitmentHints(case_folder=case_folder)
    if commitment_hints.pretrain(force=True, restrict_to_names=train_name_set) is None:
        raise RuntimeError(f"Commitment-hints training failed for {case_folder}.")
    gnn_screener = GNNLineScreener(case_folder=case_folder)
    if gnn_screener.pretrain(force=True, seed=EXPERIMENT_SEED, restrict_to_names=train_name_set) is None:
        raise RuntimeError(f"GNN screening training failed for {case_folder}.")
    gru_warmstart = GRUDispatchWarmStart(case_folder=case_folder)
    if gru_warmstart.pretrain(epochs=30, force=True, seed=EXPERIMENT_SEED, restrict_to_names=train_name_set) is None:
        raise RuntimeError(f"GRU warm-start training failed for {case_folder}.")
    return CaseArtifacts(case_folder, train_name_set, warm_provider, redundancy_provider, commitment_hints, gnn_screener, gru_warmstart, STReductionProvider(case_folder=case_folder))


def _successful_train_names_for_case(paths: RunPaths, case_folder: str) -> List[str]:
    case_dir = paths.train_output_dir / case_folder
    if not case_dir.exists():
        return []
    names = []
    for json_path in sorted(case_dir.glob("*.json")):
        rel = json_path.resolve().relative_to(paths.train_output_dir.resolve()).as_posix()
        names.append(rel[:-5] if rel.endswith(".json") else rel)
    return names


def _run_train_stage(*, run_id: str, case_folder: str, train_instances: Sequence[str], paths: RunPaths, state: Dict) -> Optional[CaseArtifacts]:
    logger.info("TRAIN start: %s", case_folder)
    base_tl = int(TRAIN_TL_SEC_BY_CASE[case_folder])
    train_mode = replace(TRAIN_BASE_MODE, time_limit_sec=base_tl)
    bootstrap = CaseArtifacts(
        case_folder=case_folder, train_names=set(),
        warm_provider=WarmStartProvider(case_folder=case_folder, coverage_threshold=0.0, train_ratio=1.0, val_ratio=0.0, split_seed=EXPERIMENT_SEED),
        redundancy_provider=RCProvider(case_folder=case_folder, train_ratio=1.0, val_ratio=0.0, split_seed=EXPERIMENT_SEED),
        commitment_hints=CommitmentHints(case_folder=case_folder), gnn_screener=GNNLineScreener(case_folder=case_folder),
        gru_warmstart=GRUDispatchWarmStart(case_folder=case_folder), streduction=STReductionProvider(case_folder=case_folder),
    )
    completed = set(state.get("completed_keys", []))
    for instance_name in train_instances:
        key = _result_key("TRAIN", train_mode.mode_id, instance_name, train_mode.time_limit_sec)
        if key in completed:
            continue
        row = _run_single_solve(run_id=run_id, stage="TRAIN", instance_name=instance_name, mode=train_mode, case_artifacts=bootstrap, paths=paths)
        _record_row(paths, state, row)
        completed.add(key)
    train_names = _successful_train_names_for_case(paths, case_folder)
    if not train_names:
        logger.error("TRAIN failed: no successful solutions for %s; skipping case.", case_folder)
        case_state = state.setdefault("case_state", {}).setdefault(case_folder, {})
        case_state["skipped"] = True
        case_state["skip_reason"] = "NO_SUCCESSFUL_TRAIN"
        _save_state(paths, state)
        return None
    artifacts = _build_case_artifacts(case_folder, train_names)
    built = set(state.get("artifacts_built_cases", []))
    built.add(case_folder)
    state["artifacts_built_cases"] = sorted(built)
    _save_state(paths, state)
    logger.info("TRAIN complete: %s | successful_train=%d", case_folder, len(artifacts.train_names))
    return artifacts


def _mode_rows(rows: Sequence[Dict], mode_id: str) -> List[Dict]:
    return [row for row in rows if row.get("mode_id") == mode_id]


def _mode_success_rate(rows: Sequence[Dict], denominator: int) -> float:
    return float(sum(int(row.get("pass") or 0) for row in rows)) / float(max(1, denominator))


def _mode_median_runtime(rows: Sequence[Dict]) -> float:
    runtimes = []
    for row in rows:
        try:
            runtimes.append(float(row["runtime_sec"]))
        except Exception:
            continue
    return float("inf") if not runtimes else float(statistics.median(runtimes))


def _run_small_test_stage(*, run_id: str, case_folder: str, test_instances: Sequence[str], case_artifacts: CaseArtifacts, paths: RunPaths, state: Dict) -> None:
    logger.info("TEST start (small): %s", case_folder)
    completed = set(state.get("completed_keys", []))
    for mode in MODE_CATALOG_SMALL:
        mode_tl = replace(mode, time_limit_sec=int(TEST_TL_INIT_SEC_BY_CASE[case_folder]))
        for instance_name in test_instances:
            key = _result_key("TEST", mode_tl.mode_id, instance_name, mode_tl.time_limit_sec)
            if key in completed:
                continue
            row = _run_single_solve(run_id=run_id, stage="TEST", instance_name=instance_name, mode=mode_tl, case_artifacts=case_artifacts, paths=paths)
            _record_row(paths, state, row)
            completed.add(key)


def _run_large_test_stage(*, run_id: str, case_folder: str, test_instances: Sequence[str], case_artifacts: CaseArtifacts, paths: RunPaths, state: Dict, global_alive_mode_ids: Sequence[str], no_medlarge_drop: bool = False) -> Tuple[List[Dict], List[str], int]:
    logger.info("TEST start (medlarge): %s", case_folder)
    mode_lookup = {mode.mode_id: mode for mode in _medlarge_catalog(str(state.get("medlarge_modes_kind", "start")))}
    alive_modes = [mode_lookup[mode_id] for mode_id in global_alive_mode_ids if mode_id in mode_lookup]
    current_tl = int(state.get("case_state", {}).get(case_folder, {}).get("test_time_limit_sec", TEST_TL_INIT_SEC_BY_CASE[case_folder]))
    tl_cap = int(TEST_TL_CAP_SEC_BY_CASE[case_folder])
    pilot_instances = list(test_instances[:PILOT_N])
    case_rows: List[Dict] = []
    result_lookup = _load_result_lookup(paths.csv_path)
    while True:
        pilot_rows: List[Dict] = []
        completed = set(state.get("completed_keys", []))
        for mode in alive_modes:
            mode_tl = replace(mode, time_limit_sec=current_tl)
            for instance_name in pilot_instances:
                key = _result_key("TEST", mode_tl.mode_id, instance_name, mode_tl.time_limit_sec)
                if key in completed:
                    if key in result_lookup:
                        pilot_rows.append(result_lookup[key])
                    continue
                row = _run_single_solve(run_id=run_id, stage="TEST", instance_name=instance_name, mode=mode_tl, case_artifacts=case_artifacts, paths=paths)
                pilot_rows.append(row)
                _record_row(paths, state, row)
                result_lookup[key] = row
        case_rows = list(pilot_rows)
        if any(int(row.get("pass") or 0) == 1 for row in pilot_rows):
            break
        if current_tl >= tl_cap:
            logger.warning("Pilot failed at TL cap: case=%s tl=%d", case_folder, current_tl)
            state.setdefault("case_state", {}).setdefault(case_folder, {})["test_time_limit_sec"] = current_tl
            _save_state(paths, state)
            if no_medlarge_drop:
                break
            return case_rows, [mode.mode_id for mode in alive_modes], current_tl
        current_tl = min(current_tl * TIME_LIMIT_MULTIPLIER, tl_cap)
        state.setdefault("case_state", {}).setdefault(case_folder, {})["test_time_limit_sec"] = current_tl
        _save_state(paths, state)
    if no_medlarge_drop:
        surviving_mode_ids = [mode.mode_id for mode in alive_modes]
    else:
        surviving_mode_ids = [mode.mode_id for mode in alive_modes if sum(int(row.get("pass") or 0) for row in _mode_rows(case_rows, mode.mode_id)) > 0]
    completed = set(state.get("completed_keys", []))
    for mode_id in surviving_mode_ids:
        mode_tl = replace(mode_lookup[mode_id], time_limit_sec=current_tl)
        for instance_name in test_instances[PILOT_N:]:
            key = _result_key("TEST", mode_tl.mode_id, instance_name, mode_tl.time_limit_sec)
            if key in completed:
                if key in result_lookup:
                    case_rows.append(result_lookup[key])
                continue
            row = _run_single_solve(run_id=run_id, stage="TEST", instance_name=instance_name, mode=mode_tl, case_artifacts=case_artifacts, paths=paths)
            case_rows.append(row)
            _record_row(paths, state, row)
            completed.add(key)
            result_lookup[key] = row
    state.setdefault("case_state", {}).setdefault(case_folder, {})["test_time_limit_sec"] = current_tl
    state["case_state"][case_folder]["alive_mode_ids"] = surviving_mode_ids
    _save_state(paths, state)
    return case_rows, surviving_mode_ids, current_tl


def _update_global_alive_modes(case_rows: Sequence[Dict], previous_alive_mode_ids: Sequence[str], current_tl: int, *, no_medlarge_drop: bool = False) -> List[str]:
    if no_medlarge_drop:
        return list(previous_alive_mode_ids)
    baseline = "WARM_LAZY" if "WARM_LAZY" in previous_alive_mode_ids else "LAZY_ALL"
    next_alive = []
    for mode_id in previous_alive_mode_ids:
        mode_rows = _mode_rows(case_rows, mode_id)
        if _mode_success_rate(mode_rows, denominator=len(TEST_DATES_6)) >= float(DROP_SUCCESS_RATE_MIN) and _mode_median_runtime(mode_rows) <= float(current_tl) * float(DROP_MEDIAN_RUNTIME_FRACTION):
            next_alive.append(mode_id)
    if baseline not in next_alive and baseline in previous_alive_mode_ids:
        next_alive.insert(0, baseline)
    return next_alive


def _select_cases(profile: str) -> List[str]:
    if profile == "small":
        return list(CASES_SMALL)
    if profile == "medlarge":
        return list(CASES_MEDLARGE)
    return list(CASES_ALL)


def _run_case(*, run_id: str, case_folder: str, paths: RunPaths, state: Dict, medlarge_alive_mode_ids: Sequence[str], no_medlarge_drop: bool = False) -> List[str]:
    train_instances, test_instances = _case_instances(case_folder)
    if bool(state.get("case_state", {}).get(case_folder, {}).get("skipped", False)):
        logger.warning("CASE skipped (state): %s", case_folder)
        return list(medlarge_alive_mode_ids)
    try:
        if case_folder in set(state.get("artifacts_built_cases", [])):
            artifacts = _build_case_artifacts(case_folder, _successful_train_names_for_case(paths, case_folder))
        else:
            artifacts = _run_train_stage(run_id=run_id, case_folder=case_folder, train_instances=train_instances, paths=paths, state=state)
    except Exception as exc:
        logger.exception("CASE failed during TRAIN/artifact build: %s", case_folder)
        case_state = state.setdefault("case_state", {}).setdefault(case_folder, {})
        case_state["skipped"] = True
        case_state["skip_reason"] = f"TRAIN_OR_ARTIFACT_ERROR: {exc}"
        _save_state(paths, state)
        return list(medlarge_alive_mode_ids)
    if artifacts is None:
        return list(medlarge_alive_mode_ids)
    if case_folder in CASES_SMALL:
        _run_small_test_stage(run_id=run_id, case_folder=case_folder, test_instances=test_instances, case_artifacts=artifacts, paths=paths, state=state)
        return list(medlarge_alive_mode_ids)
    case_rows, alive_mode_ids, current_tl = _run_large_test_stage(
        run_id=run_id,
        case_folder=case_folder,
        test_instances=test_instances,
        case_artifacts=artifacts,
        paths=paths,
        state=state,
        global_alive_mode_ids=medlarge_alive_mode_ids,
        no_medlarge_drop=no_medlarge_drop,
    )
    next_alive = _update_global_alive_modes(case_rows, alive_mode_ids, current_tl, no_medlarge_drop=no_medlarge_drop)
    state["alive_mode_ids"] = list(next_alive)
    _save_state(paths, state)
    return next_alive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic SCUC experiment pipeline.")
    parser.add_argument("--run-id", required=True, help="Result folder under results/.")
    parser.add_argument("--profile", choices=("small", "medlarge", "all"), default="all", help="Which case profile to run.")
    parser.add_argument(
        "--medlarge-modes",
        choices=("start", "full"),
        default="start",
        help="Mode catalog for medium/large cases: compact lazy starter set or expanded full set.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume an existing run-id.")
    parser.add_argument(
        "--start-case",
        default=None,
        help="Start from this case folder (e.g. matpower/case300). Earlier cases are skipped.",
    )
    parser.add_argument(
        "--only-case",
        default=None,
        help="Run only this single case folder (e.g. matpower/case1354pegase).",
    )
    parser.add_argument(
        "--no-medlarge-drop",
        action="store_true",
        help="Do not drop medium/large modes after pilot filtering; run all currently alive modes on all test dates.",
    )
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = parse_args()
    _set_global_determinism(EXPERIMENT_SEED)
    paths = _configure_run_paths(args.run_id, args.resume)
    medlarge_catalog = _medlarge_catalog(args.medlarge_modes)
    medlarge_mode_ids = [mode.mode_id for mode in medlarge_catalog]
    state = _load_state(paths, args.run_id, args.resume, medlarge_mode_ids=medlarge_mode_ids)
    state_kind = str(state.get("medlarge_modes_kind", "") or "").strip().lower()
    if not args.resume or state_kind != args.medlarge_modes:
        state["medlarge_modes_kind"] = args.medlarge_modes
        existing_alive = list(state.get("alive_mode_ids", []))
        for mode_id in medlarge_mode_ids:
            if mode_id not in existing_alive:
                existing_alive.append(mode_id)
        state["alive_mode_ids"] = existing_alive or medlarge_mode_ids
    alive_mode_ids = list(state.get("alive_mode_ids", medlarge_mode_ids))
    cases = _select_cases(args.profile)
    if args.only_case:
        only = str(args.only_case).strip().strip("/\\").replace("\\", "/")
        cases = [only]
    elif args.start_case:
        start = str(args.start_case).strip().strip("/\\").replace("\\", "/")
        if start not in cases:
            raise ValueError(f"--start-case '{start}' not in selected cases: {cases}")
        cases = cases[cases.index(start):]
    for case_folder in cases:
        logger.info("CASE start: %s", case_folder)
        alive_mode_ids = _run_case(
            run_id=args.run_id,
            case_folder=case_folder,
            paths=paths,
            state=state,
            medlarge_alive_mode_ids=alive_mode_ids,
            no_medlarge_drop=bool(args.no_medlarge_drop),
        )
        logger.info("CASE complete: %s", case_folder)
    logger.info("Run complete: %s", args.run_id)


if __name__ == "__main__":
    main()
