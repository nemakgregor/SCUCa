from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import random
import re
import statistics
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

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
from src.optimization_model.SCUC_solver.solve_instances import list_local_cached_instances, list_remote_instances
from src.optimization_model.helpers.branching_hints import apply_branching_hints_from_starts
from src.optimization_model.helpers.lazy_contingency_cb import LazyContingencyConfig, attach_lazy_contingency_callback
from src.optimization_model.helpers.save_json_solution import save_solution_as_json
from src.optimization_model.helpers.verify_solution import verify_solution
from src.paper.experiment_spec import (
    CASES_ALL,
    CASES_LARGE,
    CASES_MEDLARGE,
    CASES_SMALL,
    DEFAULT_NO_REL_HEUR_TIME_RATIO,
    EXPERIMENT_SEED,
    MODE_CATALOG_MEDLARGE_FULL,
    MODE_CATALOG_SMALL,
    TEST_DATES_6,
    TEST_TL_INIT_SEC_BY_CASE,
    TRAIN_BASE_MODE,
    TRAIN_BASE_MODE_ID,
    TRAIN_DATES_24,
    TRAIN_TL_SEC_BY_CASE,
    ModeSpec,
)
from src.paper.modes import canonical_mode, mode_exactness
from src.scuc_sr.analysis import radii_for_scenario
from src.scuc_sr.novel_methods import ActiveSetConfig, RollingHorizonConfig, optimize_with_active_set, solve_rolling_horizon

logger = logging.getLogger(__name__)

DEFAULT_MAX_LARGE_TEST_MODEL_CONSTRS = 3_000_000
DEFAULT_MAX_LARGE_TEST_MODEL_NONZEROS = 12_000_000

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


def _env_flag(name: str, default: bool) -> bool:
    text = os.environ.get(name)
    if text is None:
        return bool(default)
    return str(text).strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    text = os.environ.get(name)
    if text is None or str(text).strip() == "":
        return int(default)
    try:
        return int(text)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %d", name, text, default)
        return int(default)


@dataclass
class RunPaths:
    root: Path
    csv_path: Path
    state_path: Path
    live_status_path: Path
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


class SkipSolve(RuntimeError):
    def __init__(self, status: str, message: str, metrics: Optional[Dict[str, object]] = None):
        super().__init__(message)
        self.status = status
        self.metrics = dict(metrics or {})


def _medlarge_catalog(kind: str) -> List[ModeSpec]:
    return list(MODE_CATALOG_MEDLARGE_FULL)


def _mode_lookup(catalog: Sequence[ModeSpec]) -> Dict[str, ModeSpec]:
    return {mode.mode_id: mode for mode in catalog}


def _mode_catalog_for_case(case_folder: str, medlarge_kind: str) -> List[ModeSpec]:
    return list(MODE_CATALOG_SMALL if case_folder in CASES_SMALL else _medlarge_catalog(medlarge_kind))


def _normalize_mode_ids(mode_ids: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for raw in mode_ids:
        normalized.extend(part for part in re.split(r"[\s,]+", str(raw).strip()) if part)
    return normalized


def _selected_modes_from_ids(case_folder: str, medlarge_kind: str, mode_ids: Sequence[str]) -> List[ModeSpec]:
    catalog = _mode_catalog_for_case(case_folder, medlarge_kind)
    lookup: Dict[str, ModeSpec] = {}
    for mode in catalog:
        lookup[mode.mode_id] = mode
        lookup[canonical_mode(mode.mode_id)] = mode
    selected: List[ModeSpec] = []
    missing: List[str] = []
    seen: Set[str] = set()
    for raw_mode_id in _normalize_mode_ids(mode_ids):
        key = canonical_mode(raw_mode_id)
        mode = lookup.get(str(raw_mode_id).strip()) or lookup.get(key)
        if mode is None:
            missing.append(str(raw_mode_id))
            continue
        if mode.mode_id in seen:
            continue
        seen.add(mode.mode_id)
        selected.append(mode)
    if missing:
        available = ", ".join(mode.mode_id for mode in catalog)
        raise ValueError(f"Unknown mode(s) for {case_folder}: {missing}. Available: {available}")
    return selected


def _planned_modes_for_case(
    *,
    case_folder: str,
    mode_ids: Optional[Sequence[str]],
) -> List[ModeSpec]:
    if mode_ids:
        return _selected_modes_from_ids(case_folder, "full", mode_ids)
    if case_folder in CASES_SMALL:
        return list(MODE_CATALOG_SMALL)
    raise ValueError("Non-small case runs require explicit --only-modes.")


def _mode_requires_training_artifacts(mode: ModeSpec) -> bool:
    return bool(
        mode.use_warm_start
        or mode.use_branch_hints
        or mode.use_commit_hints
        or mode.use_gnn_screening
        or mode.use_gru_warmstart
        or mode.mode_family in {"WARM_PRUNE", "WARM_PRUNE_LAZY", "WARM_LPSCREEN", "WARM_LPSCREEN_LAZY", "STREDUCE", "STREDUCE_LAZY"}
    )


def _modes_require_training_artifacts(modes: Sequence[ModeSpec]) -> bool:
    return any(_mode_requires_training_artifacts(mode) for mode in modes)


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


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _resolve_no_rel_heur_time(mode: ModeSpec) -> float:
    if mode.no_rel_heur_time is not None:
        return max(0.0, float(mode.no_rel_heur_time))
    time_limit = max(0.0, float(mode.time_limit_sec))
    scaled = time_limit * float(DEFAULT_NO_REL_HEUR_TIME_RATIO)
    return max(0.0, min(scaled, time_limit))


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


def _write_live_status(paths: RunPaths, payload: Dict) -> None:
    payload = dict(payload)
    payload.setdefault("schema_version", 1)
    payload.setdefault("timestamp_utc", _utc_now_text())
    payload.setdefault("results_csv", str(paths.csv_path))
    try:
        _atomic_write_json(paths.live_status_path, payload)
    except Exception as exc:
        logger.warning("Could not update live status file %s: %s", paths.live_status_path, exc)


def _mark_solve_started(*, paths: RunPaths, run_id: str, stage: str, instance_name: str, mode: ModeSpec, case_folder: str) -> None:
    payload = {
        "event": "solve_started",
        "run_id": run_id,
        "result_key": _result_key(stage, mode.mode_id, instance_name, mode.time_limit_sec),
        "stage": stage,
        "case_folder": case_folder,
        "instance_name": instance_name,
        "mode_id": mode.mode_id,
        "mode_family": mode.mode_family,
        "time_limit_sec": int(mode.time_limit_sec),
        "mip_gap_target": float(mode.mip_gap),
    }
    if mode.mode_family != "SHRINK_LAZY":
        payload["gurobi_log"] = str(_gurobi_log_path(paths, stage, mode, instance_name))
    _write_live_status(paths, payload)


def _mark_result_recorded(paths: RunPaths, row: Dict) -> None:
    fields = [
        "run_id",
        "result_key",
        "stage",
        "case_folder",
        "instance_name",
        "mode_id",
        "mode_family",
        "time_limit_sec",
        "status",
        "runtime_sec",
        "wall_sec",
        "mip_gap",
        "has_incumbent",
        "feasible_ok",
        "violations",
        "pass",
        "candidate_solution_json",
        "train_solution_json",
        "error_message",
    ]
    _write_live_status(
        paths,
        {
            "event": "result_recorded",
            "last_result": {field: row.get(field, "") for field in fields},
        },
    )


def _configure_run_paths(run_id: str, resume: bool) -> RunPaths:
    root = (Path("results") / run_id).resolve()
    csv_path = root / "results.csv"
    state_path = root / "state.json"
    live_status_path = root / "live_status.json"
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
    return RunPaths(root, csv_path, state_path, live_status_path, logs_dir, solutions_dir, train_output_dir, artifacts_dir, warm_dir)


def _default_state(run_id: str, medlarge_mode_ids: Optional[Sequence[str]] = None) -> Dict:
    return {
        "run_id": run_id,
        "seed": EXPERIMENT_SEED,
        "comparison_plan": "explicit",
        "completed_keys": [],
        "artifacts_built_cases": [],
        "case_state": {},
        "alive_mode_ids": list(medlarge_mode_ids or [mode.mode_id for mode in MODE_CATALOG_MEDLARGE_FULL]),
    }


def _load_completed_keys_from_csv(csv_path: Path) -> Set[str]:
    if not csv_path.exists():
        return set()
    latest_status: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("result_key") or "").strip()
            status = (row.get("status") or "").strip().upper()
            if key:
                latest_status[key] = status
    return {key for key, status in latest_status.items() if status != "ERROR"}


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


def _logical_result_key(stage: str, mode_id: str, instance_name: str) -> str:
    return f"{stage}::{mode_id}::{instance_name}"


def _load_completed_result_lookup_by_logical_key(csv_path: Path) -> Dict[str, Dict]:
    if not csv_path.exists():
        return {}
    latest: Dict[str, Dict] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            stage = str(row.get("stage") or "").strip()
            mode_id = str(row.get("mode_id") or "").strip()
            instance_name = str(row.get("instance_name") or "").strip()
            if stage and mode_id and instance_name:
                latest[_logical_result_key(stage, mode_id, instance_name)] = row
    return {
        key: row
        for key, row in latest.items()
        if str(row.get("status") or "").strip().upper() != "ERROR"
    }


def _instances_from_dates(case_folder: str, dates_or_names: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in dates_or_names:
        text = str(item).strip().strip("/\\").replace("\\", "/")
        if not text:
            continue
        out.append(text if "/" in text else f"{case_folder}/{text}")
    return out


def _discover_case_instances(case_folder: str, instance_policy: str) -> List[str]:
    policy = str(instance_policy or "paper").strip().lower()
    if policy == "paper":
        return []
    try:
        if policy == "all-local":
            names = list_local_cached_instances(include_filters=[case_folder])
        elif policy == "remote":
            names = list_remote_instances(include_filters=[case_folder], roots=["matpower"], max_depth=4)
        else:
            names = []
    except Exception as exc:
        logger.warning("Instance discovery failed for %s using policy=%s: %s", case_folder, policy, exc)
        return []
    prefix = f"{case_folder}/"
    return sorted(name for name in names if str(name).startswith(prefix))


def _case_instances(
    case_folder: str,
    *,
    train_dates: Optional[Sequence[str]] = None,
    test_dates: Optional[Sequence[str]] = None,
    only_test_instances: Optional[Sequence[str]] = None,
    instance_policy: str = "paper",
    limit_train: int = 0,
    limit_test: int = 0,
) -> Tuple[List[str], List[str]]:
    discovered = _discover_case_instances(case_folder, instance_policy)
    discovered_set = set(discovered)
    policy = str(instance_policy or "paper").strip().lower()

    train_source = list(TRAIN_DATES_24 if train_dates is None else train_dates)
    train_instances = _instances_from_dates(case_folder, train_source)
    if policy in {"all-local", "remote"} and discovered:
        train_instances = [name for name in train_instances if name in discovered_set]

    if only_test_instances:
        test_instances = _instances_from_dates(case_folder, only_test_instances)
    elif test_dates is not None:
        test_instances = _instances_from_dates(case_folder, test_dates)
    elif policy in {"all-local", "remote"} and discovered:
        train_set = set(train_instances)
        test_instances = [name for name in discovered if name not in train_set]
    else:
        test_instances = _instances_from_dates(case_folder, TEST_DATES_6)

    if limit_train > 0:
        train_instances = train_instances[: int(limit_train)]
    if limit_test > 0:
        test_instances = test_instances[: int(limit_test)]
    return train_instances, test_instances


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


def _pair_masks_from_predicate(scenario, predicate) -> Optional[Dict[str, Set[Tuple[str, str]]]]:
    if predicate is None:
        return None
    line_pairs: Set[Tuple[str, str]] = set()
    gen_pairs: Set[Tuple[str, str]] = set()
    for contingency in scenario.contingencies or []:
        for out_line in getattr(contingency, "lines", None) or []:
            for line_l in scenario.lines or []:
                try:
                    if bool(predicate("line", line_l, out_line, 0, 0.0, float(line_l.emergency_limit[0]))):
                        line_pairs.add((line_l.name, out_line.name))
                except Exception:
                    continue
        for gen in getattr(contingency, "units", None) or []:
            for line_l in scenario.lines or []:
                try:
                    if bool(predicate("gen", line_l, gen, 0, 0.0, float(line_l.emergency_limit[0]))):
                        gen_pairs.add((line_l.name, gen.name))
                except Exception:
                    continue
    if not line_pairs and not gen_pairs:
        return None
    return {"line": line_pairs, "gen": gen_pairs}


def _verify_tol(check_id: str) -> float:
    return 1e-3 if check_id == "C-108" else 1e-5


def _verify_model(scenario, model) -> Tuple[Optional[bool], Optional[float], Optional[float], Optional[str]]:
    if not _has_incumbent(model):
        return None, None, None, None
    _, checks, _ = verify_solution(scenario, model)
    milp_feasible_ok = True
    max_constraint_residual = 0.0
    objective_inconsistency = 0.0
    bad_ids: List[str] = []
    for check in checks:
        check_id = str(getattr(check, "idx", "") or "")
        try:
            value = float(check.value)
        except Exception:
            value = float("inf")
        tol = _verify_tol(check_id)
        if check_id.startswith(("C-", "V-")) and (not np.isfinite(value) or value > tol):
            milp_feasible_ok = False
        if check_id.startswith("C-"):
            max_constraint_residual = max(max_constraint_residual, value if np.isfinite(value) else float("inf"))
            if not np.isfinite(value) or value > tol:
                bad_ids.append(check_id.replace("-", ""))
        if check_id == "O-301":
            objective_inconsistency = value
    return milp_feasible_ok, max_constraint_residual, objective_inconsistency, ("OK" if not bad_ids else " ".join(sorted(set(bad_ids))))


def _make_error_row(*, run_id: str, stage: str, case_folder: str, instance_name: str, mode: ModeSpec, started_at: float, error_message: str) -> Dict:
    return {
        "timestamp_utc": _utc_now_text(),
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
        "exact_method": 1 if mode_exactness(mode.mode_id) == "exact" else 0,
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


def _make_skip_row(*, run_id: str, stage: str, case_folder: str, instance_name: str, mode: ModeSpec, started_at: float, status: str, error_message: str, metrics: Optional[Dict[str, object]] = None) -> Dict:
    row = _make_error_row(
        run_id=run_id,
        stage=stage,
        case_folder=case_folder,
        instance_name=instance_name,
        mode=mode,
        started_at=started_at,
        error_message=error_message,
    )
    row["status"] = status
    for key, value in (metrics or {}).items():
        if key in row and value is not None:
            row[key] = value
    return row


def _build_success_row(*, run_id: str, stage: str, case_folder: str, instance_name: str, mode: ModeSpec, started_at: float, payload: SolvePayload, candidate_solution_json: Optional[Path], train_solution_json: Optional[Path]) -> Dict:
    feasible_ok, max_constraint_residual, objective_inconsistency, violations = _verify_model(payload.scenario, payload.model)
    status_code = int(getattr(payload.model, "Status", -1))
    runtime_sec = float(
        getattr(payload.model, "_paper_runtime_sec", getattr(payload.model, "Runtime", 0.0))
        or 0.0
    )
    mip_gap = getattr(payload.model, "MIPGap", None)
    obj_val = getattr(payload.model, "ObjVal", None)
    obj_bound = getattr(payload.model, "ObjBound", None)
    nodes = getattr(payload.model, "NodeCount", None)
    has_incumbent = _has_incumbent(payload.model)
    if not has_incumbent:
        mip_gap = None
        obj_val = None
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
    pass_flag = int(has_incumbent and feasible_ok is True and str(violations or "").strip().upper() == "OK")
    return {
        "timestamp_utc": _utc_now_text(),
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
        "exact_method": 1 if mode_exactness(mode.mode_id) == "exact" else 0,
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
    gnn_keep_masks = None
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
        st_keep_masks = {}
        if st_profile.keep_line_pairs is not None:
            st_keep_masks["line"] = st_profile.keep_line_pairs
        if st_profile.keep_gen_pairs is not None:
            st_keep_masks["gen"] = st_profile.keep_gen_pairs
        if not st_keep_masks:
            st_keep_masks = None
    if mode.use_gnn_screening:
        gnn_pred = case_artifacts.gnn_screener.make_pruning_predicate(scenario, thr_pred=float(mode.gnn_thr))
        if mode.use_lazy_callback:
            gnn_keep_masks = _pair_masks_from_predicate(scenario, gnn_pred)
    return scenario, scenario_model, st_profile, st_keep_masks, monitored_line_whitelist, screen_setup_sec, screen_monitored_lines, rc_pred, rc_keep_masks, lp_pred, lp_keep_masks, gnn_pred, gnn_keep_masks


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
    model = None
    try:
        model = build_model(
            scenario=scenario_model,
            contingency_filter=None if (omit_explicit or mode.mode_family == "WARM_SR_LAZY") else contingency_filter,
            contingency_keep_masks=st_keep_masks,
            use_lazy_contingencies=bool(omit_explicit),
            radius_line_whitelist=monitored_line_whitelist,
        )
        model.update()
        return model
    except Exception:
        if model is not None:
            _dispose_model_quietly(model)
        raise


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


def _attach_lazy_if_needed(mode: ModeSpec, model, scenario_model, rc_keep_masks, lp_keep_masks, gnn_keep_masks) -> None:
    if not mode.use_lazy_callback:
        return
    lazy_keep_masks = rc_keep_masks or lp_keep_masks or gnn_keep_masks
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
    if os.environ.get("GUROBI_SEED"):
        model.Params.Seed = int(os.environ["GUROBI_SEED"])
    if os.environ.get("GUROBI_THREADS"):
        model.Params.Threads = int(os.environ["GUROBI_THREADS"])
    no_rel_heur_time = _resolve_no_rel_heur_time(mode)
    if no_rel_heur_time > 0.0:
        model.Params.NoRelHeurTime = float(no_rel_heur_time)
    if mode.use_lazy_callback:
        model.Params.LazyConstraints = 1


def _dispose_model_quietly(model) -> None:
    try:
        model.dispose()
    except Exception:
        pass


def _is_large_test_case(stage: str, case_folder: str) -> bool:
    return str(stage or "").strip().upper() == "TEST" and case_folder in CASES_LARGE


def _skip_large_mode_before_build(mode: ModeSpec, case_folder: str, stage: str) -> None:
    if not _is_large_test_case(stage, case_folder):
        return
    if mode.mode_family in {"ACTIVESET", "ACTIVESET_LAZY"} and _env_flag("PAPER_SKIP_LARGE_ACTIVESET", True):
        raise SkipSolve(
            "SKIPPED_TOO_LARGE",
            (
                f"Skipped {mode.mode_id} on {case_folder}: active-set candidate enumeration "
                "is too large for the overnight large-case run. Set PAPER_SKIP_LARGE_ACTIVESET=0 to force it."
            ),
        )
    if mode.mode_family == "WARM_SR_LAZY" and _env_flag("PAPER_SKIP_LARGE_SR_LAZY", True):
        raise SkipSolve(
            "SKIPPED_TOO_LARGE",
            (
                f"Skipped {mode.mode_id} on {case_folder}: stability-radius explicit model is too large "
                "for the overnight large-case run. Set PAPER_SKIP_LARGE_SR_LAZY=0 and raise the "
                "large model-size limits to force it."
            ),
        )


def _skip_large_model_if_needed(*, mode: ModeSpec, case_folder: str, stage: str, model, screen_setup_sec: float, screen_monitored_lines: Optional[int]) -> None:
    if not _is_large_test_case(stage, case_folder):
        return
    max_constrs = _env_int("PAPER_MAX_LARGE_TEST_MODEL_CONSTRS", DEFAULT_MAX_LARGE_TEST_MODEL_CONSTRS)
    max_nonzeros = _env_int("PAPER_MAX_LARGE_TEST_MODEL_NONZEROS", DEFAULT_MAX_LARGE_TEST_MODEL_NONZEROS)
    num_vars_root = int(getattr(model, "NumVars", 0) or 0)
    num_constrs_root = int(getattr(model, "NumConstrs", 0) or 0)
    num_nonzeros_root = int(getattr(model, "NumNZs", 0) or 0)
    too_many_constrs = max_constrs > 0 and num_constrs_root > max_constrs
    too_many_nonzeros = max_nonzeros > 0 and num_nonzeros_root > max_nonzeros
    if not (too_many_constrs or too_many_nonzeros):
        return
    message = (
        f"Skipped {mode.mode_id} on {case_folder}: model too large before optimize "
        f"(vars={num_vars_root}, constrs={num_constrs_root}, nonzeros={num_nonzeros_root}; "
        f"limits constrs<={max_constrs}, nonzeros<={max_nonzeros}). "
        "Raise PAPER_MAX_LARGE_TEST_MODEL_CONSTRS/PAPER_MAX_LARGE_TEST_MODEL_NONZEROS to force it."
    )
    metrics = {
        "screen_setup_sec": f"{float(screen_setup_sec):.6f}",
        "num_vars_root": num_vars_root,
        "num_constrs_root": num_constrs_root,
        "num_vars_final": num_vars_root,
        "num_constrs_final": num_constrs_root,
        "screen_monitored_lines": "" if screen_monitored_lines is None else int(screen_monitored_lines),
        "explicit_added_cont": int(getattr(model, "_explicit_total_cont_constraints", 0) or 0),
    }
    _dispose_model_quietly(model)
    raise SkipSolve("SKIPPED_TOO_LARGE", message, metrics)


def _solve_payload(instance_name: str, mode: ModeSpec, case_artifacts: CaseArtifacts, *, paths: Optional[RunPaths] = None, stage: str = "TEST") -> SolvePayload:
    _skip_large_mode_before_build(mode, case_artifacts.case_folder, stage)
    scenario, scenario_model, st_profile, st_keep_masks, monitored_line_whitelist, screen_setup_sec, screen_monitored_lines, rc_pred, rc_keep_masks, lp_pred, lp_keep_masks, gnn_pred, gnn_keep_masks = _prepare_build_components(mode, instance_name, case_artifacts)
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
    try:
        num_vars_root = int(getattr(model, "NumVars", 0) or 0)
        num_constrs_root = int(getattr(model, "NumConstrs", 0) or 0)
        _skip_large_model_if_needed(
            mode=mode,
            case_folder=case_artifacts.case_folder,
            stage=stage,
            model=model,
            screen_setup_sec=screen_setup_sec,
            screen_monitored_lines=screen_monitored_lines,
        )
        warm_start_applied, branch_hints_applied = _apply_training_artifacts(mode, model, scenario, scenario_model, instance_name, case_artifacts)
        _attach_lazy_if_needed(mode, model, scenario_model, rc_keep_masks, lp_keep_masks, gnn_keep_masks)
        _set_solver_params(model, mode, gurobi_log_path=None if paths is None else _gurobi_log_path(paths, stage, mode, instance_name))
    except Exception:
        _dispose_model_quietly(model)
        raise
    active_set_iters = 0
    active_set_added = 0
    active_set_dropped = 0
    try:
        if mode.mode_family in {"ACTIVESET", "ACTIVESET_LAZY"}:
            report = optimize_with_active_set(model, scenario, ActiveSetConfig(
                time_limit=int(mode.time_limit_sec), mip_gap=float(mode.mip_gap), lodf_tol=float(mode.lazy_lodf_tol), isf_tol=float(mode.lazy_isf_tol),
                violation_tol=float(mode.lazy_viol_tol), batch_size=int(mode.active_set_batch), max_rounds=int(mode.active_set_max_rounds),
                cleanup_inactive=bool(mode.active_set_cleanup), cleanup_tol=float(mode.active_set_cleanup_tol),
                no_rel_heur_time=float(_resolve_no_rel_heur_time(mode)), output_flag=1,
            ))
            active_set_iters = int(getattr(report, "iterations", 0) or 0)
            active_set_added = int(getattr(report, "added_constraints", 0) or 0)
            active_set_dropped = int(getattr(report, "dropped_constraints", 0) or 0)
            model._paper_runtime_sec = float(getattr(report, "total_runtime", 0.0) or 0.0)
        else:
            callback = getattr(model, "_lazy_contingency_callback", None)
            model.optimize() if callback is None else model.optimize(callback)
    except Exception:
        _dispose_model_quietly(model)
        raise
    try:
        constr_total_cont = None
        constr_kept_cont = None
        constr_ratio_cont_explicit = None
        if mode.mode_family in {"WARM_PRUNE","WARM_LPSCREEN","WARM_PRUNE_LAZY","WARM_LPSCREEN_LAZY","WARM_SR_LAZY","STREDUCE","STREDUCE_LAZY"} or gnn_keep_masks is not None:
            estimate_pred = rc_pred if mode.mode_family == "WARM_PRUNE" else lp_pred if mode.mode_family == "WARM_LPSCREEN" else None
            estimate_masks = rc_keep_masks if mode.mode_family == "WARM_PRUNE_LAZY" else lp_keep_masks if mode.mode_family == "WARM_LPSCREEN_LAZY" else st_keep_masks if mode.mode_family in {"STREDUCE","STREDUCE_LAZY"} else gnn_keep_masks
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
    except Exception:
        _dispose_model_quietly(model)
        raise


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
    payload: Optional[SolvePayload] = None
    _mark_solve_started(
        paths=paths,
        run_id=run_id,
        stage=stage,
        instance_name=instance_name,
        mode=mode,
        case_folder=case_artifacts.case_folder,
    )
    try:
        payload = _solve_payload(instance_name, mode, case_artifacts, paths=paths, stage=stage)
        candidate_solution_json = _save_candidate_solution(paths, stage, mode, instance_name, payload)
        provisional = _build_success_row(run_id=run_id, stage=stage, case_folder=case_artifacts.case_folder, instance_name=instance_name, mode=mode, started_at=started_at, payload=payload, candidate_solution_json=candidate_solution_json, train_solution_json=None)
        train_solution_json = _save_train_solution(paths, instance_name, payload) if _is_train_artifact_eligible(provisional) else None
        row = _build_success_row(run_id=run_id, stage=stage, case_folder=case_artifacts.case_folder, instance_name=instance_name, mode=mode, started_at=started_at, payload=payload, candidate_solution_json=candidate_solution_json, train_solution_json=train_solution_json)
        return row
    except SkipSolve as exc:
        logger.warning(
            "Solve skipped: stage=%s case=%s instance=%s mode=%s reason=%s",
            stage,
            case_artifacts.case_folder,
            instance_name,
            mode.mode_id,
            str(exc),
        )
        return _make_skip_row(
            run_id=run_id,
            stage=stage,
            case_folder=case_artifacts.case_folder,
            instance_name=instance_name,
            mode=mode,
            started_at=started_at,
            status=exc.status,
            error_message=str(exc),
            metrics=exc.metrics,
        )
    except Exception as exc:
        logger.exception("Solve failed: stage=%s case=%s instance=%s mode=%s", stage, case_artifacts.case_folder, instance_name, mode.mode_id)
        return _make_error_row(run_id=run_id, stage=stage, case_folder=case_artifacts.case_folder, instance_name=instance_name, mode=mode, started_at=started_at, error_message=str(exc))
    finally:
        if payload is not None:
            _dispose_model_quietly(payload.model)
        gc.collect()


def _record_row(paths: RunPaths, state: Dict, row: Dict) -> None:
    _append_result_row(paths, row)
    completed = set(state.get("completed_keys", []))
    if str(row.get("status") or "").strip().upper() != "ERROR":
        completed.add(row["result_key"])
    state["completed_keys"] = sorted(completed)
    _save_state(paths, state)
    _mark_result_recorded(paths, row)


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


def _bootstrap_case_artifacts(case_folder: str) -> CaseArtifacts:
    return CaseArtifacts(
        case_folder=case_folder,
        train_names=set(),
        warm_provider=WarmStartProvider(case_folder=case_folder, coverage_threshold=0.0, train_ratio=1.0, val_ratio=0.0, split_seed=EXPERIMENT_SEED),
        redundancy_provider=RCProvider(case_folder=case_folder, train_ratio=1.0, val_ratio=0.0, split_seed=EXPERIMENT_SEED),
        commitment_hints=CommitmentHints(case_folder=case_folder),
        gnn_screener=GNNLineScreener(case_folder=case_folder),
        gru_warmstart=GRUDispatchWarmStart(case_folder=case_folder),
        streduction=STReductionProvider(case_folder=case_folder),
    )


def _run_train_stage(
    *,
    run_id: str,
    case_folder: str,
    train_instances: Sequence[str],
    paths: RunPaths,
    state: Dict,
    train_time_limit_sec: Optional[int] = None,
) -> Optional[CaseArtifacts]:
    logger.info("TRAIN start: %s", case_folder)
    base_tl = int(train_time_limit_sec or TRAIN_TL_SEC_BY_CASE.get(case_folder, 7200))
    train_mode = replace(TRAIN_BASE_MODE, time_limit_sec=base_tl)
    bootstrap = _bootstrap_case_artifacts(case_folder)
    completed = set(state.get("completed_keys", []))
    completed_by_logical_key = _load_completed_result_lookup_by_logical_key(paths.csv_path)
    jobs = []
    skipped = 0
    for instance_name in train_instances:
        key = _result_key("TRAIN", train_mode.mode_id, instance_name, train_mode.time_limit_sec)
        logical_key = _logical_result_key("TRAIN", train_mode.mode_id, instance_name)
        if key in completed or logical_key in completed_by_logical_key:
            skipped += 1
            continue
        jobs.append((instance_name, key, logical_key))
    logger.info("TRAIN jobs: case=%s skipped=%d pending=%d", case_folder, skipped, len(jobs))
    progress = tqdm(jobs, desc=f"{case_folder} TRAIN", unit="solve", dynamic_ncols=True)
    ok_count = 0
    fail_count = 0
    for instance_name, key, logical_key in progress:
        progress.set_postfix_str(instance_name.rsplit("/", 1)[-1])
        started = time.time()
        row = _run_single_solve(run_id=run_id, stage="TRAIN", instance_name=instance_name, mode=train_mode, case_artifacts=bootstrap, paths=paths)
        elapsed = time.time() - started
        _record_row(paths, state, row)
        completed.add(key)
        if str(row.get("status") or "").strip().upper() != "ERROR":
            completed_by_logical_key[logical_key] = row
        if int(row.get("pass") or 0) == 1:
            ok_count += 1
        else:
            fail_count += 1
        progress.set_postfix(ok=ok_count, fail=fail_count, skip=skipped, sec=f"{elapsed:.1f}")
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


def _run_test_modes(
    *,
    run_id: str,
    case_folder: str,
    test_instances: Sequence[str],
    modes: Sequence[ModeSpec],
    case_artifacts: CaseArtifacts,
    paths: RunPaths,
    state: Dict,
    time_limit_sec: int,
    force_rerun: bool = False,
) -> List[Dict]:
    completed = set(state.get("completed_keys", []))
    case_rows: List[Dict] = []
    result_lookup = _load_result_lookup(paths.csv_path)
    completed_by_logical_key = _load_completed_result_lookup_by_logical_key(paths.csv_path)
    jobs = []
    skipped = 0
    for mode in modes:
        mode_tl = replace(mode, time_limit_sec=int(time_limit_sec))
        for instance_name in test_instances:
            key = _result_key("TEST", mode_tl.mode_id, instance_name, mode_tl.time_limit_sec)
            logical_key = _logical_result_key("TEST", mode_tl.mode_id, instance_name)
            if not force_rerun and (key in completed or logical_key in completed_by_logical_key):
                row = result_lookup.get(key) or completed_by_logical_key.get(logical_key)
                if row is not None:
                    case_rows.append(row)
                skipped += 1
                continue
            jobs.append((mode_tl, instance_name, key, logical_key))
    logger.info("TEST jobs: case=%s skipped=%d pending=%d", case_folder, skipped, len(jobs))
    progress = tqdm(jobs, desc=f"{case_folder} TEST", unit="solve", dynamic_ncols=True)
    ok_count = 0
    fail_count = 0
    for mode_tl, instance_name, key, logical_key in progress:
        progress.set_postfix_str(f"{mode_tl.mode_id} {instance_name.rsplit('/', 1)[-1]}")
        started = time.time()
        row = _run_single_solve(run_id=run_id, stage="TEST", instance_name=instance_name, mode=mode_tl, case_artifacts=case_artifacts, paths=paths)
        elapsed = time.time() - started
        case_rows.append(row)
        _record_row(paths, state, row)
        completed.add(key)
        result_lookup[key] = row
        if str(row.get("status") or "").strip().upper() != "ERROR":
            completed_by_logical_key[logical_key] = row
        if int(row.get("pass") or 0) == 1:
            ok_count += 1
        else:
            fail_count += 1
        progress.set_postfix(
            mode=mode_tl.mode_id,
            ok=ok_count,
            fail=fail_count,
            skip=skipped,
            sec=f"{elapsed:.1f}",
        )
    return case_rows


def _run_small_test_stage(
    *,
    run_id: str,
    case_folder: str,
    test_instances: Sequence[str],
    modes: Sequence[ModeSpec],
    case_artifacts: CaseArtifacts,
    paths: RunPaths,
    state: Dict,
    time_limit_sec: int,
    force_rerun: bool = False,
) -> None:
    logger.info("TEST start (small): %s", case_folder)
    _run_test_modes(
        run_id=run_id,
        case_folder=case_folder,
        test_instances=test_instances,
        modes=modes,
        case_artifacts=case_artifacts,
        paths=paths,
        state=state,
        time_limit_sec=int(time_limit_sec),
        force_rerun=force_rerun,
    )


def _run_large_test_stage_staged(
    *,
    run_id: str,
    case_folder: str,
    test_instances: Sequence[str],
    modes: Sequence[ModeSpec],
    case_artifacts: CaseArtifacts,
    paths: RunPaths,
    state: Dict,
    time_limit_sec: int,
    force_rerun: bool = False,
) -> Tuple[List[Dict], List[str], int]:
    logger.info("TEST start (medlarge staged): %s", case_folder)
    selected_modes = list(modes)
    current_tl = int(time_limit_sec)
    case_rows = _run_test_modes(
        run_id=run_id,
        case_folder=case_folder,
        test_instances=test_instances,
        modes=selected_modes,
        case_artifacts=case_artifacts,
        paths=paths,
        state=state,
        time_limit_sec=current_tl,
        force_rerun=force_rerun,
    )
    case_state = state.setdefault("case_state", {}).setdefault(case_folder, {})
    case_state["test_time_limit_sec"] = current_tl
    case_state["selected_mode_ids"] = [mode.mode_id for mode in selected_modes]
    _save_state(paths, state)
    return case_rows, [mode.mode_id for mode in selected_modes], current_tl


def _select_cases(profile: str) -> List[str]:
    if profile == "small":
        return list(CASES_SMALL)
    if profile == "medlarge":
        return list(CASES_MEDLARGE)
    if profile in {"large", "large6800"}:
        return list(CASES_LARGE)
    return list(CASES_ALL)


def _run_case(
    *,
    run_id: str,
    case_folder: str,
    paths: RunPaths,
    state: Dict,
    mode_ids: Optional[Sequence[str]] = None,
    train_dates: Optional[Sequence[str]] = None,
    test_dates: Optional[Sequence[str]] = None,
    only_test_instances: Optional[Sequence[str]] = None,
    instance_policy: str = "paper",
    limit_train: int = 0,
    limit_test: int = 0,
    train_time_limit_sec: Optional[int] = None,
    test_time_limit_sec: Optional[int] = None,
    force_rerun: bool = False,
) -> List[str]:
    train_instances, test_instances = _case_instances(
        case_folder,
        train_dates=train_dates,
        test_dates=test_dates,
        only_test_instances=only_test_instances,
        instance_policy=instance_policy,
        limit_train=limit_train,
        limit_test=limit_test,
    )
    if bool(state.get("case_state", {}).get(case_folder, {}).get("skipped", False)):
        logger.warning("CASE skipped (state): %s", case_folder)
        return []
    planned_modes = _planned_modes_for_case(
        case_folder=case_folder,
        mode_ids=mode_ids,
    )
    needs_artifacts = _modes_require_training_artifacts(planned_modes)
    try:
        if not needs_artifacts:
            artifacts = _bootstrap_case_artifacts(case_folder)
        else:
            # Always pass through the resume-aware TRAIN stage. It skips the
            # latest successful logical cells but retries missing/ERROR cells;
            # an old artifacts_built_cases marker must not hide such holes.
            artifacts = _run_train_stage(
                run_id=run_id,
                case_folder=case_folder,
                train_instances=train_instances,
                paths=paths,
                state=state,
                train_time_limit_sec=train_time_limit_sec,
            )
    except Exception as exc:
        logger.exception("CASE failed during TRAIN/artifact build: %s", case_folder)
        case_state = state.setdefault("case_state", {}).setdefault(case_folder, {})
        case_state["skipped"] = True
        case_state["skip_reason"] = f"TRAIN_OR_ARTIFACT_ERROR: {exc}"
        _save_state(paths, state)
        return []
    if artifacts is None:
        return []
    if case_folder in CASES_SMALL:
        _run_small_test_stage(
            run_id=run_id,
            case_folder=case_folder,
            test_instances=test_instances,
            modes=planned_modes,
            case_artifacts=artifacts,
            paths=paths,
            state=state,
            time_limit_sec=int(test_time_limit_sec or TEST_TL_INIT_SEC_BY_CASE.get(case_folder, 7200)),
            force_rerun=force_rerun,
        )
        return []
    _, mode_ids_run, _ = _run_large_test_stage_staged(
        run_id=run_id,
        case_folder=case_folder,
        test_instances=test_instances,
        modes=planned_modes,
        case_artifacts=artifacts,
        paths=paths,
        state=state,
        time_limit_sec=int(test_time_limit_sec or TEST_TL_INIT_SEC_BY_CASE.get(case_folder, 7200)),
        force_rerun=force_rerun,
    )
    state["alive_mode_ids"] = list(mode_ids_run)
    _save_state(paths, state)
    return list(mode_ids_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic SCUC experiment pipeline.")
    parser.add_argument("--run-id", required=True, help="Result folder under results/.")
    parser.add_argument(
        "--profile",
        choices=("small", "medlarge", "large", "large6800", "all"),
        default="all",
        help="Which case profile to run. `large`/`large6800` covers MATPOWER cases from case1354 through case6515.",
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
        "--only-modes",
        nargs="+",
        default=None,
        help="Run only these mode IDs. Accepts separate args or one whitespace/comma-separated string.",
    )
    parser.add_argument(
        "--train-dates",
        nargs="+",
        default=None,
        help="Override training dates or full train instance names. Dates are appended to each case folder.",
    )
    parser.add_argument(
        "--test-dates",
        nargs="+",
        default=None,
        help="Override test dates or full test instance names. Dates are appended to each case folder.",
    )
    parser.add_argument(
        "--only-test-instances",
        nargs="+",
        default=None,
        help="Run exactly these test instances. A date-only value is appended to the current case folder.",
    )
    parser.add_argument(
        "--instance-policy",
        choices=("paper", "all-local", "remote"),
        default="paper",
        help="`paper` uses fixed TRAIN_DATES_24/TEST_DATES_6; `all-local` tests cached local instances; `remote` lists remote case instances.",
    )
    parser.add_argument("--limit-train", type=int, default=0, help="Limit train instances after selection; 0 means no limit.")
    parser.add_argument("--limit-test", type=int, default=0, help="Limit test instances after selection; 0 means no limit.")
    parser.add_argument("--train-time-limit", type=int, default=None, help="Override per-case TRAIN time limit in seconds.")
    parser.add_argument("--test-time-limit", type=int, default=None, help="Override per-case TEST time limit in seconds.")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Append fresh TEST rows even when the result key already exists. Analysis keeps the latest row per case-instance-mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved cases, train/test instances, and modes without solving.",
    )
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = parse_args()
    _set_global_determinism(EXPERIMENT_SEED)
    paths = _configure_run_paths(args.run_id, args.resume)
    medlarge_catalog = _medlarge_catalog("full")
    medlarge_mode_ids = [mode.mode_id for mode in medlarge_catalog]
    state = _load_state(paths, args.run_id, args.resume, medlarge_mode_ids=medlarge_mode_ids)
    state["medlarge_modes_kind"] = "full"
    state["comparison_plan"] = "explicit"
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
    if args.dry_run:
        for case_folder in cases:
            train_instances, test_instances = _case_instances(
                case_folder,
                train_dates=args.train_dates,
                test_dates=args.test_dates,
                only_test_instances=args.only_test_instances,
                instance_policy=args.instance_policy,
                limit_train=int(args.limit_train),
                limit_test=int(args.limit_test),
            )
            planned_modes = _planned_modes_for_case(
                case_folder=case_folder,
                mode_ids=args.only_modes,
            )
            logger.info(
                "DRY-RUN case=%s train=%d test=%d modes=%s needs_artifacts=%s",
                case_folder,
                len(train_instances),
                len(test_instances),
                [mode.mode_id for mode in planned_modes],
                _modes_require_training_artifacts(planned_modes),
            )
            logger.info("DRY-RUN test_instances=%s", test_instances)
        return
    for case_folder in cases:
        logger.info("CASE start: %s", case_folder)
        alive_mode_ids = _run_case(
            run_id=args.run_id,
            case_folder=case_folder,
            paths=paths,
            state=state,
            mode_ids=args.only_modes,
            train_dates=args.train_dates,
            test_dates=args.test_dates,
            only_test_instances=args.only_test_instances,
            instance_policy=args.instance_policy,
            limit_train=int(args.limit_train),
            limit_test=int(args.limit_test),
            train_time_limit_sec=args.train_time_limit,
            test_time_limit_sec=args.test_time_limit,
            force_rerun=bool(args.force_rerun),
        )
        logger.info("CASE complete: %s", case_folder)
    logger.info("Run complete: %s", args.run_id)


if __name__ == "__main__":
    main()
