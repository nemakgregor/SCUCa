"""
Benchmark pipeline: compare raw Gurobi vs ML-based (warm-start + redundant-constraint pruning).

What it does:
- Stage 1 (TRAIN): solve with raw Gurobi, save outputs (skip if --skip-existing).
- Stage 2: pretrain warm-start index from TRAIN outputs.
- Stage 2.5 (optional): pretrain redundancy index from TRAIN outputs.
- Stage 3 (TEST): for each instance, solve RAW and WARM(+optional pruning);
  verify feasibility and write:
    • a per-case results CSV under src/data/output/<case>/compare_<tag>_<timestamp>.csv
    • a general logs CSV under src/data/logs/compare_logs_<tag>_<timestamp>.csv
      with (instance, split, method, num vars, num constrs, runtime, max constraint violation, etc.)

Notes:
- Redundancy pruning controls (all OFF unless you pass --rc-enable):
    --rc-enable               Enable pruning
    --rc-thr-abs 20.0         Absolute margin threshold [MW]
    --rc-thr-rel 0.10         Relative margin threshold (fraction of emergency limit)
    --rc-use-train-db         Restrict redundancy k-NN to TRAIN split (default True)
"""

from __future__ import annotations

import argparse
import csv
import math
import time
import requests
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gurobipy import GRB

from src.data_preparation.params import DataParams
from src.data_preparation.read_data import read_benchmark
from src.optimization_model.SCUC_solver.scuc_model_builder import build_model
from src.optimization_model.SCUC_solver.solve_instances import (
    list_remote_instances,
    list_local_cached_instances,
)
from src.optimization_model.helpers.save_json_solution import (
    save_solution_as_json,
    compute_output_path,
)
from src.optimization_model.helpers.verify_solution import (
    verify_solution,
    verify_solution_to_log,
)
from src.optimization_model.helpers.run_utils import allocate_run_id, make_log_filename
from src.ml_models.warm_start import WarmStartProvider, _hash01
from src.ml_models.redundant_constraints import (
    RedundancyProvider as RCProvider,
)  # ML-based constraint pruning provider


def _status_str(code: int) -> str:
    return DataParams.SOLVER_STATUS_STR.get(code, f"STATUS_{code}")


def _case_tag(case_folder: str) -> str:
    t = case_folder.strip().strip("/\\").replace("\\", "/")
    return "".join(ch if ch.isalnum() else "_" for ch in t).strip("_").lower()


def _list_case_instances(case_folder: str) -> List[str]:
    items = list_remote_instances(
        include_filters=[case_folder], roots=["matpower", "test"], max_depth=4
    )
    items = [x for x in items if x.startswith(case_folder)]
    if not items:
        items = list_local_cached_instances(include_filters=[case_folder])
        items = [x for x in items if x.startswith(case_folder)]
    return sorted(set(items))


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


@dataclass
class SolveResult:
    instance_name: str
    split: str
    method: str  # raw | warm
    status: str
    status_code: int
    runtime_sec: float
    mip_gap: Optional[float]
    obj_val: Optional[float]
    obj_bound: Optional[float]
    nodes: Optional[float]
    feasible_ok: Optional[bool]
    warm_start_applied_vars: Optional[int] = None
    # New fields for logging/analysis
    num_vars: Optional[int] = None
    num_bin_vars: Optional[int] = None
    num_int_vars: Optional[int] = None
    num_constrs: Optional[int] = None
    max_constraint_violation: Optional[float] = None


def _metrics_from_model(
    model,
) -> Tuple[
    int, float, Optional[float], Optional[float], Optional[float], Optional[float]
]:
    code = int(getattr(model, "Status", -1))
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
    return code, runtime, mip_gap, obj_val, obj_bound, nodes


def _model_size(model) -> Tuple[int, int, int, int]:
    """
    Return (num_vars, num_bin_vars, num_int_vars, num_constrs)
    """
    try:
        nvars = int(getattr(model, "NumVars", 0))
    except Exception:
        nvars = 0
    try:
        nbin = int(getattr(model, "NumBinVars", 0))
    except Exception:
        nbin = 0
    try:
        nint = int(getattr(model, "NumIntVars", 0))
    except Exception:
        nint = 0
    try:
        ncons = int(getattr(model, "NumConstrs", 0))
    except Exception:
        ncons = 0
    return nvars, nbin, nint, ncons


def _max_constraint_violation(checks) -> Optional[float]:
    """
    Compute max violation among constraint checks (IDs starting with 'C-').
    Returns None if checks missing; 0.0 if all within tolerance.
    """
    if not checks:
        return None
    vals: List[float] = []
    for c in checks:
        try:
            if isinstance(c.idx, str) and c.idx.startswith("C-"):
                v = float(c.value)
                if math.isfinite(v):
                    vals.append(v)
        except Exception:
            continue
    return max(vals) if vals else 0.0


def _save_logs_if_requested(sc, model, save_logs: bool) -> None:
    if not save_logs:
        return
    run_id = allocate_run_id(sc.name or "scenario")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        sol_fname = make_log_filename("solution", sc.name, run_id, ts)
        _ = Path(DataParams._LOGS / sol_fname)
        from src.optimization_model.helpers.save_solution import (
            save_solution_to_log as _save,
        )

        _save(sc, model, filename=sol_fname)
    except Exception:
        pass
    try:
        ver_fname = make_log_filename("verify", sc.name, run_id, ts)
        verify_solution_to_log(sc, model, filename=ver_fname)
    except Exception:
        pass


def _instance_cache_path_and_url(instance_name: str) -> Tuple[Path, str]:
    gz_name = f"{instance_name}.json.gz"
    local_path = (DataParams._CACHE / gz_name).resolve()
    url = f"{DataParams.INSTANCES_URL.rstrip('/')}/{gz_name}"
    return local_path, url


def _robust_download(instance_name: str, attempts: int, timeout: int) -> bool:
    """
    Try to download instance_name.json.gz to the input cache with retry and timeout.
    Returns True on success, False on final failure.
    """
    local_path, url = _instance_cache_path_and_url(instance_name)
    if local_path.is_file():
        return True

    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")

    backoff_base = 2.0
    for k in range(1, max(1, attempts) + 1):
        try:
            with requests.get(url, stream=True, timeout=max(1, int(timeout))) as r:
                r.raise_for_status()
                with tmp_path.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
            tmp_path.replace(local_path)
            return True
        except Exception:
            if k < attempts:
                sleep_s = min(60.0, backoff_base ** (k - 1))
                time.sleep(sleep_s)
            else:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                return False
    return False


def _prepare_results_csv(case_folder: str) -> Path:
    """
    Create CSV file (under src/data/output/<case>/) and write header. Return path.
    """
    tag = _case_tag(case_folder)
    out_dir = (DataParams._OUTPUT / case_folder).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"compare_{tag}_{ts}.csv"

    header = [
        "timestamp",
        "instance_name",
        "split",
        "method",
        "status",
        "status_code",
        "runtime_sec",
        "mip_gap",
        "obj_val",
        "obj_bound",
        "nodes",
        "feasible_ok",
        "warm_start_applied_vars",
        "num_vars",
        "num_bin_vars",
        "num_int_vars",
        "num_constrs",
        "max_constraint_violation",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
    return path


def _append_result_to_csv(csv_path: Path, r: SolveResult) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                now,
                r.instance_name,
                r.split,
                r.method,
                r.status,
                r.status_code,
                f"{r.runtime_sec:.6f}",
                "" if r.mip_gap is None else f"{r.mip_gap:.8f}",
                "" if r.obj_val is None else f"{r.obj_val:.6f}",
                "" if r.obj_bound is None else f"{r.obj_bound:.6f}",
                "" if r.nodes is None else f"{r.nodes:.0f}",
                "" if r.feasible_ok is None else ("OK" if r.feasible_ok else "FAIL"),
                ""
                if r.warm_start_applied_vars is None
                else int(r.warm_start_applied_vars),
                "" if r.num_vars is None else int(r.num_vars),
                "" if r.num_bin_vars is None else int(r.num_bin_vars),
                "" if r.num_int_vars is None else int(r.num_int_vars),
                "" if r.num_constrs is None else int(r.num_constrs),
                ""
                if r.max_constraint_violation is None
                else f"{float(r.max_constraint_violation):.8f}",
            ]
        )


def _prepare_logs_csv(case_folder: str) -> Path:
    """
    Create a general logs CSV under src/data/logs and write header. Return path.
    Includes general info that the user requested (instance, method, sizes, times, violations).
    """
    tag = _case_tag(case_folder)
    logs_dir = DataParams._LOGS.resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = logs_dir / f"compare_logs_{tag}_{ts}.csv"
    header = [
        "timestamp",
        "instance_name",
        "split",
        "method",
        "status",
        "runtime_sec",
        "obj_val",
        "num_vars",
        "num_constrs",
        "num_bin_vars",
        "num_int_vars",
        "max_constraint_violation",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(header)
    return path


def _append_result_to_logs_csv(csv_path: Path, r: SolveResult) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                now,
                r.instance_name,
                r.split,
                r.method,
                r.status,
                f"{r.runtime_sec:.6f}",
                "" if r.obj_val is None else f"{r.obj_val:.6f}",
                "" if r.num_vars is None else int(r.num_vars),
                "" if r.num_constrs is None else int(r.num_constrs),
                "" if r.num_bin_vars is None else int(r.num_bin_vars),
                "" if r.num_int_vars is None else int(r.num_int_vars),
                ""
                if r.max_constraint_violation is None
                else f"{float(r.max_constraint_violation):.8f}",
            ]
        )


def _build_contingency_filter(
    rc: Optional[RCProvider],
    sc,
    instance_name: str,
    enable: bool,
    thr_abs: float,
    thr_rel: float,
    use_train_db: bool,
) -> Optional[callable]:
    if not enable or rc is None:
        return None
    try:
        res = rc.make_filter_for_instance(
            sc,
            instance_name,
            thr_abs=thr_abs,
            thr_rel=thr_rel,
            use_train_index_only=use_train_db,
            exclude_self=True,
        )
        if res is None:
            return None
        predicate, stats = res
        print(
            f"[redundancy] Using neighbor {stats.get('neighbor')} (dist={stats.get('distance'):.3f}); "
            f"constraints skipped will be counted in builder logs."
        )
        return predicate
    except Exception:
        return None


def _solve_raw(
    instance_name: str,
    time_limit: int,
    mip_gap: float,
    split: str,
    save_logs: bool,
    skip_existing: bool,
    download_attempts: int,
    download_timeout: int,
    *,
    rc_provider: Optional[RCProvider] = None,
    rc_enable: bool = False,
    rc_thr_abs: float = 20.0,
    rc_thr_rel: float = 0.10,
    rc_use_train_db: bool = True,
) -> SolveResult:
    out_json_path = compute_output_path(instance_name)
    if skip_existing and out_json_path.is_file():
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="raw",
            status="SKIPPED_EXISTING",
            status_code=-2,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )

    ok_dl = _robust_download(
        instance_name=instance_name,
        attempts=max(1, int(download_attempts)),
        timeout=max(1, int(download_timeout)),
    )
    if not ok_dl:
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="raw",
            status="DOWNLOAD_FAIL",
            status_code=-3,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )

    try:
        inst = read_benchmark(instance_name, quiet=True)
    except Exception:
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="raw",
            status="READ_FAIL",
            status_code=-4,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )

    sc = inst.deterministic
    cont_filter = _build_contingency_filter(
        rc_provider,
        sc,
        instance_name,
        rc_enable,
        rc_thr_abs,
        rc_thr_rel,
        rc_use_train_db,
    )
    model = build_model(scenario=sc, contingency_filter=cont_filter)
    try:
        model.Params.OutputFlag = 1
        model.Params.MIPGap = mip_gap
        model.Params.TimeLimit = time_limit
        model.Params.NumericFocus = 1
    except Exception:
        pass
    model.optimize()

    try:
        save_solution_as_json(sc, model, instance_name=instance_name)
    except Exception:
        pass

    feasible_ok = None
    max_viol = None
    try:
        feasible_ok, checks, _ = verify_solution(sc, model)
        max_viol = _max_constraint_violation(checks)
    except Exception:
        feasible_ok = None
        max_viol = None

    _save_logs_if_requested(sc, model, save_logs)

    code, runtime, gap, obj, bound, nodes = _metrics_from_model(model)
    nvars, nbin, nint, ncons = _model_size(model)
    return SolveResult(
        instance_name=instance_name,
        split=split,
        method="raw",
        status=_status_str(code),
        status_code=code,
        runtime_sec=runtime,
        mip_gap=gap,
        obj_val=obj,
        obj_bound=bound,
        nodes=nodes,
        feasible_ok=feasible_ok,
        warm_start_applied_vars=None,
        num_vars=nvars,
        num_bin_vars=nbin,
        num_int_vars=nint,
        num_constrs=ncons,
        max_constraint_violation=max_viol,
    )


def _solve_warm(
    instance_name: str,
    wsp: WarmStartProvider,
    time_limit: int,
    mip_gap: float,
    split: str,
    warm_mode: str,
    save_logs: bool,
    skip_existing: bool,
    download_attempts: int,
    download_timeout: int,
    *,
    rc_provider: Optional[RCProvider] = None,
    rc_enable: bool = False,
    rc_thr_abs: float = 20.0,
    rc_thr_rel: float = 0.10,
    rc_use_train_db: bool = True,
) -> SolveResult:
    out_json_path = compute_output_path(instance_name)
    if skip_existing and out_json_path.is_file():
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="warm",
            status="SKIPPED_EXISTING",
            status_code=-2,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )

    ok_dl = _robust_download(
        instance_name=instance_name,
        attempts=max(1, int(download_attempts)),
        timeout=max(1, int(download_timeout)),
    )
    if not ok_dl:
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="warm",
            status="DOWNLOAD_FAIL",
            status_code=-3,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )

    try:
        inst = read_benchmark(instance_name, quiet=True)
    except Exception:
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="warm",
            status="READ_FAIL",
            status_code=-4,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )
    sc = inst.deterministic

    try:
        wsp.generate_and_save_warm_start(
            instance_name, use_train_index_only=True, exclude_self=True, auto_fix=True
        )
    except Exception:
        pass

    cont_filter = _build_contingency_filter(
        rc_provider,
        sc,
        instance_name,
        rc_enable,
        rc_thr_abs,
        rc_thr_rel,
        rc_use_train_db,
    )
    model = build_model(scenario=sc, contingency_filter=cont_filter)

    mode = warm_mode.strip().lower()
    if mode == "fixed":
        mode = "repair"
    applied = 0
    try:
        applied = wsp.apply_warm_start_to_model(model, sc, instance_name, mode=mode)
    except Exception:
        applied = 0

    try:
        model.Params.OutputFlag = 1
        model.Params.MIPGap = mip_gap
        model.Params.TimeLimit = time_limit
        model.Params.NumericFocus = 1
    except Exception:
        pass

    model.optimize()

    try:
        save_solution_as_json(sc, model, instance_name=instance_name)
    except Exception:
        pass

    feasible_ok = None
    max_viol = None
    try:
        feasible_ok, checks, _ = verify_solution(sc, model)
        max_viol = _max_constraint_violation(checks)
    except Exception:
        feasible_ok = None
        max_viol = None

    _save_logs_if_requested(sc, model, save_logs)

    code, runtime, gap, obj, bound, nodes = _metrics_from_model(model)
    nvars, nbin, nint, ncons = _model_size(model)
    return SolveResult(
        instance_name=instance_name,
        split=split,
        method="warm",
        status=_status_str(code),
        status_code=code,
        runtime_sec=runtime,
        mip_gap=gap,
        obj_val=obj,
        obj_bound=bound,
        nodes=nodes,
        feasible_ok=feasible_ok,
        warm_start_applied_vars=applied,
        num_vars=nvars,
        num_bin_vars=nbin,
        num_int_vars=nint,
        num_constrs=ncons,
        max_constraint_violation=max_viol,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Compare raw Gurobi vs warm-start (+ optional redundancy pruning) on TRAIN/TEST for a case."
    )
    ap.add_argument("--case", required=True, help="Case folder, e.g., matpower/case118")
    ap.add_argument(
        "--time-limit",
        type=int,
        default=600,
        help="Time limit [s] for both RAW and WARM",
    )
    ap.add_argument(
        "--mip-gap",
        type=float,
        default=0.05,
        help="Relative MIP gap for both RAW and WARM",
    )
    ap.add_argument(
        "--train-ratio", type=float, default=0.70, help="Train ratio for split"
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation ratio; remainder is test",
    )
    ap.add_argument("--seed", type=int, default=42, help="Split seed")
    ap.add_argument(
        "--limit-train",
        type=int,
        default=0,
        help="Limit number of TRAIN instances (0 => no limit)",
    )
    ap.add_argument(
        "--limit-test",
        type=int,
        default=0,
        help="Limit number of TEST instances (0 => no limit)",
    )
    ap.add_argument(
        "--save-logs",
        action="store_true",
        default=False,
        help="Save human logs to src/data/logs (solution and verification reports)",
    )
    ap.add_argument(
        "--warm-mode",
        choices=["fixed", "commit-only", "as-is"],
        default="fixed",
        help="Warm-start application mode for evaluation",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="TRAIN ONLY: skip instances that already have a JSON solution in src/data/output",
    )
    ap.add_argument(
        "--download-attempts",
        type=int,
        default=3,
        help="Number of attempts to download a missing instance (default: 3)",
    )
    ap.add_argument(
        "--download-timeout",
        type=int,
        default=60,
        help="Per-attempt HTTP timeout for instance download in seconds (default: 60)",
    )
    # Redundancy pruning
    ap.add_argument(
        "--rc-enable",
        action="store_true",
        default=False,
        help="Enable ML-based redundant contingency pruning (OFF by default)",
    )
    ap.add_argument(
        "--rc-thr-abs",
        type=float,
        default=20.0,
        help="Absolute margin threshold [MW] to prune a constraint (default 20 MW)",
    )
    ap.add_argument(
        "--rc-thr-rel",
        type=float,
        default=0.50,
        help="Relative margin threshold (fraction of F_em) to prune (default 0.50)",
    )
    ap.add_argument(
        "--rc-use-train-db",
        action="store_true",
        default=True,
        help="Restrict redundancy k-NN to TRAIN split (default True)",
    )

    args = ap.parse_args()

    case_folder = args.case.strip().strip("/\\").replace("\\", "/")
    all_instances = _list_case_instances(case_folder)
    if not all_instances:
        print(f"No instances found for {case_folder}.")
        return

    train, val, test = _split_instances(
        all_instances, args.train_ratio, args.val_ratio, args.seed
    )
    if not train:
        print("Empty TRAIN split; cannot pretrain warm-start index. Aborting.")
        return
    if not test:
        print("Empty TEST split; nothing to compare. Aborting.")
        return

    if args.limit_train and args.limit_train > 0:
        train = train[: args.limit_train]
    if args.limit_test and args.limit_test > 0:
        test = test[: args.limit_test]

    print(f"Case: {case_folder}")
    print(f"- Train: {len(train)}")
    print(f"- Val  : {len(val)}")
    print(f"- Test : {len(test)}")
    if args.skip_existing:
        print(
            "Note: --skip-existing applies to TRAIN stage only. TEST runs will re-solve even if outputs exist."
        )

    # Prepare CSVs
    csv_path = _prepare_results_csv(case_folder)
    logs_csv_path = _prepare_logs_csv(case_folder)
    print(f"Results will be appended to: {csv_path}")
    print(f"General logs CSV will be appended to: {logs_csv_path}")

    results: List[SolveResult] = []

    # Stage 1: RAW Train (allow skipping existing)
    print("Stage 1: Solving TRAIN with raw Gurobi (no warm start) ...")
    for nm in train:
        r = _solve_raw(
            nm,
            args.time_limit,
            args.mip_gap,
            split="train",
            save_logs=args.save_logs,
            skip_existing=args.skip_existing,
            download_attempts=args.download_attempts,
            download_timeout=args.download_timeout,
            rc_provider=None,
            rc_enable=False,  # Do not prune while building train outputs
        )
        _append_result_to_csv(csv_path, r)
        _append_result_to_logs_csv(logs_csv_path, r)
        results.append(r)

    # Stage 2: Pretrain warm-start index from outputs (existing solutions are used)
    print("Stage 2: Pretraining warm-start index from TRAIN outputs ...")
    wsp = WarmStartProvider(case_folder=case_folder)
    wsp.pretrain(force=True)
    trained, cov = wsp.ensure_trained(case_folder, allow_build_if_missing=False)
    print(f"- Warm-start index: trained={trained}, coverage={cov:.3f}")
    if not trained:
        print(
            "Index not trained (insufficient outputs). Evaluation will still try warm, but coverage may be low."
        )

    # Stage 2.5: Pretrain redundancy index from TRAIN outputs (if enabled)
    rc_provider = None
    if args.rc_enable:
        print("Stage 2.5: Pretraining redundancy index from TRAIN outputs ...")
        rc_provider = RCProvider(case_folder=case_folder)
        rc_provider.pretrain(force=True)
        ok, rc_cov = rc_provider.ensure_trained(
            case_folder, allow_build_if_missing=False
        )
        print(f"- Redundancy index: available={ok}, coverage={rc_cov:.3f}")
        if not ok:
            print("Redundancy index not available. Pruning will be OFF.")
            rc_provider = None

    # Stage 3: Compare on TEST (RAW vs WARM) — always re-solve; ignore skip-existing
    print("Stage 3: Comparing on TEST (RAW vs WARM) ...")
    for nm in test:
        # RAW
        r_raw = _solve_raw(
            nm,
            args.time_limit,
            args.mip_gap,
            split="test",
            save_logs=args.save_logs,
            skip_existing=False,
            download_attempts=args.download_attempts,
            download_timeout=args.download_timeout,
            rc_provider=rc_provider,
            rc_enable=args.rc_enable,
            rc_thr_abs=args.rc_thr_abs,
            rc_thr_rel=args.rc_thr_rel,
            rc_use_train_db=args.rc_use_train_db,
        )
        _append_result_to_csv(csv_path, r_raw)
        _append_result_to_logs_csv(logs_csv_path, r_raw)
        results.append(r_raw)

        # WARM
        r_warm = _solve_warm(
            nm,
            wsp,
            args.time_limit,
            args.mip_gap,
            split="test",
            warm_mode=args.warm_mode,
            save_logs=args.save_logs,
            skip_existing=False,
            download_attempts=args.download_attempts,
            download_timeout=args.download_timeout,
            rc_provider=rc_provider,
            rc_enable=args.rc_enable,
            rc_thr_abs=args.rc_thr_abs,
            rc_thr_rel=args.rc_thr_rel,
            rc_use_train_db=args.rc_use_train_db,
        )
        _append_result_to_csv(csv_path, r_warm)
        _append_result_to_logs_csv(logs_csv_path, r_warm)
        results.append(r_warm)

    print(f"Results appended to: {csv_path}")
    print(f"General logs appended to: {logs_csv_path}")

    # Simple summary
    test_pairs = {}
    for r in results:
        if r.split != "test":
            continue
        test_pairs.setdefault(r.instance_name, {})[r.method] = r
    speed_wins = 0
    obj_wins = 0
    count_pairs = 0
    for nm, pair in test_pairs.items():
        if "raw" in pair and "warm" in pair:
            if pair["raw"].status not in (
                "SKIPPED_EXISTING",
                "DOWNLOAD_FAIL",
                "READ_FAIL",
            ) and pair["warm"].status not in (
                "SKIPPED_EXISTING",
                "DOWNLOAD_FAIL",
                "READ_FAIL",
            ):
                count_pairs += 1
                if pair["warm"].runtime_sec < pair["raw"].runtime_sec:
                    speed_wins += 1
                if (pair["warm"].obj_val is not None) and (
                    pair["raw"].obj_val is not None
                ):
                    if pair["warm"].obj_val <= pair["raw"].obj_val + 1e-6:
                        obj_wins += 1
    if count_pairs > 0:
        print(
            f"Warm faster on {speed_wins}/{count_pairs} test instances; objective <= raw on {obj_wins}/{count_pairs}."
        )


if __name__ == "__main__":
    main()
