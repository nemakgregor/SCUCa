from __future__ import annotations
import argparse
import csv
import threading
import time
from datetime import datetime
from pathlib import Path
import sys
import shutil
import builtins
import logging
import os

import numpy as np
import gurobipy as gp

from src.optimization_model.SCUC_solver.solve_instances import (
    list_remote_instances,
    list_local_cached_instances,
)
from src.ml_models.redundant_constraints import RedundancyProvider
from src.ml_models.warm_start import _hash01
from src.optimization_model.helpers.verify_solution import verify_solution
from src.data_preparation.read_data import read_benchmark

from src.scuc_sr.constraint_enumerator import enumerate_potential_constraints
from src.scuc_sr.solve_modes import (
    run_raw,
    run_prune,
    run_lazy,
    run_prune_lazy,
    run_sr,
    run_sr_lazy,
    _compute_final_flows_and_generation,
)
from src.scuc_sr.utils import case_results_dir, case_tag

VERIFY_TOL = 1e-6
PRINT_LOCK = threading.Lock()


def safe_print(*args, **kwargs):
    """
    Thread-safe print that always flushes. Used instead of bare print()
    to avoid interleaved output when multiple threads/loggers are active.
    """
    kwargs.setdefault("flush", True)
    with PRINT_LOCK:
        builtins.print(*args, **kwargs)


def _list_case_instances(case_folder):
    """
    List instances for a given case folder, preferring local cache and
    falling back to remote listing if needed.
    """
    items = list_local_cached_instances(include_filters=[case_folder])
    items = [x for x in items if x.startswith(case_folder)]
    if not items:
        safe_print(f"[{case_folder}] No local instances. Querying remote...")
        try:
            remote = list_remote_instances(
                include_filters=[case_folder], roots=["matpower", "test"], max_depth=4
            )
            items = [x for x in remote if x.startswith(case_folder)]
        except Exception as e:
            safe_print(f"Remote error: {e}")
    return sorted(set(items))


def _split_instances(names, train_ratio, val_ratio, seed):
    """
    Deterministic train/val/test split based on hash of instance name.
    """
    tr, va, te = [], [], []
    for nm in sorted(names):
        r = _hash01(nm, seed)
        if r < train_ratio:
            tr.append(nm)
        elif r < train_ratio + val_ratio:
            va.append(nm)
        else:
            te.append(nm)
    return tr, va, te


def _log_event(log_file: Path, msg: str) -> None:
    """
    Append a single log line with a wall-clock timestamp to the pipeline log
    and echo it to stdout via safe_print.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    safe_print(line)
    try:
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
    except Exception as e:
        safe_print(f"Log error: {e}")


def _log_table_row(log_file: Path, row: str) -> None:
    """
    Append a preformatted table row to the pipeline log WITHOUT timestamp
    prefix. Used for the tabular summary section so that the file parses
    cleanly as a table.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(row.rstrip("\n") + "\n")
            fh.flush()
    except Exception as e:
        safe_print(f"Log error (table row): {e}")


def _identify_dropped_essential(sc, model, lodf_tol, isf_tol, viol_tol) -> int:
    """
    Post-hoc check: count how many potential N-1 constraints (enumerated from
    topology) are violated by the final solution. Used only for SR/prune modes.
    """
    count = 0
    try:
        f_val, p_total = _compute_final_flows_and_generation(sc, model)
        events = enumerate_potential_constraints(sc, lodf_tol=lodf_tol, isf_tol=isf_tol)

        for ev in events:
            if ev.kind == "line":
                fl = f_val.get((ev.line_name, ev.t), 0.0)
                fo = f_val.get((ev.out_name, ev.t), 0.0)
                post = fl + ev.coeff * fo
            else:
                fl = f_val.get((ev.line_name, ev.t), 0.0)
                pg = p_total.get((ev.out_name, ev.t), 0.0)
                post = fl - ev.coeff * pg

            if abs(post) > ev.F_em + viol_tol:
                count += 1
    except Exception:
        pass
    return count


class Heartbeat:
    """
    Background thread printing a periodic heartbeat line to show that the
    solver is still running.
    """

    def __init__(self, interval=10.0):
        self.interval = interval
        self._stop = threading.Event()
        self._label = ""
        self._thr = None

    def start(self, label: str) -> None:
        self._label = label
        self._t0 = time.time()
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thr:
            self._thr.join(0.5)

    def _run(self) -> None:
        while not self._stop.is_set():
            safe_print(f"[heartbeat] {self._label} ... {time.time() - self._t0:.1f}s")
            self._stop.wait(self.interval)


def _configure_python_logging() -> None:
    """
    Configure root Python logging so that INFO-level messages from SCUC modules
    (and others) are emitted to sys.stdout.

    Called AFTER installing the Tee on sys.stdout/sys.stderr, so that all
    logging output is also captured into run_*.log.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if called multiple times.
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) for h in root.handlers
    )
    if not has_stream_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        fmt = logging.Formatter("%(levelname)s [%(name)s] %(message)s")
        handler.setFormatter(fmt)
        root.addHandler(handler)

    # Reduce noise from the Gurobi Python logger. We rely on the solver's own
    # console log for parameter messages, so we silence gurobipy INFO logs
    # while keeping our own SCUC INFO logs.
    try:
        logging.getLogger("gurobipy").setLevel(logging.WARNING)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument(
        "--run-split", choices=["train", "val", "test", "all"], default="val"
    )
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["raw", "lazy", "prune_lazy", "sr_lazy"],
        choices=["raw", "sr", "prune", "lazy", "prune_lazy", "sr_lazy"],
    )
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--time-limit", type=int, default=300)
    ap.add_argument("--mip-gap", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--lodf-tol", type=float, default=1e-2)
    ap.add_argument("--isf-tol", type=float, default=1e-3)
    ap.add_argument("--viol-tol", type=float, default=1e-2)
    ap.add_argument("--rc-thr-rel", type=float, default=0.40)
    ap.add_argument("--rc-use-train-db", action="store_true", default=True)
    ap.add_argument("--sigma-sr", type=float, default=0.3)
    ap.add_argument("--sr-l2-thr", type=float, default=60.0)
    ap.add_argument("--sr-sigma-thr", type=float, default=0.0)
    ap.add_argument("--lazy-top-k", type=int, default=0)
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--tee-logs", action="store_true", default=True)
    ap.add_argument("--overwrite-logs-results", action="store_true", default=True)

    args = ap.parse_args()
    case_folder = args.case.strip().strip("/\\").replace("\\", "/")

    # If both flags are true, clear the terminal to give a clean view.
    if args.tee_logs and args.overwrite_logs_results:
        try:
            if os.name == "nt":
                os.system("cls")
            else:
                os.system("clear")
        except Exception:
            pass

    all_instances = _list_case_instances(case_folder)
    if not all_instances:
        return

    if args.run_split == "all":
        to_run = all_instances
    else:
        tr, va, te = _split_instances(
            all_instances, args.train_ratio, args.val_ratio, args.seed
        )
        to_run = {"train": tr, "val": va, "test": te}[args.run_split]

    if args.limit_val > 0:
        to_run = to_run[: args.limit_val]
    if not to_run:
        safe_print(
            f"No instances for split '{args.run_split}' (Seed {args.seed}). Total {len(all_instances)}."
        )
        return

    logs_root = Path(__file__).resolve().parent / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    case_dir = case_results_dir(case_folder)

    case_tag_value = case_tag(case_folder)
    timestamp_label = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Results base tag and log file names depend on overwrite flag.
    if args.overwrite_logs_results:
        shutil.rmtree(case_dir, ignore_errors=True)
        case_dir.mkdir(parents=True, exist_ok=True)
        results_base = case_tag_value
        run_log_suffix = case_tag_value
        pipeline_log_name = f"pipeline_{case_tag_value}.log"
    else:
        results_base = f"{case_tag_value}_{timestamp_label}"
        run_log_suffix = results_base
        pipeline_log_name = f"pipeline_{case_tag_value}_{timestamp_label}.log"

    run_log_name = f"run_{run_log_suffix}.log"

    rc = RedundancyProvider(
        case_folder=case_folder,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_seed=args.seed,
    )
    rc_ok, _ = rc.ensure_trained(case_folder, allow_build_if_missing=False)

    gurobi_log = logs_root / run_log_name
    evt_log = logs_root / pipeline_log_name

    # If overwriting, ensure previous log files are removed.
    if args.overwrite_logs_results:
        try:
            if gurobi_log.exists():
                gurobi_log.unlink()
        except Exception:
            pass
        try:
            if evt_log.exists():
                evt_log.unlink()
        except Exception:
            pass

    csv_cols = [
        "run_id",
        "timestamp",
        "instance_name",
        "method",
        "status",
        "status_code",
        "runtime_sec",
        "mip_gap",
        "obj_val",
        "num_vars",
        "num_constrs",
        "num_bin",
        "num_int",
        "n_n1_explicit",
        "n_n1_lazy",
        "n_violations",
        "dropped_essential",
        "verified_ok",
    ]
    res_csv = case_dir / f"compare_detailed_{results_base}.csv"
    with res_csv.open("w", newline="") as f:
        csv.writer(f).writerow(csv_cols)

    # Configure a Gurobi environment.
    # When tee-logs is disabled, we let Gurobi write its own LogFile.
    # When tee-logs is enabled (default), we do NOT set LogFile to avoid
    # duplicate solver logs (Gurobi -> file and Tee -> same file).
    env = gp.Env(empty=0)
    if not args.tee_logs:
        env.setParam("LogFile", str(gurobi_log))
    env.setParam("OutputFlag", 1)
    env.setParam("NumericFocus", 1)

    # Tee stdout/stderr to run_*.log if requested
    tee_fh = None
    if args.tee_logs:
        mode = "w" if args.overwrite_logs_results else "a"
        tee_fh = open(gurobi_log, mode, encoding="utf-8")

        class Tee:
            def __init__(self, s):
                self.s = s

            def write(self, d):
                self.s.write(d)
                tee_fh.write(d)
                tee_fh.flush()

            def flush(self):
                self.s.flush()
                tee_fh.flush()

        sys.stdout = Tee(sys.stdout)
        sys.stderr = Tee(sys.stderr)

    # Configure Python logging after Tee installation so logs also go to run_*.log.
    _configure_python_logging()

    hb = Heartbeat(10.0)
    safe_print(f"Running {len(to_run)} instances on modes: {args.modes}")
    run_counter = 0

    # Column header for pipeline_*.log table (without leading timestamp).
    table_header_cols = [
        "run_id",
        "instance",
        "mode",
        "status",
        "status_code",
        "runtime_sec",
        "mip_gap",
        "obj_val",
        "num_vars",
        "num_constrs",
        "num_bin",
        "num_int",
        "n_n1_explicit",
        "n_n1_lazy",
        "violations",
        "dropped_essential",
        "verified",
    ]

    for i, nm in enumerate(to_run, 1):
        safe_print(f"--- {i}/{len(to_run)}: {nm} ---")
        # _log_event(evt_log, f"START {nm}")
        try:
            sc = read_benchmark(nm, quiet=True).deterministic
        except Exception as e:
            safe_print(f"Err: {e}")
            _log_event(evt_log, f"ERROR reading instance {nm}: {e}")
            continue

        for mode in args.modes:
            run_counter += 1
            run_label = f"Run #{run_counter:04d}"
            hb.start(f"{run_label} {nm} [{mode}]")
            safe_print(f"\n\n{run_label} | {nm} | mode={mode}")
            # _log_event(evt_log, f"{run_label} START {nm} mode={mode}")
            m, met, verified = None, None, False
            violations = 0
            ess_dropped = 0
            n_expl, n_lazy = 0, 0

            try:
                if mode == "raw":
                    m, met = run_raw(
                        nm, args.time_limit, args.mip_gap, env=env, threads=args.threads
                    )
                elif mode == "sr":
                    m, met, n_expl = run_sr(
                        nm,
                        args.time_limit,
                        args.mip_gap,
                        env=env,
                        threads=args.threads,
                        sigma_sr=args.sigma_sr,
                        sr_l2_thr=args.sr_l2_thr,
                        sr_sigma_thr=args.sr_sigma_thr,
                    )
                elif mode == "prune":
                    m, met, rep = run_prune(
                        nm,
                        args.time_limit,
                        args.mip_gap,
                        args.lodf_tol,
                        args.isf_tol,
                        rc_provider=rc if rc_ok else None,
                        rc_thr_rel=args.rc_thr_rel,
                        env=env,
                        threads=args.threads,
                    )
                    n_expl = rep.get("explicit_added_total", 0)
                elif mode == "lazy":
                    m, met, rep = run_lazy(
                        nm,
                        args.time_limit,
                        args.mip_gap,
                        args.lodf_tol,
                        args.isf_tol,
                        args.viol_tol,
                        lazy_top_k=args.lazy_top_k,
                        env=env,
                        threads=args.threads,
                    )
                    n_lazy = rep.get("lazy_added", 0)
                elif mode == "prune_lazy":
                    m, met, rep = run_prune_lazy(
                        nm,
                        args.time_limit,
                        args.mip_gap,
                        args.lodf_tol,
                        args.isf_tol,
                        args.viol_tol,
                        rc_provider=rc if rc_ok else None,
                        rc_thr_rel=args.rc_thr_rel,
                        lazy_top_k=args.lazy_top_k,
                        env=env,
                        threads=args.threads,
                    )
                    n_lazy = rep.get("lazy_added", 0)
                    n_expl = rep.get("explicit_added_total", 0)
                elif mode == "sr_lazy":
                    m, met, rep = run_sr_lazy(
                        nm,
                        args.time_limit,
                        args.mip_gap,
                        args.lodf_tol,
                        args.isf_tol,
                        args.viol_tol,
                        lazy_top_k=args.lazy_top_k,
                        env=env,
                        threads=args.threads,
                        sigma_sr=args.sigma_sr,
                        sr_l2_thr=args.sr_l2_thr,
                        sr_sigma_thr=args.sr_sigma_thr,
                    )
                    n_lazy = rep.get("lazy_added", 0)
                    n_expl = rep.get("explicit_added", 0)

                verified, checks, verify_report = verify_solution(sc, m)
                violations = sum(1 for c in checks if c.value > VERIFY_TOL)

                # Console verification report (always)
                safe_print(f"{run_label} verification report:\n{verify_report}\n")

                # If validation failed, also log the full report into pipeline_*.log
                if not verified:
                    _log_event(evt_log, f"{run_label} verification report:")
                    for line in verify_report.splitlines():
                        _log_event(evt_log, line)

                if (not verified) and mode in ["sr", "prune"]:
                    ess_dropped = _identify_dropped_essential(
                        sc, m, args.lodf_tol, args.isf_tol, args.viol_tol
                    )

            except Exception as e:
                safe_print(f"FAIL {mode}: {e}")
                _log_event(evt_log, f"{run_label} {nm} mode={mode} EXCEPTION: {e}")
                # Fill metrics with sentinel values so downstream logging still works
                met = type(
                    "o",
                    (),
                    {
                        "status": "FAIL",
                        "status_code": -1,
                        "runtime": 0.0,
                        "mip_gap": None,
                        "obj_val": None,
                        "num_vars": 0,
                        "num_constrs": 0,
                        "num_bin": 0,
                        "num_int": 0,
                    },
                )()
                verified = False
                violations = 0
                ess_dropped = 0
            finally:
                hb.stop()

            # Tabular line into pipeline_*.log; repeat header every 50 runs
            if run_counter % 50 == 1:
                _log_table_row(evt_log, " | ".join(table_header_cols))

            mip_gap_str = (
                f"{met.mip_gap:.6f}"
                if getattr(met, "mip_gap", None) is not None
                else ""
            )
            obj_str = (
                f"{met.obj_val:.6f}"
                if getattr(met, "obj_val", None) is not None
                else ""
            )
            row_values = [
                str(run_counter),
                nm,
                mode,
                getattr(met, "status", "UNKNOWN"),
                str(getattr(met, "status_code", -1)),
                f"{getattr(met, 'runtime', 0.0):.4f}",
                mip_gap_str,
                obj_str,
                str(getattr(met, "num_vars", 0)),
                str(getattr(met, "num_constrs", 0)),
                str(getattr(met, "num_bin", 0)),
                str(getattr(met, "num_int", 0)),
                str(n_expl),
                str(n_lazy),
                str(violations),
                str(ess_dropped),
                "OK" if verified else "FAIL",
            ]
            _log_table_row(evt_log, " | ".join(row_values))

            # Detailed CSV row
            with res_csv.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        run_counter,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        nm,
                        mode,
                        getattr(met, "status", "UNKNOWN"),
                        getattr(met, "status_code", -1),
                        f"{getattr(met, 'runtime', 0.0):.4f}",
                        mip_gap_str,
                        obj_str,
                        getattr(met, "num_vars", 0),
                        getattr(met, "num_constrs", 0),
                        getattr(met, "num_bin", 0),
                        getattr(met, "num_int", 0),
                        n_expl,
                        n_lazy,
                        violations,
                        ess_dropped,
                        1 if verified else 0,
                    ]
                )

    if tee_fh:
        tee_fh.close()


if __name__ == "__main__":
    main()
