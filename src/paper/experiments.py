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
from typing import Dict, List, Optional, Tuple

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
    LazyContingencyConfig,
)
from src.optimization_model.helpers.save_json_solution import (
    save_solution_as_json,
    compute_output_path,
)
from src.optimization_model.helpers.verify_solution import verify_solution


def _configure_logging():
    """
    Make logs readable and avoid duplicate gurobipy messages.
    """
    # Configure root logger if nobody did it yet
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s:%(name)s:%(message)s",
        )
    # Silence Python-side gurobipy log (Gurobi still prints its own single line to console)
    logging.getLogger("gurobipy").setLevel(logging.WARNING)
    # Quiet common HTTP libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def _case_tag(case_folder: str) -> str:
    cf = case_folder.strip().strip("/\\").replace("\\", "/")
    return "".join(ch if ch.isalnum() else "_" for ch in cf).strip("_").lower()


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
    mode: str  # RAW | WARM | WARM+HINTS | WARM+LAZY | WARM+PRUNE
    tau: Optional[float] = None  # for WARM+PRUNE
    time_limit: int = 600
    mip_gap: float = 0.05
    lazy_top_k: int = 0
    lazy_lodf_tol: float = 1e-4
    lazy_isf_tol: float = 1e-8
    lazy_viol_tol: float = 1e-6
    save_human_logs: bool = False


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


def _estimate_cont_counts(
    sc, filter_predicate: Optional[callable]
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
                if filter_predicate is None:
                    kept_line += 2 * T
                else:
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
                    if filter_predicate is None:
                        kept_gen += 2 * T
                    else:
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
                    if filter_predicate is None:
                        kept_gen += 2 * T
                    else:
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


def solve_one(
    instance_name: str,
    mode_cfg: RunConfig,
    wsp: Optional[WarmStartProvider],
    rc: Optional[RCProvider],
    use_train_db_for_rc: bool,
) -> Dict:
    started_at = time.time()
    # quiet=True suppresses download progress bars now
    inst = read_benchmark(instance_name, quiet=True)
    sc = inst.deterministic

    # Prepare pruning predicate (PRUNE only)
    cont_filter = None
    if mode_cfg.mode.startswith("WARM+PRUNE"):
        tau = 0.5 if mode_cfg.tau is None else float(mode_cfg.tau)
        cont_filter = _build_contingency_filter(
            rc,
            sc,
            instance_name,
            enable=True,
            thr_rel=tau,
            use_train_db=use_train_db_for_rc,
        )

    # Build model with lazy flag only when mode is LAZY
    use_lazy = mode_cfg.mode == "WARM+LAZY"
    model = build_model(
        scenario=sc,
        contingency_filter=None if use_lazy else cont_filter,
        use_lazy_contingencies=bool(use_lazy),
    )

    # Root size
    num_vars_root = int(getattr(model, "NumVars", 0))
    num_constrs_root = int(getattr(model, "NumConstrs", 0))

    # Apply warm starts (all WARM* modes)
    applied_starts = 0
    if mode_cfg.mode in ("WARM", "WARM+HINTS", "WARM+LAZY", "WARM+PRUNE"):
        if wsp is not None:
            try:
                wsp.generate_and_save_warm_start(
                    instance_name,
                    use_train_index_only=True,
                    exclude_self=True,
                    auto_fix=True,
                )
            except Exception:
                pass
            try:
                applied_starts = wsp.apply_warm_start_to_model(
                    model, sc, instance_name, mode="repair"
                )
            except Exception:
                applied_starts = 0

    # Branching hints only for WARM+HINTS, WARM+LAZY, WARM+PRUNE
    hints_applied = 0
    if mode_cfg.mode in ("WARM+HINTS", "WARM+LAZY", "WARM+PRUNE"):
        try:
            hints_applied = apply_branching_hints_from_starts(model)
        except Exception:
            hints_applied = 0

    # Attach lazy callback for LAZY
    if use_lazy and sc.contingencies:
        cfg = LazyContingencyConfig(
            lodf_tol=mode_cfg.lazy_lodf_tol,
            isf_tol=mode_cfg.lazy_isf_tol,
            violation_tol=mode_cfg.lazy_viol_tol,
            add_top_k=int(mode_cfg.lazy_top_k)
            if mode_cfg.lazy_top_k and mode_cfg.lazy_top_k > 0
            else 0,
            verbose=True,
        )
        attach_lazy_contingency_callback(model, sc, cfg)

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

    # Memory sampling + optimize
    with PeakMemSampler(interval_sec=0.2) as mems:
        model.optimize()
        peak_mem_gb = mems.peak_gb

    # Save JSON for ML indexes and provenance
    try:
        save_solution_as_json(sc, model, instance_name=instance_name)
    except Exception:
        pass

    # Verification
    feasible_ok = None
    max_constraint_residual = None
    obj_inconsistency = None
    violations_str = None
    try:
        feasible_ok, checks, _ = verify_solution(sc, model)
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

    # Metrics
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

    # Final model size
    num_vars_final = int(getattr(model, "NumVars", 0))
    num_constrs_final = int(getattr(model, "NumConstrs", 0))

    # Constraint ratio for PRUNE (static estimate)
    constr_total = None
    constr_kept = None
    constr_ratio = None
    if mode_cfg.mode == "WARM+PRUNE":
        tot, kept, _, _ = _estimate_cont_counts(sc, cont_filter)
        constr_total = int(tot)
        constr_kept = int(kept)
        constr_ratio = float(kept) / float(tot) if tot > 0 else 1.0

    # Dispose model to release memory
    try:
        model.dispose()
    except Exception:
        pass

    finished_at = time.time()

    return {
        # Use timezone-aware UTC timestamp (fixes deprecation warning)
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "instance_name": instance_name,
        "case_folder": "/".join(instance_name.split("/")[:2]),
        "mode": mode_cfg.mode
        if mode_cfg.tau is None
        else f"{mode_cfg.mode}-{mode_cfg.tau:.2f}",
        "tau": "" if mode_cfg.tau is None else f"{mode_cfg.tau:.2f}",
        "status": _status_str(status_code),
        "status_code": status_code,
        "runtime_sec": runtime,
        "wall_sec": finished_at - started_at,
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
        "warm_start_applied_vars": applied_starts,
        "branch_hints_applied": hints_applied,
        "constr_total_cont": "" if constr_total is None else int(constr_total),
        "constr_kept_cont": "" if constr_kept is None else int(constr_kept),
        "constr_ratio_cont": "" if constr_ratio is None else f"{constr_ratio:.4f}",
    }


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
        "warm_start_applied_vars",
        "branch_hints_applied",
        "constr_total_cont",
        "constr_kept_cont",
        "constr_ratio_cont",
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
                    "warm_start_applied_vars",
                    "branch_hints_applied",
                    "constr_total_cont",
                    "constr_kept_cont",
                    "constr_ratio_cont",
                ]
            ]
        )


def main():
    _configure_logging()

    ap = argparse.ArgumentParser(
        description="Run SCUC+ML experiments across modes and cases."
    )
    ap.add_argument(
        "--cases",
        nargs="*",
        default=[
            # "matpower/case14",
            # "matpower/case30",
            # "matpower/case57",
            # "matpower/case89pegase",
            # "matpower/case118",
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
        action="store_true",
        default=True,
        help="Skip solving TRAIN if JSON exists",
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
        default=["RAW", "WARM", "WARM+HINTS", "WARM+LAZY", "WARM+PRUNE"],
        help="Which modes to run",
    )
    ap.add_argument(
        "--taus",
        nargs="*",
        type=float,
        default=[0.3, 0.5, 0.7],
        help="Tau sweep for WARM+PRUNE",
    )
    ap.add_argument(
        "--lazy-top-k",
        type=int,
        default=0,
        help="Lazy: add top-K violated constraints per incumbent (0=all)",
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
        "--rc-max-items",
        type=int,
        default=0,
        help="Redundancy index: limit number of outputs when building (0=all)",
    )
    ap.add_argument(
        "--rc-restrict-train",
        action="store_true",
        default=True,
        help="Redundancy index: restrict to TRAIN split only",
    )
    args = ap.parse_args()

    # Do we need redundancy at all?
    prune_requested = any(
        m.strip().upper().startswith("WARM+PRUNE") for m in args.modes
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

        # Stage A: Solve TRAIN with RAW explicit N-1 if missing (to build warm/screening DB)
        if train:
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
                    cfg = RunConfig(
                        mode="RAW",
                        time_limit=args.time_limit,
                        mip_gap=args.mip_gap,
                        save_human_logs=False,
                        lazy_top_k=args.lazy_top_k,
                        lazy_lodf_tol=args.lazy_lodf_tol,
                        lazy_isf_tol=args.lazy_isf_tol,
                        lazy_viol_tol=args.lazy_viol_tol,
                    )
                    _ = solve_one(nm, cfg, wsp=None, rc=None, use_train_db_for_rc=True)

        # Stage B: Pretrain warm-start index (lightweight)
        wsp = WarmStartProvider(
            case_folder=case_folder,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            split_seed=args.seed,
        )
        wsp.pretrain(force=True)
        trained, cov = wsp.ensure_trained(case_folder, allow_build_if_missing=False)
        tqdm.write(f"[{case_folder}] Warm index trained={trained}, coverage={cov:.3f}")

        # Stage B2 (optional): Pretrain redundancy index ONLY if PRUNE requested
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

        # Stage C: Run TEST across modes
        log_path = _prepare_csv_log(case_folder)
        tqdm.write(f"[{case_folder}] logging to {log_path}")

        mode_list: List[RunConfig] = []
        for m in args.modes:
            m = m.strip().upper()
            if m == "RAW":
                mode_list.append(
                    RunConfig(
                        mode="RAW",
                        time_limit=args.time_limit,
                        mip_gap=args.mip_gap,
                        save_human_logs=args.save_human_logs,
                    )
                )
            elif m == "WARM":
                mode_list.append(
                    RunConfig(
                        mode="WARM",
                        time_limit=args.time_limit,
                        mip_gap=args.mip_gap,
                        save_human_logs=args.save_human_logs,
                    )
                )
            elif m == "WARM+HINTS":
                mode_list.append(
                    RunConfig(
                        mode="WARM+HINTS",
                        time_limit=args.time_limit,
                        mip_gap=args.mip_gap,
                        save_human_logs=args.save_human_logs,
                    )
                )
            elif m == "WARM+LAZY":
                mode_list.append(
                    RunConfig(
                        mode="WARM+LAZY",
                        time_limit=args.time_limit,
                        mip_gap=args.mip_gap,
                        lazy_top_k=args.lazy_top_k,
                        lazy_lodf_tol=args.lazy_lodf_tol,
                        lazy_isf_tol=args.lazy_isf_tol,
                        lazy_viol_tol=args.lazy_viol_tol,
                        save_human_logs=args.save_human_logs,
                    )
                )
            elif m == "WARM+PRUNE":
                for tau in args.taus:
                    mode_list.append(
                        RunConfig(
                            mode="WARM+PRUNE",
                            tau=float(tau),
                            time_limit=args.time_limit,
                            mip_gap=args.mip_gap,
                            save_human_logs=args.save_human_logs,
                        )
                    )
            else:
                tqdm.write(f"Unknown mode '{m}' - ignored")

        for nm in tqdm(test, desc=f"{case_folder} TEST", unit="inst"):
            for cfg in mode_list:
                row = solve_one(nm, cfg, wsp=wsp, rc=rc, use_train_db_for_rc=True)
                _append_csv(log_path, row)
                tqdm.write(
                    f"  {nm} | {row['mode']:>12} | status={row['status']:<12} time={row['runtime_sec']}s nodes={row['nodes']}"
                )

        tqdm.write(f"[{case_folder}] done. Log: {log_path}")


if __name__ == "__main__":
    main()
