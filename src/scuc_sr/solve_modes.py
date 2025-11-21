from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import logging

import gurobipy as gp

from src.data_preparation.read_data import read_benchmark
from src.data_preparation.params import DataParams
from src.optimization_model.SCUC_solver.scuc_model_builder import build_model
from src.optimization_model.helpers.lazy_contingency_cb import (
    attach_lazy_contingency_callback,
    LazyContingencyConfig,
)
from src.ml_models.redundant_constraints import RedundancyProvider
from src.optimization_model.helpers.verify_solution import verify_solution
from src.scuc_sr.analysis import radii_for_scenario

logger = logging.getLogger(__name__)


@dataclass
class SolveMetrics:
    status_code: int
    status: str
    runtime: float
    mip_gap: Optional[float]
    obj_val: Optional[float]
    num_vars: int
    num_constrs: int
    num_bin: int
    num_int: int


def _metrics_from_model(model: gp.Model) -> SolveMetrics:
    code = int(getattr(model, "Status", -1))
    try:
        gap = float(model.MIPGap)
    except Exception:
        gap = None
    try:
        obj = float(model.ObjVal)
    except Exception:
        obj = None
    return SolveMetrics(
        status_code=code,
        status=DataParams.SOLVER_STATUS_STR.get(code, f"STATUS_{code}"),
        runtime=float(getattr(model, "Runtime", 0.0)),
        mip_gap=gap,
        obj_val=obj,
        num_vars=getattr(model, "NumVars", 0),
        num_constrs=getattr(model, "NumConstrs", 0),
        num_bin=getattr(model, "NumBinVars", 0),
        num_int=getattr(model, "NumIntVars", 0),
    )


def _sum_dict_values(data: Optional[Dict]) -> int:
    if not data:
        return 0
    return int(sum(int(v) for v in data.values()))


def _collect_constraint_stats(model: gp.Model) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    pair_line = getattr(model, "_explicit_pair_line_counts", {}) or {}
    pair_gen = getattr(model, "_explicit_pair_gen_counts", {}) or {}
    stats["explicit_line"] = _sum_dict_values(pair_line)
    stats["explicit_gen"] = _sum_dict_values(pair_gen)
    stats["explicit_total"] = int(
        getattr(
            model,
            "_explicit_total_cont_constraints",
            stats["explicit_line"] + stats["explicit_gen"],
        )
    )
    lazy_pair_line = getattr(model, "_lazy_added_pair_line", {}) or {}
    lazy_pair_gen = getattr(model, "_lazy_added_pair_gen", {}) or {}
    stats["lazy_line"] = _sum_dict_values(lazy_pair_line)
    stats["lazy_gen"] = _sum_dict_values(lazy_pair_gen)
    lazy_stats = getattr(model, "_lazy_contingency_stats", {}) or {}
    stats["lazy_total"] = int(
        lazy_stats.get("lazy_added", stats["lazy_line"] + stats["lazy_gen"])
    )
    stats["lazy_incumbents"] = int(lazy_stats.get("incumbents_seen", 0))
    return stats


def _log_constraint_stats(stats: Dict[str, int], scenario, mode_tag: str) -> None:
    if not stats or scenario is None:
        return
    logger.info(
        "[%s] Constraints: explicit_line=%d, explicit_gen=%d (total=%d); "
        "lazy_line=%d, lazy_gen=%d (total=%d, incumbents=%d); lines=%d, T=%d",
        mode_tag,
        stats.get("explicit_line", 0),
        stats.get("explicit_gen", 0),
        stats.get("explicit_total", 0),
        stats.get("lazy_line", 0),
        stats.get("lazy_gen", 0),
        stats.get("lazy_total", 0),
        stats.get("lazy_incumbents", 0),
        len(scenario.lines or []),
        scenario.time,
    )


def _apply_common_gurobi_params(
    model, time_limit, mip_gap, output_log, log_file, threads=0
):
    try:
        model.setParam("LogToConsole", 1 if output_log else 0)
    except Exception:
        pass
    try:
        model.Params.OutputFlag = 1 if output_log else 0
    except Exception:
        pass
    model.setParam("TimeLimit", float(time_limit))
    model.setParam("MIPGap", float(mip_gap))
    model.setParam("NumericFocus", 1)
    if threads > 0:
        model.setParam("Threads", threads)


def _allowed_lines_by_radius(
    sc, l2_thr=400.0, sigma_thr=5.0, balance="agc", sigma_sr=0.3
):
    allowed = set()
    try:
        rd = radii_for_scenario(sc, t=0, balance=balance, sigma_sr=float(sigma_sr))
        l2_map = rd.get("l2", {})
        sig_map = rd.get("sigma", {})

        for ln in sc.lines:
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
        for ln in sc.lines:
            allowed.add(ln.name)
    return allowed


def run_raw(
    instance_name,
    time_limit,
    mip_gap,
    output_log=True,
    log_file=None,
    env=None,
    threads=0,
):
    inst = read_benchmark(instance_name, quiet=True)
    sc = inst.deterministic
    model = build_model(
        sc,
        use_lazy_contingencies=False,
        env=env,
        name_constraints=False,
        radius_line_whitelist=None,
    )
    _apply_common_gurobi_params(
        model, time_limit, mip_gap, output_log, log_file, threads=threads
    )
    model.optimize()
    stats = _collect_constraint_stats(model)
    _log_constraint_stats(stats, sc, f"raw:{instance_name}")
    return model, _metrics_from_model(model)


def run_sr(
    instance_name,
    time_limit,
    mip_gap,
    output_log=True,
    log_file=None,
    env=None,
    threads=0,
    sigma_sr=0.3,
    sr_l2_thr=400.0,
    sr_sigma_thr=5.0,
):
    inst = read_benchmark(instance_name, quiet=True)
    sc = inst.deterministic
    whitelist = _allowed_lines_by_radius(
        sc, l2_thr=sr_l2_thr, sigma_thr=sr_sigma_thr, sigma_sr=sigma_sr
    )
    model = build_model(
        sc,
        use_lazy_contingencies=False,
        env=env,
        name_constraints=False,
        radius_line_whitelist=whitelist,
    )
    _apply_common_gurobi_params(
        model, time_limit, mip_gap, output_log, log_file, threads=threads
    )
    model.optimize()
    stats = _collect_constraint_stats(model)
    _log_constraint_stats(stats, sc, f"sr:{instance_name}")
    added = stats.get("explicit_total", 0)
    return model, _metrics_from_model(model), added


def run_prune(
    instance_name,
    time_limit,
    mip_gap,
    lodf_tol,
    isf_tol,
    rc_provider=None,
    rc_thr_rel=0.10,
    rc_use_train_db=True,
    output_log=True,
    log_file=None,
    collect_stats=False,
    env=None,
    threads=0,
    viol_tol=1e-6,
):
    inst = read_benchmark(instance_name, quiet=True)
    sc = inst.deterministic
    keep_masks = None
    rc_stats = {}
    if rc_provider:
        try:
            res = rc_provider.make_masks_for_instance(
                sc,
                instance_name,
                thr_rel=rc_thr_rel,
                use_train_index_only=rc_use_train_db,
                exclude_self=True,
            )
            if res:
                keep_masks = {"line": res[0][0], "gen": res[0][1]}
                rc_stats = res[1]
        except Exception:
            pass

    model = build_model(
        sc,
        contingency_keep_masks=keep_masks,
        use_lazy_contingencies=False,
        name_constraints=False,
        env=env,
        radius_line_whitelist=None,
    )
    _apply_common_gurobi_params(
        model, time_limit, mip_gap, output_log, log_file, threads=threads
    )
    model.optimize()

    stats = _collect_constraint_stats(model)
    _log_constraint_stats(stats, sc, f"prune:{instance_name}")
    repair_added = 0
    report = {
        "explicit_added_total": stats.get("explicit_total", 0),
        "explicit_line": stats.get("explicit_line", 0),
        "explicit_gen": stats.get("explicit_gen", 0),
        "lazy_added": stats.get("lazy_total", 0),
        "repair_added_total": repair_added,
        "rc_stats": rc_stats,
    }
    return model, _metrics_from_model(model), report


def run_lazy(
    instance_name,
    time_limit,
    mip_gap,
    lodf_tol,
    isf_tol,
    viol_tol,
    lazy_top_k=0,
    output_log=True,
    log_file=None,
    env=None,
    threads=0,
):
    inst = read_benchmark(instance_name, quiet=True)
    sc = inst.deterministic
    model = build_model(sc, use_lazy_contingencies=True, env=env)
    cfg = LazyContingencyConfig(
        lodf_tol=lodf_tol,
        isf_tol=isf_tol,
        violation_tol=viol_tol,
        add_top_k=lazy_top_k,
        allowed_monitored_lines=None,
        verbose=True,
    )
    attach_lazy_contingency_callback(model, sc, cfg)
    _apply_common_gurobi_params(
        model, time_limit, mip_gap, output_log, log_file, threads=threads
    )
    try:
        model.Params.LazyConstraints = 1
    except Exception:
        pass
    model.optimize()
    stats = _collect_constraint_stats(model)
    _log_constraint_stats(stats, sc, f"lazy:{instance_name}")
    metrics = _metrics_from_model(model)
    report = {
        "lazy_added": stats.get("lazy_total", 0),
        "lazy_line": stats.get("lazy_line", 0),
        "lazy_gen": stats.get("lazy_gen", 0),
    }
    return model, metrics, report


def run_prune_lazy(
    instance_name,
    time_limit,
    mip_gap,
    lodf_tol,
    isf_tol,
    viol_tol,
    rc_provider=None,
    rc_thr_rel=0.10,
    rc_use_train_db=True,
    lazy_top_k=0,
    output_log=True,
    log_file=None,
    env=None,
    threads=0,
):
    inst = read_benchmark(instance_name, quiet=True)
    sc = inst.deterministic
    keep_masks = None
    if rc_provider:
        try:
            res = rc_provider.make_masks_for_instance(
                sc,
                instance_name,
                thr_rel=rc_thr_rel,
                use_train_index_only=rc_use_train_db,
                exclude_self=True,
            )
            if res:
                keep_masks = {"line": res[0][0], "gen": res[0][1]}
        except Exception:
            pass

    model = build_model(sc, use_lazy_contingencies=True, env=env)
    cfg = LazyContingencyConfig(
        lodf_tol=lodf_tol,
        isf_tol=isf_tol,
        violation_tol=viol_tol,
        add_top_k=lazy_top_k,
        keep_line_pairs=keep_masks["line"] if keep_masks else None,
        keep_gen_pairs=keep_masks["gen"] if keep_masks else None,
        allowed_monitored_lines=None,
        verbose=True,
    )
    attach_lazy_contingency_callback(model, sc, cfg)
    _apply_common_gurobi_params(
        model, time_limit, mip_gap, output_log, log_file, threads=threads
    )
    try:
        model.Params.LazyConstraints = 1
    except Exception:
        pass
    model.optimize()
    stats = _collect_constraint_stats(model)
    _log_constraint_stats(stats, sc, f"prune_lazy:{instance_name}")
    metrics = _metrics_from_model(model)
    report = {
        "lazy_added": stats.get("lazy_total", 0),
        "lazy_line": stats.get("lazy_line", 0),
        "lazy_gen": stats.get("lazy_gen", 0),
        "explicit_added_total": stats.get("explicit_total", 0),
        "explicit_line": stats.get("explicit_line", 0),
        "explicit_gen": stats.get("explicit_gen", 0),
    }
    return model, metrics, report


def run_sr_lazy(
    instance_name,
    time_limit,
    mip_gap,
    lodf_tol,
    isf_tol,
    viol_tol,
    output_log=True,
    log_file=None,
    env=None,
    threads=0,
    sigma_sr=0.3,
    sr_l2_thr=400.0,
    sr_sigma_thr=5.0,
    lazy_top_k=0,
):
    inst = read_benchmark(instance_name, quiet=True)
    sc = inst.deterministic

    whitelist = _allowed_lines_by_radius(
        sc, sigma_sr=sigma_sr, l2_thr=sr_l2_thr, sigma_thr=sr_sigma_thr
    )
    model = build_model(
        sc,
        use_lazy_contingencies=False,
        env=env,
        name_constraints=False,
        radius_line_whitelist=whitelist,
    )

    cfg = LazyContingencyConfig(
        lodf_tol=lodf_tol,
        isf_tol=isf_tol,
        violation_tol=viol_tol,
        add_top_k=lazy_top_k,
        allowed_monitored_lines=None,
        verbose=True,
    )
    attach_lazy_contingency_callback(model, sc, cfg)

    _apply_common_gurobi_params(
        model, time_limit, mip_gap, output_log, log_file, threads=threads
    )
    try:
        model.Params.LazyConstraints = 1
    except Exception:
        pass
    model.optimize()

    stats = _collect_constraint_stats(model)
    _log_constraint_stats(stats, sc, f"sr_lazy:{instance_name}")
    metrics = _metrics_from_model(model)
    report = {
        "explicit_added": stats.get("explicit_total", 0),
        "explicit_line": stats.get("explicit_line", 0),
        "explicit_gen": stats.get("explicit_gen", 0),
        "lazy_added": stats.get("lazy_total", 0),
        "lazy_line": stats.get("lazy_line", 0),
        "lazy_gen": stats.get("lazy_gen", 0),
    }
    return model, metrics, report


def _compute_final_flows_and_generation(sc, model):
    T = sc.time
    lines = sc.lines or []
    f_val = {}
    p_total = {}

    try:
        line_flow = model.line_flow
        commit = model.commit
        seg = getattr(model, "gen_segment_power", None)

        for ln in lines:
            for t in range(T):
                f_val[(ln.name, t)] = line_flow[ln.name, t].X

        for gen in sc.thermal_units:
            for t in range(T):
                u = commit[gen.name, t].X
                p = u * gen.min_power[t]
                if seg:
                    for s in range(len(gen.segments)):
                        p += seg[gen.name, t, s].X
                p_total[(gen.name, t)] = p
    except Exception:
        pass
    return f_val, p_total
