"""
Comprehensive analysis on existing logs (no re-run of experiments needed).

What this script does:
- Reads completed rows from active results/*/results.csv files;
  falls back to results/raw_logs/*.csv only when no run results are present.
- Merges them into results/merged_results.csv
- Computes:
  • Objective delta vs RAW baseline (ppm)
  • Per-case/per-mode robust summaries with bootstrap CI
  • Pairwise (within-instance) comparisons vs RAW for all modes (runtime, nodes, etc.)
  • Effect sizes (WARM_LAZY vs RAW): Hodges–Lehmann median paired difference, rank-biserial correlation
  • MILP feasibility rates, solver-status diagnostics, MIP gap stats
  • Warm-start and branching-hint utilization summaries
  • PRUNE tau sweep summaries (constraint ratio vs runtime)
  • NEW: full pairwise comparisons among all modes on the same instance (pairs_all.csv)
  • NEW: fastest-mode share per case (mode_fastest_share.csv)
  • NEW: per-mode speed ratio (mode over RAW) stats for bar-plots (mode_speed_stats.csv)
  • NEW: per-row flags (with_LAZY/GNN/PRUNE/COMMIT/GRU/BANDIT) for group-wise plots (flags added in merged_results.csv)
  • NEW: exact vs heuristic method tags (for publication-safe comparisons)
  • NEW: explicit vs realized contingency ratio fields for hybrid methods
"""

from __future__ import annotations

import glob
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paper.modes import RAW_BASELINE_MODE, canonical_mode, mode_exactness
from src.paper.experiment_spec import CASES_LARGE, CASES_SMALL

RESULTS_DIR = Path("results")
RAW_DIR = RESULTS_DIR / "raw_logs"
OUT_SUMMARY = RESULTS_DIR / "summary.csv"
OUT_MERGED = RESULTS_DIR / "merged_results.csv"
OUT_PAIRS = RESULTS_DIR / "pairs.csv"
OUT_PAIRS_ALL = RESULTS_DIR / "pairs_all.csv"
OUT_FASTEST = RESULTS_DIR / "mode_fastest_share.csv"
OUT_SUMMARY_EXT = RESULTS_DIR / "summary_extended.csv"
OUT_EFFECTS_WLZ = RESULTS_DIR / "effects_wlz_vs_raw.csv"
OUT_OVERALL = RESULTS_DIR / "overall_summary.csv"
OUT_MODE_SPEED = RESULTS_DIR / "mode_speed_stats.csv"  # NEW: for bar-plot speed ratios
OUT_RAW_FAILURES = RESULTS_DIR / "raw_failure_summary.csv"
OUT_PAIRS_REFERENCE = RESULTS_DIR / "pairs_vs_reference.csv"
OUT_LARGE_LAZY_COMPARISON = RESULTS_DIR / "large_lazy_comparison.csv"
OUT_CASE_RUNTIME_SCALE = RESULTS_DIR / "case_runtime_scale.csv"
REPORT_CASES = list(CASES_SMALL) + list(CASES_LARGE)
LARGE_REFERENCE_MODE = "LAZY_ALL"
LARGE_LAZY_COLUMNS = [
    "case_folder",
    "reference_mode",
    "mode",
    "N",
    "runtime_median",
    "speedup_vs_lazy",
    "success_rate",
    "feasible_rate",
    "mip_gap_median",
    "method_exactness",
]
_NOREL_ELAPSED_RE = re.compile(r"Elapsed time for NoRel heuristic:\s*([0-9.]+)s")
_LOG_METRIC_CACHE: Dict[str, float] = {}


def _mode_exactness(mode: str) -> str:
    return mode_exactness(mode)


def _runtime_col(df: pd.DataFrame) -> str:
    if "runtime_report_sec" in df.columns:
        return "runtime_report_sec"
    if "wall_sec" in df.columns:
        return "wall_sec"
    return "runtime_sec"


def _clean_text(value) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"nan", "none", "na"} else text


_OK_TEXT = {"OK", "TRUE", "1", "YES"}
_FAIL_TEXT = {"FAIL", "FALSE", "0", "NO"}


def _norm_text_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df[col].astype(str).str.strip().str.upper().replace(
        {"NAN": "", "NONE": "", "NA": ""}
    )


def _milp_feasible_mask(df: pd.DataFrame) -> pd.Series:
    """Verified MILP feasibility, independent of solver status and target MIP gap."""
    violations = _norm_text_col(df, "violations")
    feasible = _norm_text_col(df, "feasible_ok")

    has_violations_signal = violations.ne("")
    ok = has_violations_signal & violations.isin(_OK_TEXT)

    has_feasible_signal = feasible.ne("")
    feasible_ok = feasible.isin(_OK_TEXT)
    feasible_fail = feasible.isin(_FAIL_TEXT)

    ok = ok.where(has_violations_signal, feasible_ok)
    ok = ok & ~(has_feasible_signal & feasible_fail)

    if "has_incumbent" in df.columns:
        incumbent = pd.to_numeric(df["has_incumbent"], errors="coerce")
        ok = ok & (incumbent.isna() | (incumbent > 0.0))

    return ok.fillna(False)


def _safe_gurobi_log_path(row: pd.Series) -> str:
    run_id = _clean_text(row.get("run_id", ""))
    stage = _clean_text(row.get("stage", "TEST")).lower() or "test"
    instance_name = _clean_text(row.get("instance_name", ""))
    mode_id = _clean_text(row.get("mode_id", "")) or _clean_text(row.get("mode_original", "")) or _clean_text(row.get("mode", ""))
    if not run_id or not instance_name or not mode_id:
        return ""
    safe_case = instance_name.strip("/\\").replace("/", "_").replace("\\", "_")
    safe_mode = mode_id.lower()
    path = RESULTS_DIR / run_id / "logs" / "gurobi" / stage / f"{safe_case}__{safe_mode}.log"
    return path.as_posix()


def _parse_norel_elapsed_sec(log_path: str) -> float:
    if not log_path:
        return np.nan
    cached = _LOG_METRIC_CACHE.get(log_path)
    if cached is not None:
        return cached
    path = Path(log_path)
    if not path.is_file():
        _LOG_METRIC_CACHE[log_path] = np.nan
        return np.nan
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        _LOG_METRIC_CACHE[log_path] = np.nan
        return np.nan
    values = [float(x) for x in _NOREL_ELAPSED_RE.findall(text)]
    value = max(values) if values else 0.0
    _LOG_METRIC_CACHE[log_path] = value
    return value


def _add_gurobi_log_metrics(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["gurobi_log_path"] = out.apply(_safe_gurobi_log_path, axis=1)
    out["norel_heur_elapsed_sec"] = out["gurobi_log_path"].map(_parse_norel_elapsed_sec)
    runtime = pd.to_numeric(out["runtime_report_sec"], errors="coerce")
    norel = pd.to_numeric(out["norel_heur_elapsed_sec"], errors="coerce")
    out["runtime_ex_norel_sec"] = runtime - norel
    out.loc[out["runtime_ex_norel_sec"] < 0.0, "runtime_ex_norel_sec"] = 0.0
    out.loc[norel.isna(), "runtime_ex_norel_sec"] = np.nan
    return out


def _discover_result_sources() -> List[str]:
    files: List[str] = []
    files.extend(sorted(glob.glob(str(RESULTS_DIR / "*" / "results.csv"))))
    if not files:
        files.extend(sorted(glob.glob(str(RAW_DIR / "*.csv"))))
    return list(dict.fromkeys(files))


def _read_all_logs() -> pd.DataFrame:
    files = _discover_result_sources()
    if not files:
        raise FileNotFoundError(
            "No result CSVs found. Looked for "
            f"{RESULTS_DIR / '*' / 'results.csv'} and {RAW_DIR / '*.csv'}."
        )
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, on_bad_lines="skip")
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df["logfile"] = str(Path(f).as_posix())
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(
            f"All discovered result CSVs were empty: {files}. "
            f"Run experiments.py first or remove header-only logs."
        )
    data = pd.concat(dfs, ignore_index=True)
    if "stage" in data.columns:
        data = data[data["stage"].astype(str).str.upper().eq("TEST")].copy()
        if data.empty:
            raise FileNotFoundError("No TEST rows found in the selected result source.")

    # Ensure correct dtypes
    float_cols = [
        "runtime_sec",
        "nodes",
        "obj_val",
        "obj_bound",
        "num_constrs_root",
        "num_vars_root",
        "num_constrs_final",
        "num_vars_final",
        "peak_memory_gb",
        "branch_hints_applied",
        "warm_start_applied_vars",
        "constr_total_cont",
        "constr_kept_cont",
        "constr_ratio_cont",
        "constr_ratio_cont_explicit",
        "constr_realized_cont",
        "constr_ratio_cont_realized",
        "wall_sec",
        "runtime_report_sec",
        "screen_setup_sec",
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
        "mip_gap",
        "max_constraint_residual",
        "objective_inconsistency",
    ]
    for col in float_cols:
        if col not in data.columns:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if "mode" not in data.columns:
        if "mode_id" in data.columns:
            data["mode"] = data["mode_id"].astype(str)
        else:
            raise KeyError("Missing required column 'mode' (or 'mode_id').")
    data["mode_original"] = data["mode"].astype(str)
    data["mode"] = data["mode"].map(canonical_mode)
    if "timestamp" not in data.columns:
        if "timestamp_utc" in data.columns:
            data["timestamp"] = data["timestamp_utc"].astype(str)
        else:
            raise KeyError("Missing required column 'timestamp' (or 'timestamp_utc').")

    # Standard categorical/text cols
    for col in [
        "timestamp",
        "case_folder",
        "mode",
        "instance_name",
        "violations",
        "status",
        "feasible_ok",
    ]:
        if col not in data.columns:
            data[col] = ""
        data[col] = data[col].astype(str)
        data[col] = data[col].replace({"nan": "", "None": "", "NaN": ""})

    data["_timestamp_dt"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data["_row_order"] = np.arange(len(data), dtype=int)

    if "runtime_report_sec" not in data.columns:
        data["runtime_report_sec"] = np.nan
    data["runtime_report_sec"] = pd.to_numeric(
        data["runtime_report_sec"], errors="coerce"
    )
    if "wall_sec" in data.columns:
        data["runtime_report_sec"] = data["runtime_report_sec"].fillna(
            pd.to_numeric(data["wall_sec"], errors="coerce")
        )
    if "runtime_sec" in data.columns:
        data["runtime_report_sec"] = data["runtime_report_sec"].fillna(
            pd.to_numeric(data["runtime_sec"], errors="coerce")
        )
    no_incumbent = pd.to_numeric(data.get("has_incumbent", np.nan), errors="coerce").fillna(0.0) <= 0.0
    data.loc[no_incumbent, ["mip_gap", "obj_val"]] = np.nan
    skipped = data["status"].astype(str).str.strip().str.upper().str.startswith("SKIPPED")
    data.loc[skipped, "runtime_report_sec"] = np.nan
    data = _add_gurobi_log_metrics(data)
    data.loc[skipped, ["norel_heur_elapsed_sec", "runtime_ex_norel_sec"]] = np.nan

    if "constr_ratio_cont_explicit" not in data.columns:
        data["constr_ratio_cont_explicit"] = np.nan
    if "constr_ratio_cont_realized" not in data.columns:
        data["constr_ratio_cont_realized"] = np.nan
    if "constr_realized_cont" not in data.columns:
        data["constr_realized_cont"] = np.nan
    if "method_exactness" not in data.columns:
        data["method_exactness"] = ""

    # Normalize mode casing a bit
    data["mode_clean"] = data["mode"].map(canonical_mode)
    # Older result rows stored `method_exactness` before the classification was
    # tightened for screened/reduced modes. The paper reports the canonical
    # current classification, so normalize from the mode id instead of trusting
    # stale per-row metadata.
    data["method_exactness"] = data["mode_clean"].apply(_mode_exactness)

    # Add technique flags for grouping/plots
    data["with_LAZY"] = data["mode_clean"].str.contains("LAZY", case=False, na=False)
    data["with_PRUNE"] = data["mode_clean"].str.contains("PRUNE", case=False, na=False)
    data["with_LPSCREEN"] = data["mode_clean"].str.contains("LPSCREEN", case=False, na=False)
    data["with_SR"] = data["mode_clean"].str.contains("SR_LAZY", case=False, na=False)
    data["with_ACTIVESET"] = data["mode_clean"].str.contains("ACTIVESET", case=False, na=False)
    data["with_SHRINK"] = data["mode_clean"].str.contains("SHRINK", case=False, na=False)
    data["with_STREDUCE"] = data["mode_clean"].str.contains("STREDUCE", case=False, na=False)
    data["with_GNN"] = data["mode_clean"].str.contains("GNN", case=False, na=False)
    data["with_COMMIT"] = data["mode_clean"].str.contains("COMMIT", case=False, na=False)
    data["with_GRU"] = data["mode_clean"].str.contains("GRU", case=False, na=False)
    data["with_BANDIT"] = data["mode_clean"].str.contains("BANDIT", case=False, na=False)
    data["is_heuristic_method"] = data["method_exactness"] == "heuristic"
    data["is_exact_method"] = data["method_exactness"] == "exact"
    data["is_raw_baseline"] = data["mode_clean"] == RAW_BASELINE_MODE

    # Keep the latest row per (case, instance, mode) to avoid double counting reruns.
    data = (
        data.sort_values(["_timestamp_dt", "logfile", "_row_order"], kind="stable")
        .groupby(["case_folder", "instance_name", "mode"], as_index=False, sort=False)
        .tail(1)
        .copy()
    )
    data = data.drop(columns=["_timestamp_dt", "_row_order"], errors="ignore")
    data = data[data["case_folder"].isin(REPORT_CASES)].copy()

    return data


def _compute_obj_ppm_vs_raw(df: pd.DataFrame) -> pd.DataFrame:
    # For each instance_name, get RAW obj as baseline
    raw = df[df["mode_clean"] == RAW_BASELINE_MODE][
        ["instance_name", "obj_val"]
    ].copy()
    raw = raw.rename(columns={"obj_val": "obj_raw"})
    out = df.merge(raw, on="instance_name", how="left")

    def ppm(row):
        try:
            a = float(row["obj_val"])
            b = float(row["obj_raw"])
            if not math.isfinite(a) or not math.isfinite(b) or b == 0:
                return np.nan
            return 1e6 * (a - b) / abs(b)
        except Exception:
            return np.nan

    out["obj_ppm_vs_raw"] = out.apply(ppm, axis=1)
    return out


def _raw_failure_summary(df: pd.DataFrame) -> pd.DataFrame:
    raw = df[df["mode_clean"] == RAW_BASELINE_MODE].copy()
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "case_folder",
                "raw_runs",
                "raw_time_limit_runs",
                "raw_time_limit_no_incumbent",
                "raw_time_limit_with_incumbent",
                "raw_no_feasible_within_limit",
                "raw_pass",
                "raw_no_feasible_rate",
            ]
        )
    status = raw["status"].astype(str).str.upper()
    milp_feasible = _milp_feasible_mask(raw)
    if "has_incumbent" in raw.columns:
        incumbent = pd.to_numeric(raw["has_incumbent"], errors="coerce")
    else:
        incumbent = pd.Series(np.nan, index=raw.index)
    inferred_incumbent = status.isin(["OPTIMAL", "SUBOPTIMAL"]) | milp_feasible
    raw["has_incumbent_num"] = incumbent.where(incumbent.notna(), inferred_incumbent.astype(float)).fillna(0.0)
    raw["pass_num"] = milp_feasible.astype(float)
    raw["time_limit_run"] = status.str.contains("TIME_LIMIT", na=False)
    raw["time_limit_no_incumbent"] = raw["time_limit_run"] & (raw["has_incumbent_num"] <= 0.0)
    raw["time_limit_with_incumbent"] = raw["time_limit_run"] & (raw["has_incumbent_num"] > 0.0)
    raw["no_feasible_within_limit"] = ~milp_feasible
    grouped = raw.groupby("case_folder", dropna=False)
    out = grouped.agg(
        raw_runs=("instance_name", "count"),
        raw_time_limit_runs=("time_limit_run", "sum"),
        raw_time_limit_no_incumbent=("time_limit_no_incumbent", "sum"),
        raw_time_limit_with_incumbent=("time_limit_with_incumbent", "sum"),
        raw_no_feasible_within_limit=("no_feasible_within_limit", "sum"),
        raw_pass=("pass_num", "sum"),
    ).reset_index()
    out["raw_pass"] = out["raw_pass"].astype(int)
    out["raw_no_feasible_rate"] = out["raw_no_feasible_within_limit"] / out["raw_runs"].replace(0, np.nan)
    return out


def _bootstrap_ci95_median(
    x: np.ndarray, nboot: int = 2000, seed: int = 123
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan
    meds = []
    for _ in range(nboot):
        samp = rng.choice(x, size=x.size, replace=True)
        meds.append(np.median(samp))
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def _nanmedian_safe(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return np.nan
    if not np.isfinite(arr).any():
        return np.nan
    return float(np.nanmedian(arr))


def _iqr(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def _pair_with_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-instance pairing vs RAW baseline for all other modes.
    Returns per-instance rows with deltas/speedups:
      - runtime_speedup = raw_runtime / mode_runtime
      - runtime_ratio   = mode_runtime / raw_runtime (values < 1 => faster)
      - runtime_delta   = mode_runtime - raw_runtime (negative means faster)
      - nodes_delta     = mode_nodes - raw_nodes
      - mem_delta_gb    = mode_mem - raw_mem
      - root_constr_ratio = mode_root_constr / raw_root_constr
      - final_to_root_ratio = mode_num_constrs_final / mode_num_constrs_root
    """
    rows = []
    runtime_col = _runtime_col(df)
    for (case, inst), g in df.groupby(["case_folder", "instance_name"]):
        g_raw = g[g["mode_clean"] == RAW_BASELINE_MODE]
        if g_raw.empty:
            continue
        raw = g_raw.iloc[0]
        raw_rt = float(raw.get(runtime_col, np.nan))
        raw_nodes = float(raw.get("nodes", np.nan))
        raw_mem = float(raw.get("peak_memory_gb", np.nan))
        raw_root = float(raw.get("num_constrs_root", np.nan))
        for _, r in g.iterrows():
            mode = r["mode"]
            if str(mode).upper() == RAW_BASELINE_MODE:
                continue
            rt = float(r.get(runtime_col, np.nan))
            nodes = float(r.get("nodes", np.nan))
            mem = float(r.get("peak_memory_gb", np.nan))
            root = float(r.get("num_constrs_root", np.nan))
            final = float(r.get("num_constrs_final", np.nan))

            spd = (
                raw_rt / rt
                if (rt and rt > 0 and math.isfinite(rt) and math.isfinite(raw_rt))
                else np.nan
            )
            ratio = (
                rt / raw_rt
                if (rt and raw_rt and rt > 0 and raw_rt > 0 and math.isfinite(raw_rt))
                else np.nan
            )
            d_rt = (
                rt - raw_rt if (math.isfinite(rt) and math.isfinite(raw_rt)) else np.nan
            )
            d_nd = (
                nodes - raw_nodes
                if (math.isfinite(nodes) and math.isfinite(raw_nodes))
                else np.nan
            )
            d_mem = (
                mem - raw_mem
                if (math.isfinite(mem) and math.isfinite(raw_mem))
                else np.nan
            )
            root_ratio = (
                (root / raw_root)
                if (math.isfinite(root) and math.isfinite(raw_root) and raw_root > 0)
                else np.nan
            )
            fr_ratio = (
                (final / root)
                if (math.isfinite(final) and math.isfinite(root) and root > 0)
                else np.nan
            )

            rows.append(
                {
                    "case_folder": case,
                    "instance_name": inst,
                    "mode": mode,
                    "runtime_speedup": spd,
                    "runtime_ratio": ratio,  # NEW
                    "runtime_delta": d_rt,
                    "nodes_delta": d_nd,
                    "mem_delta_gb": d_mem,
                    "root_constr_ratio": root_ratio,
                    "final_to_root_ratio": fr_ratio,
                    "obj_ppm_vs_raw": r.get("obj_ppm_vs_raw", np.nan),
                    "feasible_ok": r.get("feasible_ok", ""),
                    "violations": r.get("violations", ""),
                    "status": r.get("status", ""),
                    "mip_gap": r.get("mip_gap", np.nan),
                    "warm_start_applied_vars": r.get("warm_start_applied_vars", np.nan),
                    "branch_hints_applied": r.get("branch_hints_applied", np.nan),
                }
            )
    return pd.DataFrame(rows)


def _reference_mode_for_case(case_folder: str) -> str:
    return LARGE_REFERENCE_MODE if case_folder in set(CASES_LARGE) else RAW_BASELINE_MODE


def _pair_with_reference(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    runtime_col = _runtime_col(df)
    for (case, inst), g in df.groupby(["case_folder", "instance_name"]):
        ref_mode = _reference_mode_for_case(str(case))
        g_ref = g[g["mode_clean"] == ref_mode]
        if g_ref.empty:
            continue
        ref = g_ref.iloc[0]
        ref_rt = float(ref.get(runtime_col, np.nan))
        for _, r in g.iterrows():
            mode = str(r["mode"])
            rt = float(r.get(runtime_col, np.nan))
            speedup = (
                ref_rt / rt
                if (math.isfinite(ref_rt) and math.isfinite(rt) and ref_rt > 0 and rt > 0)
                else np.nan
            )
            rows.append(
                {
                    "case_folder": case,
                    "instance_name": inst,
                    "reference_mode": ref_mode,
                    "mode": mode,
                    "runtime_speedup_vs_reference": speedup,
                    "runtime_delta_vs_reference": (
                        rt - ref_rt
                        if (math.isfinite(ref_rt) and math.isfinite(rt))
                        else np.nan
                    ),
                    "status": r.get("status", ""),
                    "feasible_ok": r.get("feasible_ok", ""),
                    "violations": r.get("violations", ""),
                    "mip_gap": r.get("mip_gap", np.nan),
                }
            )
    return pd.DataFrame(rows)


def _wilcoxon_pairs(
    df: pd.DataFrame, case: str, a_mode: str, b_mode: str, col: str
) -> Tuple[float, float, float]:
    """
    Paired Wilcoxon signed-rank between a_mode and b_mode on column 'col' for one case.
    Returns (p-value, HL median of diffs a-b, rank-biserial correlation).
      - HL: for paired samples, we use median of paired differences (robust effect).
      - RBC: 2*W/T - 1, where W is sum of ranks for positive diffs, T=n(n+1)/2.
             Positive RBC => a_mode tends larger than b_mode.
    """
    ga = df[(df["case_folder"] == case) & (df["mode"] == a_mode)][
        ["instance_name", col]
    ].dropna()
    gb = df[(df["case_folder"] == case) & (df["mode"] == b_mode)][
        ["instance_name", col]
    ].dropna()
    merged = ga.merge(gb, on="instance_name", suffixes=("_a", "_b"))
    if merged.empty:
        return (np.nan, np.nan, np.nan)
    x = merged[f"{col}_a"].to_numpy(dtype=float)
    y = merged[f"{col}_b"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return (np.nan, np.nan, np.nan)
    x = x[finite]
    y = y[finite]
    diffs = x - y
    if diffs.size == 0:
        return (np.nan, np.nan, np.nan)
    hl_med = _nanmedian_safe(diffs)

    # Degenerate paired samples are common for `nodes` on easy cases:
    # all differences can be exactly zero, which makes SciPy's Wilcoxon
    # raise and/or emit runtime warnings. Treat identical paired samples as
    # "no detectable difference" instead of polluting stdout with warnings.
    nonzero_mask = np.abs(diffs) > 1e-12
    diffs_nz = diffs[nonzero_mask]
    if diffs_nz.size == 0:
        return (1.0, 0.0, 0.0)

    # Rank-biserial correlation from signed ranks of nonzero differences.
    try:
        ranks = rankdata(np.abs(diffs_nz), method="average")
        w_pos = float(np.sum(ranks[diffs_nz > 0]))
        w_neg = float(np.sum(ranks[diffs_nz < 0]))
        denom = w_pos + w_neg
        rbc = ((w_pos - w_neg) / denom) if denom > 0 else 0.0
    except Exception:
        rbc = np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stat = wilcoxon(
                diffs_nz,
                zero_method="wilcox",
                alternative="two-sided",
            )
        pval = float(stat.pvalue)
    except Exception:
        pval = np.nan
    return (pval, hl_med, rbc)


def _build_extended_summary(
    df: pd.DataFrame, pairs: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extended per-case/per-mode summary with:
      - success/feasibility rates
      - mip gap stats
      - warm-start and branching-hints utilization
      - Wilcoxon vs RAW (p, HL, RBC) for runtime and nodes (for modes ≠ RAW)
    Also returns effects_wlz (WARM_LAZY vs RAW) per case for plotting.
    """
    rows = []
    effects_wlz = []
    runtime_col = _runtime_col(df)

    for case, g_case in df.groupby("case_folder"):
        # Baseline medians for speedup
        base = g_case[g_case["mode_clean"] == RAW_BASELINE_MODE]
        base_rt_med = (
            _nanmedian_safe(pd.to_numeric(base[runtime_col], errors="coerce").to_numpy())
            if not base.empty
            else np.nan
        )
        base_rt_ex_norel_med = (
            _nanmedian_safe(pd.to_numeric(base.get("runtime_ex_norel_sec", np.nan), errors="coerce").to_numpy())
            if not base.empty
            else np.nan
        )
        reference_mode = _reference_mode_for_case(str(case))
        reference = g_case[g_case["mode_clean"] == reference_mode]
        reference_rt_med = (
            _nanmedian_safe(pd.to_numeric(reference[runtime_col], errors="coerce").to_numpy())
            if not reference.empty
            else np.nan
        )

        for mode, g_mode in g_case.groupby("mode"):
            # Success/feasibility for paper plots is MILP feasibility only:
            # solver status and target MIPGap are diagnostic, not pass/fail criteria.
            milp_ok = _milp_feasible_mask(g_mode)
            violations = _norm_text_col(g_mode, "violations")
            has_violations_signal = violations.ne("")
            constraint_ok = violations.isin(_OK_TEXT).where(has_violations_signal, milp_ok)
            feas = float(constraint_ok.mean()) if not g_mode.empty else np.nan
            succ = float(milp_ok.mean()) if not g_mode.empty else np.nan
            # MIP gap
            mip = pd.to_numeric(g_mode.get("mip_gap", np.nan), errors="coerce")
            mip_med = _nanmedian_safe(
                mip.to_numpy() if hasattr(mip, "to_numpy") else mip
            )
            # Warm start / hints usage
            ws = pd.to_numeric(
                g_mode.get("warm_start_applied_vars", np.nan), errors="coerce"
            )
            ws_med = _nanmedian_safe(ws.to_numpy() if hasattr(ws, "to_numpy") else ws)
            bh = pd.to_numeric(
                g_mode.get("branch_hints_applied", np.nan), errors="coerce"
            )
            bh_med = _nanmedian_safe(bh.to_numpy() if hasattr(bh, "to_numpy") else bh)

            # Median/IQR/CI metrics
            rt = pd.to_numeric(g_mode[runtime_col], errors="coerce").to_numpy()
            rt_ex_norel = pd.to_numeric(g_mode.get("runtime_ex_norel_sec", np.nan), errors="coerce").to_numpy()
            norel_elapsed = pd.to_numeric(g_mode.get("norel_heur_elapsed_sec", np.nan), errors="coerce").to_numpy()
            nd = pd.to_numeric(g_mode["nodes"], errors="coerce").to_numpy()
            mem = pd.to_numeric(g_mode["peak_memory_gb"], errors="coerce").to_numpy()
            ppm = pd.to_numeric(g_mode["obj_ppm_vs_raw"], errors="coerce").to_numpy()
            root = pd.to_numeric(g_mode["num_constrs_root"], errors="coerce").to_numpy()
            final = pd.to_numeric(
                g_mode["num_constrs_final"], errors="coerce"
            ).to_numpy()

            def med_iqr_ci(x):
                return (
                    _nanmedian_safe(x),
                    _iqr(x),
                    *_bootstrap_ci95_median(x),
                )

            med_rt, iqr_rt, lo_rt, hi_rt = med_iqr_ci(rt)
            med_rt_ex_norel = _nanmedian_safe(rt_ex_norel)
            med_norel_elapsed = _nanmedian_safe(norel_elapsed)
            med_nd, iqr_nd, lo_nd, hi_nd = med_iqr_ci(nd)
            med_mem, iqr_mem, lo_mem, hi_mem = med_iqr_ci(mem)
            med_ppm, iqr_ppm, lo_ppm, hi_ppm = med_iqr_ci(ppm)
            med_root, _, _, _ = med_iqr_ci(root)
            med_final, _, _, _ = med_iqr_ci(final)
            fin_root_ratio = (
                (med_final / med_root)
                if (
                    math.isfinite(med_final)
                    and math.isfinite(med_root)
                    and med_root > 0
                )
                else np.nan
            )

            # Speed-up vs RAW (use medians)
            su = (
                base_rt_med / med_rt
                if (math.isfinite(base_rt_med) and math.isfinite(med_rt) and med_rt > 0)
                else np.nan
            )
            su_ex_norel = (
                base_rt_ex_norel_med / med_rt_ex_norel
                if (
                    math.isfinite(base_rt_ex_norel_med)
                    and math.isfinite(med_rt_ex_norel)
                    and med_rt_ex_norel > 0
                )
                else np.nan
            )
            su_ref = (
                reference_rt_med / med_rt
                if (
                    math.isfinite(reference_rt_med)
                    and math.isfinite(med_rt)
                    and med_rt > 0
                )
                else np.nan
            )

            # Paired Wilcoxon vs RAW on runtime and nodes
            p_rt, hl_rt, rbc_rt = (np.nan, np.nan, np.nan)
            p_nd, hl_nd, rbc_nd = (np.nan, np.nan, np.nan)
            if not str(mode).upper() == RAW_BASELINE_MODE:
                p_rt, hl_rt, rbc_rt = _wilcoxon_pairs(
                    df, case, mode, RAW_BASELINE_MODE, runtime_col
                )
                p_nd, hl_nd, rbc_nd = _wilcoxon_pairs(
                    df, case, mode, RAW_BASELINE_MODE, "nodes"
                )
                # store WLZ-only effects for plotting
                if mode == "WARM_LAZY":
                    effects_wlz.append(
                        {
                            "case_folder": case,
                            "p_runtime": p_rt,
                            "HL_runtime_delta": hl_rt,
                            "RBC_runtime": rbc_rt,
                            "p_nodes": p_nd,
                            "HL_nodes_delta": hl_nd,
                            "RBC_nodes": rbc_nd,
                        }
                    )

            rows.append(
                {
                    "case_folder": case,
                    "mode": mode,
                    "method_exactness": _mode_exactness(mode),
                    "N": int(len(g_mode)),
                    "runtime_median": med_rt,
                    "runtime_ex_norel_median": med_rt_ex_norel,
                    "norel_heur_elapsed_median": med_norel_elapsed,
                    "runtime_IQR": iqr_rt,
                    "runtime_CI95_lo": lo_rt,
                    "runtime_CI95_hi": hi_rt,
                    "nodes_median": med_nd,
                    "nodes_IQR": iqr_nd,
                    "nodes_CI95_lo": lo_nd,
                    "nodes_CI95_hi": hi_nd,
                    "mem_gb_median": med_mem,
                    "mem_gb_IQR": iqr_mem,
                    "mem_gb_CI95_lo": lo_mem,
                    "mem_gb_CI95_hi": hi_mem,
                    "obj_ppm_median": med_ppm,
                    "obj_ppm_IQR": iqr_ppm,
                    "obj_ppm_CI95_lo": lo_ppm,
                    "obj_ppm_CI95_hi": hi_ppm,
                    "root_constrs_median": med_root,
                    "final_constrs_median": med_final,
                    "final_to_root_constr_ratio_median": fin_root_ratio,
                    "success_rate": float(succ),
                    "feasible_rate": float(feas),
                    "mip_gap_median": mip_med,
                    "warm_applied_vars_median": ws_med,
                    "branch_hints_median": bh_med,
                    "speedup_vs_raw": su,
                    "speedup_vs_raw_ex_norel": su_ex_norel,
                    "reference_mode": reference_mode,
                    "speedup_vs_reference": su_ref,
                    "wilcoxon_p_runtime": p_rt,
                    "HL_delta_runtime": hl_rt,
                    "RBC_runtime": rbc_rt,
                    "wilcoxon_p_nodes": p_nd,
                    "HL_delta_nodes": hl_nd,
                    "RBC_nodes": rbc_nd,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(effects_wlz)


def _overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapsed across cases: per-mode medians and IQR for main metrics.
    """
    rows = []
    runtime_col = _runtime_col(df)
    for mode, g in df.groupby("mode"):
        rt = pd.to_numeric(g[runtime_col], errors="coerce").to_numpy()
        nd = pd.to_numeric(g["nodes"], errors="coerce").to_numpy()
        ppm = pd.to_numeric(g["obj_ppm_vs_raw"], errors="coerce").to_numpy()
        mem = pd.to_numeric(g["peak_memory_gb"], errors="coerce").to_numpy()

        def med_iqr(x):
            return _nanmedian_safe(x), _iqr(x)

        med_rt, iqr_rt = med_iqr(rt)
        med_nd, iqr_nd = med_iqr(nd)
        med_ppm, iqr_ppm = med_iqr(ppm)
        med_mem, iqr_mem = med_iqr(mem)
        rows.append(
            {
                "mode": mode,
                "method_exactness": _mode_exactness(mode),
                "N": int(len(g)),
                "runtime_median": med_rt,
                "runtime_IQR": iqr_rt,
                "nodes_median": med_nd,
                "nodes_IQR": iqr_nd,
                "obj_ppm_median": med_ppm,
                "obj_ppm_IQR": iqr_ppm,
                "mem_gb_median": med_mem,
                "mem_gb_IQR": iqr_mem,
            }
        )
    return pd.DataFrame(rows)


def _pairs_all_modes(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (case, instance), compute all pairwise comparisons between modes present.
    Emits one row per ordered pair (mode_a, mode_b) with:
      - speedup_a_over_b = runtime_b / runtime_a
      - delta_runtime = runtime_a - runtime_b (negative => a is slower)
      - delta_nodes = nodes_a - nodes_b
    """
    out = []
    runtime_col = _runtime_col(df)
    for (case, inst), g in df.groupby(["case_folder", "instance_name"]):
        # Keep only rows with runtime present
        gg = g.dropna(subset=[runtime_col]).copy()
        if len(gg) < 2:
            continue
        # build dicts
        runs = dict(zip(gg["mode"], gg[runtime_col]))
        nodes = dict(zip(gg["mode"], gg["nodes"]))
        modes = list(runs.keys())
        for i in range(len(modes)):
            for j in range(len(modes)):
                if i == j:
                    continue
                a = modes[i]
                b = modes[j]
                rt_a = runs.get(a, np.nan)
                rt_b = runs.get(b, np.nan)
                if not (
                    math.isfinite(rt_a)
                    and math.isfinite(rt_b)
                    and rt_a > 0
                    and rt_b > 0
                ):
                    continue
                nd_a = nodes.get(a, np.nan)
                nd_b = nodes.get(b, np.nan)
                out.append(
                    {
                        "case_folder": case,
                        "instance_name": inst,
                        "mode_a": a,
                        "mode_b": b,
                        "speedup_a_over_b": rt_b / rt_a,
                        "delta_runtime": rt_a - rt_b,
                        "delta_nodes": (nd_a - nd_b)
                        if (math.isfinite(nd_a) and math.isfinite(nd_b))
                        else np.nan,
                    }
                )
    return pd.DataFrame(out)


def _fastest_share(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (case, instance) pick the mode with minimal runtime.
    Aggregate per case to compute share of being fastest.
    """
    rows = []
    runtime_col = _runtime_col(df)
    for (case, inst), g in df.groupby(["case_folder", "instance_name"]):
        g2 = g.dropna(subset=[runtime_col])
        if g2.empty:
            continue
        best_row = g2.loc[g2[runtime_col].idxmin()]
        rows.append(
            {"case_folder": case, "instance_name": inst, "best_mode": best_row["mode"]}
        )
    if not rows:
        return pd.DataFrame(columns=["case_folder", "mode", "fastest_share"])
    best = pd.DataFrame(rows)
    share = best.groupby(["case_folder", "best_mode"]).size().reset_index(name="count")
    total = share.groupby("case_folder")["count"].transform("sum")
    share["fastest_share"] = share["count"] / total
    share = share.rename(columns={"best_mode": "mode"})
    return share[["case_folder", "mode", "fastest_share"]]


def _mode_speed_stats(pairs: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-mode speed statistics for plots:
     - mean_runtime_ratio (mode_runtime / raw_runtime) — values < 1 => faster
     - std_runtime_ratio
     - N (paired count)
     - success_rate_strict (verified MILP feasible)
     - success_rate_status (status in OPTIMAL/SUBOPTIMAL/TIME_LIMIT)
     - method_exactness (exact / heuristic)
    """
    if pairs.empty:
        return pd.DataFrame(
            columns=[
                "mode",
                "mean_runtime_ratio",
                "std_runtime_ratio",
                "N",
                "success_rate_strict",
                "success_rate_status",
                "method_exactness",
            ]
        )

    # per-mode ratio stats from pairs
    pr = pairs.dropna(subset=["runtime_ratio"]).copy()
    grp = pr.groupby("mode")["runtime_ratio"]
    stats = grp.agg(["mean", "std", "count"]).reset_index()
    stats = stats.rename(
        columns={
            "mean": "mean_runtime_ratio",
            "std": "std_runtime_ratio",
            "count": "N",
        }
    )

    # per-mode success from merged (all rows)
    d = merged.copy()
    d["ok_strict"] = _milp_feasible_mask(d).astype(float)
    d["ok_status"] = (
        d["status"].isin(["OPTIMAL", "SUBOPTIMAL", "TIME_LIMIT"]).astype(float)
    )
    succ = (
        d.groupby("mode")
        .agg(
            success_rate_strict=("ok_strict", "mean"),
            success_rate_status=("ok_status", "mean"),
        )
        .reset_index()
    )

    mode_exact = (
        d[["mode", "method_exactness"]]
        .dropna()
        .drop_duplicates(subset=["mode"], keep="last")
    )
    out = stats.merge(succ, on="mode", how="left").merge(mode_exact, on="mode", how="left")
    out["method_exactness"] = out["method_exactness"].fillna(
        out["mode"].map(_mode_exactness)
    )
    return out


def _large_lazy_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    large = summary[summary["case_folder"].isin(CASES_LARGE)].copy()
    if large.empty:
        return pd.DataFrame(columns=LARGE_LAZY_COLUMNS)
    rows = []
    for case, g in large.groupby("case_folder"):
        g = g.copy()
        g["runtime_median"] = pd.to_numeric(g["runtime_median"], errors="coerce")
        g = g.dropna(subset=["runtime_median"])
        if g.empty:
            continue
        ref = g[g["mode"] == LARGE_REFERENCE_MODE]
        if ref.empty:
            continue
        ref_rt = float(ref.iloc[0]["runtime_median"])
        candidates = g[g["mode"] != LARGE_REFERENCE_MODE].copy()
        if candidates.empty:
            selected = ref.copy()
        else:
            selected = pd.concat([ref, candidates.nsmallest(1, "runtime_median")], ignore_index=True)
        for _, r in selected.iterrows():
            rows.append(
                {
                    "case_folder": case,
                    "reference_mode": LARGE_REFERENCE_MODE,
                    "mode": r["mode"],
                    "N": int(r.get("N", 0)),
                    "runtime_median": float(r["runtime_median"]),
                    "speedup_vs_lazy": (
                        ref_rt / float(r["runtime_median"])
                        if float(r["runtime_median"]) > 0
                        else np.nan
                    ),
                    "success_rate": float(r.get("success_rate", np.nan)),
                    "feasible_rate": float(r.get("feasible_rate", np.nan)),
                    "mip_gap_median": float(r.get("mip_gap_median", np.nan)),
                    "method_exactness": r.get("method_exactness", _mode_exactness(r["mode"])),
                }
            )
    return pd.DataFrame(rows, columns=LARGE_LAZY_COLUMNS)


def _case_size(case_folder: str) -> int:
    tag = str(case_folder).split("/")[-1]
    digits = "".join(ch for ch in tag if ch.isdigit())
    return int(digits) if digits else 0


def _case_runtime_scale(df: pd.DataFrame) -> pd.DataFrame:
    runtime_col = _runtime_col(df)
    d = df.copy()
    d["runtime_sec"] = pd.to_numeric(d[runtime_col], errors="coerce")
    d = d.dropna(subset=["runtime_sec"])
    if d.empty:
        return pd.DataFrame()
    out = (
        d.groupby(["case_folder", "mode"], as_index=False)
        .agg(
            case_size=("case_folder", lambda x: _case_size(str(x.iloc[0]))),
            avg_runtime_sec=("runtime_sec", "mean"),
            median_runtime_sec=("runtime_sec", "median"),
            runs=("runtime_sec", "size"),
        )
        .sort_values(["case_size", "mode"])
    )
    out["regime"] = np.where(out["case_folder"].isin(CASES_LARGE), "large", "small")
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Read and enrich
    df = _read_all_logs()
    df = _compute_obj_ppm_vs_raw(df)

    # Write enriched merged (keeps with_* flags)
    df.to_csv(OUT_MERGED, index=False)

    # Pairs vs RAW
    pairs = _pair_with_raw(df)
    pairs.to_csv(OUT_PAIRS, index=False)

    pairs_ref = _pair_with_reference(df)
    pairs_ref.to_csv(OUT_PAIRS_REFERENCE, index=False)

    # Extended summary and effects
    summary_ext, effects_wlz = _build_extended_summary(df, pairs)
    summary_ext.to_csv(OUT_SUMMARY, index=False)
    summary_ext.to_csv(OUT_SUMMARY_EXT, index=False)
    effects_wlz.to_csv(OUT_EFFECTS_WLZ, index=False)

    # Overall collapsed (across cases)
    overall = _overall_summary(df)
    overall.to_csv(OUT_OVERALL, index=False)

    # NEW: all pairwise comparisons among modes
    pairs_all = _pairs_all_modes(df)
    pairs_all.to_csv(OUT_PAIRS_ALL, index=False)

    # NEW: fastest-mode share per case
    fastest = _fastest_share(df)
    fastest.to_csv(OUT_FASTEST, index=False)

    # NEW: per-mode speed stats for bar-plot & filtering
    mode_speed = _mode_speed_stats(pairs, df)
    mode_speed.to_csv(OUT_MODE_SPEED, index=False)

    raw_failures = _raw_failure_summary(df)
    raw_failures.to_csv(OUT_RAW_FAILURES, index=False)

    large_lazy = _large_lazy_comparison(summary_ext)
    large_lazy.to_csv(OUT_LARGE_LAZY_COMPARISON, index=False)

    runtime_scale = _case_runtime_scale(df)
    runtime_scale.to_csv(OUT_CASE_RUNTIME_SCALE, index=False)

    print(f"Wrote per-instance merged results to: {OUT_MERGED}")
    print(f"Wrote per-instance pairs vs RAW to:   {OUT_PAIRS}")
    print(f"Wrote all pairwise mode pairs to:     {OUT_PAIRS_ALL}")
    print(f"Wrote fastest-mode share to:          {OUT_FASTEST}")
    print(f"Wrote summary to:                     {OUT_SUMMARY}")
    print(f"Wrote extended summary to:            {OUT_SUMMARY_EXT}")
    print(f"Wrote WLZ vs RAW effects to:          {OUT_EFFECTS_WLZ}")
    print(f"Wrote overall (across cases) to:      {OUT_OVERALL}")
    print(f"Wrote per-mode speed stats to:        {OUT_MODE_SPEED}")
    print(f"Wrote RAW failure summary to:         {OUT_RAW_FAILURES}")
    print(f"Wrote reference-mode pairs to:        {OUT_PAIRS_REFERENCE}")
    print(f"Wrote large LAZY comparison to:       {OUT_LARGE_LAZY_COMPARISON}")
    print(f"Wrote runtime scaling data to:        {OUT_CASE_RUNTIME_SCALE}")


if __name__ == "__main__":
    main()
