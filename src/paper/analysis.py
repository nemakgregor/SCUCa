"""
Comprehensive analysis on existing logs (no re-run of experiments needed).

What this script does:
- Reads results/raw_logs/*.csv (per-instance logs produced by experiments.py)
- Merges them into results/merged_results.csv
- Computes:
  • Objective delta vs RAW baseline (ppm)
  • Per-case/per-mode robust summaries with bootstrap CI
  • Pairwise (within-instance) comparisons vs RAW for all modes (runtime, nodes, etc.)
  • Effect sizes (WARM+LAZY vs RAW): Hodges–Lehmann median paired difference, rank-biserial correlation
  • Success rates (OPTIMAL/SUBOPTIMAL), feasibility rates (violations=OK), MIP gap stats
  • Warm-start and branching-hint utilization summaries
  • PRUNE tau sweep summaries (constraint ratio vs runtime)
  • NEW: full pairwise comparisons among all modes on the same instance (pairs_all.csv)
  • NEW: fastest-mode share per case (mode_fastest_share.csv)
  • NEW: per-mode speed ratio (mode over RAW) stats for bar-plots (mode_speed_stats.csv)
  • NEW: per-row flags (with_LAZY/GNN/PRUNE/COMMIT/GRU/BANDIT) for group-wise plots (flags added in merged_results.csv)
"""

from __future__ import annotations

import glob
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = Path("results")
RAW_DIR = RESULTS_DIR / "raw_logs"
OUT_SUMMARY = RESULTS_DIR / "summary.csv"  # legacy (kept)
OUT_MERGED = RESULTS_DIR / "merged_results.csv"
OUT_PAIRS = RESULTS_DIR / "pairs.csv"
OUT_PAIRS_ALL = RESULTS_DIR / "pairs_all.csv"
OUT_FASTEST = RESULTS_DIR / "mode_fastest_share.csv"
OUT_SUMMARY_EXT = RESULTS_DIR / "summary_extended.csv"
OUT_EFFECTS_WLZ = RESULTS_DIR / "effects_wlz_vs_raw.csv"
OUT_OVERALL = RESULTS_DIR / "overall_summary.csv"
OUT_MODE_SPEED = RESULTS_DIR / "mode_speed_stats.csv"  # NEW: for bar-plot speed ratios


def _read_all_logs() -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No log CSVs found in {RAW_DIR}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["logfile"] = os.path.basename(f)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

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
        "wall_sec",
        "mip_gap",
        "max_constraint_residual",
        "objective_inconsistency",
    ]
    for col in float_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Standard categorical/text cols
    for col in [
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

    # Normalize mode casing a bit
    data["mode_clean"] = data["mode"].str.upper()

    # Add technique flags for grouping/plots
    data["with_LAZY"] = data["mode"].str.contains("LAZY", case=False, na=False)
    data["with_PRUNE"] = data["mode"].str.contains("PRUNE", case=False, na=False)
    data["with_GNN"] = data["mode"].str.contains("GNN", case=False, na=False)
    data["with_COMMIT"] = data["mode"].str.contains("COMMIT", case=False, na=False)
    data["with_GRU"] = data["mode"].str.contains("GRU", case=False, na=False)
    data["with_BANDIT"] = data["mode"].str.contains("BANDIT", case=False, na=False)

    return data


def _compute_obj_ppm_vs_raw(df: pd.DataFrame) -> pd.DataFrame:
    # For each instance_name, get RAW obj as baseline
    raw = df[df["mode_clean"].str.startswith("RAW")][
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
    for (case, inst), g in df.groupby(["case_folder", "instance_name"]):
        g_raw = g[g["mode_clean"].str.startswith("RAW")]
        if g_raw.empty:
            continue
        raw = g_raw.iloc[0]
        raw_rt = float(raw.get("runtime_sec", np.nan))
        raw_nodes = float(raw.get("nodes", np.nan))
        raw_mem = float(raw.get("peak_memory_gb", np.nan))
        raw_root = float(raw.get("num_constrs_root", np.nan))
        for _, r in g.iterrows():
            mode = r["mode"]
            if str(mode).upper().startswith("RAW"):
                continue
            rt = float(r.get("runtime_sec", np.nan))
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


def _aggregate_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy aggregate (kept for backward compatibility):
      - per case/mode medians/IQR/CI for runtime, nodes, obj ppm, mem, root_constrs
    """
    rows = []
    for case, g_case in df.groupby("case_folder"):
        for mode, g_mode in g_case.groupby("mode"):
            rt = pd.to_numeric(g_mode["runtime_sec"], errors="coerce").to_numpy()
            nd = pd.to_numeric(g_mode["nodes"], errors="coerce").to_numpy()
            ppm = pd.to_numeric(g_mode["obj_ppm_vs_raw"], errors="coerce").to_numpy()
            mem = pd.to_numeric(g_mode["peak_memory_gb"], errors="coerce").to_numpy()
            ncr = pd.to_numeric(g_mode["num_constrs_root"], errors="coerce").to_numpy()

            def med_iqr_ci(x):
                return (
                    float(np.nanmedian(x)),
                    _iqr(x),
                    *_bootstrap_ci95_median(x),
                )

            med_rt, iqr_rt, lo_rt, hi_rt = med_iqr_ci(rt)
            med_nd, iqr_nd, lo_nd, hi_nd = med_iqr_ci(nd)
            med_ppm, iqr_ppm, lo_ppm, hi_ppm = med_iqr_ci(ppm)
            med_mem, iqr_mem, lo_mem, hi_mem = med_iqr_ci(mem)
            med_ncr, iqr_ncr, lo_ncr, hi_ncr = med_iqr_ci(ncr)

            rows.append(
                {
                    "case_folder": case,
                    "mode": mode,
                    "N": int(len(g_mode)),
                    "runtime_median": med_rt,
                    "runtime_IQR": iqr_rt,
                    "runtime_CI95_lo": lo_rt,
                    "runtime_CI95_hi": hi_rt,
                    "nodes_median": med_nd,
                    "nodes_IQR": iqr_nd,
                    "nodes_CI95_lo": lo_nd,
                    "nodes_CI95_hi": hi_nd,
                    "obj_ppm_median": med_ppm,
                    "obj_ppm_IQR": iqr_ppm,
                    "obj_ppm_CI95_lo": lo_ppm,
                    "obj_ppm_CI95_hi": hi_ppm,
                    "mem_gb_median": med_mem,
                    "mem_gb_IQR": iqr_mem,
                    "mem_gb_CI95_lo": lo_mem,
                    "mem_gb_CI95_hi": hi_mem,
                    "root_constrs_median": med_ncr,
                    "root_constrs_IQR": iqr_ncr,
                    "root_constrs_CI95_lo": lo_ncr,
                    "root_constrs_CI95_hi": hi_ncr,
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
    diffs = x - y
    try:
        stat = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        pval = float(stat.pvalue)
        # Rank-biserial correlation
        n = len(diffs)
        T = n * (n + 1) / 2.0
        W = float(stat.statistic)
        rbc = 2.0 * W / T - 1.0
    except Exception:
        pval = np.nan
        rbc = np.nan
    hl_med = float(np.nanmedian(diffs)) if diffs.size > 0 else np.nan
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
    Also returns effects_wlz (WARM+LAZY vs RAW) per case for plotting.
    """
    rows = []
    effects_wlz = []

    for case, g_case in df.groupby("case_folder"):
        # Baseline medians for speedup (same as legacy)
        base = g_case[g_case["mode_clean"].str.startswith("RAW")]
        base_rt_med = (
            float(np.nanmedian(pd.to_numeric(base["runtime_sec"], errors="coerce")))
            if not base.empty
            else np.nan
        )

        for mode, g_mode in g_case.groupby("mode"):
            # Feasibility rate (strict: violations == OK)
            feas = (
                (g_mode["violations"].astype(str) == "OK").mean()
                if not g_mode.empty
                else np.nan
            )
            # Success rate: OPTIMAL or SUBOPTIMAL or TIME_LIMIT can be considered "success"
            succ = (
                g_mode["status"].isin(["OPTIMAL", "SUBOPTIMAL", "TIME_LIMIT"]).mean()
                if not g_mode.empty
                else np.nan
            )
            # MIP gap
            mip = pd.to_numeric(g_mode.get("mip_gap", np.nan), errors="coerce")
            mip_med = (
                float(np.nanmedian(mip)) if mip is not None and mip.size > 0 else np.nan
            )
            # Warm start / hints usage
            ws = pd.to_numeric(
                g_mode.get("warm_start_applied_vars", np.nan), errors="coerce"
            )
            ws_med = (
                float(np.nanmedian(ws)) if ws is not None and ws.size > 0 else np.nan
            )
            bh = pd.to_numeric(
                g_mode.get("branch_hints_applied", np.nan), errors="coerce"
            )
            bh_med = (
                float(np.nanmedian(bh)) if bh is not None and bh.size > 0 else np.nan
            )

            # Legacy metrics reused (medians, CIs)
            rt = pd.to_numeric(g_mode["runtime_sec"], errors="coerce").to_numpy()
            nd = pd.to_numeric(g_mode["nodes"], errors="coerce").to_numpy()
            mem = pd.to_numeric(g_mode["peak_memory_gb"], errors="coerce").to_numpy()
            ppm = pd.to_numeric(g_mode["obj_ppm_vs_raw"], errors="coerce").to_numpy()
            root = pd.to_numeric(g_mode["num_constrs_root"], errors="coerce").to_numpy()
            final = pd.to_numeric(
                g_mode["num_constrs_final"], errors="coerce"
            ).to_numpy()

            def med_iqr_ci(x):
                return (
                    float(np.nanmedian(x)),
                    _iqr(x),
                    *_bootstrap_ci95_median(x),
                )

            med_rt, iqr_rt, lo_rt, hi_rt = med_iqr_ci(rt)
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

            # Paired Wilcoxon vs RAW on runtime and nodes
            p_rt, hl_rt, rbc_rt = (np.nan, np.nan, np.nan)
            p_nd, hl_nd, rbc_nd = (np.nan, np.nan, np.nan)
            if not str(mode).upper().startswith("RAW"):
                p_rt, hl_rt, rbc_rt = _wilcoxon_pairs(
                    df, case, mode, "RAW", "runtime_sec"
                )
                p_nd, hl_nd, rbc_nd = _wilcoxon_pairs(df, case, mode, "RAW", "nodes")
                # store WLZ-only effects for plotting
                if mode == "WARM+LAZY":
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
                    "N": int(len(g_mode)),
                    "runtime_median": med_rt,
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
    for mode, g in df.groupby("mode"):
        rt = pd.to_numeric(g["runtime_sec"], errors="coerce").to_numpy()
        nd = pd.to_numeric(g["nodes"], errors="coerce").to_numpy()
        ppm = pd.to_numeric(g["obj_ppm_vs_raw"], errors="coerce").to_numpy()
        mem = pd.to_numeric(g["peak_memory_gb"], errors="coerce").to_numpy()

        def med_iqr(x):
            return float(np.nanmedian(x)), _iqr(x)

        med_rt, iqr_rt = med_iqr(rt)
        med_nd, iqr_nd = med_iqr(nd)
        med_ppm, iqr_ppm = med_iqr(ppm)
        med_mem, iqr_mem = med_iqr(mem)
        rows.append(
            {
                "mode": mode,
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
    for (case, inst), g in df.groupby(["case_folder", "instance_name"]):
        # Keep only rows with runtime present
        gg = g.dropna(subset=["runtime_sec"]).copy()
        if len(gg) < 2:
            continue
        # build dicts
        runs = dict(zip(gg["mode"], gg["runtime_sec"]))
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
    for (case, inst), g in df.groupby(["case_folder", "instance_name"]):
        g2 = g.dropna(subset=["runtime_sec"])
        if g2.empty:
            continue
        best_row = g2.loc[g2["runtime_sec"].idxmin()]
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
     - success_rate_strict (violations == 'OK')
     - success_rate_status (status in OPTIMAL/SUBOPTIMAL/TIME_LIMIT)
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
    d["ok_strict"] = (d["violations"].astype(str) == "OK").astype(float)
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

    out = stats.merge(succ, on="mode", how="left")
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Read and enrich
    df = _read_all_logs()
    df = _compute_obj_ppm_vs_raw(df)

    # Write enriched merged (keeps with_* flags)
    df.to_csv(OUT_MERGED, index=False)

    # Legacy aggregate
    agg_basic = _aggregate_basic(df)

    # Add speedups vs RAW to legacy
    rows = []
    for case in sorted(df["case_folder"].unique()):
        base = agg_basic[
            (agg_basic["case_folder"] == case)
            & (agg_basic["mode"].str.upper().str.startswith("RAW"))
        ]
        if base.empty:
            continue
        base_rt = float(base["runtime_median"].values[0])
        for _, row in agg_basic[agg_basic["case_folder"] == case].iterrows():
            su = (
                base_rt / row["runtime_median"]
                if (row["runtime_median"] and row["runtime_median"] > 0)
                else np.nan
            )
            d = row.to_dict()
            d["speedup_vs_raw"] = su
            rows.append(d)
    agg2 = pd.DataFrame(rows)

    # Wilcoxon for WARM+LAZY vs RAW (legacy)
    def _wilc(df_full, case_name):
        # fallback wrapper: reuse extended later; for summary.csv we keep p-values stub
        return np.nan, np.nan

    wilcoxon_rows = []
    for case in sorted(df["case_folder"].unique()):
        # Compute via pairs down below; keep placeholder here
        wilcoxon_rows.append(
            {
                "case_folder": case,
                "p_wilcoxon_runtime": np.nan,
                "p_wilcoxon_nodes": np.nan,
            }
        )
    wilc = pd.DataFrame(wilcoxon_rows)

    summary_legacy = agg2.merge(wilc, on="case_folder", how="left")
    summary_legacy.to_csv(OUT_SUMMARY, index=False)

    # New: pairs vs RAW
    pairs = _pair_with_raw(df)
    pairs.to_csv(OUT_PAIRS, index=False)

    # New extended summary and effects
    summary_ext, effects_wlz = _build_extended_summary(df, pairs)
    # Patch p-values for summary.csv from extended (WLZ vs RAW only); keep for backward compat
    try:
        wlz = summary_ext[summary_ext["mode"] == "WARM+LAZY"][
            ["case_folder", "wilcoxon_p_runtime", "wilcoxon_p_nodes"]
        ]
        summary_legacy = summary_legacy.drop(
            columns=["p_wilcoxon_runtime", "p_wilcoxon_nodes"], errors="ignore"
        ).merge(
            wlz.rename(
                columns={
                    "wilcoxon_p_runtime": "p_wilcoxon_runtime",
                    "wilcoxon_p_nodes": "p_wilcoxon_nodes",
                }
            ),
            on="case_folder",
            how="left",
        )
        summary_legacy.to_csv(OUT_SUMMARY, index=False)
    except Exception:
        pass

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

    print(f"Wrote per-instance merged results to: {OUT_MERGED}")
    print(f"Wrote per-instance pairs vs RAW to:   {OUT_PAIRS}")
    print(f"Wrote all pairwise mode pairs to:     {OUT_PAIRS_ALL}")
    print(f"Wrote fastest-mode share to:          {OUT_FASTEST}")
    print(f"Wrote legacy summary to:              {OUT_SUMMARY}")
    print(f"Wrote extended summary to:            {OUT_SUMMARY_EXT}")
    print(f"Wrote WLZ vs RAW effects to:          {OUT_EFFECTS_WLZ}")
    print(f"Wrote overall (across cases) to:      {OUT_OVERALL}")
    print(f"Wrote per-mode speed stats to:        {OUT_MODE_SPEED}")


if __name__ == "__main__":
    main()
