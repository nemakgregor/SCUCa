"""
Unified publication plots.

Concept:
  - aggregate plots: show all modes with RAW pairings
  - large-case validation plots: show only the selected modes that were actually run
  - keep one consistent ordering and one consistent color per mode
  - use speedup vs RAW where paired RAW exists, and feasibility/gap diagnostics on large cases

Generated PNG figures under results/figures:
  - avg_speedup_vs_success_pareto.png
  - bar_speed_ratio_by_mode.png
  - constraints_runtime_tradeoff.png
  - ecdf.png
  - heatmap_speedup_by_case_mode.png
  - heatmap_speedup_ex_norel_by_case_mode.png
  - status_success_heatmap.png
  - large_cases_speedup_vs_lazy_by_case_mode.png
  - large_cases_mip_gap_by_case_mode.png
  - large_cases_feasibility_by_case_mode.png
  - large_cases_time_to_first_incumbent_by_case_mode.png
  - large_cases_gap_progress.png
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paper.modes import (
    RAW_BASELINE_MODE,
    canonical_mode,
    mode_display,
    mode_exactness,
    ordered_modes,
)
from src.paper.experiment_spec import (
    CASES_LARGE,
    CASES_SMALL,
    CASE_TIME_LIMIT_SEC,
    MODE_CATALOG_MEDLARGE_FULL,
)

FIG_DIR = Path("results") / "figures"
SUMMARY = Path("results") / "summary.csv"
SUMMARY_EXT = Path("results") / "summary_extended.csv"
MERGED = Path("results") / "merged_results.csv"
PAIRS = Path("results") / "pairs.csv"
SLEEP_OVERLAP_CURRENT = Path("results") / "sleep_overlap_current.tsv"
MODE_PARETO_SPEED_SUCCESS = Path("results") / "mode_pareto_speed_success.csv"
MODE_FAMILY_BEST_SPEED_SUCCESS = Path("results") / "mode_family_best_speed_success.csv"
LARGE_MODE_SELECTION_PARETO = Path("results") / "large_mode_selection_pareto.txt"
CASE1354_MODE_LIST = Path("results") / "case1354_probe_modes.txt"
CASE2383_MODE_LIST = Path("results") / "case2383_modes_from_case1354_completed.txt"
CASE3375_CASE6515_MODE_LIST = Path("results") / "case3375_case6515_modes_from_case2383_completed.txt"
LARGE_GAP_RUNTIME_SUMMARY = Path("results") / "large_cases_gap_solution_time_summary.csv"
LARGE_GAP_RUNTIME_ROWS = Path("results") / "large_cases_gap_solution_time_rows.csv"
LARGE_GUROBI_PROGRESS_ROWS = Path("results") / "large_cases_gurobi_progress_rows.csv"
LARGE_GUROBI_PROGRESS_SUMMARY = Path("results") / "large_cases_gurobi_progress_summary.csv"

LARGE_REFERENCE_MODE = "LAZY_ALL"
LARGE_CASE_TAGS = {str(c).split("/")[-1] for c in CASES_LARGE}
PARETO_SUCCESS_THRESHOLD = 0.95
FULL_SUCCESS_TOL = 1e-12
ECDF_HIGHLIGHT_N = 10
ECDF_REQUIRED_MODES = [
    "RAW",
    "WARM_GRU",
    "RAW_COMMIT_HINTS",
    "LAZY_ALL",
    "LAZY_COMMIT_HINTS",
    "WARM_LAZY_GRU",
    "WARM_PRUNE_T030",
    "WARM_PRUNE_LAZY_T080",
    "LAZY_GNN_T080",
    "LAZY_BANDIT",
    "STREDUCE_LAZY_GRU",
]
MAIN_HEATMAP_MODE_LIMIT = 15
MAIN_HEATMAP_REQUIRED_MODES = {
    "RAW",
    "WARM_GRU",
    "RAW_COMMIT_HINTS",
    "LAZY_ALL",
    "LAZY_COMMIT_HINTS",
    "WARM_LAZY_GRU",
    "WARM_PRUNE_T030",
    "WARM_PRUNE_LAZY_T080",
    "LAZY_GNN_T080",
    "LAZY_BANDIT",
    "STREDUCE_LAZY_GRU",
}
PARETO_PLOT_FAMILY_COLORS = {
    "Active set": "#0072B2",
    "Learning": "#CC79A7",
    "LP screen": "#009E73",
    "Pruning": "#D55E00",
    "Reduction": "#E69F00",
    "Lazy / hints": "#56B4E9",
    "RAW / warm": "#8C8C8C",
}
_TIME_TOKEN_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)s$")
_NOREL_TIME_RE = re.compile(
    r"Elapsed time for NoRel heuristic:\s*([0-9.]+)s\s*"
    r"(?:\(best bound\s+([^\)]+)\))?"
)
_HEURISTIC_SOLUTION_RE = re.compile(
    r"Found heuristic solution:\s*objective\s+([-+0-9.eE]+)"
)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 150


def _ensure() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _cleanup_figure_dir(keep_png: set[str]) -> None:
    for p in FIG_DIR.iterdir():
        if p.name == ".gitkeep":
            continue
        if p.is_file() and p.suffix.lower() in {".png", ".pdf"}:
            if p.suffix.lower() == ".png" and p.name in keep_png:
                continue
            p.unlink(missing_ok=True)


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run analysis.py first.")
    return pd.read_csv(path)


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_summary() -> pd.DataFrame:
    if SUMMARY_EXT.is_file():
        return pd.read_csv(SUMMARY_EXT)
    return _read_csv_required(SUMMARY)


def _runtime_col(df: pd.DataFrame) -> str:
    if "runtime_report_sec" in df.columns:
        return "runtime_report_sec"
    if "wall_sec" in df.columns:
        return "wall_sec"
    return "runtime_sec"


def _normalize_runtime_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["runtime_sec"] = pd.to_numeric(out[_runtime_col(out)], errors="coerce")
    return out


def _case_tag(case_folder: str) -> str:
    return str(case_folder).split("/")[-1]


def _case_sort_key(case_folder: str) -> tuple[int, str]:
    tag = _case_tag(case_folder)
    order = {
        "case14": 14,
        "case30": 30,
        "case57": 57,
        "case89pegase": 89,
        "case118": 118,
        "case300": 300,
        "case1354pegase": 1354,
    }
    if tag in order:
        return (order[tag], tag)
    digits = "".join(ch for ch in tag if ch.isdigit())
    return (int(digits) if digits else 10_000, tag)


def _is_large_case(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda x: _case_tag(x) in LARGE_CASE_TAGS)


def _small_case_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "case_folder" not in df.columns:
        return df.copy()
    return df[df["case_folder"].astype(str).isin(CASES_SMALL)].copy()


def _mode_exactness(mode: str) -> str:
    return mode_exactness(mode)


def _mode_label(mode: str) -> str:
    m = mode_display(mode)
    return f"{m}$^\\dagger$" if _mode_exactness(m) == "heuristic" else m


def _mode_rank_map(mode_order: list[str]) -> dict[str, int]:
    return {canonical_mode(mode): idx + 1 for idx, mode in enumerate(mode_order)}


def _ranked_mode_label(mode: str, rank_map: dict[str, int]) -> str:
    canonical = canonical_mode(mode)
    label = _mode_label(canonical)
    rank = rank_map.get(canonical)
    return f"{rank:02d}. {label}" if rank is not None else label


def _mode_family_label(mode: str) -> str:
    m = canonical_mode(mode)
    if "ACTIVESET" in m:
        return "ACTIVESET"
    if "GNN" in m:
        return "GNN"
    if "LPSCREEN" in m:
        return "LPSCREEN"
    if "PRUNE" in m:
        return "PRUNE"
    if "STREDUCE" in m:
        return "STREDUCE"
    if "SHRINK" in m:
        return "SHRINK"
    if "_SR_" in m or m.endswith("_SR") or m == "WARM_SR_LAZY":
        return "SR"
    if "GRU" in m:
        return "GRU"
    if "COMMIT" in m:
        return "COMMIT"
    if "BANDIT" in m:
        return "BANDIT"
    if "TOPK" in m:
        return "TOPK"
    if m.startswith("WARM_LAZY"):
        return "WARM_LAZY"
    if m.startswith("LAZY"):
        return "LAZY"
    if m.startswith("WARM"):
        return "WARM"
    if m == RAW_BASELINE_MODE:
        return "RAW"
    return m.split("_", 1)[0]


def _pareto_plot_family_label(mode: str) -> str:
    family = _mode_family_label(mode)
    if family == "ACTIVESET":
        return "Active set"
    if family in {"GNN", "GRU"}:
        return "Learning"
    if family == "LPSCREEN":
        return "LP screen"
    if family == "PRUNE":
        return "Pruning"
    if family in {"SHRINK", "STREDUCE", "SR"}:
        return "Reduction"
    if family in {"LAZY", "COMMIT", "BANDIT", "TOPK", "WARM_LAZY"}:
        return "Lazy / hints"
    return "RAW / warm"


def _clip_plot_label(label: str, max_chars: int = 31) -> str:
    text = str(label)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "."


def _format_count(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def _format_percent(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) < 10.0:
        return f"{value:.1f}%"
    return f"{value:.0f}%"


def _speedup_sorted_mode_order(
    stats: pd.DataFrame,
    present_modes: set[str],
) -> list[str]:
    ordered: list[str] = []
    if not stats.empty and {"mode", "mean_speedup"}.issubset(stats.columns):
        ranked = (
            stats.dropna(subset=["mean_speedup"])
            .sort_values(
                ["mean_speedup", "strict_success", "mode"],
                ascending=[False, False, True],
            )["mode"]
            .astype(str)
            .tolist()
        )
        ordered.extend([m for m in ranked if m in present_modes and m not in ordered])
    for mode in ordered_modes(present_modes):
        if mode in present_modes and mode not in ordered:
            ordered.append(mode)
    return ordered


def _top_speed_modes(mode_order: list[str], present_modes: set[str], n: int) -> list[str]:
    return [m for m in mode_order if m in present_modes][:n]


def _main_heatmap_mode_order(mode_order: list[str], present_modes: set[str]) -> list[str]:
    selected: list[str] = []
    for mode in mode_order:
        if mode in present_modes and mode not in selected:
            selected.append(mode)
        if len(selected) >= MAIN_HEATMAP_MODE_LIMIT:
            break
    for mode in mode_order:
        if (
            mode in present_modes
            and mode in MAIN_HEATMAP_REQUIRED_MODES
            and mode not in selected
        ):
            selected.append(mode)
    return selected


def _ecdf_mode_order(mode_order: list[str], present_modes: set[str]) -> list[str]:
    selected = [mode for mode in ECDF_REQUIRED_MODES if mode in present_modes]
    for mode in mode_order:
        if mode in present_modes and mode not in selected:
            selected.append(mode)
        if len(selected) >= ECDF_HIGHLIGHT_N:
            break
    return selected[:ECDF_HIGHLIGHT_N]


def _verified_ok_mask(df: pd.DataFrame) -> pd.Series:
    ok_text = {"OK", "TRUE", "1", "YES"}
    fail_text = {"FAIL", "FALSE", "0", "NO"}
    if "violations" in df.columns:
        violations = df["violations"].astype(str).str.strip().str.upper().replace(
            {"NAN": "", "NONE": "", "NA": ""}
        )
    else:
        violations = pd.Series("", index=df.index, dtype="object")
    if "feasible_ok" in df.columns:
        feasible = df["feasible_ok"].astype(str).str.strip().str.upper().replace(
            {"NAN": "", "NONE": "", "NA": ""}
        )
    else:
        feasible = pd.Series("", index=df.index, dtype="object")

    has_violations_signal = violations.ne("")
    ok = has_violations_signal & violations.isin(ok_text)
    has_feasible_signal = feasible.ne("")
    ok = ok.where(has_violations_signal, feasible.isin(ok_text))
    ok = ok & ~(has_feasible_signal & feasible.isin(fail_text))

    if "has_incumbent" in df.columns:
        incumbent = pd.to_numeric(df["has_incumbent"], errors="coerce")
        ok = ok & (incumbent.isna() | (incumbent > 0.0))
    return ok.fillna(False)


def _strict_success_mask(df: pd.DataFrame) -> pd.Series:
    return _verified_ok_mask(df)


def _mode_color_map(mode_order: list[str]) -> dict[str, tuple[float, float, float]]:
    if not mode_order:
        return {}
    palette = sns.color_palette("husl", n_colors=len(mode_order))
    return {mode: palette[i] for i, mode in enumerate(mode_order)}


def _outline_color(strict_success: float) -> str:
    if not np.isfinite(strict_success):
        return "#444444"
    if strict_success < 0.85:
        return "#b00020"
    if strict_success < 0.95:
        return "#d17a00"
    return "#222222"


def _line_style(mode: str, strict_success: float) -> str:
    if _mode_exactness(mode) == "heuristic":
        return "-."
    if not np.isfinite(strict_success):
        return "-"
    if strict_success < 0.85:
        return ":"
    if strict_success < 0.95:
        return "--"
    return "-"


def _build_small_mode_stats(
    merged: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    merged_small = merged.copy()
    pairs_small = pairs.copy()

    pairs_small["runtime_speedup"] = pd.to_numeric(
        pairs_small["runtime_speedup"], errors="coerce"
    )
    pairs_small = pairs_small[np.isfinite(pairs_small["runtime_speedup"])].copy()

    speed = (
        pairs_small.groupby("mode", as_index=False)
        .agg(
            mean_speedup=("runtime_speedup", "mean"),
            median_speedup=("runtime_speedup", "median"),
            N_pairs=("runtime_speedup", "size"),
        )
        .copy()
    )

    merged_small["strict_success"] = _strict_success_mask(merged_small).astype(float)
    qual = (
        merged_small.groupby("mode", as_index=False)
        .agg(
            strict_success=("strict_success", "mean"),
            N_rows=("strict_success", "size"),
        )
        .copy()
    )

    out = speed.merge(qual, on="mode", how="outer")
    raw_rows = merged_small.loc[merged_small["mode"].astype(str) == RAW_BASELINE_MODE]
    if not raw_rows.empty:
        raw_values = {
            "mode": RAW_BASELINE_MODE,
            "mean_speedup": 1.0,
            "median_speedup": 1.0,
            "N_pairs": len(raw_rows),
            "strict_success": float(_strict_success_mask(raw_rows).mean()),
            "N_rows": len(raw_rows),
        }
        raw_mask = out["mode"].astype(str) == RAW_BASELINE_MODE
        if raw_mask.any():
            for col, value in raw_values.items():
                out.loc[raw_mask, col] = value
        else:
            out = pd.concat([out, pd.DataFrame([raw_values])], ignore_index=True)

    out["method_exactness"] = out["mode"].astype(str).map(_mode_exactness)
    out["mean_speedup"] = pd.to_numeric(out["mean_speedup"], errors="coerce")
    out["strict_success"] = pd.to_numeric(out["strict_success"], errors="coerce")
    out["N_rows"] = pd.to_numeric(out["N_rows"], errors="coerce")
    out = out.dropna(subset=["mean_speedup"]).copy()
    out = out.sort_values(
        ["mean_speedup", "strict_success", "mode"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return out


def _pareto_front_mask(df: pd.DataFrame) -> pd.Series:
    values = df[["mean_speedup", "strict_success"]].to_numpy(dtype=float)
    flags: list[bool] = []
    for speed, success in values:
        dominated = (
            (values[:, 0] >= speed)
            & (values[:, 1] >= success)
            & ((values[:, 0] > speed) | (values[:, 1] > success))
        ).any()
        flags.append(not bool(dominated))
    return pd.Series(flags, index=df.index, dtype=bool)


def _family_best_mask(stats: pd.DataFrame) -> pd.Series:
    if stats.empty or "family" not in stats.columns:
        return pd.Series(False, index=stats.index, dtype=bool)
    flags = pd.Series(False, index=stats.index, dtype=bool)
    for _, group in stats.groupby("family", sort=False):
        ranked = group.sort_values(
            ["strict_success", "mean_speedup", "median_speedup", "mode"],
            ascending=[False, False, False, True],
        )
        if not ranked.empty:
            flags.loc[ranked.index[0]] = True
    return flags


def _sleep_overlap_keys() -> set[tuple[str, str, str]]:
    if not SLEEP_OVERLAP_CURRENT.is_file() or SLEEP_OVERLAP_CURRENT.stat().st_size <= 1:
        return set()
    try:
        invalid = pd.read_csv(SLEEP_OVERLAP_CURRENT, sep="\t")
    except pd.errors.EmptyDataError:
        return set()
    req = {"case_folder", "instance_name", "mode"}
    if not req.issubset(invalid.columns):
        return set()
    return {
        (str(r.case_folder), str(r.instance_name), canonical_mode(r.mode))
        for r in invalid.itertuples(index=False)
    }


def _drop_sleep_overlap_rows(
    df: pd.DataFrame,
    invalid_keys: set[tuple[str, str, str]],
) -> pd.DataFrame:
    if not invalid_keys or not {"case_folder", "instance_name", "mode"}.issubset(df.columns):
        return df.copy()
    keep = [
        (str(r.case_folder), str(r.instance_name), canonical_mode(r.mode)) not in invalid_keys
        for r in df.itertuples(index=False)
    ]
    return df.loc[keep].copy()


def _benchmark_valid_small_frames(
    merged: pd.DataFrame,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    small_cases = set(CASES_SMALL)
    m = merged.loc[merged["case_folder"].astype(str).isin(small_cases)].copy()
    p = pairs.loc[pairs["case_folder"].astype(str).isin(small_cases)].copy()

    raw = m.loc[m["mode"].astype(str) == RAW_BASELINE_MODE].copy()
    raw_instances_total = int(raw["instance_name"].nunique())
    if raw.empty:
        raw_valid_instances = set(m["instance_name"].astype(str).unique())
    else:
        raw_ok = _strict_success_mask(raw)
        raw_valid_instances = set(raw.loc[raw_ok, "instance_name"].astype(str))

    m = m.loc[m["instance_name"].astype(str).isin(raw_valid_instances)].copy()
    p = p.loc[p["instance_name"].astype(str).isin(raw_valid_instances)].copy()

    invalid_keys = _sleep_overlap_keys()
    before_m = len(m)
    before_p = len(p)
    m = _drop_sleep_overlap_rows(m, invalid_keys)
    p = _drop_sleep_overlap_rows(p, invalid_keys)

    info = {
        "raw_instances_total": raw_instances_total,
        "raw_valid_instances": len(raw_valid_instances),
        "sleep_overlap_rows_excluded": before_m - len(m),
        "sleep_overlap_pairs_excluded": before_p - len(p),
    }
    return m, p, info


def _save_png(
    fig: plt.Figure,
    name: str,
    aliases: list[str] | None = None,
    *,
    tight: bool = True,
) -> list[str]:
    if tight:
        fig.tight_layout()
    names = [name] + list(aliases or [])
    for nm in names:
        fig.savefig(FIG_DIR / f"{nm}.png")
    plt.close(fig)
    return [f"{nm}.png" for nm in names]


def _safe_float_token(token: object) -> float:
    text = str(token or "").strip().rstrip(",")
    if not text or text in {"-", "--"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _gap_pct_from_incumbent_bound(incumbent: float, bound: float) -> float:
    if not (np.isfinite(incumbent) and np.isfinite(bound)):
        return np.nan
    denom = abs(incumbent)
    if denom <= 0.0:
        return np.nan
    return 100.0 * abs(incumbent - bound) / denom


def _parse_gurobi_progress_line(line: str) -> dict[str, float | str] | None:
    parts = line.strip().split()
    if len(parts) < 6:
        return None
    time_match = _TIME_TOKEN_RE.match(parts[-1])
    if not time_match:
        return None
    gap_text = parts[-3]
    if not gap_text.endswith("%"):
        return None
    gap_pct = _safe_float_token(gap_text[:-1])
    if not np.isfinite(gap_pct):
        return None
    incumbent = _safe_float_token(parts[-5]) if len(parts) >= 5 else np.nan
    best_bound = _safe_float_token(parts[-4]) if len(parts) >= 4 else np.nan
    marker = parts[0] if parts[0] in {"H", "*"} else ""
    return {
        "time_sec": float(time_match.group(1)),
        "incumbent": incumbent,
        "best_bound": best_bound,
        "gap_pct": gap_pct,
        "source": "mip_progress",
        "is_incumbent_update": bool(marker),
    }


def _parse_gurobi_progress_log(path: Path) -> list[dict[str, float | str | bool]]:
    rows: list[dict[str, float | str | bool]] = []
    if not path.is_file():
        return rows
    last_norel_time = np.nan
    last_norel_bound = np.nan
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rows

    for line in lines:
        norel = _NOREL_TIME_RE.search(line)
        if norel:
            last_norel_time = _safe_float_token(norel.group(1))
            if norel.group(2) is not None:
                last_norel_bound = _safe_float_token(norel.group(2))
            continue

        heuristic = _HEURISTIC_SOLUTION_RE.search(line)
        if heuristic:
            incumbent = _safe_float_token(heuristic.group(1))
            gap_pct = _gap_pct_from_incumbent_bound(incumbent, last_norel_bound)
            if np.isfinite(last_norel_time) and np.isfinite(gap_pct):
                rows.append(
                    {
                        "time_sec": float(last_norel_time),
                        "incumbent": incumbent,
                        "best_bound": float(last_norel_bound),
                        "gap_pct": gap_pct,
                        "source": "norel_heuristic",
                        "is_incumbent_update": True,
                    }
                )
            continue

        parsed = _parse_gurobi_progress_line(line)
        if parsed is not None:
            rows.append(parsed)

    if not rows:
        return rows
    d = pd.DataFrame(rows)
    d["time_sec"] = pd.to_numeric(d["time_sec"], errors="coerce")
    d["gap_pct"] = pd.to_numeric(d["gap_pct"], errors="coerce")
    d = d[np.isfinite(d["time_sec"]) & np.isfinite(d["gap_pct"])].copy()
    if d.empty:
        return []
    d = d.sort_values(["time_sec", "gap_pct"], kind="stable")
    d = d.groupby("time_sec", as_index=False, sort=False).first()
    d["gap_pct"] = d["gap_pct"].cummin()
    return d.to_dict("records")


def _large_mode_order(present_modes: set[str], global_order: list[str]) -> list[str]:
    order: list[str] = []
    for path in (CASE1354_MODE_LIST, CASE2383_MODE_LIST, CASE3375_CASE6515_MODE_LIST):
        if not path.is_file():
            continue
        try:
            planned = path.read_text(encoding="utf-8").replace(",", " ").split()
        except OSError:
            planned = []
        for mode in planned:
            canonical = canonical_mode(mode)
            if canonical in present_modes and canonical not in order:
                order.append(canonical)
    for mode in global_order:
        canonical = canonical_mode(mode)
        if canonical in present_modes and canonical not in order:
            order.append(canonical)
    for mode in sorted(present_modes):
        if mode not in order:
            order.append(mode)
    return order


def _large_selected_mode_order(
    large_summary: pd.DataFrame,
    global_order: list[str],
) -> list[str]:
    if large_summary.empty or "mode" not in large_summary.columns:
        return []
    d = large_summary.copy()
    rows = pd.to_numeric(d.get("rows", np.nan), errors="coerce").fillna(0.0)
    skipped = pd.to_numeric(d.get("skipped", np.nan), errors="coerce").fillna(0.0)
    runtime = pd.to_numeric(d.get("runtime_median_sec", np.nan), errors="coerce")
    gap = pd.to_numeric(d.get("mip_gap_median_pct", np.nan), errors="coerce")
    logs = pd.to_numeric(d.get("logs_with_incumbent", np.nan), errors="coerce").fillna(0.0)
    has_solver_signal = (rows > skipped) | np.isfinite(runtime) | np.isfinite(gap) | (logs > 0)
    present = set(d.loc[has_solver_signal, "mode"].astype(str))
    return _large_mode_order(present, global_order)


def _large_cases_test_frame(merged: pd.DataFrame) -> pd.DataFrame:
    req = {"case_folder", "mode", "instance_name", "status"}
    if merged.empty or not req.issubset(merged.columns):
        return pd.DataFrame()
    d = merged.loc[_is_large_case(merged["case_folder"])].copy()
    if "stage" in d.columns:
        d = d[d["stage"].astype(str).str.upper() == "TEST"].copy()
    if d.empty:
        return d
    d["mode"] = d["mode"].map(canonical_mode)
    d["case_label"] = d["case_folder"].astype(str).map(_case_tag)
    d["case_size"] = d["case_folder"].astype(str).map(lambda x: _case_sort_key(x)[0])
    d["date"] = d["instance_name"].astype(str).str.rsplit("/", n=1).str[-1]
    d["status_upper"] = d["status"].astype(str).str.strip().str.upper()
    d["solve_runtime_sec"] = pd.to_numeric(d.get("runtime_sec", np.nan), errors="coerce")
    d["time_limit_sec"] = pd.to_numeric(d.get("time_limit_sec", np.nan), errors="coerce")
    d["time_limit_sec"] = d["time_limit_sec"].fillna(
        d["case_folder"].astype(str).map(CASE_TIME_LIMIT_SEC)
    )
    d["mip_gap"] = pd.to_numeric(d.get("mip_gap", np.nan), errors="coerce")
    d["mip_gap_pct"] = 100.0 * d["mip_gap"]
    d.loc[~np.isfinite(d["mip_gap_pct"]), "mip_gap_pct"] = np.nan
    if "pass" in d.columns:
        d["pass_num"] = pd.to_numeric(d["pass"], errors="coerce").fillna(0.0)
        d["is_verified"] = d["pass_num"] > 0.0
    else:
        d["is_verified"] = _strict_success_mask(d)
    d["is_skipped"] = d["status_upper"].str.startswith("SKIPPED")
    d["hit_time_limit"] = d["status_upper"].str.contains("TIME_LIMIT", na=False)
    return d


def _large_gurobi_progress_rows(large_rows: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    if large_rows.empty or "gurobi_log_path" not in large_rows.columns:
        return pd.DataFrame()
    for _, row in large_rows.iterrows():
        log_path = Path(str(row.get("gurobi_log_path", "")))
        parsed = _parse_gurobi_progress_log(log_path)
        final_runtime = float(row.get("solve_runtime_sec", np.nan))
        final_gap = float(row.get("mip_gap_pct", np.nan))
        if np.isfinite(final_runtime) and np.isfinite(final_gap):
            parsed.append(
                {
                    "time_sec": final_runtime,
                    "incumbent": np.nan,
                    "best_bound": np.nan,
                    "gap_pct": final_gap,
                    "source": "result_final",
                    "is_incumbent_update": False,
                }
            )
        if not parsed:
            continue
        time_limit = float(row.get("time_limit_sec", np.nan))
        for point in parsed:
            time_sec = float(point.get("time_sec", np.nan))
            out_rows.append(
                {
                    "case_folder": row["case_folder"],
                    "case_label": row["case_label"],
                    "case_size": row["case_size"],
                    "instance_name": row["instance_name"],
                    "date": row["date"],
                    "mode": row["mode"],
                    "status": row["status"],
                    "is_verified": bool(row["is_verified"]),
                    "time_limit_sec": time_limit,
                    "time_sec": time_sec,
                    "time_budget_pct": (
                        100.0 * time_sec / time_limit
                        if np.isfinite(time_sec) and np.isfinite(time_limit) and time_limit > 0
                        else np.nan
                    ),
                    "incumbent": point.get("incumbent", np.nan),
                    "best_bound": point.get("best_bound", np.nan),
                    "gap_pct": point.get("gap_pct", np.nan),
                    "source": point.get("source", ""),
                    "is_incumbent_update": bool(point.get("is_incumbent_update", False)),
                    "gurobi_log_path": str(log_path),
                }
            )
    if not out_rows:
        return pd.DataFrame()
    d = pd.DataFrame(out_rows)
    for col in ("time_sec", "time_budget_pct", "gap_pct", "incumbent", "best_bound"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d[np.isfinite(d["time_sec"]) & np.isfinite(d["gap_pct"])].copy()
    d = d.sort_values(
        ["case_size", "case_folder", "mode", "instance_name", "time_sec", "gap_pct"],
        kind="stable",
    )
    d = (
        d.groupby(["case_folder", "mode", "instance_name", "time_sec"], as_index=False, sort=False)
        .first()
        .copy()
    )
    d["gap_pct"] = d.groupby(["case_folder", "mode", "instance_name"], sort=False)["gap_pct"].cummin()
    return d


def _large_cases_summary(
    large_rows: pd.DataFrame,
    progress_rows: pd.DataFrame,
) -> pd.DataFrame:
    if large_rows.empty:
        return pd.DataFrame()
    grouped = large_rows.groupby(["case_folder", "case_label", "case_size", "mode"], dropna=False)
    summary = grouped.agg(
        rows=("instance_name", "size"),
        feasible=("is_verified", "sum"),
        skipped=("is_skipped", "sum"),
        time_limit_hits=("hit_time_limit", "sum"),
        time_limit_sec=("time_limit_sec", "median"),
        runtime_median_sec=("solve_runtime_sec", "median"),
        runtime_mean_sec=("solve_runtime_sec", "mean"),
        mip_gap_median_pct=("mip_gap_pct", "median"),
        mip_gap_mean_pct=("mip_gap_pct", "mean"),
    ).reset_index()
    summary["failed"] = (
        summary["rows"].astype(int)
        - summary["feasible"].astype(int)
        - summary["skipped"].astype(int)
    )
    summary.loc[summary["failed"] < 0, "failed"] = 0
    summary["feasible_rate"] = summary["feasible"] / summary["rows"].replace(0, np.nan)
    summary["runtime_budget_pct"] = (
        100.0 * summary["runtime_median_sec"] / summary["time_limit_sec"].replace(0, np.nan)
    )

    if not progress_rows.empty:
        first = (
            progress_rows.groupby(["case_folder", "case_label", "case_size", "mode", "instance_name"], as_index=False)
            .agg(
                first_incumbent_time_sec=("time_sec", "min"),
                best_log_gap_pct=("gap_pct", "min"),
                log_points=("gap_pct", "size"),
            )
        )
        target = (
            progress_rows[progress_rows["gap_pct"] <= 5.0]
            .groupby(["case_folder", "mode", "instance_name"], as_index=False)
            .agg(time_to_5pct_gap_sec=("time_sec", "min"))
        )
        first = first.merge(target, on=["case_folder", "mode", "instance_name"], how="left")
        prog_summary = (
            first.groupby(["case_folder", "case_label", "case_size", "mode"], as_index=False)
            .agg(
                logs_with_incumbent=("instance_name", "nunique"),
                progress_points=("log_points", "sum"),
                first_incumbent_time_median_sec=("first_incumbent_time_sec", "median"),
                first_incumbent_time_min_sec=("first_incumbent_time_sec", "min"),
                time_to_5pct_gap_median_sec=("time_to_5pct_gap_sec", "median"),
                best_log_gap_median_pct=("best_log_gap_pct", "median"),
            )
        )
        summary = summary.merge(
            prog_summary,
            on=["case_folder", "case_label", "case_size", "mode"],
            how="left",
        )

    ref = summary[summary["mode"] == LARGE_REFERENCE_MODE][
        ["case_folder", "runtime_median_sec"]
    ].rename(columns={"runtime_median_sec": "reference_runtime_median_sec"})
    summary = summary.merge(ref, on="case_folder", how="left")
    summary["speedup_vs_lazy"] = (
        summary["reference_runtime_median_sec"]
        / summary["runtime_median_sec"].replace(0, np.nan)
    )
    return summary.sort_values(["case_size", "mode"]).reset_index(drop=True)


def _large_gurobi_progress_summary(
    progress_rows: pd.DataFrame,
    large_rows: pd.DataFrame,
) -> pd.DataFrame:
    if large_rows.empty:
        return pd.DataFrame()
    base = large_rows.copy()
    if "is_skipped" in base.columns:
        base = base[~base["is_skipped"]].copy()
    if base.empty:
        return pd.DataFrame()
    grids = np.linspace(0.0, 100.0, 21)
    rows: list[dict[str, object]] = []
    for (case, case_label, case_size, mode), group in base.groupby(
        ["case_folder", "case_label", "case_size", "mode"],
        sort=False,
    ):
        instances = sorted(group["instance_name"].astype(str).unique())
        if not instances:
            continue
        progress_group = pd.DataFrame()
        if not progress_rows.empty:
            progress_group = progress_rows[
                (progress_rows["case_folder"].astype(str) == str(case))
                & (progress_rows["mode"].astype(str) == str(mode))
            ].copy()
        by_inst = {
            str(inst): inst_rows.sort_values("time_budget_pct")
            for inst, inst_rows in progress_group.groupby("instance_name", sort=False)
            if inst_rows["time_budget_pct"].notna().any()
        }
        for budget_pct in grids:
            values: list[float] = []
            instances_with_gap = 0
            for inst_name in instances:
                inst = by_inst.get(inst_name)
                if inst is None:
                    values.append(100.0)
                    continue
                seen = inst[inst["time_budget_pct"] <= budget_pct]
                if seen.empty:
                    values.append(100.0)
                    continue
                instances_with_gap += 1
                gap = float(seen["gap_pct"].iloc[-1])
                values.append(min(100.0, gap) if np.isfinite(gap) else 100.0)
            rows.append(
                {
                    "case_folder": case,
                    "case_label": case_label,
                    "case_size": case_size,
                    "mode": mode,
                    "time_budget_pct": float(budget_pct),
                    "median_gap_pct": float(np.nanmedian(values)) if values else np.nan,
                    "logs_with_incumbent": instances_with_gap,
                    "logs_total": len(instances),
                }
            )
    return pd.DataFrame(rows)


def bar_speed_ratio_by_mode(
    small_mode_stats: pd.DataFrame,
    mode_order: list[str],
    color_map: dict[str, tuple[float, float, float]],
) -> list[str] | None:
    if small_mode_stats.empty or not mode_order:
        return None

    d = small_mode_stats.set_index("mode").reindex(mode_order).reset_index()
    d = d.dropna(subset=["mean_speedup"]).copy()
    fig_h = max(8.0, 0.20 * len(d) + 2.0)
    fig, ax = plt.subplots(figsize=(11.0, fig_h))

    y = np.arange(len(d))
    colors = [color_map.get(m, "#4c78a8") for m in d["mode"].astype(str)]
    edges = [_outline_color(float(v)) for v in d["strict_success"].to_numpy(dtype=float)]
    alphas = [
        0.75 if _mode_exactness(str(m)) == "heuristic" else 0.95
        for m in d["mode"].astype(str)
    ]
    bars = ax.barh(
        y,
        d["mean_speedup"],
        color=colors,
        edgecolor=edges,
        linewidth=1.1,
    )
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    ax.axvline(1.0, ls="--", color="gray", lw=1)
    ax.set_xlabel("Mean speedup vs RAW (higher is better)")
    ax.set_ylabel("Mode")
    ax.set_title("Modes Sorted by Average Speedup")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{idx + 1:02d}. {_mode_label(mode)}"
            for idx, mode in enumerate(d["mode"].astype(str))
        ]
    )
    ax.tick_params(axis="y", labelsize=6.2 if len(d) > 35 else 7.5)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    for bar, speed in zip(bars, d["mean_speedup"].to_numpy(dtype=float)):
        if np.isfinite(speed) and speed >= 1.0:
            ax.text(
                speed + 0.12,
                bar.get_y() + bar.get_height() / 2.0,
                f"{speed:.1f}x",
                va="center",
                ha="left",
                fontsize=5.7,
                color="#333333",
            )

    legend_items = [
        Patch(facecolor="#999999", edgecolor="#222222", label="MILP feasible >= 95%"),
        Patch(facecolor="#999999", edgecolor="#d17a00", label="MILP feasible 85-95%"),
        Patch(facecolor="#999999", edgecolor="#b00020", label="MILP feasible < 85%"),
        Patch(facecolor="#999999", edgecolor="#222222", alpha=0.75, label=r"$^\dagger$ heuristic"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8, ncols=2)
    return _save_png(fig, "bar_speed_ratio_by_mode")


def heatmap_speedup_by_case_mode(
    summary: pd.DataFrame,
    mode_order: list[str],
) -> list[str] | None:
    req = {"case_folder", "mode", "speedup_vs_raw"}
    if not req.issubset(summary.columns):
        return None

    d = _small_case_frame(summary)
    d["speedup_vs_raw"] = pd.to_numeric(d["speedup_vs_raw"], errors="coerce")
    d = d.dropna(subset=["speedup_vs_raw"]).copy()
    if d.empty:
        return None

    case_order = sorted(d["case_folder"].astype(str).unique(), key=_case_sort_key)
    pivot = d.pivot_table(
        index="mode",
        columns="case_folder",
        values="speedup_vs_raw",
        aggfunc="first",
    ).reindex(index=mode_order, columns=case_order)
    pivot = pivot.dropna(how="all")
    if pivot.empty:
        return None

    rank_map = _mode_rank_map(mode_order)
    fig_w = max(7.2, 1.5 * len(case_order) + 3.2)
    fig_h = max(9.5, 0.33 * len(pivot.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        linewidths=0.25,
        linecolor="white",
        mask=pivot.isna(),
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Speedup vs RAW (higher is better)"},
        ax=ax,
    )
    ax.set_title("Small Cases: Speedup by Case and Mode")
    ax.set_xlabel("Case")
    ax.set_ylabel("Mode")
    ax.set_xticklabels([_case_tag(c) for c in pivot.columns], rotation=0)
    ax.set_yticklabels([_ranked_mode_label(str(m), rank_map) for m in pivot.index], rotation=0)
    ax.tick_params(axis="y", labelsize=7 if len(pivot.index) > 18 else 8)
    return _save_png(fig, "heatmap_speedup_by_case_mode")


def heatmap_speedup_ex_norel_by_case_mode(
    summary: pd.DataFrame,
    mode_order: list[str],
) -> list[str] | None:
    req = {"case_folder", "mode", "speedup_vs_raw_ex_norel"}
    if not req.issubset(summary.columns):
        return None

    d = _small_case_frame(summary)
    d["speedup_vs_raw_ex_norel"] = pd.to_numeric(
        d["speedup_vs_raw_ex_norel"], errors="coerce"
    )
    d = d.dropna(subset=["speedup_vs_raw_ex_norel"]).copy()
    if d.empty:
        return None

    case_order = sorted(d["case_folder"].astype(str).unique(), key=_case_sort_key)
    pivot = d.pivot_table(
        index="mode",
        columns="case_folder",
        values="speedup_vs_raw_ex_norel",
        aggfunc="first",
    ).reindex(index=mode_order, columns=case_order)
    pivot = pivot.dropna(how="all")
    if pivot.empty:
        return None

    rank_map = _mode_rank_map(mode_order)
    fig_w = max(7.2, 1.5 * len(case_order) + 3.2)
    fig_h = max(9.5, 0.33 * len(pivot.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        linewidths=0.25,
        linecolor="white",
        mask=pivot.isna(),
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Speedup vs RAW after subtracting NoRel time"},
        ax=ax,
    )
    ax.set_title("Small Cases: Speedup Excluding NoRel Heuristic Time")
    ax.set_xlabel("Case")
    ax.set_ylabel("Mode")
    ax.set_xticklabels([_case_tag(c) for c in pivot.columns], rotation=0)
    ax.set_yticklabels([_ranked_mode_label(str(m), rank_map) for m in pivot.index], rotation=0)
    ax.tick_params(axis="y", labelsize=7 if len(pivot.index) > 18 else 8)
    return _save_png(fig, "heatmap_speedup_ex_norel_by_case_mode")


def status_success_heatmap(
    merged: pd.DataFrame,
    mode_order: list[str],
) -> list[str] | None:
    req = {"case_folder", "mode"}
    if not req.issubset(merged.columns):
        return None

    d = _small_case_frame(merged)
    d["strict_success"] = _strict_success_mask(d).astype(float)
    case_order = sorted(d["case_folder"].astype(str).unique(), key=_case_sort_key)
    all_modes = [m for m in mode_order if m in set(d["mode"].astype(str))]
    if not all_modes or not case_order:
        return None

    pivot = (
        d.pivot_table(
            index="mode",
            columns="case_folder",
            values="strict_success",
            aggfunc="mean",
        )
        .reindex(index=all_modes, columns=case_order)
        .dropna(how="all")
    )
    if pivot.empty:
        return None

    rank_map = _mode_rank_map(all_modes)
    fig_w = max(7.5, 1.35 * len(case_order) + 3.0)
    fig_h = max(9.0, 0.33 * len(pivot.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        100.0 * pivot,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=100.0,
        linewidths=0.25,
        linecolor="white",
        mask=pivot.isna(),
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Verified MILP feasible [%]"},
        ax=ax,
    )
    ax.set_title("Small Cases: Verified MILP Feasibility by Mode")
    ax.set_xlabel("Case")
    ax.set_ylabel("Mode")
    ax.set_xticklabels([_case_tag(c) for c in pivot.columns], rotation=0)
    ax.set_yticklabels([_ranked_mode_label(str(m), rank_map) for m in pivot.index], rotation=0)
    ax.tick_params(axis="y", labelsize=7 if len(pivot.index) > 18 else 8)
    return _save_png(fig, "status_success_heatmap")


def ecdf_speedup(
    pairs: pd.DataFrame,
    small_mode_stats: pd.DataFrame,
    mode_order: list[str],
    color_map: dict[str, tuple[float, float, float]],
) -> list[str] | None:
    req = {"case_folder", "mode", "runtime_speedup"}
    if not req.issubset(pairs.columns):
        return None

    d = _small_case_frame(pairs)
    d["runtime_speedup"] = pd.to_numeric(d["runtime_speedup"], errors="coerce")
    d = d[np.isfinite(d["runtime_speedup"]) & (d["runtime_speedup"] > 0)].copy()
    if d.empty:
        return None

    small_stats = small_mode_stats.set_index("mode")
    present = set(d["mode"].astype(str))
    top_modes = _ecdf_mode_order(mode_order, present)
    if not top_modes:
        return None

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    ecdf_palette = sns.color_palette("tab20", n_colors=max(len(top_modes), 3))
    ecdf_colors = {mode: ecdf_palette[i] for i, mode in enumerate(top_modes)}
    for mode in top_modes:
        x = np.sort(
            d.loc[d["mode"].astype(str) == mode, "runtime_speedup"].to_numpy(dtype=float)
        )
        if x.size == 0:
            continue
        y = np.arange(1, x.size + 1, dtype=float) / float(x.size)
        strict = float(small_stats.loc[mode, "strict_success"]) if mode in small_stats.index else np.nan
        ax.step(
            x,
            y,
            where="post",
            lw=2.4 if mode != RAW_BASELINE_MODE else 2.0,
            color=ecdf_colors.get(mode, "#4c78a8"),
            linestyle="-",
            label=(
                f"{_mode_label(mode)} ({strict:.0%})"
                if np.isfinite(strict)
                else _mode_label(mode)
            ),
        )

    ax.axvline(1.0, ls="--", color="gray", lw=1)
    speed_values = d["runtime_speedup"].to_numpy(dtype=float)
    x_left = max(0.1, float(np.nanmin(speed_values)) * 0.9)
    x_right_full = float(np.nanmax(speed_values)) * 1.03
    x_right_clip = float(np.nanquantile(speed_values, 0.98)) * 1.08
    x_right = x_right_clip if x_right_full > x_right_clip * 1.2 else x_right_full
    ax.set_xlim(left=x_left, right=x_right)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Per-instance speedup vs RAW (higher is better)")
    ax.set_ylabel("Empirical cumulative share")
    ax.set_title(f"ECDF of Speedup for {len(top_modes)} Representative Modes")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.2, ncols=2 if len(top_modes) > 7 else 1)
    fig.text(
        0.01,
        0.01,
        "All ECDF curves are solid; the vertical dashed line marks parity with RAW. Legend values are verified-feasibility rates."
        + (" X-axis is clipped at the 98th percentile." if x_right < x_right_full else ""),
        fontsize=8,
        ha="left",
    )
    return _save_png(fig, "ecdf")


def large_cases_speedup_vs_lazy_by_case_mode(
    large_summary: pd.DataFrame,
    mode_order: list[str],
) -> list[str] | None:
    req = {"case_folder", "case_label", "mode", "speedup_vs_lazy"}
    if large_summary.empty or not req.issubset(large_summary.columns):
        return None
    d = large_summary.copy()
    d["speedup_vs_lazy"] = pd.to_numeric(d["speedup_vs_lazy"], errors="coerce")

    present = set(large_summary["mode"].astype(str))
    order = [mode for mode in mode_order if mode in present]
    if not order:
        return None
    case_order = sorted(d["case_folder"].astype(str).unique(), key=_case_sort_key)
    pivot = d.pivot_table(
        index="mode",
        columns="case_folder",
        values="speedup_vs_lazy",
        aggfunc="first",
    ).reindex(index=order, columns=case_order)
    if pivot.empty:
        return None

    fig_w = max(8.6, 1.35 * len(case_order) + 4.0)
    fig_h = max(5.2, 0.45 * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        cmap="RdYlGn",
        center=1.0,
        linewidths=0.35,
        linecolor="white",
        mask=pivot.isna(),
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Median runtime speedup vs LAZY"},
        ax=ax,
    )
    ax.set_title("Large Cases: Median Runtime Speedup vs LAZY")
    ax.set_xlabel("Case")
    ax.set_ylabel("Mode")
    ax.set_xticklabels([_case_tag(c) for c in pivot.columns], rotation=0)
    ax.set_yticklabels([_mode_label(str(m)) for m in pivot.index], rotation=0)
    ax.tick_params(axis="y", labelsize=8 if len(pivot.index) <= 12 else 7)
    fig.text(0.01, 0.01, "Values above 1.0 are faster than LAZY on the same case.", fontsize=8, ha="left")
    return _save_png(fig, "large_cases_speedup_vs_lazy_by_case_mode")


def large_cases_mip_gap_by_case_mode(
    large_summary: pd.DataFrame,
    mode_order: list[str],
) -> list[str] | None:
    req = {"case_folder", "mode", "mip_gap_median_pct"}
    if large_summary.empty or not req.issubset(large_summary.columns):
        return None
    d = large_summary.copy()
    d["mip_gap_median_pct"] = pd.to_numeric(d["mip_gap_median_pct"], errors="coerce")

    present = set(large_summary["mode"].astype(str))
    order = [mode for mode in mode_order if mode in present]
    if not order:
        return None
    case_order = sorted(d["case_folder"].astype(str).unique(), key=_case_sort_key)
    pivot = d.pivot_table(
        index="mode",
        columns="case_folder",
        values="mip_gap_median_pct",
        aggfunc="first",
    ).reindex(index=order, columns=case_order)
    if pivot.empty:
        return None
    plot_values = pivot.clip(lower=0.0, upper=100.0)

    fig_w = max(8.6, 1.35 * len(case_order) + 4.0)
    fig_h = max(5.2, 0.45 * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        plot_values,
        cmap="YlOrRd",
        linewidths=0.35,
        linecolor="white",
        mask=pivot.isna(),
        annot=pivot,
        fmt=".1f",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Median final MIP gap [%], clipped at 100 for color"},
        ax=ax,
    )
    ax.set_title("Large Cases: Median Final MIP Gap")
    ax.set_xlabel("Case")
    ax.set_ylabel("Mode")
    ax.set_xticklabels([_case_tag(c) for c in pivot.columns], rotation=0)
    ax.set_yticklabels([_mode_label(str(m)) for m in pivot.index], rotation=0)
    ax.tick_params(axis="y", labelsize=8 if len(pivot.index) <= 12 else 7)
    fig.text(0.01, 0.01, "Lower is better; 5% is the configured target gap.", fontsize=8, ha="left")
    return _save_png(fig, "large_cases_mip_gap_by_case_mode")


def large_cases_feasibility_by_case_mode(
    large_summary: pd.DataFrame,
    mode_order: list[str],
) -> list[str] | None:
    req = {"case_folder", "mode", "feasible_rate"}
    if large_summary.empty or not req.issubset(large_summary.columns):
        return None
    d = large_summary.copy()
    d["feasible_rate"] = pd.to_numeric(d["feasible_rate"], errors="coerce")

    present = set(large_summary["mode"].astype(str))
    order = [mode for mode in mode_order if mode in present]
    if not order:
        return None
    case_order = sorted(d["case_folder"].astype(str).unique(), key=_case_sort_key)
    pivot = d.pivot_table(
        index="mode",
        columns="case_folder",
        values="feasible_rate",
        aggfunc="first",
    ).reindex(index=order, columns=case_order)
    if pivot.empty:
        return None

    fig_w = max(8.6, 1.35 * len(case_order) + 4.0)
    fig_h = max(5.2, 0.45 * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        cmap="YlGn",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.35,
        linecolor="white",
        mask=pivot.isna(),
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Verified MILP-feasible share"},
        ax=ax,
    )
    ax.set_title("Large Cases: Verified Feasibility Rate")
    ax.set_xlabel("Case")
    ax.set_ylabel("Mode")
    ax.set_xticklabels([_case_tag(c) for c in pivot.columns], rotation=0)
    ax.set_yticklabels([_mode_label(str(m)) for m in pivot.index], rotation=0)
    ax.tick_params(axis="y", labelsize=8 if len(pivot.index) <= 12 else 7)
    return _save_png(fig, "large_cases_feasibility_by_case_mode")


def large_cases_time_to_first_incumbent_by_case_mode(
    large_summary: pd.DataFrame,
    mode_order: list[str],
) -> list[str] | None:
    req = {"case_folder", "mode", "first_incumbent_time_median_sec"}
    if large_summary.empty or not req.issubset(large_summary.columns):
        return None
    d = large_summary.copy()
    d["first_incumbent_time_median_sec"] = pd.to_numeric(
        d["first_incumbent_time_median_sec"], errors="coerce"
    )

    present = set(large_summary["mode"].astype(str))
    order = [mode for mode in mode_order if mode in present]
    if not order:
        return None
    case_order = sorted(d["case_folder"].astype(str).unique(), key=_case_sort_key)
    pivot = d.pivot_table(
        index="mode",
        columns="case_folder",
        values="first_incumbent_time_median_sec",
        aggfunc="first",
    ).reindex(index=order, columns=case_order)
    if pivot.empty:
        return None

    fig_w = max(8.6, 1.35 * len(case_order) + 4.0)
    fig_h = max(5.2, 0.45 * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        cmap="viridis_r",
        linewidths=0.35,
        linecolor="white",
        mask=pivot.isna(),
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Median time to first incumbent [s]"},
        ax=ax,
    )
    ax.set_title("Large Cases: Time To First Incumbent From Gurobi Logs")
    ax.set_xlabel("Case")
    ax.set_ylabel("Mode")
    ax.set_xticklabels([_case_tag(c) for c in pivot.columns], rotation=0)
    ax.set_yticklabels([_mode_label(str(m)) for m in pivot.index], rotation=0)
    ax.tick_params(axis="y", labelsize=8 if len(pivot.index) <= 12 else 7)
    fig.text(0.01, 0.01, "Includes timestamped NoRel heuristic incumbents and MIP progress-table incumbents.", fontsize=8, ha="left")
    return _save_png(fig, "large_cases_time_to_first_incumbent_by_case_mode")


def large_cases_gap_progress(
    progress_summary: pd.DataFrame,
    mode_order: list[str],
    color_map: dict[str, tuple[float, float, float]],
) -> list[str] | None:
    req = {"case_folder", "mode", "time_budget_pct", "median_gap_pct", "logs_with_incumbent"}
    if progress_summary.empty or not req.issubset(progress_summary.columns):
        return None
    d = progress_summary.copy()
    d["median_gap_pct"] = pd.to_numeric(d["median_gap_pct"], errors="coerce")
    d["time_budget_pct"] = pd.to_numeric(d["time_budget_pct"], errors="coerce")
    d["logs_with_incumbent"] = pd.to_numeric(d["logs_with_incumbent"], errors="coerce").fillna(0)
    d = d[np.isfinite(d["median_gap_pct"]) & np.isfinite(d["time_budget_pct"])].copy()
    if d.empty:
        return None

    cases = sorted(d["case_folder"].astype(str).unique(), key=_case_sort_key)
    present = set(d["mode"].astype(str))
    order = [mode for mode in mode_order if mode in present]
    if not order:
        return None
    ncols = 2 if len(cases) > 1 else 1
    nrows = int(np.ceil(len(cases) / ncols))
    fig_w = 12.0 if ncols == 2 else 7.0
    fig_h = max(4.2, 3.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False, sharex=True)
    used_modes: list[str] = []

    for ax, case in zip(axes.ravel(), cases):
        case_data = d[d["case_folder"].astype(str) == case]
        y_values: list[float] = []
        for mode in order:
            sub = case_data[case_data["mode"].astype(str) == mode].sort_values("time_budget_pct")
            if sub.empty:
                continue
            if mode not in used_modes:
                used_modes.append(mode)
            y = sub["median_gap_pct"].to_numpy(dtype=float)
            y_values.extend([float(v) for v in y if np.isfinite(v)])
            ax.plot(
                sub["time_budget_pct"].to_numpy(dtype=float),
                y,
                color=color_map.get(mode, "#555555"),
                linewidth=1.8,
                marker="o",
                markersize=2.5,
                label=_mode_label(mode),
                alpha=0.9,
            )
        ax.axhline(5.0, color="#555555", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_title(_case_tag(case))
        ax.set_xlim(0.0, 100.0)
        ax.set_ylim(0.0, 100.0)
        ax.grid(alpha=0.25)

    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Elapsed share of case time limit [%]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Median best log gap [%]")

    handles = [
        Line2D([0], [0], color=color_map.get(mode, "#555555"), lw=2, label=_mode_label(mode))
        for mode in used_modes
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=min(4, len(handles)),
            fontsize=8,
            frameon=False,
        )
    fig.suptitle("Large Cases: Gurobi Gap Progress From Logs", y=0.985)
    fig.text(
        0.01,
        0.01,
        "Each line is the median best incumbent gap over TEST dates; gaps are held forward between logged progress points.",
        fontsize=8,
        ha="left",
    )
    fig.subplots_adjust(top=0.80 if handles else 0.92, bottom=0.10, hspace=0.35, wspace=0.18)
    return _save_png(fig, "large_cases_gap_progress", tight=False)


def constraints_runtime_tradeoff(
    summary: pd.DataFrame,
    merged: pd.DataFrame,
    mode_order: list[str],
) -> list[str] | None:
    req_merged = {"case_folder", "mode", "num_constrs_final"}
    if not req_merged.issubset(merged.columns):
        return None

    m = _small_case_frame(merged)
    m["num_constrs_final"] = pd.to_numeric(m["num_constrs_final"], errors="coerce")
    if "fixed_commit_vars" in m.columns:
        m["fixed_commit_vars"] = (
            pd.to_numeric(m["fixed_commit_vars"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
    else:
        m["fixed_commit_vars"] = 0.0
    m["core_num_constrs_final"] = (
        m["num_constrs_final"] - m["fixed_commit_vars"]
    ).where(lambda s: s > 0)
    d = (
        m.groupby(["case_folder", "mode"], as_index=False)
        .agg(
            final_constrs_median=("core_num_constrs_final", "median"),
            fixed_commit_constrs_median=("fixed_commit_vars", "median"),
        )
    )
    d = d.dropna(subset=["final_constrs_median"])
    d = d[d["final_constrs_median"] > 0].copy()
    if d.empty:
        return None

    raw_by_case = (
        d.loc[d["mode"].astype(str) == RAW_BASELINE_MODE]
        .set_index("case_folder")["final_constrs_median"]
    )
    if raw_by_case.empty:
        return None
    d["raw_final_constrs_median"] = d["case_folder"].map(raw_by_case)
    d = d[d["raw_final_constrs_median"] > 0].copy()
    if d.empty:
        return None
    d["final_constrs_vs_raw_pct"] = (
        100.0 * d["final_constrs_median"] / d["raw_final_constrs_median"]
    )
    d = d[np.isfinite(d["final_constrs_vs_raw_pct"])].copy()
    if d.empty:
        return None

    case_order = sorted(d["case_folder"].astype(str).unique(), key=_case_sort_key)
    present_modes = [m for m in mode_order if m in set(d["mode"].astype(str))]
    pivot = (
        d.pivot_table(
            index="mode",
            columns="case_folder",
            values="final_constrs_vs_raw_pct",
            aggfunc="first",
        )
        .reindex(index=present_modes, columns=case_order)
        .dropna(how="all")
    )
    if pivot.empty:
        return None

    annot = pivot.apply(lambda col: col.map(_format_percent))
    fig_w = max(8.0, 1.35 * len(case_order) + 3.5)
    fig_h = max(9.0, 0.25 * len(pivot.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=100.0,
        linewidths=0.25,
        linecolor="white",
        mask=pivot.isna(),
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 6.2},
        cbar_kws={"label": "Core final constraints vs RAW [%]"},
        ax=ax,
    )
    ax.set_title("Small Cases: Median Core Final Constraint Count Relative to RAW")
    ax.set_xlabel("Case")
    ax.set_ylabel("Mode")
    ax.set_xticklabels([_case_tag(c) for c in pivot.columns], rotation=0)
    ax.set_yticklabels(
        [
            f"{mode_order.index(mode) + 1:02d}. {_mode_label(mode)}"
            if mode in mode_order
            else _mode_label(mode)
            for mode in pivot.index
        ],
        rotation=0,
    )
    ax.tick_params(axis="y", labelsize=6.4 if len(pivot.index) > 35 else 7.5)
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 1.0))
    fig.text(
        0.01,
        0.01,
        "100% is the RAW final constraint count for the same case; fixed-commit auxiliary constraints are excluded from the percentage.",
        fontsize=8,
        ha="left",
    )
    return _save_png(fig, "constraints_runtime_tradeoff", tight=False)


def avg_speedup_vs_success_pareto(
    merged: pd.DataFrame,
    pairs: pd.DataFrame,
    mode_order: list[str],
    color_map: dict[str, tuple[float, float, float]],
) -> list[str] | None:
    merged_valid, pairs_valid, info = _benchmark_valid_small_frames(merged, pairs)
    stats = _build_small_mode_stats(merged_valid, pairs_valid)
    if stats.empty:
        return None

    stats = stats.copy()
    stats["family"] = stats["mode"].astype(str).map(_mode_family_label)
    stats["plot_family"] = stats["mode"].astype(str).map(_pareto_plot_family_label)
    stats["pareto_front"] = _pareto_front_mask(stats)
    stats["full_success_boundary"] = (
        stats["strict_success"] >= (1.0 - FULL_SUCCESS_TOL)
    )
    stats["family_best"] = _family_best_mask(stats)
    acceptable = stats["strict_success"] >= PARETO_SUCCESS_THRESHOLD
    stats["acceptable_pareto_front"] = False
    if acceptable.any():
        stats.loc[acceptable, "acceptable_pareto_front"] = _pareto_front_mask(
            stats.loc[acceptable]
        )

    present_stats_modes = set(stats["mode"].astype(str))
    rank_order = [m for m in mode_order if m in present_stats_modes]
    for mode in (
        stats.dropna(subset=["mean_speedup"])
        .sort_values(
            ["mean_speedup", "strict_success", "mode"],
            ascending=[False, False, True],
        )["mode"]
        .astype(str)
    ):
        if mode not in rank_order:
            rank_order.append(mode)
    rank_map = {mode: idx + 1 for idx, mode in enumerate(rank_order)}
    stats["speedup_rank"] = stats["mode"].astype(str).map(rank_map)
    stats["numbered_highlight"] = True

    stats = stats.sort_values(
        ["pareto_front", "strict_success", "mean_speedup", "mode"],
        ascending=[False, False, False, True],
    )
    stats.rename(
        columns={
            "mean_speedup": "avg_speedup",
            "strict_success": "avg_success",
        }
    ).to_csv(MODE_PARETO_SPEED_SUCCESS, index=False)
    stats.loc[stats["family_best"]].sort_values(
        ["family", "strict_success", "mean_speedup"],
        ascending=[True, False, False],
    ).rename(
        columns={
            "mean_speedup": "avg_speedup",
            "strict_success": "avg_success",
        }
    ).to_csv(MODE_FAMILY_BEST_SPEED_SUCCESS, index=False)

    front_modes = (
        stats.loc[stats["pareto_front"]]
        .sort_values(["strict_success", "mean_speedup"], ascending=[False, False])
        ["mode"]
        .astype(str)
        .tolist()
    )
    if front_modes:
        large_available = {canonical_mode(spec.mode_id) for spec in MODE_CATALOG_MEDLARGE_FULL}
        preferred = [
            m
            for m in [LARGE_REFERENCE_MODE]
            if m in set(stats["mode"].astype(str)) and m in large_available
        ]
        selected = list(
            dict.fromkeys(preferred + [m for m in front_modes if m in large_available])
        )
        LARGE_MODE_SELECTION_PARETO.write_text(" ".join(selected) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11.8, 7.1))
    front = stats.loc[stats["pareto_front"]].copy()

    stats["plot_success"] = stats["strict_success"].astype(float)
    stats["plot_speedup"] = stats["mean_speedup"].astype(float)
    speed = stats["mean_speedup"].astype(float)
    speed_bin = np.select(
        [speed < 3.0, speed < 5.0, speed < 10.0],
        [(speed / 0.90).round(), (speed / 0.70).round(), (speed / 1.20).round()],
        default=speed.round(),
    ).astype(int)
    stats["_dense_key"] = list(
        zip((34.0 * stats["strict_success"]).round().astype(int), speed_bin)
    )
    for _, dense in stats.groupby("_dense_key", sort=False):
        if len(dense) <= 1:
            continue
        ordered_idx = dense.sort_values(["speedup_rank", "mode"]).index.tolist()
        cols = int(np.ceil(np.sqrt(len(ordered_idx))))
        rows = int(np.ceil(len(ordered_idx) / cols))
        center_x = float(dense["strict_success"].median())
        center_y = float(dense["mean_speedup"].median())
        for pos, idx in enumerate(ordered_idx):
            col = pos % cols - (cols - 1) / 2.0
            row = pos // cols - (rows - 1) / 2.0
            dx = 0.020 if center_x >= 0.96 else 0.014
            dy = 0.70 if center_y < 3.0 else 0.58 if center_y < 5.0 else 0.64 if center_y < 10.0 else 0.48
            stats.at[idx, "plot_success"] = min(max(center_x + col * dx, 0.0), 1.045)
            stats.at[idx, "plot_speedup"] = max(center_y + row * dy, 0.08)

    xmin = max(0.0, float(stats["strict_success"].min()) - 0.05)
    xmax = 1.055
    ymax = max(1.2, float(stats["mean_speedup"].max()) * 1.14)
    ymin = 0.0

    plot_xy = stats[["plot_success", "plot_speedup"]].to_numpy(dtype=float)
    x_span = max(xmax - xmin, 1e-9)
    y_span = max(ymax - ymin, 1e-9)
    axes_width_in = 11.8 * (0.985 - 0.095)
    axes_height_in = 7.1 * (0.82 - 0.15)
    y_floor = max(ymin + 0.32, 0.45)
    y_ceiling = ymax - 0.25
    y_floor_display = (y_floor - ymin) / y_span * axes_height_in
    y_ceiling_display = (y_ceiling - ymin) / y_span * axes_height_in
    display_xy = np.column_stack(
        (
            (plot_xy[:, 0] - xmin) / x_span * axes_width_in,
            (plot_xy[:, 1] - ymin) / y_span * axes_height_in,
        )
    )
    anchor_xy = display_xy.copy()
    ranks = stats["speedup_rank"].fillna(0).to_numpy(dtype=int)
    min_display_distance = 0.245
    for _ in range(220):
        moved = False
        for i in range(len(display_xy)):
            for j in range(i + 1, len(display_xy)):
                delta = display_xy[i] - display_xy[j]
                dist = float(np.hypot(delta[0], delta[1]))
                if dist >= min_display_distance:
                    continue
                if dist < 1e-9:
                    angle = np.deg2rad((37 * ranks[i] + 17 * ranks[j]) % 360)
                    unit = np.array([np.cos(angle), np.sin(angle)])
                else:
                    unit = delta / dist
                push = 0.52 * (min_display_distance - dist)
                display_xy[i] += unit * push
                display_xy[j] -= unit * push
                moved = True
        display_xy += 0.010 * (anchor_xy - display_xy)
        display_xy[:, 0] = np.clip(display_xy[:, 0], 0.02, axes_width_in - 0.02)
        display_xy[:, 1] = np.clip(display_xy[:, 1], y_floor_display, y_ceiling_display)
        if not moved:
            break

    stats["plot_success"] = xmin + (display_xy[:, 0] / axes_width_in) * x_span
    stats["plot_speedup"] = ymin + (display_xy[:, 1] / axes_height_in) * y_span

    present_plot_families = [
        label
        for label in PARETO_PLOT_FAMILY_COLORS
        if label in set(stats["plot_family"].astype(str))
    ]
    for family in present_plot_families:
        group = stats[stats["plot_family"] == family]
        ax.scatter(
            group["plot_success"],
            group["plot_speedup"],
            s=165,
            marker="o",
            facecolor=PARETO_PLOT_FAMILY_COLORS[family],
            edgecolor="#34404c",
            linewidth=0.75,
            alpha=0.74,
            zorder=2,
        )

    family_best = stats.loc[stats["family_best"]].copy()
    if not family_best.empty:
        ax.scatter(
            family_best["plot_success"],
            family_best["plot_speedup"],
            s=190,
            marker="o",
            facecolor=[
                PARETO_PLOT_FAMILY_COLORS.get(label, "#8C8C8C")
                for label in family_best["plot_family"]
            ],
            edgecolor="#111111",
            linewidth=1.15,
            alpha=0.96,
            zorder=4,
        )
        ax.scatter(
            family_best["plot_success"],
            family_best["plot_speedup"],
            s=260,
            marker="o",
            facecolors="none",
            edgecolors="#F0A202",
            linewidth=2.1,
            zorder=4,
        )

    if not front.empty:
        front_line = front.sort_values(["strict_success", "mean_speedup"])
        ax.plot(
            front_line["strict_success"],
            front_line["mean_speedup"],
            color="#222222",
            lw=1.8,
            alpha=0.85,
            zorder=3,
        )

    numbered = stats.loc[stats["numbered_highlight"]].copy()
    numbered = numbered.dropna(subset=["speedup_rank"]).sort_values(
        ["speedup_rank", "mode"]
    )
    for _, row in numbered.iterrows():
        x = float(row["plot_success"])
        y_val = float(row["plot_speedup"])
        rank = int(row["speedup_rank"])
        ax.text(
            x,
            y_val,
            f"{rank:02d}",
            ha="center",
            va="center",
            fontsize=5.1,
            fontweight="bold",
            color="#111111",
            zorder=5,
        )

    ax.axvspan(
        PARETO_SUCCESS_THRESHOLD,
        1.05,
        color="#e7f2ed",
        alpha=0.42,
        zorder=0,
    )
    ax.axvline(
        PARETO_SUCCESS_THRESHOLD,
        ls="--",
        color="#008060",
        lw=1.0,
        alpha=0.8,
    )
    ax.axhline(1.0, ls=":", color="#777777", lw=1.0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Average verified MILP success")
    ax.set_ylabel("Average speedup vs RAW")
    ax.grid(alpha=0.24)
    ax.xaxis.set_major_formatter(lambda x, _: f"{100*x:.0f}%")

    footer = f"Benchmark-valid: {info['raw_valid_instances']}/{info['raw_instances_total']} RAW-feasible small TEST instances."
    if info["sleep_overlap_rows_excluded"]:
        footer += f" Excluded sleep-overlap rows: {info['sleep_overlap_rows_excluded']}."
    footer += " Numbers are speedup ranks inside mode dots; tight clusters are locally separated for readability."
    fig.text(0.08, 0.034, footer, fontsize=7.3, ha="left", va="bottom")

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label=family,
            markerfacecolor=PARETO_PLOT_FAMILY_COLORS[family],
            markeredgecolor="#34404c",
            markeredgewidth=0.8,
            markersize=6.4,
            alpha=0.78,
        )
        for family in present_plot_families
    ]
    special_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label="Best in family",
            markerfacecolor="white",
            markeredgecolor="#F0A202",
            markeredgewidth=2.1,
            markersize=8.0,
        ),
        Line2D([0], [0], color="#222222", lw=1.8, label="Pareto front"),
        Line2D(
            [0],
            [0],
            color="#008060",
            lw=1.0,
            ls="--",
            label=f"{PARETO_SUCCESS_THRESHOLD:.0%} success cutoff",
        ),
    ]
    ax.legend(
        handles=family_handles + special_handles,
        fontsize=7.2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=5,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.82, bottom=0.15)
    return _save_png(fig, "avg_speedup_vs_success_pareto", tight=False)


def main() -> None:
    _ensure()
    summary = _load_summary()
    merged = _normalize_runtime_frame(_read_csv_required(MERGED))
    pairs = _read_csv_required(PAIRS)
    for df in (summary, merged, pairs):
        for col in ("mode", "mode_a", "mode_b"):
            if col in df.columns:
                df[col] = df[col].map(canonical_mode)

    small_summary = _small_case_frame(summary)
    small_merged = _small_case_frame(merged)
    small_pairs = _small_case_frame(pairs)

    valid_merged, valid_pairs, _ = _benchmark_valid_small_frames(small_merged, small_pairs)
    small_mode_stats = _build_small_mode_stats(valid_merged, valid_pairs)
    present_modes = set(small_merged["mode"].astype(str)) | set(small_summary["mode"].astype(str))
    mode_order = _speedup_sorted_mode_order(small_mode_stats, present_modes)
    paired_mode_order = [m for m in mode_order if m in set(small_mode_stats["mode"].astype(str))]
    heatmap_mode_order = _main_heatmap_mode_order(mode_order, present_modes)
    large_rows = _large_cases_test_frame(merged)
    large_progress_rows = _large_gurobi_progress_rows(large_rows)
    large_summary = _large_cases_summary(large_rows, large_progress_rows)
    large_progress_summary = _large_gurobi_progress_summary(large_progress_rows, large_rows)
    large_plot_mode_order = _large_selected_mode_order(large_summary, mode_order) or mode_order
    color_order = list(dict.fromkeys(mode_order + large_plot_mode_order))
    color_map = _mode_color_map(color_order)
    large_rows.to_csv(LARGE_GAP_RUNTIME_ROWS, index=False)
    large_summary.to_csv(LARGE_GAP_RUNTIME_SUMMARY, index=False)
    large_progress_rows.to_csv(LARGE_GUROBI_PROGRESS_ROWS, index=False)
    large_progress_summary.to_csv(LARGE_GUROBI_PROGRESS_SUMMARY, index=False)

    generated: set[str] = set()
    outputs = [
        bar_speed_ratio_by_mode(small_mode_stats, paired_mode_order, color_map),
        heatmap_speedup_by_case_mode(small_summary, heatmap_mode_order),
        heatmap_speedup_ex_norel_by_case_mode(small_summary, heatmap_mode_order),
        status_success_heatmap(small_merged, heatmap_mode_order),
        ecdf_speedup(small_pairs, small_mode_stats, paired_mode_order, color_map),
        constraints_runtime_tradeoff(small_summary, small_merged, mode_order),
        avg_speedup_vs_success_pareto(small_merged, small_pairs, paired_mode_order, color_map),
        large_cases_speedup_vs_lazy_by_case_mode(large_summary, large_plot_mode_order),
        large_cases_mip_gap_by_case_mode(large_summary, large_plot_mode_order),
        large_cases_feasibility_by_case_mode(large_summary, large_plot_mode_order),
        large_cases_time_to_first_incumbent_by_case_mode(large_summary, large_plot_mode_order),
        large_cases_gap_progress(large_progress_summary, large_plot_mode_order, color_map),
    ]

    for out in outputs:
        if not out:
            continue
        if isinstance(out, list):
            generated.update(out)
        else:
            generated.add(out)

    _cleanup_figure_dir(generated)
    print(f"Figures saved under {FIG_DIR}")
    print("Generated:", ", ".join(sorted(generated)) if generated else "<none>")


if __name__ == "__main__":
    main()
