"""
All-mode, publication-friendly table generation.

Generates full-coverage LaTeX tables:
  - results/tables/optimality_by_case.tex
  - results/tables/speedup_by_case.tex
  - results/tables/runtime_by_case.tex
  - results/tables/memory_by_case.tex
  - results/tables/quality_by_mode.tex
  - results/tables/constraints_by_mode.tex
  - results/tables/fastest_share_by_case.tex
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

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
from src.paper.experiment_spec import CASES_LARGE

SUMMARY = Path("results") / "summary.csv"
MERGED = Path("results") / "merged_results.csv"
MODE_SPEED = Path("results") / "mode_speed_stats.csv"
LARGE_LAZY = Path("results") / "large_lazy_comparison.csv"
OUT_DIR = Path("results") / "tables"
PAPER_TABLE_DIR = OUT_DIR


def _ensure() -> None:
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _cleanup_table_dir(keep_tex: set[str]) -> None:
    for p in PAPER_TABLE_DIR.iterdir():
        if p.name == ".gitkeep":
            continue
        if p.is_file() and p.suffix.lower() == ".tex" and p.name not in keep_tex:
            p.unlink(missing_ok=True)


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


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _mode_exactness(mode: str) -> str:
    return mode_exactness(mode)


def _mode_display(mode: str) -> str:
    m = mode_display(mode)
    return rf"{m}$^\dagger$" if _mode_exactness(m) == "heuristic" else m


def _case_label(case_folder: str) -> str:
    return str(case_folder).split("/")[-1]


def _case_sort_key(case_folder: str) -> tuple[int, str]:
    tag = _case_label(case_folder)
    order = {
        "case14": 14,
        "case30": 30,
        "case57": 57,
        "case89pegase": 89,
        "case118": 118,
        "case300": 300,
        "case1354pegase": 1354,
    }
    return (order.get(tag, 10_000), tag)


def _tabularx_spec(num_case_cols: int) -> str:
    return "|p{0.23\\textwidth}||" + "|".join(["X"] * num_case_cols) + "|"


def _ordered_modes(present_modes: list[str] | set[str], mode_speed: pd.DataFrame | None = None) -> list[str]:
    present = {
        canonical_mode(m)
        for m in present_modes
        if str(m or "").strip()
    }
    return [mode for mode in ordered_modes(present) if mode in present]


def _fmt_num(x, fmt="{:.2f}") -> str:
    if not np.isfinite(x):
        return "--"
    return fmt.format(float(x))


def _fmt_pct(x, decimals: int = 0) -> str:
    if not np.isfinite(x):
        return "--"
    return f"{100.0 * float(x):.{decimals}f}\\%"


def _best_by_case(summary: pd.DataFrame, value_col: str, higher_is_better: bool) -> dict[str, float]:
    d = summary.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    out: dict[str, float] = {}
    for case, g in d.groupby("case_folder"):
        vals = g[value_col].dropna()
        if vals.empty:
            continue
        out[str(case)] = float(vals.max() if higher_is_better else vals.min())
    return out


def _maybe_bold(text: str, value: float, best: float | None) -> str:
    if best is None or not np.isfinite(value) or not np.isfinite(best):
        return text
    if abs(float(value) - float(best)) <= max(1e-9, abs(float(best)) * 1e-9):
        return rf"\textbf{{{text}}}"
    return text


def optimality_by_case(merged: pd.DataFrame, mode_speed: pd.DataFrame) -> str | None:
    cases = sorted(merged["case_folder"].dropna().astype(str).unique(), key=_case_sort_key)
    modes = _ordered_modes(merged["mode"].dropna().astype(str).unique(), mode_speed=mode_speed)
    if not cases or not modes:
        return None

    tex = []
    tex.append(
        rf"\begin{{tabularx}}{{0.98\textwidth}}{{{_tabularx_spec(len(cases))}}}\hline"
    )
    header = " & ".join(
        [r"\textbf{Mode}"] + [rf"\textbf{{{_case_label(case)}}}" for case in cases]
    )
    tex.append(header + r" \\ \hline \hline")

    for mode in modes:
        row = [_mode_display(mode)]
        for case in cases:
            g = merged[(merged["case_folder"] == case) & (merged["mode"] == mode)]
            if g.empty:
                row.append("--")
                continue
            status = g["status"].astype(str).str.upper()
            total = len(status)
            opt = int((status == "OPTIMAL").sum())
            infeas = int(
                status.isin(["INFEASIBLE", "INF_OR_UNBD", "INFEASIBLE_OR_UNBOUNDED"]).sum()
            )
            tl = int(status.isin(["TIME_LIMIT", "SUBOPTIMAL"]).sum())
            err = int(status.isin(["ERROR"]).sum())
            row.append(
                f"{100.0 * opt / total:.0f}/{100.0 * tl / total:.0f}/{100.0 * infeas / total:.0f}/{100.0 * err / total:.0f}"
            )
        tex.append(" & ".join(row) + r" \\ \hline")

    tex.append(
        rf"\multicolumn{{{1 + len(cases)}}}{{l}}{{\footnotesize Entries are Optimal / TimeLimit-or-Suboptimal / Infeasible / Error shares in percent.}} \\ \hline"
    )
    if any(_mode_exactness(m) == "heuristic" for m in modes):
        tex.append(
            rf"\multicolumn{{{1 + len(cases)}}}{{l}}{{\footnotesize $^\dagger$ Heuristic mode (not exact MILP-equivalent).}} \\ \hline"
        )
    tex.append(r"\end{tabularx}")

    out = PAPER_TABLE_DIR / "optimality_by_case.tex"
    out.write_text("\n".join(tex), encoding="utf-8")
    return out.name


def speedup_by_case(summary: pd.DataFrame, mode_speed: pd.DataFrame) -> str | None:
    cases = sorted(summary["case_folder"].dropna().astype(str).unique(), key=_case_sort_key)
    modes = _ordered_modes(summary["mode"].dropna().astype(str).unique(), mode_speed=mode_speed)
    if not cases or not modes:
        return None
    best = _best_by_case(summary, "speedup_vs_raw", higher_is_better=True)

    tex = []
    tex.append(
        rf"\begin{{tabularx}}{{0.98\textwidth}}{{{_tabularx_spec(len(cases))}}}\hline"
    )
    header = " & ".join(
        [r"\textbf{Mode}"] + [rf"\textbf{{{_case_label(case)}}}" for case in cases]
    )
    tex.append(header + r" \\ \hline \hline")

    for mode in modes:
        row = [_mode_display(mode)]
        for case in cases:
            s = summary[(summary["case_folder"] == case) & (summary["mode"] == mode)]
            if s.empty:
                row.append("--")
                continue
            val = pd.to_numeric(s["speedup_vs_raw"], errors="coerce").iloc[0]
            cell = f"{val:.2f}x" if np.isfinite(val) else "--"
            row.append(_maybe_bold(cell, val, best.get(case)))
        tex.append(" & ".join(row) + r" \\ \hline")

    if any(_mode_exactness(m) == "heuristic" for m in modes):
        tex.append(
            rf"\multicolumn{{{1 + len(cases)}}}{{l}}{{\footnotesize $^\dagger$ Heuristic mode (not exact MILP-equivalent).}} \\ \hline"
        )
    tex.append(r"\end{tabularx}")

    out = PAPER_TABLE_DIR / "speedup_by_case.tex"
    out.write_text("\n".join(tex), encoding="utf-8")
    return out.name


def metric_by_case(
    summary: pd.DataFrame,
    mode_speed: pd.DataFrame,
    *,
    value_col: str,
    out_name: str,
    fmt: str,
    best_low: bool = True,
) -> str | None:
    req = {"case_folder", "mode", value_col}
    if not req.issubset(summary.columns):
        return None
    cases = sorted(summary["case_folder"].dropna().astype(str).unique(), key=_case_sort_key)
    modes = _ordered_modes(summary["mode"].dropna().astype(str).unique(), mode_speed=mode_speed)
    if not cases or not modes:
        return None
    best = _best_by_case(summary, value_col, higher_is_better=not best_low)

    tex = []
    tex.append(rf"\begin{{tabularx}}{{0.98\textwidth}}{{{_tabularx_spec(len(cases))}}}\hline")
    header = " & ".join([r"\textbf{Mode}"] + [rf"\textbf{{{_case_label(case)}}}" for case in cases])
    tex.append(header + r" \\ \hline \hline")
    for mode in modes:
        row = [_mode_display(mode)]
        for case in cases:
            s = summary[(summary["case_folder"] == case) & (summary["mode"] == mode)]
            if s.empty:
                row.append("--")
                continue
            val = pd.to_numeric(s[value_col], errors="coerce").iloc[0]
            cell = _fmt_num(val, fmt)
            row.append(_maybe_bold(cell, val, best.get(case)))
        tex.append(" & ".join(row) + r" \\ \hline")
    tex.append(r"\end{tabularx}")

    out = PAPER_TABLE_DIR / out_name
    out.write_text("\n".join(tex), encoding="utf-8")
    return out.name


def quality_by_mode(summary: pd.DataFrame, mode_speed: pd.DataFrame) -> str | None:
    req = {"mode", "N", "runtime_median", "speedup_vs_raw", "success_rate", "feasible_rate", "mip_gap_median", "obj_ppm_median", "mem_gb_median"}
    if not req.issubset(summary.columns):
        return None

    d = summary.copy()
    numeric_cols = ["N", "runtime_median", "speedup_vs_raw", "success_rate", "feasible_rate", "mip_gap_median", "obj_ppm_median", "mem_gb_median"]
    for col in numeric_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    agg = (
        d.groupby("mode", as_index=False)
        .agg(
            cases=("case_folder", "nunique"),
            runs=("N", "sum"),
            runtime_median=("runtime_median", "median"),
            speedup_median=("speedup_vs_raw", "median"),
            success_rate=("success_rate", "mean"),
            feasible_rate=("feasible_rate", "mean"),
            mip_gap_median=("mip_gap_median", "median"),
            obj_ppm_abs_median=("obj_ppm_median", lambda x: float(np.nanmedian(np.abs(x)))),
            mem_gb_median=("mem_gb_median", "median"),
        )
        .copy()
    )
    modes = _ordered_modes(agg["mode"].dropna().astype(str).unique(), mode_speed=mode_speed)
    agg = agg.set_index("mode").reindex(modes).reset_index()

    tex = []
    tex.append(r"\begin{tabularx}{0.98\textwidth}{|p{0.22\textwidth}||X|X|X|X|X|X|X|X|}\hline")
    tex.append(
        r"\textbf{Mode} & \textbf{Cases} & \textbf{Runs} & \textbf{Med. runtime [s]} & \textbf{Med. speedup} & \textbf{MILP feas.} & \textbf{Constraint OK} & \textbf{Med. MIP gap} & \textbf{Med. mem. [GB]} \\ \hline \hline"
    )
    for _, row in agg.iterrows():
        tex.append(
            " & ".join(
                [
                    _mode_display(str(row["mode"])),
                    _fmt_num(row["cases"], "{:.0f}"),
                    _fmt_num(row["runs"], "{:.0f}"),
                    _fmt_num(row["runtime_median"], "{:.2f}"),
                    (_fmt_num(row["speedup_median"], "{:.2f}") + "x") if np.isfinite(row["speedup_median"]) else "--",
                    _fmt_pct(row["success_rate"]),
                    _fmt_pct(row["feasible_rate"]),
                    _fmt_num(row["mip_gap_median"], "{:.1e}"),
                    _fmt_num(row["mem_gb_median"], "{:.2f}"),
                ]
            )
            + r" \\ \hline"
        )
    tex.append(r"\end{tabularx}")

    out = PAPER_TABLE_DIR / "quality_by_mode.tex"
    out.write_text("\n".join(tex), encoding="utf-8")
    return out.name


def constraints_by_mode(merged: pd.DataFrame, mode_speed: pd.DataFrame) -> str | None:
    req = {"mode", "num_constrs_final", "constr_total_cont", "constr_kept_cont", "constr_ratio_cont"}
    if not req.issubset(merged.columns):
        return None
    d = merged.copy()
    for col in ["num_constrs_final", "constr_total_cont", "constr_kept_cont", "constr_ratio_cont", "lazy_added_cont", "screen_setup_sec"]:
        if col not in d.columns:
            d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors="coerce")
    agg = (
        d.groupby("mode", as_index=False)
        .agg(
            final_constrs_median=("num_constrs_final", "median"),
            total_cont_median=("constr_total_cont", "median"),
            kept_cont_median=("constr_kept_cont", "median"),
            kept_ratio_median=("constr_ratio_cont", "median"),
            lazy_added_median=("lazy_added_cont", "median"),
            screen_setup_median=("screen_setup_sec", "median"),
        )
        .copy()
    )
    modes = _ordered_modes(agg["mode"].dropna().astype(str).unique(), mode_speed=mode_speed)
    agg = agg.set_index("mode").reindex(modes).reset_index()

    tex = []
    tex.append(r"\begin{tabularx}{0.98\textwidth}{|p{0.24\textwidth}||X|X|X|X|X|X|}\hline")
    tex.append(
        r"\textbf{Mode} & \textbf{Final constr.} & \textbf{Candidate N-1} & \textbf{Kept N-1} & \textbf{Kept ratio} & \textbf{Lazy cuts} & \textbf{Screen setup [s]} \\ \hline \hline"
    )
    for _, row in agg.iterrows():
        tex.append(
            " & ".join(
                [
                    _mode_display(str(row["mode"])),
                    _fmt_num(row["final_constrs_median"], "{:.0f}"),
                    _fmt_num(row["total_cont_median"], "{:.0f}"),
                    _fmt_num(row["kept_cont_median"], "{:.0f}"),
                    _fmt_pct(row["kept_ratio_median"], decimals=1),
                    _fmt_num(row["lazy_added_median"], "{:.0f}"),
                    _fmt_num(row["screen_setup_median"], "{:.2f}"),
                ]
            )
            + r" \\ \hline"
        )
    tex.append(r"\end{tabularx}")

    out = PAPER_TABLE_DIR / "constraints_by_mode.tex"
    out.write_text("\n".join(tex), encoding="utf-8")
    return out.name


def fastest_share_by_case(fastest: pd.DataFrame, mode_speed: pd.DataFrame) -> str | None:
    req = {"case_folder", "mode", "fastest_share"}
    if fastest.empty or not req.issubset(fastest.columns):
        return None
    cases = sorted(fastest["case_folder"].dropna().astype(str).unique(), key=_case_sort_key)
    modes = _ordered_modes(fastest["mode"].dropna().astype(str).unique(), mode_speed=mode_speed)
    if not cases or not modes:
        return None
    tex = []
    tex.append(rf"\begin{{tabularx}}{{0.98\textwidth}}{{{_tabularx_spec(len(cases))}}}\hline")
    header = " & ".join([r"\textbf{Mode}"] + [rf"\textbf{{{_case_label(case)}}}" for case in cases])
    tex.append(header + r" \\ \hline \hline")
    for mode in modes:
        row = [_mode_display(mode)]
        for case in cases:
            s = fastest[(fastest["case_folder"] == case) & (fastest["mode"] == mode)]
            if s.empty:
                row.append("--")
                continue
            row.append(_fmt_pct(pd.to_numeric(s["fastest_share"], errors="coerce").iloc[0], decimals=0))
        tex.append(" & ".join(row) + r" \\ \hline")
    tex.append(r"\end{tabularx}")
    out = PAPER_TABLE_DIR / "fastest_share_by_case.tex"
    out.write_text("\n".join(tex), encoding="utf-8")
    return out.name


def large_lazy_comparison_table(large_lazy: pd.DataFrame) -> str | None:
    req = {
        "case_folder",
        "mode",
        "runtime_median",
        "speedup_vs_lazy",
        "success_rate",
        "feasible_rate",
        "mip_gap_median",
    }
    if large_lazy.empty or not req.issubset(large_lazy.columns):
        return None
    d = large_lazy[large_lazy["case_folder"].isin(CASES_LARGE)].copy()
    if d.empty:
        return None
    for col in ["runtime_median", "speedup_vs_lazy", "success_rate", "feasible_rate", "mip_gap_median"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["mode"] = d["mode"].map(canonical_mode)
    d = d.sort_values(["case_folder", "runtime_median", "mode"])

    tex = []
    tex.append(r"\begin{tabularx}{0.74\textwidth}{|p{0.22\textwidth}||X|X|X|X|X|}\hline")
    tex.append(
        r"\textbf{Mode} & \textbf{Med. runtime [s]} & \textbf{Speedup vs LAZY} & \textbf{MILP feasible} & \textbf{Constraint OK} & \textbf{Med. MIP gap} \\ \hline \hline"
    )
    for _, row in d.iterrows():
        tex.append(
            " & ".join(
                [
                    _mode_display(str(row["mode"])),
                    _fmt_num(row["runtime_median"], "{:.2f}"),
                    (_fmt_num(row["speedup_vs_lazy"], "{:.2f}") + "x") if np.isfinite(row["speedup_vs_lazy"]) else "--",
                    _fmt_pct(row["success_rate"]),
                    _fmt_pct(row["feasible_rate"]),
                    _fmt_num(row["mip_gap_median"], "{:.1e}"),
                ]
            )
            + r" \\ \hline"
        )
    tex.append(r"\end{tabularx}")
    out = PAPER_TABLE_DIR / "large_lazy_comparison.tex"
    out.write_text("\n".join(tex), encoding="utf-8")
    return out.name


def main() -> None:
    _ensure()
    if not SUMMARY.is_file() or not MERGED.is_file() or not MODE_SPEED.is_file():
        raise FileNotFoundError("Run analysis.py first.")

    summary = pd.read_csv(SUMMARY)
    merged = _normalize_runtime_frame(pd.read_csv(MERGED))
    mode_speed = pd.read_csv(MODE_SPEED)
    fastest = _read_csv_optional(Path("results") / "mode_fastest_share.csv")
    large_lazy = _read_csv_optional(LARGE_LAZY)
    for df in (summary, merged, mode_speed, fastest):
        if "mode" in df.columns:
            df["mode"] = df["mode"].map(canonical_mode)

    generated = set()
    for out in (
        optimality_by_case(merged, mode_speed),
        speedup_by_case(summary, mode_speed),
        metric_by_case(summary, mode_speed, value_col="runtime_median", out_name="runtime_by_case.tex", fmt="{:.2f}", best_low=True),
        metric_by_case(summary, mode_speed, value_col="mem_gb_median", out_name="memory_by_case.tex", fmt="{:.2f}", best_low=True),
        quality_by_mode(summary, mode_speed),
        constraints_by_mode(merged, mode_speed),
        fastest_share_by_case(fastest, mode_speed),
        large_lazy_comparison_table(large_lazy),
    ):
        if out:
            generated.add(out)

    _cleanup_table_dir(generated)
    print(f"Tables written to {PAPER_TABLE_DIR}")
    print("Generated:", ", ".join(sorted(generated)) if generated else "<none>")


if __name__ == "__main__":
    main()
