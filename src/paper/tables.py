"""
Minimal, publication-focused table generation.

Generates only relevant tables:
  - results/tables/optimality_by_case.tex
  - results/tables/speedup_by_case.tex
  - results/tables/runtime_selected.tex
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

SUMMARY = Path("results") / "summary.csv"
MERGED = Path("results") / "merged_results.csv"
OUT_DIR = Path("results") / "tables"
RAW_BASELINE_MODE = "RAW"
HEURISTIC_MODES = {"SHRINK+LAZY"}

# Keep only publication-relevant methods for clear tables.
FOCUS_MODES = [
    "RAW",
    "WARM",
    "WARM+LAZY",
    "WARM+PRUNE-0.10",
    "WARM+LPSCREEN-0.10",
    "WARM+SR+LAZY",
    "ACTIVESET",
    "ACTIVESET+LAZY",
    "STREDUCE",
    "STREDUCE+LAZY",
    "SHRINK+LAZY",
]


def _ensure() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _cleanup_table_dir(keep_tex: set[str]) -> None:
    for p in OUT_DIR.iterdir():
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


def _mode_exactness(mode: str) -> str:
    return "heuristic" if str(mode).upper() in HEURISTIC_MODES else "exact"


def _mode_display(mode: str) -> str:
    m = str(mode)
    return rf"{m}$^\dagger$" if _mode_exactness(m) == "heuristic" else m


def _present_focus_modes(values: pd.Series) -> list[str]:
    present = set(values.dropna().astype(str))
    return [m for m in FOCUS_MODES if m in present]


def _case_label(case_folder: str) -> str:
    return str(case_folder).split("/")[-1]


def _tabularx_spec(num_case_cols: int) -> str:
    return "|l||" + "|".join(["X"] * num_case_cols) + "|"


def _fmt_ci(m, lo, hi, fmt="{:.2f}") -> str:
    if np.isnan(m) or np.isnan(lo) or np.isnan(hi):
        return "--"
    return f"{fmt.format(m)} ({fmt.format(lo)}--{fmt.format(hi)})"


def optimality_by_case(merged: pd.DataFrame) -> str | None:
    cases = sorted(merged["case_folder"].dropna().astype(str).unique())
    modes = _present_focus_modes(merged["mode"])
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
            other = max(total - opt - infeas, 0)
            row.append(
                f"{100.0 * opt / total:.0f}/{100.0 * infeas / total:.0f}/{100.0 * other / total:.0f}"
            )
        tex.append(" & ".join(row) + r" \\ \hline")

    if any(_mode_exactness(m) == "heuristic" for m in modes):
        tex.append(
            rf"\multicolumn{{{1 + len(cases)}}}{{l}}{{\footnotesize $^\dagger$ Heuristic mode (not exact MILP-equivalent).}} \\ \hline"
        )
    tex.append(r"\end{tabularx}")

    out = OUT_DIR / "optimality_by_case.tex"
    out.write_text("\n".join(tex), encoding="utf-8")
    return out.name


def speedup_by_case(summary: pd.DataFrame) -> str | None:
    cases = sorted(summary["case_folder"].dropna().astype(str).unique())
    modes = _present_focus_modes(summary["mode"])
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
            s = summary[(summary["case_folder"] == case) & (summary["mode"] == mode)]
            if s.empty:
                row.append("--")
                continue
            val = pd.to_numeric(s["speedup_vs_raw"], errors="coerce").iloc[0]
            row.append(f"{val:.2f}x" if np.isfinite(val) else "--")
        tex.append(" & ".join(row) + r" \\ \hline")

    if any(_mode_exactness(m) == "heuristic" for m in modes):
        tex.append(
            rf"\multicolumn{{{1 + len(cases)}}}{{l}}{{\footnotesize $^\dagger$ Heuristic mode (not exact MILP-equivalent).}} \\ \hline"
        )
    tex.append(r"\end{tabularx}")

    out = OUT_DIR / "speedup_by_case.tex"
    out.write_text("\n".join(tex), encoding="utf-8")
    return out.name


def runtime_selected(summary: pd.DataFrame) -> str | None:
    modes = _present_focus_modes(summary["mode"])
    if not modes:
        return None
    cases = sorted(summary["case_folder"].dropna().astype(str).unique())
    if not cases:
        return None

    tex = []
    tex.append(
        rf"\begin{{tabularx}}{{0.98\textwidth}}{{{_tabularx_spec(len(modes))}}}\hline"
    )
    header = " & ".join([r"\textbf{Case}"] + [rf"\textbf{{{_mode_display(m)}}}" for m in modes])
    tex.append(header + r" \\ \hline \hline")

    for case in cases:
        row = [_case_label(case)]
        for mode in modes:
            s = summary[(summary["case_folder"] == case) & (summary["mode"] == mode)]
            if s.empty:
                row.append("--")
                continue
            row.append(
                _fmt_ci(
                    pd.to_numeric(s["runtime_median"], errors="coerce").iloc[0],
                    pd.to_numeric(s["runtime_CI95_lo"], errors="coerce").iloc[0],
                    pd.to_numeric(s["runtime_CI95_hi"], errors="coerce").iloc[0],
                    "{:.2f}",
                )
            )
        tex.append(" & ".join(row) + r" \\ \hline")

    if any(_mode_exactness(m) == "heuristic" for m in modes):
        tex.append(
            rf"\multicolumn{{{1 + len(modes)}}}{{l}}{{\footnotesize $^\dagger$ Heuristic mode (not exact MILP-equivalent).}} \\ \hline"
        )
    tex.append(r"\end{tabularx}")

    out = OUT_DIR / "runtime_selected.tex"
    out.write_text("\n".join(tex), encoding="utf-8")
    return out.name


def main() -> None:
    _ensure()
    if not SUMMARY.is_file() or not MERGED.is_file():
        raise FileNotFoundError("Run analysis.py first.")

    summary = pd.read_csv(SUMMARY)
    merged = _normalize_runtime_frame(pd.read_csv(MERGED))

    generated = set()
    for fn in (optimality_by_case, speedup_by_case, runtime_selected):
        out = fn(merged if fn is optimality_by_case else summary)
        if out:
            generated.add(out)

    _cleanup_table_dir(generated)
    print(f"Tables written to {OUT_DIR}")
    print("Generated:", ", ".join(sorted(generated)) if generated else "<none>")


if __name__ == "__main__":
    main()
