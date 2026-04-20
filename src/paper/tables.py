"""
All-mode, publication-friendly table generation.

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
MODE_SPEED = Path("results") / "mode_speed_stats.csv"
OUT_DIR = Path("results") / "tables"
RAW_BASELINE_MODE = "RAW"
HEURISTIC_MODES = {"SHRINK+LAZY"}


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


def _case_label(case_folder: str) -> str:
    return str(case_folder).split("/")[-1]


def _tabularx_spec(num_case_cols: int) -> str:
    return "|l||" + "|".join(["X"] * num_case_cols) + "|"


def _ordered_modes(
    present_modes: list[str] | set[str],
    mode_speed: pd.DataFrame | None = None,
) -> list[str]:
    present = {str(m) for m in present_modes if pd.notna(m)}
    ordered: list[str] = []
    if RAW_BASELINE_MODE in present:
        ordered.append(RAW_BASELINE_MODE)
    if mode_speed is not None and not mode_speed.empty and "mode" in mode_speed.columns:
        ms = mode_speed.copy()
        if "mean_runtime_ratio" in ms.columns:
            ms["mean_runtime_ratio"] = pd.to_numeric(
                ms["mean_runtime_ratio"], errors="coerce"
            )
            ms = ms.sort_values(["mean_runtime_ratio", "mode"], na_position="last")
        for mode in ms["mode"].astype(str):
            if mode in present and mode not in ordered:
                ordered.append(mode)
    for mode in sorted(present):
        if mode not in ordered:
            ordered.append(mode)
    return ordered


def _fmt_num(x, fmt="{:.2f}") -> str:
    if not np.isfinite(x):
        return "--"
    return fmt.format(float(x))


def optimality_by_case(merged: pd.DataFrame, mode_speed: pd.DataFrame) -> str | None:
    cases = sorted(merged["case_folder"].dropna().astype(str).unique())
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
            row.append(
                f"{100.0 * opt / total:.0f}/{100.0 * tl / total:.0f}/{100.0 * infeas / total:.0f}"
            )
        tex.append(" & ".join(row) + r" \\ \hline")

    tex.append(
        rf"\multicolumn{{{1 + len(cases)}}}{{l}}{{\footnotesize Entries are Optimal / TimeLimit-or-Suboptimal / Infeasible shares in percent.}} \\ \hline"
    )
    if any(_mode_exactness(m) == "heuristic" for m in modes):
        tex.append(
            rf"\multicolumn{{{1 + len(cases)}}}{{l}}{{\footnotesize $^\dagger$ Heuristic mode (not exact MILP-equivalent).}} \\ \hline"
        )
    tex.append(r"\end{tabularx}")

    out = OUT_DIR / "optimality_by_case.tex"
    out.write_text("\n".join(tex), encoding="utf-8")
    return out.name


def speedup_by_case(summary: pd.DataFrame, mode_speed: pd.DataFrame) -> str | None:
    cases = sorted(summary["case_folder"].dropna().astype(str).unique())
    modes = _ordered_modes(summary["mode"].dropna().astype(str).unique(), mode_speed=mode_speed)
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


def runtime_selected(mode_speed: pd.DataFrame) -> str | None:
    req = {"mode", "mean_runtime_ratio", "std_runtime_ratio", "N"}
    if not req.issubset(set(mode_speed.columns)):
        return None

    d = mode_speed.copy()
    d["mean_runtime_ratio"] = pd.to_numeric(d["mean_runtime_ratio"], errors="coerce")
    d["std_runtime_ratio"] = pd.to_numeric(d["std_runtime_ratio"], errors="coerce")
    d["success_rate_strict"] = pd.to_numeric(
        d.get("success_rate_strict", np.nan), errors="coerce"
    )
    d["method_exactness"] = d.get("method_exactness", "exact").fillna("exact")
    if d.empty:
        return None

    d = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "mode": RAW_BASELINE_MODE,
                        "mean_runtime_ratio": 1.0,
                        "std_runtime_ratio": 0.0,
                        "N": int(pd.to_numeric(d["N"], errors="coerce").max())
                        if "N" in d.columns and not d.empty
                        else 0,
                        "success_rate_strict": 1.0,
                        "method_exactness": "exact",
                    }
                ]
            ),
            d,
        ],
        ignore_index=True,
    )
    d = d.drop_duplicates(subset=["mode"], keep="first")
    d = d.sort_values(["mean_runtime_ratio", "mode"], na_position="last").reset_index(drop=True)

    tex = []
    tex.append(r"\begin{tabularx}{0.98\textwidth}{|l||X|X|X|X|}\hline")
    tex.append(
        r"\textbf{Mode} & \textbf{Type} & \textbf{Mean Runtime Ratio vs RAW} & \textbf{Std} & \textbf{Strict Success} \\ \hline \hline"
    )
    for _, row in d.iterrows():
        tex.append(
            " & ".join(
                [
                    _mode_display(str(row["mode"])),
                    str(row.get("method_exactness", "exact")),
                    _fmt_num(float(row["mean_runtime_ratio"]), "{:.3f}"),
                    _fmt_num(float(row["std_runtime_ratio"]), "{:.3f}"),
                    _fmt_num(100.0 * float(row.get("success_rate_strict", np.nan)), "{:.0f}") + r"\%",
                ]
            )
            + r" \\ \hline"
        )
    if any(_mode_exactness(m) == "heuristic" for m in d["mode"].astype(str)):
        tex.append(
            r"\multicolumn{5}{l}{\footnotesize $^\dagger$ Heuristic mode (not exact MILP-equivalent).} \\ \hline"
        )
    tex.append(r"\end{tabularx}")

    out = OUT_DIR / "runtime_selected.tex"
    out.write_text("\n".join(tex), encoding="utf-8")
    return out.name


def main() -> None:
    _ensure()
    if not SUMMARY.is_file() or not MERGED.is_file() or not MODE_SPEED.is_file():
        raise FileNotFoundError("Run analysis.py first.")

    summary = pd.read_csv(SUMMARY)
    merged = _normalize_runtime_frame(pd.read_csv(MERGED))
    mode_speed = pd.read_csv(MODE_SPEED)

    generated = set()
    for out in (
        optimality_by_case(merged, mode_speed),
        speedup_by_case(summary, mode_speed),
        runtime_selected(mode_speed),
    ):
        if out:
            generated.add(out)

    _cleanup_table_dir(generated)
    print(f"Tables written to {OUT_DIR}")
    print("Generated:", ", ".join(sorted(generated)) if generated else "<none>")


if __name__ == "__main__":
    main()
