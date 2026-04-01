"""
Generate LaTeX-ready tables from aggregated summary and per-instance results.

Produces:
- results/tables/topline.tex         Top-line performance (RAW vs WARM+LAZY, with CI)
- results/tables/ablation_case118.tex
- results/tables/screening_case118.tex
- results/tables/memory.tex
- results/tables/wilcoxon.tex
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

SUMMARY = Path("results") / "summary.csv"
MERGED = Path("results") / "merged_results.csv"
OUT_DIR = Path("results") / "tables"
RAW_BASELINE_MODE = "RAW"
MODE_ORDER = [
    "RAW",
    "RAW+COMMIT",
    "RAW+GNN",
    "WARM",
    "WARM+HINTS",
    "WARM+HINTS+COMMIT",
    "WARM+HINTS+GRU",
    "WARM+HINTS+COMMIT+GRU",
    "WARM+LAZY",
    "WARM+LAZY+BANDIT",
    "WARM+LAZY+COMMIT",
    "WARM+LAZY+GRU",
    "WARM+SR+LAZY",
    "ACTIVESET",
    "ACTIVESET+LAZY",
    "SHRINK+LAZY",
    "STREDUCE",
    "STREDUCE+LAZY",
    "WARM+PRUNE-0.10",
    "WARM+PRUNE-0.10+GNN",
    "WARM+PRUNE-0.10+LAZY",
    "WARM+PRUNE-0.20",
    "WARM+PRUNE-0.20+GNN",
    "WARM+PRUNE-0.20+LAZY",
    "WARM+PRUNE-0.30",
    "WARM+PRUNE-0.30+GNN",
    "WARM+PRUNE-0.30+LAZY",
    "WARM+PRUNE-0.50",
    "WARM+PRUNE-0.50+GNN",
    "WARM+PRUNE-0.50+LAZY",
    "WARM+PRUNE-0.80",
    "WARM+PRUNE-0.80+GNN",
    "WARM+PRUNE-0.80+LAZY",
    "WARM+LPSCREEN-0.10",
    "WARM+LPSCREEN-0.10+LAZY",
    "WARM+LPSCREEN-0.20",
    "WARM+LPSCREEN-0.20+LAZY",
    "WARM+LPSCREEN-0.30",
    "WARM+LPSCREEN-0.30+LAZY",
    "WARM+LPSCREEN-0.50",
    "WARM+LPSCREEN-0.50+LAZY",
    "WARM+LPSCREEN-0.80",
    "WARM+LPSCREEN-0.80+LAZY",
]


def _ensure():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def _ordered_modes(values) -> list[str]:
    order = {mode: idx for idx, mode in enumerate(MODE_ORDER)}
    unique = []
    seen = set()
    for value in values:
        mode = str(value)
        if mode in seen:
            continue
        seen.add(mode)
        unique.append(mode)
    return sorted(unique, key=lambda mode: (order.get(mode, len(order)), mode))


def _case_label(case_folder: str) -> str:
    return str(case_folder).split("/")[-1]


def _tabularx_spec(num_case_cols: int) -> str:
    return "|l||" + "|".join(["X"] * num_case_cols) + "|"


def _parse_tau(mode: str):
    try:
        mt = re.search(r"(?:PRUNE|LPSCREEN)-([0-9]+(?:\.[0-9]+)?)", str(mode).upper())
        if mt:
            return float(mt.group(1))
    except Exception:
        return np.nan
    return np.nan


def _fmt_ci(m, lo, hi, fmt="{:.0f}"):
    if np.isnan(m) or np.isnan(lo) or np.isnan(hi):
        return "--"
    return f"{fmt.format(m)} ({fmt.format(lo)}–{fmt.format(hi)})"


def _fmt_ci(m, lo, hi, fmt="{:.0f}"):
    if np.isnan(m) or np.isnan(lo) or np.isnan(hi):
        return "--"
    return f"{fmt.format(m)} ({fmt.format(lo)}--{fmt.format(hi)})"


def optimality_by_case(merged: pd.DataFrame):
    cases = sorted(merged["case_folder"].dropna().astype(str).unique())
    modes = _ordered_modes(merged["mode"].dropna().astype(str).unique())
    tex = []
    tex.append(
        rf"\begin{{tabularx}}{{0.98\textwidth}}{{{_tabularx_spec(len(cases))}}}\hline"
    )
    header = " & ".join(
        [r"\textbf{Mode}"] + [rf"\textbf{{{_case_label(case)}}}" for case in cases]
    )
    tex.append(header + r" \\ \hline \hline")
    for mode in modes:
        cells = [mode]
        for case in cases:
            group = merged[(merged["case_folder"] == case) & (merged["mode"] == mode)]
            if group.empty:
                cells.append("--")
                continue
            status = group["status"].astype(str).str.upper()
            total = len(status)
            optimal = int((status == "OPTIMAL").sum())
            infeasible = int(
                status.isin(
                    ["INFEASIBLE", "INF_OR_UNBD", "INFEASIBLE_OR_UNBOUNDED"]
                ).sum()
            )
            other = max(total - optimal - infeasible, 0)
            cells.append(
                f"{100.0 * optimal / total:.0f}/{100.0 * infeasible / total:.0f}/{100.0 * other / total:.0f}"
            )
        tex.append(" & ".join(cells) + r" \\ \hline")
    tex.append(r"\end{tabularx}")
    (OUT_DIR / "optimality_by_case.tex").write_text("\n".join(tex), encoding="utf-8")


def speedup_by_case(summary: pd.DataFrame):
    cases = sorted(summary["case_folder"].dropna().astype(str).unique())
    modes = _ordered_modes(summary["mode"].dropna().astype(str).unique())
    tex = []
    tex.append(
        rf"\begin{{tabularx}}{{0.98\textwidth}}{{{_tabularx_spec(len(cases))}}}\hline"
    )
    header = " & ".join(
        [r"\textbf{Mode}"] + [rf"\textbf{{{_case_label(case)}}}" for case in cases]
    )
    tex.append(header + r" \\ \hline \hline")
    for mode in modes:
        cells = [mode]
        for case in cases:
            row = summary[(summary["case_folder"] == case) & (summary["mode"] == mode)]
            if row.empty:
                cells.append("--")
                continue
            val = pd.to_numeric(row["speedup_vs_raw"], errors="coerce").iloc[0]
            cells.append(f"{val:.2f}x" if np.isfinite(val) else "--")
        tex.append(" & ".join(cells) + r" \\ \hline")
    tex.append(r"\end{tabularx}")
    (OUT_DIR / "speedup_by_case.tex").write_text("\n".join(tex), encoding="utf-8")


def runtime_selected(summary: pd.DataFrame):
    selected_modes = [
        mode
        for mode in [
            "RAW",
            "WARM+LAZY",
            "ACTIVESET",
            "ACTIVESET+LAZY",
            "SHRINK+LAZY",
            "STREDUCE",
            "STREDUCE+LAZY",
            "WARM+PRUNE-0.10+LAZY",
            "WARM+LPSCREEN-0.10+LAZY",
            "WARM+SR+LAZY",
            "WARM+LAZY+BANDIT",
            "WARM+PRUNE-0.10",
        ]
        if mode in set(summary["mode"].astype(str))
    ]
    if not selected_modes:
        return
    tex = []
    tex.append(
        rf"\begin{{tabularx}}{{0.98\textwidth}}{{{_tabularx_spec(len(selected_modes))}}}\hline"
    )
    header = " & ".join(
        [r"\textbf{Case}"] + [rf"\textbf{{{mode}}}" for mode in selected_modes]
    )
    tex.append(header + r" \\ \hline \hline")
    for case in sorted(summary["case_folder"].dropna().astype(str).unique()):
        cells = [_case_label(case)]
        for mode in selected_modes:
            row = summary[(summary["case_folder"] == case) & (summary["mode"] == mode)]
            if row.empty:
                cells.append("--")
                continue
            cells.append(
                _fmt_ci(
                    pd.to_numeric(row["runtime_median"], errors="coerce").iloc[0],
                    pd.to_numeric(row["runtime_CI95_lo"], errors="coerce").iloc[0],
                    pd.to_numeric(row["runtime_CI95_hi"], errors="coerce").iloc[0],
                    "{:.2f}",
                )
            )
        tex.append(" & ".join(cells) + r" \\ \hline")
    tex.append(r"\end{tabularx}")
    (OUT_DIR / "runtime_selected.tex").write_text("\n".join(tex), encoding="utf-8")


def topline(summary: pd.DataFrame):
    # RAW vs WARM+LAZY per case — show medians and CI for runtime and obj ppm
    rows = []
    for case in sorted(summary["case_folder"].unique()):
        s_raw = summary[
            (summary["case_folder"] == case)
            & (summary["mode"].str.upper() == RAW_BASELINE_MODE)
        ]
        s_wlz = summary[
            (summary["case_folder"] == case) & (summary["mode"] == "WARM+LAZY")
        ]
        if s_raw.empty or s_wlz.empty:
            continue
        raw_rt = _fmt_ci(
            s_raw["runtime_median"].values[0],
            s_raw["runtime_CI95_lo"].values[0],
            s_raw["runtime_CI95_hi"].values[0],
            "{:.0f}",
        )
        wlz_rt = _fmt_ci(
            s_wlz["runtime_median"].values[0],
            s_wlz["runtime_CI95_lo"].values[0],
            s_wlz["runtime_CI95_hi"].values[0],
            "{:.0f}",
        )
        su = s_wlz["speedup_vs_raw"].values[0]
        su_str = f"{su:.2f}×" if np.isfinite(su) else "--"
        su_str = f"{su:.2f}x" if np.isfinite(su) else "--"
        ppm = s_wlz["obj_ppm_median"].values[0]
        ppm_str = f"{ppm:.1f}" if np.isfinite(ppm) else "--"
        rows.append((case.split("/")[-1], raw_rt, wlz_rt, su_str, ppm_str))
    tex = []
    tex.append(r"\begin{tabular}{|l|r|r|r|r|}\hline")
    tex.append(
        r"\textbf{Case} & \textbf{RAW [s]} & \textbf{WARM+LAZY [s]} & \textbf{Speed-up} & \textbf{Obj $\Delta$ [ppm]}\\ \hline"
    )
    for r in rows:
        tex.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} \\\\")
    tex.append(r"\hline\end{tabular}")
    (OUT_DIR / "topline.tex").write_text("\n".join(tex), encoding="utf-8")


def ablation_case(
    summary: pd.DataFrame, merged: pd.DataFrame, case_focus: str = "matpower/case118"
):
    # Show medians for RAW, WARM, WARM+HINTS, WARM+LAZY
    order = ["RAW", "WARM", "WARM+HINTS", "WARM+LAZY"]
    rows = []
    for m in order:
        s = summary[(summary["case_folder"] == case_focus) & (summary["mode"] == m)]
        if s.empty:
            continue
        rows.append(
            (
                m,
                f"{s['runtime_median'].values[0]:.0f}",
                f"{s['nodes_median'].values[0]:.0f}",
            )
        )
    tex = []
    tex.append(r"\begin{tabular}{|l|r|r|}\hline")
    tex.append(r"\textbf{Method} & \textbf{Runtime [s]} & \textbf{Nodes [k]}\\ \hline")
    for r in rows:
        tex.append(f"{r[0]} & {r[1]} & {r[2]} \\\\")
    tex.append(r"\hline\end{tabular}")
    (OUT_DIR / "ablation_case118.tex").write_text("\n".join(tex), encoding="utf-8")


def screening_table(merged: pd.DataFrame, case_focus: str = "matpower/case118"):
    g = merged[
        (merged["case_folder"] == case_focus)
        & (merged["mode"].str.startswith("WARM+PRUNE"))
        & (~merged["mode"].str.contains("LAZY", case=False, na=False))
    ].copy()
    if g.empty:
        return

    # parse tau
    g["tau"] = g["mode"].apply(_parse_tau)
    # ratio + time gain vs RAW
    raw_med = merged[
        (merged["case_folder"] == case_focus)
        & (merged["mode"].str.upper() == RAW_BASELINE_MODE)
    ]["runtime_sec"].median()
    rows = []
    for tau, group in g.groupby("tau"):
        ratio = pd.to_numeric(group["constr_ratio_cont"], errors="coerce")
        if ratio.isna().all():
            # fallback: root constraints vs RAW
            ratio = group["num_constrs_root"] / float(
                merged[
                    (merged["case_folder"] == case_focus)
                    & (merged["mode"].str.upper() == RAW_BASELINE_MODE)
                ]["num_constrs_root"].median()
            )
        time_med = group["runtime_sec"].median()
        gain = raw_med / time_med if time_med and time_med > 0 else np.nan
        rows.append((tau, float(np.nanmedian(ratio)), gain))
    rows = sorted(rows, key=lambda x: x[0] if x[0] is not None else 0.0)
    tex = []
    tex.append(r"\begin{tabular}{|l|c|c|}\hline")
    tex.append(r"$\tau$ & \textbf{Constr. Ratio} & \textbf{Time Gain}\\ \hline")
    for tau, r, gain in rows:
        tau_str = f"{tau:.2f}" if np.isfinite(tau) else "--"
        r_str = f"{r:.2f}" if np.isfinite(r) else "--"
        gain_str = f"{gain:.2f}×" if np.isfinite(gain) else "--"
        gain_str = f"{gain:.2f}x" if np.isfinite(gain) else "--"
        tex.append(f"{tau_str} & {r_str} & {gain_str} \\\\")
    tex.append(r"\hline\end{tabular}")
    (OUT_DIR / "screening_case118.tex").write_text("\n".join(tex), encoding="utf-8")


def memory_table(summary: pd.DataFrame):
    rows = []
    for case, g in summary.groupby("case_folder"):
        for mode in [
            "RAW",
            "WARM+LAZY",
            "ACTIVESET",
            "ACTIVESET+LAZY",
            "SHRINK+LAZY",
            "STREDUCE",
            "STREDUCE+LAZY",
            "WARM+PRUNE-0.50",
            "WARM+PRUNE-0.50+LAZY",
            "WARM+LPSCREEN-0.10+LAZY",
            "WARM+SR+LAZY",
        ]:
            gg = g[g["mode"] == mode]
            if gg.empty:
                continue
            med = gg["mem_gb_median"].values[0]
            iqr = gg["mem_gb_IQR"].values[0]
            rows.append((case.split("/")[-1], mode, f"{med:.2f} [{iqr:.2f}]"))
    tex = []
    tex.append(r"\begin{tabular}{|l|l|c|}\hline")
    tex.append(
        r"\textbf{Case} & \textbf{Method} & \textbf{Peak mem (GB) [IQR]}\\ \hline"
    )
    for r in rows:
        tex.append(f"{r[0]} & {r[1]} & {r[2]} \\\\")
    tex.append(r"\hline\end{tabular}")
    (OUT_DIR / "memory.tex").write_text("\n".join(tex), encoding="utf-8")


def wilcoxon_table(summary: pd.DataFrame):
    # Extract precomputed Wilcoxon p-values from summary.csv if present
    if not {"p_wilcoxon_runtime", "p_wilcoxon_nodes"}.issubset(set(summary.columns)):
        return
    # Keep unique per case
    g = summary.groupby("case_folder").first().reset_index()
    tex = []
    tex.append(r"\begin{tabular}{|l|c|c|}\hline")
    tex.append(
        r"\textbf{Case} & \textbf{$p$ (runtime)} & \textbf{$p$ (nodes)}\\ \hline"
    )
    for _, r in g.iterrows():
        tex.append(
            f"{r['case_folder'].split('/')[-1]} & {r['p_wilcoxon_runtime']:.3g} & {r['p_wilcoxon_nodes']:.3g} \\\\"
        )
    tex.append(r"\hline\end{tabular}")
    (OUT_DIR / "wilcoxon.tex").write_text("\n".join(tex), encoding="utf-8")


def main():
    _ensure()
    if not SUMMARY.is_file() or not MERGED.is_file():
        raise FileNotFoundError("Run analysis.py first.")
    summary = pd.read_csv(SUMMARY)
    merged = _normalize_runtime_frame(pd.read_csv(MERGED))
    optimality_by_case(merged)
    speedup_by_case(summary)
    runtime_selected(summary)
    topline(summary)
    ablation_case(summary, merged, "matpower/case118")
    screening_table(merged, "matpower/case118")
    memory_table(summary)
    wilcoxon_table(summary)
    print(f"Tables written to {OUT_DIR}")


if __name__ == "__main__":
    main()
