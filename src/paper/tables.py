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
import numpy as np
import pandas as pd

SUMMARY = Path("results") / "summary.csv"
MERGED = Path("results") / "merged_results.csv"
OUT_DIR = Path("results") / "tables"


def _ensure():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _fmt_ci(m, lo, hi, fmt="{:.0f}"):
    if np.isnan(m) or np.isnan(lo) or np.isnan(hi):
        return "--"
    return f"{fmt.format(m)} ({fmt.format(lo)}–{fmt.format(hi)})"


def topline(summary: pd.DataFrame):
    # RAW vs WARM+LAZY per case — show medians and CI for runtime and obj ppm
    rows = []
    for case in sorted(summary["case_folder"].unique()):
        s_raw = summary[
            (summary["case_folder"] == case)
            & (summary["mode"].str.upper().str.startswith("RAW"))
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
    ]
    if g.empty:
        return

    # parse tau
    def parse_tau(m):
        try:
            if "-" in m:
                return float(m.split("-")[-1])
        except Exception:
            return np.nan
        return np.nan

    g["tau"] = g["mode"].apply(parse_tau)
    # ratio + time gain vs RAW
    raw_med = merged[
        (merged["case_folder"] == case_focus)
        & (merged["mode"].str.upper().str.startswith("RAW"))
    ]["runtime_sec"].median()
    rows = []
    for tau, group in g.groupby("tau"):
        ratio = pd.to_numeric(group["constr_ratio_cont"], errors="coerce")
        if ratio.isna().all():
            # fallback: root constraints vs RAW
            ratio = group["num_constrs_root"] / float(
                merged[
                    (merged["case_folder"] == case_focus)
                    & (merged["mode"].str.upper().str.startswith("RAW"))
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
        tex.append(f"{tau_str} & {r_str} & {gain_str} \\\\")
    tex.append(r"\hline\end{tabular}")
    (OUT_DIR / "screening_case118.tex").write_text("\n".join(tex), encoding="utf-8")


def memory_table(summary: pd.DataFrame):
    rows = []
    for case, g in summary.groupby("case_folder"):
        for mode in ["RAW", "WARM+LAZY", "WARM+PRUNE-0.50"]:
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
    merged = pd.read_csv(MERGED)
    topline(summary)
    ablation_case(summary, merged, "matpower/case118")
    screening_table(merged, "matpower/case118")
    memory_table(summary)
    wilcoxon_table(summary)
    print(f"Tables written to {OUT_DIR}")


if __name__ == "__main__":
    main()
