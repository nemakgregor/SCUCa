"""Generate case1354pegase results table (CSV + LaTeX) and bar-chart figure."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
FIGURES = HERE.parent / "figures"
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

MODE_ORDER = [
    "WARM",
    "WARM+LAZY",
    "WARM+LAZY+K64",
    "WARM+LAZY+K128",
    "WARM+LAZY+BANDIT",
    "WARM+LAZY+COMMIT",
    "WARM+LAZY+GRU",
]

MODE_LABELS = {
    "WARM":              r"\texttt{WARM}",
    "WARM+LAZY":         r"\texttt{LAZY-All} ($K{=}0$)",
    "WARM+LAZY+K64":     r"\texttt{LAZY-K64}",
    "WARM+LAZY+K128":    r"\texttt{LAZY-K128}",
    "WARM+LAZY+BANDIT":  r"\texttt{LAZY-BANDIT}",
    "WARM+LAZY+COMMIT":  r"\texttt{LAZY+COMMIT}",
    "WARM+LAZY+GRU":     r"\texttt{LAZY+GRU}",
}

MODE_LABELS_SHORT = {
    "WARM":              "WARM",
    "WARM+LAZY":         "LAZY-All\n(K=0)",
    "WARM+LAZY+K64":     "LAZY\nK64",
    "WARM+LAZY+K128":    "LAZY\nK128",
    "WARM+LAZY+BANDIT":  "LAZY\nBANDIT",
    "WARM+LAZY+COMMIT":  "LAZY+\nCOMMIT",
    "WARM+LAZY+GRU":     "LAZY+\nGRU",
}

# Violation-status colours
COL_OK   = "#54a24b"   # green  — N-1 feasible
COL_VIOL = "#e45756"   # red    — violations
COL_NONE = "#bab0ac"   # gray   — no feasible solution found


def _load_results() -> pd.DataFrame:
    files = sorted(glob.glob(
        str(HERE.parent.parent / "results" / "raw_logs" / "exp_matpower_case1354pegase_*.csv")
    ))
    if not files:
        raise FileNotFoundError("No case1354pegase result CSVs found in results/raw_logs/")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # keep latest run per mode
    df = df.sort_values("timestamp").drop_duplicates(["mode"], keep="last")
    df = df[df["mode"].isin(MODE_ORDER)].copy()
    df["mode"] = pd.Categorical(df["mode"], categories=MODE_ORDER, ordered=True)
    return df.sort_values("mode").reset_index(drop=True)


def _viol_status(row) -> str:
    v = str(row.get("violations", "")).strip()
    if v == "OK":
        return "OK"
    if v and v != "nan":
        return "VIOL"
    return "NONE"


def gen_csv(df: pd.DataFrame) -> Path:
    out = df[["mode", "status", "runtime_sec", "mip_gap", "obj_val", "violations"]].copy()
    p = TABLES / "case1354_results.csv"
    out.to_csv(p, index=False)
    print(f"[case1354] wrote {p}")
    return p


def gen_latex(df: pd.DataFrame) -> Path:
    rows = []
    for _, r in df.iterrows():
        mode_tex = MODE_LABELS.get(r["mode"], r["mode"])
        status = str(r["status"])
        status_tex = r"TL" if "TIME" in status.upper() else r"OPT"
        runtime = f"{r['runtime_sec']:.0f}"
        try:
            gap_val = float(r["mip_gap"])
            gap_tex = f"{gap_val*100:.1f}\\%" if np.isfinite(gap_val) else "---"
        except Exception:
            gap_tex = "---"
        try:
            obj = float(r["obj_val"])
            obj_tex = f"${obj/1e6:.2f}$M" if np.isfinite(obj) else "---"
        except Exception:
            obj_tex = "---"
        vstat = _viol_status(r)
        if vstat == "OK":
            viol_tex = r"\checkmark"
        elif vstat == "VIOL":
            viol_tex = r"\texttimes"
        else:
            viol_tex = "---"
        rows.append(
            f"        {mode_tex} & {status_tex} & {runtime} & {gap_tex} & {obj_tex} & {viol_tex} \\\\"
        )

    body = "\n".join(rows)
    tex = rf"""\begin{{table}}[t]\centering
\caption{{Results on \texttt{{case1354pegase}} (1354 buses, 1991 lines, 1288 contingencies,
  36 time steps, 600\,s limit).
  Status: OPT\,=\,optimal within gap; TL\,=\,time limit.
  N-1 feasibility: \checkmark\,=\,OK; \texttimes\,=\,violations detected; ---\,=\,no solution found.
  Objective values shown only for N-1-feasible or non-violating solutions.}}
\label{{tab:case1354}}
\small
\begin{{tabular}}{{lllrrl}}
\hline
\textbf{{Mode}} & \textbf{{Status}} & \textbf{{Time [s]}} & \textbf{{Gap}} & \textbf{{Obj.}} & \textbf{{N-1}} \\
\hline
{body}
\hline
\end{{tabular}}
\end{{table}}
"""
    p = TABLES / "case1354_results.tex"
    p.write_text(tex, encoding="utf-8")
    print(f"[case1354] wrote {p}")
    return p


def gen_figure(df: pd.DataFrame) -> Path:
    modes = list(df["mode"])
    n = len(modes)
    xs = np.arange(n)

    runtimes = []
    colors = []
    hatches = []
    for _, r in df.iterrows():
        try:
            rt = float(r["runtime_sec"])
        except Exception:
            rt = 0.0
        runtimes.append(rt if np.isfinite(rt) else 0.0)
        vstat = _viol_status(r)
        if vstat == "OK":
            colors.append(COL_OK)
            hatches.append("")
        elif vstat == "VIOL":
            colors.append(COL_VIOL)
            hatches.append("///")
        else:
            colors.append(COL_NONE)
            hatches.append("xxx")

    fig, ax = plt.subplots(figsize=(7, 3.5))

    bars = ax.bar(
        xs, runtimes, color=colors, edgecolor="white", linewidth=0.8, width=0.6
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
        bar.set_edgecolor("#555555")

    # MIP gap annotation on top of each bar
    for i, (_, r) in enumerate(df.iterrows()):
        try:
            gap = float(r["mip_gap"])
            rt = float(r["runtime_sec"])
        except Exception:
            continue
        if not np.isfinite(gap) or not np.isfinite(rt):
            continue
        ax.text(
            xs[i], rt + 8, f"{gap*100:.1f}%",
            ha="center", va="bottom", fontsize=7.5, color="#333333"
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [MODE_LABELS_SHORT.get(m, m) for m in modes],
        fontsize=8.5, ha="center"
    )
    ax.set_ylabel("Runtime [s]", fontsize=10)
    ax.set_ylim(0, 700)
    ax.axhline(600, color="#888888", linestyle="--", linewidth=0.8, label="600 s limit")
    ax.set_title(
        r"\texttt{case1354pegase} — runtime and N-1 feasibility (600\,s limit)",
        fontsize=10
    )

    legend_handles = [
        mpatches.Patch(facecolor=COL_OK,   edgecolor="#555", label="N-1 feasible (OK)"),
        mpatches.Patch(facecolor=COL_VIOL, edgecolor="#555", hatch="///", label="N-1 violations"),
        mpatches.Patch(facecolor=COL_NONE, edgecolor="#555", hatch="xxx", label="No solution found"),
        plt.Line2D([0], [0], color="#888888", linestyle="--", linewidth=0.8, label="600 s limit"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    p = FIGURES / "case1354_runtime.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[case1354] wrote {p}")
    return p


if __name__ == "__main__":
    df = _load_results()
    gen_csv(df)
    gen_latex(df)
    gen_figure(df)
    print("[case1354] done.")
