"""Post-processing of already-computed revision results.

Reads pre-computed artifacts from `paper/data/` and `paper/tables/` and
produces additional summaries requested by reviewers (CI-enriched speedup
table, GNN vs 1-NN ablation, auxiliary runtime table, Gurobi tuning and
fixed-K lazy-cut ablations, quantitative literature context).

This script performs NO new solver runs. It only re-aggregates existing
CSVs into LaTeX-ready tables and a machine-readable JSON summary.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TABLES = HERE.parent / "tables"
FIGURES = HERE.parent / "figures"
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REVISION = HERE.parent.parent / "results_revision"
CASE_ORDER = ["case14", "case30", "case57", "case89pegase", "case118", "case300"]
PAPER_MODES = [
    "RAW+COMMIT",
    "RAW+GNN",
    "WARM",
    "WARM+HINTS",
    "WARM+HINTS+COMMIT",
    "WARM+HINTS+COMMIT+GRU",
    "WARM+HINTS+GRU",
    "WARM+LAZY",
    "WARM+LAZY+BANDIT",
    "WARM+LAZY+COMMIT",
    "WARM+LAZY+GRU",
    "WARM+PRUNE-0.10",
    "WARM+PRUNE-0.10+GNN",
    "WARM+PRUNE-0.20",
    "WARM+PRUNE-0.20+GNN",
    "WARM+PRUNE-0.30",
    "WARM+PRUNE-0.30+GNN",
    "WARM+PRUNE-0.50",
    "WARM+PRUNE-0.50+GNN",
    "WARM+PRUNE-0.80",
    "WARM+PRUNE-0.80+GNN",
]
ADDITIONAL_REVIEWER_MODES = [
    "WARM+LAZY+K64",
    "WARM+LAZY+K128",
    "WARM+LAZY+K256",
    "GRBTUNE",
]
COMPARISON_MODES = ["RAW"] + PAPER_MODES + ADDITIONAL_REVIEWER_MODES
MODE_FAMILY_COLORS = {
    "RAW": "#4d4d4d",
    "Warm / hints": "#4c78a8",
    "Lazy": "#f58518",
    "Prune": "#54a24b",
    "Prune + GNN": "#b279a2",
    "Solver tuning": "#9d755d",
}
KEY_MODES = [
    "RAW+COMMIT",
    "WARM",
    "WARM+HINTS+COMMIT",
    "WARM+LAZY",
    "WARM+LAZY+BANDIT",
    "WARM+PRUNE-0.10",
    "WARM+PRUNE-0.10+GNN",
    "WARM+PRUNE-0.30",
    "WARM+PRUNE-0.50",
    "WARM+PRUNE-0.80",
]


def case_short(s: pd.Series) -> pd.Series:
    return s.str.replace("matpower/", "", regex=False)


def mode_family(mode: str) -> str:
    if mode == "RAW":
        return "RAW"
    if mode == "GRBTUNE":
        return "Solver tuning"
    if mode.startswith("RAW+"):
        return "RAW"
    if "+PRUNE-" in mode and mode.endswith("+GNN"):
        return "Prune + GNN"
    if "+PRUNE-" in mode:
        return "Prune"
    if "+LAZY" in mode:
        return "Lazy"
    return "Warm / hints"


def comparison_speedup_long() -> pd.DataFrame:
    """Per-case speed-up values for the common comparison mode set.

    Original paper modes come from the main benchmark summary. Fixed-K lazy
    budgets and grbtune are reviewer-added sanity checks; they are available
    only for the cases where those additional experiments were run.
    """
    spd = pd.read_csv(DATA / "speedup_summary.csv")
    spd["case_short"] = case_short(spd["case_folder"])
    rows = spd[["case_short", "mode", "speedup_mean", "opt_rate"]].copy()

    raw_median = (
        spd[spd["mode"] == "RAW"]
        .set_index("case_folder")["runtime_median_sec"]
        .to_dict()
    )
    topk_path = TABLES / "lazy_topk_ablation.csv"
    if topk_path.exists():
        topk = pd.read_csv(topk_path)
        topk = topk[topk["mode"].isin(["WARM+LAZY+K64", "WARM+LAZY+K128", "WARM+LAZY+K256"])].copy()
        topk["case_short"] = case_short(topk["case_folder"])
        topk["speedup_mean"] = topk.apply(
            lambda r: raw_median.get(r["case_folder"], float("nan")) / r["runtime_median_sec"],
            axis=1,
        )
        topk["opt_rate"] = topk["strict_feasible_rate"]
        rows = pd.concat(
            [rows, topk[["case_short", "mode", "speedup_mean", "opt_rate"]]],
            ignore_index=True,
        )

    grb_path = TABLES / "grbtune_vs_default.csv"
    if grb_path.exists():
        grb = pd.read_csv(grb_path)
        grb["case_short"] = case_short(grb["case"])
        grb["mode"] = "GRBTUNE"
        grb["speedup_mean"] = grb["default_runtime_s"] / grb["tuned_runtime_s"]
        grb["opt_rate"] = 1.0
        rows = pd.concat(
            [rows, grb[["case_short", "mode", "speedup_mean", "opt_rate"]]],
            ignore_index=True,
        )

    rows = rows[rows["mode"].isin(COMPARISON_MODES)]
    return rows


def comparison_speedup_wide() -> pd.DataFrame:
    long = comparison_speedup_long()
    wide = long.pivot_table(
        index="mode", columns="case_short", values="speedup_mean", aggfunc="mean"
    )
    return wide.reindex(COMPARISON_MODES)[CASE_ORDER]


# ---------------------------------------------------------------------------
# 1. Mean speedup with 95% bootstrap CI for the key modes (R1-C9)
# ---------------------------------------------------------------------------
def speedup_with_ci() -> pd.DataFrame:
    df = pd.read_csv(DATA / "speedup_summary.csv")
    df = df[df["mode"] != "RAW"].copy()
    df["case_short"] = df["case_folder"].str.replace("matpower/", "", regex=False)
    df["speedup_ci"] = df.apply(
        lambda r: f"{r['speedup_mean']:.2f} [{r['speedup_mean_ci_lo']:.2f}, {r['speedup_mean_ci_hi']:.2f}]",
        axis=1,
    )
    wide = df.pivot(index="mode", columns="case_short", values="speedup_ci")

    wide = wide.reindex(KEY_MODES)
    wide = wide[CASE_ORDER]
    wide.to_csv(TABLES / "speedup_with_ci.csv")
    return wide


def render_speedup_ci_latex(wide: pd.DataFrame) -> str:
    """Produce a compact sideways LaTeX table with CI-enriched speedups."""
    cases = list(wide.columns)
    header = (
        "\\begin{sidewaystable*}\\centering\n"
        "\\caption{Speed-up factors with bootstrap 95\\% confidence intervals "
        "(mean [CI\\textsubscript{lo}, CI\\textsubscript{hi}]).}\n"
        "\\label{tab:speedup_ci}\n"
        "\\renewcommand{\\arraystretch}{1.2}\n"
        "\\footnotesize\n"
        "\\begin{tabular}{|l||" + "c|" * len(cases) + "}\\hline\n"
        "\\textbf{Mode} & "
        + " & ".join(f"\\textbf{{{c}}}" for c in cases)
        + " \\\\ \\hline \\hline\n"
    )
    rows = []
    for mode, row in wide.iterrows():
        cells = [str(v).replace("nan", "--") for v in row.values]
        rows.append(f"{mode} & " + " & ".join(cells) + " \\\\ \\hline")
    body = "\n".join(rows)
    footer = "\n\\end{tabular}\n\\end{sidewaystable*}\n"
    tex = header + body + footer
    (TABLES / "speedup_with_ci.tex").write_text(tex, encoding="utf-8")
    return tex


def main_results_summary() -> pd.DataFrame:
    long = comparison_speedup_long()
    rows = []
    interpretations = {
        "RAW+COMMIT": "Branching guidance alone",
        "RAW+GNN": "Raw model with GNN metadata",
        "WARM": "Nearest-neighbor warm start",
        "WARM+HINTS": "Warm start plus hints",
        "WARM+HINTS+COMMIT": "Warm start plus commitment guidance",
        "WARM+HINTS+COMMIT+GRU": "GRU warm-start ablation",
        "WARM+HINTS+GRU": "GRU warm-start ablation",
        "WARM+LAZY": "Exact lazy N-1 enforcement",
        "WARM+LAZY+BANDIT": "Adaptive lazy cut budget",
        "WARM+LAZY+COMMIT": "Lazy with commitment guidance",
        "WARM+LAZY+GRU": "Lazy with GRU warm start",
        "WARM+PRUNE-0.10": "Conservative screening",
        "WARM+PRUNE-0.10+GNN": "Conservative screening + GNN",
        "WARM+PRUNE-0.20": "Screening ablation",
        "WARM+PRUNE-0.20+GNN": "Screening + GNN ablation",
        "WARM+PRUNE-0.30": "Screening ablation",
        "WARM+PRUNE-0.30+GNN": "Screening + GNN ablation",
        "WARM+PRUNE-0.50": "Aggressive screening",
        "WARM+PRUNE-0.50+GNN": "Aggressive screening + GNN",
        "WARM+PRUNE-0.80": "Very aggressive screening",
        "WARM+PRUNE-0.80+GNN": "Very aggressive screening + GNN",
        "WARM+LAZY+K64": "Fixed-K lazy ablation",
        "WARM+LAZY+K128": "Fixed-K lazy ablation",
        "WARM+LAZY+K256": "Fixed-K lazy ablation",
        "GRBTUNE": "Gurobi tuning sanity check",
    }
    for mode in COMPARISON_MODES:
        sub = long[long["mode"] == mode].copy()
        if sub.empty:
            mean_speedup = float("nan")
            best_case = "--"
            best_speedup = float("nan")
            opt_mean = float("nan")
            n_cases = 0
        else:
            mean_speedup = float(sub["speedup_mean"].mean())
            best = sub.loc[sub["speedup_mean"].idxmax()]
            best_case = best["case_short"]
            best_speedup = float(best["speedup_mean"])
            opt_mean = float(sub["opt_rate"].mean())
            n_cases = int(sub["case_short"].nunique())
        rows.append(
            {
                "mode": mode,
                "mean_case_speedup": mean_speedup,
                "best_case": best_case,
                "best_case_speedup": best_speedup,
                "opt_rate_mean": opt_mean,
                "n_cases": n_cases,
                "interpretation": interpretations.get(mode, ""),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "main_results_summary.csv", index=False)
    return out


def render_main_results_summary_latex(df: pd.DataFrame) -> str:
    header = (
        "\\begin{table}[t]\\centering\n"
        "\\caption{Summary over the common comparison mode set. "
        "Mean speed-up is averaged over the cases where the mode was evaluated; "
        "fixed-$K$ and \\texttt{grbtune} rows are reviewer-added sanity checks "
        "with fewer available cases.}\n"
        "\\label{tab:main_results}\n"
        "\\small\n"
        "\\begin{tabularx}{\\linewidth}{lrrrrX}\n\\hline\n"
        "\\textbf{Mode} & \\textbf{Mean} & \\textbf{Best case} & "
        "\\textbf{Opt.} & \\textbf{Cases} & \\textbf{Interpretation} \\\\\n\\hline\n"
    )
    rows = []
    for _, r in df.iterrows():
        mean = "--" if pd.isna(r["mean_case_speedup"]) else f"{r['mean_case_speedup']:.2f}x"
        best = (
            "--"
            if pd.isna(r["best_case_speedup"])
            else f"{r['best_case']} ({r['best_case_speedup']:.2f}x)"
        )
        opt = "--" if pd.isna(r["opt_rate_mean"]) else f"{r['opt_rate_mean']*100:.0f}\\%"
        rows.append(
            f"\\texttt{{{r['mode']}}} & "
            f"{mean} & "
            f"{best} & "
            f"{opt} & "
            f"{int(r['n_cases'])} & "
            f"{r['interpretation']} \\\\"
        )
    tex = header + "\n".join(rows) + "\n\\hline\n\\end{tabularx}\n\\end{table}\n"
    (TABLES / "main_results_summary.tex").write_text(tex, encoding="utf-8")
    return tex


def plot_speedup_heatmap() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    wide = comparison_speedup_wide()
    values = wide.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.6, 9.6), constrained_layout=True)
    im = ax.imshow(values, cmap="YlGnBu", aspect="auto", vmin=0.8, vmax=26.0)
    ax.set_xticks(np.arange(len(CASE_ORDER)), labels=CASE_ORDER, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(COMPARISON_MODES)), labels=COMPARISON_MODES)
    ax.set_title("Mean speed-up by comparison mode and benchmark case")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            txt = "n/a" if not np.isfinite(val) else f"{val:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("speed-up vs RAW")
    ax.tick_params(axis="both", labelsize=7.5)
    fig.savefig(FIGURES / "speedup_heatmap.png", dpi=300)
    plt.close(fig)

def plot_ecdf() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    logs_path = REVISION / "old_logs_paired_vs_raw.csv"
    df = pd.read_csv(logs_path)
    df = df[df["mode"].isin(PAPER_MODES)].copy()
    df = df[pd.to_numeric(df["speedup_vs_raw"], errors="coerce").notna()]
    df["speedup_vs_raw"] = df["speedup_vs_raw"].astype(float)

    panels = [
        (
            "Warm starts and branching hints",
            [
                "RAW+COMMIT",
                "RAW+GNN",
                "WARM",
                "WARM+HINTS",
                "WARM+HINTS+COMMIT",
                "WARM+HINTS+COMMIT+GRU",
                "WARM+HINTS+GRU",
            ],
            (0.75, 1.9),
        ),
        (
            "Lazy enforcement",
            [
                "WARM+LAZY",
                "WARM+LAZY+BANDIT",
                "WARM+LAZY+COMMIT",
                "WARM+LAZY+GRU",
            ],
            (0.8, 19.0),
        ),
        (
            "Constraint screening",
            [
                "WARM+PRUNE-0.10",
                "WARM+PRUNE-0.10+GNN",
                "WARM+PRUNE-0.20",
                "WARM+PRUNE-0.20+GNN",
                "WARM+PRUNE-0.30",
                "WARM+PRUNE-0.30+GNN",
                "WARM+PRUNE-0.50",
                "WARM+PRUNE-0.50+GNN",
                "WARM+PRUNE-0.80",
                "WARM+PRUNE-0.80+GNN",
            ],
            (0.8, 27.0),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    for ax, (title, modes, xlim) in zip(axes, panels):
        cmap = plt.get_cmap("tab20")
        for idx, mode in enumerate(modes):
            vals = np.sort(df.loc[df["mode"] == mode, "speedup_vs_raw"].to_numpy())
            if vals.size == 0:
                continue
            y = np.arange(1, vals.size + 1) / vals.size
            linestyle = "--" if mode.endswith("+GNN") or mode.endswith("+GRU") else "-"
            ax.step(vals, y, where="post", linewidth=1.25, linestyle=linestyle, color=cmap(idx), label=mode)
        ax.axvline(1.0, color="#333333", linewidth=0.9, alpha=0.8)
        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Speed-up vs RAW")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=6.2, loc="lower right", frameon=True)
    axes[0].set_ylabel("ECDF")
    fig.savefig(FIGURES / "ecdf.png", dpi=300)
    plt.close(fig)


def plot_constraint_reduction_speedup() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import numpy as np

    logs = pd.read_csv(REVISION / "old_logs_paired_vs_raw.csv")
    logs["case_short"] = case_short(logs["case_folder"])
    modes = COMPARISON_MODES

    size = (
        logs[logs["mode"].isin(modes)]
        .groupby(["case_short", "mode"])["num_constrs_final"]
        .mean()
        .reset_index()
    )
    raw = size[size["mode"] == "RAW"][["case_short", "num_constrs_final"]].rename(
        columns={"num_constrs_final": "raw_model_size"}
    )
    size = size.merge(raw, on="case_short", how="left")
    size["model_retained_pct"] = size["num_constrs_final"] / size["raw_model_size"] * 100.0
    retained = (
        size.pivot(index="mode", columns="case_short", values="model_retained_pct")
        .reindex(modes)[CASE_ORDER]
    )
    speed = comparison_speedup_wide()
    speed = speed.reindex(modes)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 9.4), constrained_layout=True)
    for ax, wide, title, cbar_label, vmax, fmt in [
        (
            axes[0],
            retained,
            "Final model constraint count relative to RAW [%]",
            "constraints vs RAW [%]",
            105,
            "{:.0f}",
        ),
        (
            axes[1],
            speed,
            "Mean speed-up for the same modes",
            "speed-up vs RAW",
            26,
            "{:.1f}",
        ),
    ]:
        values = wide.to_numpy(dtype=float)
        im = ax.imshow(values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(len(CASE_ORDER)), labels=CASE_ORDER, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(modes)), labels=modes)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="both", labelsize=6.8)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                txt = "n/a" if not np.isfinite(values[i, j]) else fmt.format(values[i, j])
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.8)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
    fig.savefig(FIGURES / "constraint_reduction_speedup.png", dpi=300)
    plt.close(fig)


def constraint_management_summary() -> pd.DataFrame:
    speed = comparison_speedup_long()
    speed_mean = (
        speed.groupby("mode")
        .agg(
            mean_speedup=("speedup_mean", "mean"),
            n_cases=("case_short", "nunique"),
        )
        .reset_index()
    )

    old = pd.read_csv(REVISION / "old_logs_paired_vs_raw.csv")
    old["case_short"] = case_short(old["case_folder"])
    size = (
        old[old["mode"].isin(COMPARISON_MODES)]
        .groupby(["case_short", "mode"])["num_constrs_final"]
        .mean()
        .reset_index()
    )
    raw_size = size[size["mode"] == "RAW"][["case_short", "num_constrs_final"]].rename(
        columns={"num_constrs_final": "raw_constrs"}
    )
    size = size.merge(raw_size, on="case_short", how="left")
    size["final_constraints_pct"] = size["num_constrs_final"] / size["raw_constrs"] * 100.0
    constraint_mean = (
        size.groupby("mode")
        .agg(final_constraints_pct=("final_constraints_pct", "mean"))
        .reset_index()
    )

    reduction = pd.read_csv(DATA / "constraint_reduction.csv")
    reduction["case_short"] = case_short(reduction["case_folder"])
    prune = (
        reduction[reduction["mode"].isin(COMPARISON_MODES)]
        .groupby("mode")
        .agg(explicit_removed_pct=("reduction_ratio_mean", "mean"))
        .reset_index()
    )

    new = pd.read_csv(REVISION / "new_results_all.csv")
    mode_map = {
        "WARM_LAZY": "WARM+LAZY",
        "WARM_LAZY_BANDIT": "WARM+LAZY+BANDIT",
        "WARM_LAZY_COMMIT_HINTS": "WARM+LAZY+COMMIT",
        "WARM_LAZY_GRU": "WARM+LAZY+GRU",
        "WARM_LAZY_TOPK128": "WARM+LAZY+K128",
    }
    new = new[
        (new["stage"] == "TEST")
        & (new["mode_id"].isin(mode_map))
        & (new["case_folder"].isin([f"matpower/{c}" for c in CASE_ORDER]))
    ].copy()
    new["mode"] = new["mode_id"].map(mode_map)
    new["lazy_added"] = pd.to_numeric(new["lazy_added_cont"], errors="coerce").fillna(0)
    lazy = (
        new.groupby("mode")["lazy_added"]
        .mean()
        .reset_index()
        .rename(columns={"lazy_added": "lazy_added_mean"})
    )

    out = pd.DataFrame({"mode": COMPARISON_MODES})
    out = out.merge(speed_mean, on="mode", how="left")
    out = out.merge(constraint_mean, on="mode", how="left")
    out = out.merge(prune, on="mode", how="left")
    out = out.merge(lazy, on="mode", how="left")
    out.to_csv(TABLES / "constraint_management_summary.csv", index=False)
    return out


def render_constraint_management_latex(df: pd.DataFrame) -> str:
    header = (
        "\\begin{table}[t]\\centering\n"
        "\\caption{Constraint management over the common comparison mode set. "
        "Final constraints are reported relative to RAW; explicit removal applies "
        "to PRUNE modes; lazy-added constraints are callback additions observed in "
        "the reviewer ablations where available.}\n"
        "\\label{tab:constraint_management}\n"
        "\\footnotesize\n"
        "\\begin{tabular}{lrrrrr}\n\\hline\n"
        "\\textbf{Mode} & \\textbf{Cases} & \\textbf{Speed-up} & "
        "\\textbf{Final constr.} & \\textbf{Explicit removed} & "
        "\\textbf{Lazy added} \\\\\n\\hline\n"
    )
    rows = []
    for _, r in df.iterrows():
        speed = "--" if pd.isna(r["mean_speedup"]) else f"{r['mean_speedup']:.2f}x"
        cases = "--" if pd.isna(r["n_cases"]) else f"{int(r['n_cases'])}"
        final = "--" if pd.isna(r["final_constraints_pct"]) else f"{r['final_constraints_pct']:.1f}\\%"
        removed = "--" if pd.isna(r["explicit_removed_pct"]) else f"{r['explicit_removed_pct']*100:.2f}\\%"
        lazy = "--" if pd.isna(r["lazy_added_mean"]) else f"{r['lazy_added_mean']:,.0f}"
        rows.append(
            f"\\texttt{{{r['mode']}}} & "
            f"{cases} & "
            f"{speed} & "
            f"{final} & "
            f"{removed} & "
            f"{lazy} \\\\"
        )
    tex = header + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n\\end{table}\n"
    (TABLES / "constraint_management_summary.tex").write_text(tex, encoding="utf-8")
    return tex


def plot_large_cases_runtime() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(DATA / "large_case_summary.csv")
    df["case_short"] = case_short(df["case_folder"])
    keep_modes = ["LAZY_ALL", "WARM_LAZY", "WARM_SR_LAZY", "SHRINK_LAZY"]
    df = df[df["mode_id"].isin(keep_modes)].copy()
    df["mode_id"] = pd.Categorical(df["mode_id"], keep_modes, ordered=True)
    df = df.sort_values(["case_short", "mode_id"])

    cases = ["case300"]
    fig, axes = plt.subplots(1, 1, figsize=(5.2, 3.8), constrained_layout=True)
    axes = [axes]
    palette = {
        "LAZY_ALL": "#f58518",
        "WARM_LAZY": "#e45756",
        "WARM_SR_LAZY": "#72b7b2",
        "SHRINK_LAZY": "#54a24b",
    }
    for ax, case in zip(axes, cases):
        sub = df[df["case_short"] == case].copy()
        x = list(range(len(sub)))
        heights = sub["runtime_median_sec"].fillna(0.0)
        colors = [palette[m] for m in sub["mode_id"].astype(str)]
        ax.bar(x, heights, color=colors, edgecolor="white")
        ax.set_xticks(x, labels=sub["mode_id"].astype(str), rotation=35, ha="right")
        ax.set_ylabel("Median runtime [s]")
        ax.set_title(case)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=8)
        for xx, (_, r) in zip(x, sub.iterrows()):
            if pd.notna(r["runtime_median_sec"]):
                ax.text(xx, r["runtime_median_sec"] + max(heights.max() * 0.03, 1.0), f"{r['runtime_median_sec']:.0f}", ha="center", va="bottom", fontsize=7)
            else:
                ax.text(xx, max(heights.max() * 0.05, 1.0), "n/a", ha="center", va="bottom", fontsize=7)
    fig.savefig(FIGURES / "large_cases_runtime.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. GNN vs 1-NN ablation (R1-C5): does adding GNN change retention / speedup?
# ---------------------------------------------------------------------------
def gnn_vs_1nn() -> pd.DataFrame:
    spd = pd.read_csv(DATA / "speedup_summary.csv")
    red = pd.read_csv(DATA / "constraint_reduction.csv")

    def canon_mode(m: str) -> tuple[str, bool]:
        has_gnn = m.endswith("+GNN")
        base = m.replace("+GNN", "") if has_gnn else m
        return base, has_gnn

    spd[["base_mode", "has_gnn"]] = spd["mode"].apply(
        lambda m: pd.Series(canon_mode(m))
    )
    red[["base_mode", "has_gnn"]] = red["mode"].apply(
        lambda m: pd.Series(canon_mode(m))
    )

    # keep only PRUNE-0.xx threshold families (both 1-NN and +GNN exist)
    spd = spd[spd["base_mode"].str.startswith("WARM+PRUNE-")]
    red = red[red["base_mode"].str.startswith("WARM+PRUNE-")]

    pivot_spd = spd.pivot_table(
        index=["case_folder", "base_mode"],
        columns="has_gnn",
        values="speedup_mean",
    ).rename(columns={False: "speedup_1nn", True: "speedup_gnn"})
    pivot_red = red.pivot_table(
        index=["case_folder", "base_mode"],
        columns="has_gnn",
        values="kept_ratio_mean",
    ).rename(columns={False: "kept_1nn", True: "kept_gnn"})

    merged = pivot_spd.join(pivot_red, how="inner").reset_index()
    merged["speedup_ratio_gnn_over_1nn"] = (
        merged["speedup_gnn"] / merged["speedup_1nn"]
    )
    merged["kept_delta"] = merged["kept_gnn"] - merged["kept_1nn"]
    merged["case_short"] = merged["case_folder"].str.replace(
        "matpower/", "", regex=False
    )
    merged = merged[
        [
            "case_short",
            "base_mode",
            "speedup_1nn",
            "speedup_gnn",
            "speedup_ratio_gnn_over_1nn",
            "kept_1nn",
            "kept_gnn",
            "kept_delta",
        ]
    ].sort_values(["base_mode", "case_short"])
    merged.to_csv(TABLES / "gnn_vs_1nn.csv", index=False)
    return merged


def render_gnn_vs_1nn_latex(df: pd.DataFrame) -> str:
    """Aggregate across cases: one row per PRUNE threshold with mean gap."""
    agg = (
        df.groupby("base_mode")
        .agg(
            mean_speedup_1nn=("speedup_1nn", "mean"),
            mean_speedup_gnn=("speedup_gnn", "mean"),
            mean_ratio=("speedup_ratio_gnn_over_1nn", "mean"),
            mean_kept_1nn=("kept_1nn", "mean"),
            mean_kept_gnn=("kept_gnn", "mean"),
        )
        .reset_index()
    )
    agg["base_mode"] = agg["base_mode"].str.replace("WARM+", "", regex=False)

    header = (
        "\\begin{table}[t]\\centering\n"
        "\\caption{Ablation of the GraphSAGE screener against the 1-NN baseline. "
        "Values are means across the six benchmark cases. "
        "``Kept ratio'' is the fraction of post-contingency constraints retained after screening.}\n"
        "\\label{tab:gnn_ablation}\n"
        "\\small\n"
        "\\begin{tabular}{lcccc}\n\\hline\n"
        "\\textbf{Threshold} & "
        "\\textbf{Speedup (1-NN)} & "
        "\\textbf{Speedup (GNN)} & "
        "\\textbf{Ratio} & "
        "\\textbf{Kept ratio (1-NN / GNN)} \\\\\n\\hline\n"
    )
    rows = []
    for _, r in agg.iterrows():
        rows.append(
            f"{r['base_mode']} & "
            f"{r['mean_speedup_1nn']:.2f}x & "
            f"{r['mean_speedup_gnn']:.2f}x & "
            f"{r['mean_ratio']:.3f} & "
            f"{r['mean_kept_1nn']*100:.2f}\\% / {r['mean_kept_gnn']*100:.2f}\\% \\\\"
        )
    body = "\n".join(rows)
    footer = "\n\\hline\n\\end{tabular}\n\\end{table}\n"
    tex = header + body + footer
    (TABLES / "gnn_vs_1nn.tex").write_text(tex, encoding="utf-8")
    return tex


def ablation_sanity_table(gnn: pd.DataFrame) -> pd.DataFrame:
    grb = grbtune_table()
    topk = lazy_topk_table()
    gnn_ratio_lo = gnn.groupby("base_mode")["speedup_ratio_gnn_over_1nn"].mean().min()
    gnn_ratio_hi = gnn.groupby("base_mode")["speedup_ratio_gnn_over_1nn"].mean().max()
    case118_best = topk[
        (topk["case_short"] == "case118")
        & topk["variant"].str.startswith("K")
        & (topk["strict_feasible_rate"] >= 1.0)
    ].sort_values("runtime_median_sec").iloc[0]
    case300_fixed = topk[
        (topk["case_short"] == "case300") & topk["variant"].str.startswith("K")
    ]
    grb14 = grb.loc[grb["case_short"] == "case14", "tuned_speedup"].iloc[0]
    grb118 = grb.loc[grb["case_short"] == "case118", "tuned_speedup"].iloc[0]
    grb300 = grb.loc[grb["case_short"] == "case300", "tuned_speedup"].iloc[0]
    rows = [
        {
            "check": "GNN vs 1-NN",
            "evidence": f"mean speed-up ratios {gnn_ratio_lo:.3f}--{gnn_ratio_hi:.3f}",
            "conclusion": "no consistent downstream gain",
        },
        {
            "check": "Fixed-$K$ lazy cuts",
            "evidence": (
                f"case118 best feasible {case118_best['variant']} "
                f"({case118_best['median_speedup']:.1f}x); "
                f"case300 fixed-$K$ feasibility {case300_fixed['strict_feasible_rate'].min()*100:.0f}%"
            ),
            "conclusion": "aggressive cut budgets are not robust",
        },
        {
            "check": "\\texttt{grbtune}",
            "evidence": f"case14 {grb14:.2f}x; case118 {grb118:.2f}x; case300 {grb300:.2f}x",
            "conclusion": "does not explain order-of-magnitude gains",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "ablation_sanity_summary.csv", index=False)
    return out


def render_ablation_sanity_latex(df: pd.DataFrame) -> str:
    header = (
        "\\begin{table}[t]\\centering\n"
        "\\caption{Compact reviewer-facing ablations and sanity checks. "
        "The detailed CSV summaries are released with the paper artifacts.}\n"
        "\\label{tab:ablation_sanity}\n"
        "\\small\n"
        "\\begin{tabularx}{\\linewidth}{lXX}\n\\hline\n"
        "\\textbf{Check} & \\textbf{Evidence} & \\textbf{Conclusion} \\\\\n\\hline\n"
    )
    rows = [
        f"{r['check']} & {r['evidence']} & {r['conclusion']} \\\\"
        for _, r in df.iterrows()
    ]
    tex = header + "\n".join(rows) + "\n\\hline\n\\end{tabularx}\n\\end{table}\n"
    (TABLES / "ablation_sanity_summary.tex").write_text(tex, encoding="utf-8")
    return tex


# ---------------------------------------------------------------------------
# 3. Auxiliary runtime evidence up to case300
# ---------------------------------------------------------------------------
def large_case_table() -> pd.DataFrame:
    df = pd.read_csv(DATA / "large_case_summary.csv")
    df = df[df["case_folder"] == "matpower/case300"].copy()
    df["case_short"] = df["case_folder"].str.replace("matpower/", "", regex=False)
    # baseline = LAZY_ALL (fully exact lazy enforcement over N-1)
    df = df.sort_values("runtime_median_sec")
    keep = [
        "case_folder",
        "mode_id",
        "n",
        "opt_rate",
        "time_limit_rate",
        "runtime_median_sec",
        "runtime_mean_sec",
        "root_constraints_mean",
        "lazy_added_mean",
        "mip_gap_mean",
    ]
    out = df[keep].copy()
    out.to_csv(TABLES / "large_case_table.csv", index=False)
    return out


def render_large_case_latex(df: pd.DataFrame) -> str:
    keep_modes = ["LAZY_ALL", "WARM_LAZY", "WARM_SR_LAZY", "SHRINK_LAZY"]
    df = df[df["mode_id"].isin(keep_modes)].copy()
    df = df[df["mode_id"].isin(keep_modes)].copy()
    header = (
        "\\begin{table}[t]\\centering\n"
        "\\caption{Auxiliary runtime sanity check on \\texttt{case300} for "
        "lazy and warm-lazy variants.}\n"
        "\\label{tab:large_case}\n"
        "\\small\n"
        "\\begin{tabular}{lrrrrrr}\n\\hline\n"
        "\\textbf{Mode} & "
        "\\textbf{n} & "
        "\\textbf{Opt.\\ rate} & "
        "\\textbf{TL rate} & "
        "\\textbf{Median [s]} & "
        "\\textbf{Root constr.} & "
        "\\textbf{Lazy added} \\\\\n\\hline\n"
    )
    rows = []
    for _, r in df.iterrows():
        median = (
            f"{r['runtime_median_sec']:.0f}" if pd.notna(r["runtime_median_sec"]) else "--"
        )
        root = (
            f"{r['root_constraints_mean']:,.0f}"
            if pd.notna(r["root_constraints_mean"])
            else "--"
        )
        lazy = (
            f"{r['lazy_added_mean']:,.0f}" if pd.notna(r["lazy_added_mean"]) else "--"
        )
        rows.append(
            f"\\texttt{{{r['mode_id']}}} & "
            f"{int(r['n'])} & "
            f"{r['opt_rate']*100:.0f}\\% & "
            f"{r['time_limit_rate']*100:.0f}\\% & "
            f"{median} & "
            f"{root} & "
            f"{lazy} \\\\"
        )
    body = "\n".join(rows)
    footer = "\n\\hline\n\\end{tabular}\n\\end{table}\n"
    tex = header + body + footer
    (TABLES / "large_case.tex").write_text(tex, encoding="utf-8")
    return tex


# ---------------------------------------------------------------------------
# 4. Gurobi parameter tuning sanity check (R2-C4)
# ---------------------------------------------------------------------------
def grbtune_table() -> pd.DataFrame:
    df = pd.read_csv(TABLES / "grbtune_vs_default.csv")
    df = df.copy()
    df["case_short"] = df["case"].str.replace("matpower/", "", regex=False)
    df["tuned_speedup"] = df["default_runtime_s"] / df["tuned_runtime_s"]
    return df


def render_grbtune_latex(df: pd.DataFrame) -> str:
    header = (
        "\\begin{table}[t]\\centering\n"
        "\\caption{Gurobi parameter tuning sanity check on representative "
        "instances. The speed-up is the default runtime divided by the tuned "
        "runtime; values below one indicate a slower tuned run.}\n"
        "\\label{tab:grbtune}\n"
        "\\small\n"
        "\\begin{tabular}{lrrrr}\n\\hline\n"
        "\\textbf{Case} & "
        "\\textbf{Default [s]} & "
        "\\textbf{Tuned [s]} & "
        "\\textbf{Speed-up} & "
        "\\textbf{Tune wall [s]} \\\\\n\\hline\n"
    )
    rows = []
    for _, r in df.sort_values("case_short").iterrows():
        rows.append(
            f"{r['case_short']} & "
            f"{r['default_runtime_s']:.2f} & "
            f"{r['tuned_runtime_s']:.2f} & "
            f"{r['tuned_speedup']:.2f}x & "
            f"{r['tune_wall_sec']:.0f} \\\\"
        )
    tex = header + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n\\end{table}\n"
    (TABLES / "grbtune_vs_default.tex").write_text(tex, encoding="utf-8")
    return tex


# ---------------------------------------------------------------------------
# 5. Fixed-K lazy-cut ablation against BANDIT (R1-C6)
# ---------------------------------------------------------------------------
def lazy_topk_table() -> pd.DataFrame:
    fixed = pd.read_csv(TABLES / "lazy_topk_ablation.csv")
    spd = pd.read_csv(DATA / "speedup_summary.csv")

    cases = ["matpower/case118", "matpower/case300"]
    fixed = fixed[fixed["case_folder"].isin(cases)].copy()
    fixed["case_short"] = fixed["case_folder"].str.replace(
        "matpower/", "", regex=False
    )
    fixed["variant"] = fixed["mode"].str.replace("WARM\\+LAZY\\+", "", regex=True)

    raw = (
        spd[(spd["mode"] == "RAW") & spd["case_folder"].isin(cases)]
        .set_index("case_folder")["runtime_median_sec"]
        .to_dict()
    )
    fixed["median_speedup"] = fixed.apply(
        lambda r: raw[r["case_folder"]] / r["runtime_median_sec"], axis=1
    )

    bandit = spd[
        (spd["mode"].isin(["WARM+LAZY", "WARM+LAZY+BANDIT"]))
        & spd["case_folder"].isin(cases)
    ].copy()
    bandit["case_short"] = bandit["case_folder"].str.replace(
        "matpower/", "", regex=False
    )
    bandit["variant"] = bandit["mode"].str.replace("WARM\\+LAZY\\+?", "", regex=True)
    bandit["variant"] = bandit["variant"].replace({"": "LAZY"})
    bandit["status_ok_rate"] = bandit["opt_rate"]
    bandit["strict_feasible_rate"] = bandit["opt_rate"]
    bandit["median_speedup"] = bandit["speedup_median"]

    cols = [
        "case_short",
        "variant",
        "n",
        "runtime_median_sec",
        "median_speedup",
        "status_ok_rate",
        "strict_feasible_rate",
    ]
    out = pd.concat([fixed[cols], bandit[cols]], ignore_index=True)
    order = {"LAZY": 0, "BANDIT": 1, "K64": 2, "K128": 3, "K256": 4}
    out["variant_order"] = out["variant"].map(order)
    return out.sort_values(["case_short", "variant_order"]).drop(columns="variant_order")


def render_lazy_topk_latex(df: pd.DataFrame) -> str:
    header = (
        "\\begin{table}[t]\\centering\n"
        "\\caption{Fixed-$K$ ablation for lazy post-contingency cut generation. "
        "The fixed-$K$ rows use $K\\in\\{64,128,256\\}$; feasibility is the "
        "share of runs whose post-solve violation audit returned OK.}\n"
        "\\label{tab:lazy_topk}\n"
        "\\small\n"
        "\\begin{tabular}{llrrrr}\n\\hline\n"
        "\\textbf{Case} & "
        "\\textbf{Variant} & "
        "\\textbf{n} & "
        "\\textbf{Median [s]} & "
        "\\textbf{Median speed-up} & "
        "\\textbf{Feasible} \\\\\n\\hline\n"
    )
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{r['case_short']} & "
            f"\\texttt{{{r['variant']}}} & "
            f"{int(r['n'])} & "
            f"{r['runtime_median_sec']:.2f} & "
            f"{r['median_speedup']:.2f}x & "
            f"{r['strict_feasible_rate']*100:.0f}\\% \\\\"
        )
    tex = header + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n\\end{table}\n"
    (TABLES / "lazy_topk_ablation.tex").write_text(tex, encoding="utf-8")
    return tex


# ---------------------------------------------------------------------------
# 6. Headline summary (for Response-to-Reviewers + abstract sanity check)
# ---------------------------------------------------------------------------
def headline_summary(spd: pd.DataFrame) -> dict:
    spd_paper = pd.read_csv(DATA / "speedup_summary.csv")
    mean_by_mode = (
        spd_paper[spd_paper["mode"] != "RAW"]
        .groupby("mode")["mean_case_speedup"]
        .first()
        .sort_values(ascending=False)
    )
    per_case_max = (
        spd_paper[spd_paper["mode"] != "RAW"]
        .groupby("case_folder")["speedup_mean"]
        .max()
    )
    d = {
        "best_mode_by_mean_case_speedup": mean_by_mode.index[0],
        "best_mode_value": float(mean_by_mode.iloc[0]),
        "max_case_speedup": float(per_case_max.max()),
        "max_case_name": per_case_max.idxmax(),
        "mean_case_speedup_PRUNE010": float(
            spd_paper.loc[
                spd_paper["mode"] == "WARM+PRUNE-0.10", "mean_case_speedup"
            ].iloc[0]
        ),
        "mean_case_speedup_LAZY_BANDIT": float(
            spd_paper.loc[
                spd_paper["mode"] == "WARM+LAZY+BANDIT", "mean_case_speedup"
            ].iloc[0]
        ),
    }
    (HERE.parent / "data" / "headline.json").write_text(
        json.dumps(d, indent=2), encoding="utf-8"
    )
    return d


# ---------------------------------------------------------------------------
def main() -> None:
    wide = speedup_with_ci()
    render_speedup_ci_latex(wide)
    print(f"wrote {TABLES/'speedup_with_ci.tex'}")

    main_summary = main_results_summary()
    render_main_results_summary_latex(main_summary)
    print(f"wrote {TABLES/'main_results_summary.tex'}")

    plot_speedup_heatmap()
    print(f"wrote {FIGURES/'speedup_heatmap.png'}")

    plot_ecdf()
    print(f"wrote {FIGURES/'ecdf.png'}")

    plot_constraint_reduction_speedup()
    print(f"wrote {FIGURES/'constraint_reduction_speedup.png'}")

    cm = constraint_management_summary()
    render_constraint_management_latex(cm)
    print(f"wrote {TABLES/'constraint_management_summary.tex'}")

    plot_large_cases_runtime()
    print(f"wrote {FIGURES/'large_cases_runtime.png'}")

    gnn = gnn_vs_1nn()
    render_gnn_vs_1nn_latex(gnn)
    print(f"wrote {TABLES/'gnn_vs_1nn.tex'}")

    large = large_case_table()
    render_large_case_latex(large)
    print(f"wrote {TABLES/'large_case.tex'}")

    grbtune = grbtune_table()
    render_grbtune_latex(grbtune)
    print(f"wrote {TABLES/'grbtune_vs_default.tex'}")

    lazy_topk = lazy_topk_table()
    render_lazy_topk_latex(lazy_topk)
    print(f"wrote {TABLES/'lazy_topk_ablation.tex'}")

    ablations = ablation_sanity_table(gnn)
    render_ablation_sanity_latex(ablations)
    print(f"wrote {TABLES/'ablation_sanity_summary.tex'}")

    summary = headline_summary(wide)
    print("headline:", summary)


if __name__ == "__main__":
    main()
