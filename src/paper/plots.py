"""
Improved publication-quality plots from existing merged results (no re-run).

Inputs:
  - results/merged_results.csv     (from analysis.py)
  - results/pairs.csv              (paired vs RAW)
  - results/pairs_all.csv          (all pairwise comparisons across modes)
  - results/mode_fastest_share.csv (share fastest per case)
  - results/summary_extended.csv   (extended per-case summaries)
  - results/effects_wlz_vs_raw.csv (effect sizes WARM+LAZY vs RAW)
  - results/mode_speed_stats.csv   (per-mode runtime ratio stats)  [NEW]

Outputs (.pdf and .png) under results/figures:
  - runtime_cdf_grid.{pdf,png}
  - runtime_cdf_<case>.{pdf,png}
  - nodes_boxplot_ablation_<case>.{pdf,png}
  - pareto_constraints_vs_runtime_<case>.{pdf,png}
  - heatmap_pruned_ratio.{pdf,png}
  - scaling_runtime_vs_size.{pdf,png}
  - memory_distribution.{pdf,png}
  - residual_distribution_ood.{pdf,png}
  - speedup_violin.{pdf,png}
  - slope_runtime_<case>.{pdf,png}
  - effects_forest_wlz.{pdf,png}
  - status_stacked.{pdf,png}
  - feasibility_bars.{pdf,png}
  - scatter_raw_vs_modes_<case>.{pdf,png}
  - fastest_share_bars.{pdf,png}
  - runtime_violin_by_mode.{pdf,png}

  NEW (as per task):
  - bar_speed_ratio_by_mode.{pdf,png}                 Средний speedup по модам (ratio <1 => faster)
  - scatter_runtime_vs_mipgap.{pdf,png}               Runtime vs MIP gap (+ regression)
  - scatter_runtime_vs_memory.{pdf,png}               Runtime vs Memory (OPTIMAL only)
  - box_runtime_by_group_flags.{pdf,png}              Runtime distribution by technique group flags
  - line_prune_tau_runtime_mip.{pdf,png}              Runtime and MIP Gap vs tau (PRUNE)
  - pareto_front_runtime_accuracy.{pdf,png}           Pareto front across modes (runtime vs accuracy)
  - heatmap_mode_corrs.{pdf,png}                      Correlation heatmap: runtime vs metrics by mode
  - time_breakdown_stacked.{pdf,png}                  Stacked bar (root + branch) — only if columns present
  - radar_modes_top5.{pdf,png}                        Radar chart (speed, accuracy, memory, success)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

FIG_DIR = Path("results") / "figures"
MERGED = Path("results") / "merged_results.csv"
PAIRS = Path("results") / "pairs.csv"
PAIRS_ALL = Path("results") / "pairs_all.csv"
SUMMARY_EXT = Path("results") / "summary_extended.csv"
EFFECTS_WLZ = Path("results") / "effects_wlz_vs_raw.csv"
FASTEST = Path("results") / "mode_fastest_share.csv"
MODE_SPEED = Path("results") / "mode_speed_stats.csv"

# Global aesthetics
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 140


def _ensure():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run analysis.py first.")
    return pd.read_csv(path)


def _parse_tau(m):
    try:
        if isinstance(m, str) and "-" in m:
            return float(m.split("-")[-1])
    except Exception:
        return np.nan
    return np.nan


def _set_log_if_needed(ax, vals, axis="y", q_lo=0.05, q_hi=0.95, ratio_thr=10.0):
    """
    If the range (q_hi / q_lo) exceeds ratio_thr and all values are positive,
    switch to log scale for better readability.
    """
    v = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    v = v[v > 0]
    if len(v) == 0:
        return
    lo = v.quantile(q_lo)
    hi = v.quantile(q_hi)
    if lo > 0 and hi / max(lo, 1e-12) >= ratio_thr:
        if axis == "y":
            ax.set_yscale("log")
        else:
            ax.set_xscale("log")


# --------------- Existing plots (kept) ---------------


def runtime_cdfs(df: pd.DataFrame):
    # Per-case CDFs
    for case, g in df.groupby("case_folder"):
        a = (
            g[g["mode"].str.upper().str.startswith("RAW")]["runtime_sec"]
            .dropna()
            .values
        )
        b = g[g["mode"] == "WARM+LAZY"]["runtime_sec"].dropna().values
        if len(a) == 0 or len(b) == 0:
            continue
        xs = np.sort(a)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        xs2 = np.sort(b)
        ys2 = np.arange(1, len(xs2) + 1) / len(xs2)
        plt.figure(figsize=(5.2, 4.2))
        plt.plot(xs, ys, label="RAW", lw=2.0, color="#3b5b92")
        plt.plot(xs2, ys2, label="WARM+LAZY", linestyle="--", lw=2.2, color="#cc5a49")
        plt.xlabel("Runtime [s]")
        plt.ylabel("CDF")
        plt.title(f"Runtime CDF — {case}")
        plt.grid(alpha=0.25)
        plt.legend()
        out = FIG_DIR / f"runtime_cdf_{case.replace('/', '_')}"
        plt.tight_layout()
        plt.savefig(out.with_suffix(".pdf"))
        plt.savefig(out.with_suffix(".png"))
        plt.close()


def runtime_cdf_grid(df: pd.DataFrame, max_cols: int = 2):
    # Grid of cases
    cases = sorted(df["case_folder"].unique())
    if not cases:
        return
    n = len(cases)
    cols = max_cols
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.2 * rows), squeeze=False)
    i = -1
    for i, case in enumerate(cases):
        r = i // cols
        c = i % cols
        ax = axs[r, c]
        g = df[df["case_folder"] == case]
        a = (
            g[g["mode"].str.upper().str.startswith("RAW")]["runtime_sec"]
            .dropna()
            .values
        )
        b = g[g["mode"] == "WARM+LAZY"]["runtime_sec"].dropna().values
        if len(a) == 0 or len(b) == 0:
            ax.axis("off")
            continue
        xs = np.sort(a)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        xs2 = np.sort(b)
        ys2 = np.arange(1, len(xs2) + 1) / len(xs2)
        ax.plot(xs, ys, label="RAW", lw=2.0, color="#3b5b92")
        ax.plot(xs2, ys2, label="WARM+LAZY", linestyle="--", lw=2.2, color="#cc5a49")
        ax.set_xlabel("Runtime [s]")
        ax.set_ylabel("CDF")
        ax.set_title(case)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend()
    # Clean empty cells
    for j in range(i + 1, rows * cols):
        r = j // cols
        c = j % cols
        axs[r, c].axis("off")
    out = FIG_DIR / "runtime_cdf_grid"
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def nodes_boxplot_ablation(df: pd.DataFrame, case_focus: str = "matpower/case118"):
    g = df[df["case_folder"] == case_focus].copy()
    order = ["RAW", "WARM", "WARM+HINTS", "WARM+LAZY"]
    g = g[g["mode"].isin(order)]
    if g.empty:
        return
    plt.figure(figsize=(6.6, 4.6))
    sns.boxplot(
        data=g,
        x="mode",
        y="nodes",
        order=order,
        hue="mode",
        dodge=False,
        legend=False,
        palette="Set2",
        linewidth=1,
    )
    sns.swarmplot(
        data=g,
        x="mode",
        y="nodes",
        order=order,
        color="black",
        size=2,
        alpha=0.35,
    )
    ax = plt.gca()
    _set_log_if_needed(ax, g["nodes"], axis="y", ratio_thr=20.0)
    plt.title(f"Node counts — ablation — {case_focus}")
    plt.ylabel("Node count")
    plt.xlabel("")
    out = FIG_DIR / f"nodes_boxplot_ablation_{case_focus.replace('/', '_')}"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def pareto_constraints_vs_runtime(
    df: pd.DataFrame, case_focus: str = "matpower/case118"
):
    g = df[
        (df["case_folder"] == case_focus) & (df["mode"].str.startswith("WARM+PRUNE"))
    ].copy()
    if g.empty:
        return
    g["tau"] = g["mode"].apply(_parse_tau)

    raw_med_rt = (
        df[
            (df["case_folder"] == case_focus)
            & (df["mode"].str.upper().str.startswith("RAW"))
        ]["runtime_sec"]
        .dropna()
        .median()
    )
    raw_med_root = (
        df[
            (df["case_folder"] == case_focus)
            & (df["mode"].str.upper().str.startswith("RAW"))
        ]["num_constrs_root"]
        .dropna()
        .median()
    )

    rows = []
    for tau, gg in g.groupby("tau"):
        if len(gg) == 0:
            continue
        if (
            "constr_ratio_cont" in gg.columns
            and not gg["constr_ratio_cont"].isna().all()
        ):
            ratio = pd.to_numeric(gg["constr_ratio_cont"], errors="coerce").median()
        else:
            ratio = (
                (gg["num_constrs_root"].median() / raw_med_root)
                if raw_med_root
                else np.nan
            )
        med_rt = gg["runtime_sec"].median()
        speedup = (raw_med_rt / med_rt) if (med_rt and med_rt > 0) else np.nan
        rows.append({"tau": tau, "ratio": ratio, "speedup": speedup})

    if not rows:
        return
    H = pd.DataFrame(rows).sort_values("tau")

    plt.figure(figsize=(6.6, 4.6))
    plt.plot(H["ratio"], H["speedup"], marker="o", color="#3b5b92", lw=2)
    for _, r in H.iterrows():
        plt.annotate(
            f"{r['tau']:.2f}",
            (r["ratio"], r["speedup"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color="#333333",
        )
    plt.xlabel("Constraint ratio vs RAW (median)")
    plt.ylabel("Speed-up vs RAW (×, median)")
    plt.title(f"Screening trade-off — {case_focus}")
    plt.grid(alpha=0.25)
    out = FIG_DIR / f"pareto_constraints_vs_runtime_{case_focus.replace('/', '_')}"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def heatmap_pruned_ratio(df: pd.DataFrame, cases: list):
    series_list = []
    all_taus = set()
    for case in cases:
        g = df[
            (df["case_folder"] == case) & (df["mode"].str.startswith("WARM+PRUNE"))
        ].copy()
        if g.empty:
            continue
        g["tau"] = g["mode"].apply(_parse_tau)
        if "constr_ratio_cont" in g.columns and not g["constr_ratio_cont"].isna().all():
            g["ratio"] = pd.to_numeric(g["constr_ratio_cont"], errors="coerce")
        else:
            base_root = df[
                (df["case_folder"] == case)
                & (df["mode"].str.upper().str.startswith("RAW"))
            ]["num_constrs_root"].median()
            g["ratio"] = (
                g["num_constrs_root"] / float(base_root)
                if base_root and base_root > 0
                else np.nan
            )
        s = g.groupby("tau")["ratio"].median()
        series_list.append(s.rename(case))
        all_taus.update(s.index.tolist())
    if not series_list:
        return

    all_taus = sorted(list(all_taus))
    mat = pd.concat(series_list, axis=1).reindex(index=all_taus)
    plt.figure(figsize=(6.8, 4.8))
    sns.heatmap(
        mat.T,
        annot=True,
        fmt=".2f",
        cmap="mako",
        cbar_kws={"label": "constraint ratio (median)"},
        linewidths=0.4,
        linecolor="white",
        vmin=0.0,
        vmax=1.0,
    )
    plt.xlabel("tau")
    plt.ylabel("case")
    plt.title("Pruned constraints ratio (median) by tau and case")
    out = FIG_DIR / "heatmap_pruned_ratio"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def scalability(df: pd.DataFrame):
    plt.figure(figsize=(12.5, 4.8))
    modes = [("RAW", "#3b5b92"), ("WARM+LAZY", "#cc5a49")]
    for i, (mode, color) in enumerate(modes, start=1):
        ax = plt.subplot(1, 2, i)
        g = df[df["mode"] == mode].copy()
        g = g.dropna(subset=["num_constrs_root", "runtime_sec"])
        if g.empty:
            ax.axis("off")
            continue
        sns.scatterplot(
            data=g,
            x="num_constrs_root",
            y="runtime_sec",
            hue="case_folder",
            palette="tab10",
            edgecolor="k",
            linewidth=0.25,
            s=35,
            alpha=0.8,
            ax=ax,
        )
        try:
            bins = pd.qcut(g["num_constrs_root"], q=min(8, len(g)), duplicates="drop")
            trend = g.groupby(bins)["runtime_sec"].median().reset_index()
            xs = []
            for intr in trend[bins.name]:
                try:
                    xs.append(0.5 * (intr.left + intr.right))
                except Exception:
                    xs.append(np.nan)
            ax.plot(xs, trend["runtime_sec"], color=color, lw=2, label="median trend")
        except Exception:
            pass
        _set_log_if_needed(ax, g["num_constrs_root"], axis="x", ratio_thr=10.0)
        _set_log_if_needed(ax, g["runtime_sec"], axis="y", ratio_thr=10.0)
        ax.set_xlabel("Root constraints (count)")
        if i == 1:
            ax.set_ylabel("Runtime [s]")
        else:
            ax.set_ylabel("")
        ax.set_title(mode)
        ax.grid(alpha=0.25)
        if i == 2:
            ax.legend_.remove() if ax.legend_ else None
    out = FIG_DIR / "scaling_runtime_vs_size"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def memory_distribution(df: pd.DataFrame):
    d = df.dropna(subset=["peak_memory_gb"]).copy()
    if d.empty:
        return
    plt.figure(figsize=(6.6, 4.6))
    ax = sns.violinplot(
        data=d,
        x="mode",
        y="peak_memory_gb",
        hue="mode",
        dodge=False,
        legend=False,
        density_norm="width",
        inner="quartile",
        palette="Pastel2",
        cut=0,
    )
    _set_log_if_needed(ax, d["peak_memory_gb"], axis="y", ratio_thr=5.0)
    plt.ylabel("Peak memory [GB]")
    plt.xlabel("")
    plt.title("Peak memory distribution by mode")
    out = FIG_DIR / "memory_distribution"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def residual_distribution_ood(df: pd.DataFrame):
    if "max_constraint_residual" not in df.columns:
        return
    d = df[df["mode"].isin(["RAW", "WARM+LAZY"])].copy()
    d["max_constraint_residual"] = pd.to_numeric(
        d["max_constraint_residual"], errors="coerce"
    )
    d = d.dropna(subset=["max_constraint_residual"])
    if d.empty:
        return
    d = d[d["max_constraint_residual"] > 0]
    if d.empty:
        return
    plt.figure(figsize=(6.6, 4.6))
    sns.ecdfplot(
        data=d,
        x="max_constraint_residual",
        hue="mode",
        palette={"RAW": "#3b5b92", "WARM+LAZY": "#cc5a49"},
    )
    ax = plt.gca()
    ax.set_xscale("log")
    plt.xlabel("Max constraint residual (log scale)")
    plt.ylabel("ECDF")
    plt.title("Verification residuals (ECDF, RAW vs WARM+LAZY)")
    plt.grid(alpha=0.25)
    out = FIG_DIR / "residual_distribution_ood"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def speedup_violin(pairs: pd.DataFrame):
    g = pairs.copy()
    g = g[g["runtime_speedup"].notna()]
    if g.empty:
        return
    plt.figure(figsize=(6.6, 4.6))
    sns.violinplot(
        data=g,
        x="mode",
        y="runtime_speedup",
        hue="mode",
        dodge=False,
        legend=False,
        density_norm="width",
        inner="quartile",
        palette="Set3",
        cut=0,
    )
    sns.stripplot(
        data=g,
        x="mode",
        y="runtime_speedup",
        color="k",
        size=1.8,
        alpha=0.25,
        jitter=0.25,
    )
    plt.axhline(1.0, ls="--", c="grey", lw=1)
    plt.ylabel("Speed-up vs RAW (×)")
    plt.xlabel("")
    plt.title("Runtime speed-up distribution (paired vs RAW)")
    out = FIG_DIR / "speedup_violin"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def slope_runtime(df: pd.DataFrame, case_focus: str = "matpower/case118"):
    g = df[df["case_folder"] == case_focus]
    a = g[g["mode"].str.upper().str.startswith("RAW")][
        ["instance_name", "runtime_sec"]
    ].rename(columns={"runtime_sec": "rt_raw"})
    b = g[g["mode"] == "WARM+LAZY"][["instance_name", "runtime_sec"]].rename(
        columns={"runtime_sec": "rt_wlz"}
    )
    m = a.merge(b, on="instance_name")
    if m.empty:
        return
    m = m.sort_values("rt_raw", ascending=False)
    if len(m) > 80:
        m = m.head(80)
    plt.figure(figsize=(6.8, 5.4))
    ax = plt.gca()
    for _, r in m.iterrows():
        x = [0, 1]
        y = [r["rt_raw"], r["rt_wlz"]]
        color = "#cc5a49" if r["rt_wlz"] < r["rt_raw"] else "#3b5b92"
        plt.plot(x, y, color=color, alpha=0.55, lw=1.2)
        plt.scatter(x, y, color=color, s=10, zorder=3)
    plt.xticks([0, 1], ["RAW", "WARM+LAZY"])
    _set_log_if_needed(
        ax, list(m["rt_raw"]) + list(m["rt_wlz"]), axis="y", ratio_thr=10.0
    )
    plt.ylabel("Runtime [s]")
    plt.title(f"Slopegraph of per-instance runtime — {case_focus}")
    out = FIG_DIR / f"slope_runtime_{case_focus.replace('/', '_')}"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def effects_forest(eff: pd.DataFrame):
    if eff.empty:
        return
    e = eff.copy()
    e["case"] = e["case_folder"].apply(lambda s: s.split("/")[-1])
    e = e.sort_values("HL_runtime_delta")
    plt.figure(figsize=(6.8, 4.8))
    y = np.arange(len(e))
    colors = ["#cc5a49" if v < 0 else "#3b5b92" for v in e["HL_runtime_delta"]]
    plt.hlines(y, 0, e["HL_runtime_delta"], colors=colors, alpha=0.6)
    plt.scatter(e["HL_runtime_delta"], y, color=colors, s=40)
    plt.axvline(0.0, color="k", lw=1, ls="--")
    plt.yticks(y, e["case"])
    plt.xlabel("HL median paired delta in runtime [s] (WLZ - RAW)")
    plt.title("Effect sizes by case (WARM+LAZY vs RAW)")
    out = FIG_DIR / "effects_forest_wlz"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def status_stacked(df: pd.DataFrame):
    g = df.groupby(["mode", "status"]).size().reset_index(name="count")
    if g.empty:
        return
    total = g.groupby("mode")["count"].transform("sum")
    g["share"] = g["count"] / total
    order = sorted(df["mode"].unique())
    statuses = sorted(g["status"].unique())
    mat = []
    for st in statuses:
        v = g[g["status"] == st].set_index("mode").reindex(order)["share"].fillna(0.0)
        mat.append(v.values)
    arr = np.vstack(mat)
    plt.figure(figsize=(7.2, 4.8))
    bottom = np.zeros(len(order))
    colors = sns.color_palette("tab20", n_colors=len(statuses))
    for i, st in enumerate(statuses):
        plt.bar(order, arr[i], bottom=bottom, label=st, color=colors[i])
        bottom += arr[i]
    plt.ylabel("Share")
    plt.xlabel("Mode")
    plt.title("Solver status distribution by mode")
    plt.legend(ncol=2, fontsize=8)
    out = FIG_DIR / "status_stacked"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def feasibility_bars(df: pd.DataFrame):
    g = df.copy()
    g["OK"] = (g["violations"].astype(str) == "OK").astype(float)
    t = g.groupby("mode")["OK"].mean().reset_index()
    if t.empty:
        return
    plt.figure(figsize=(6.2, 4.4))
    sns.barplot(data=t, x="mode", y="OK", color="#789262")
    plt.ylim(0, 1.0)
    plt.ylabel("Feasibility rate")
    plt.xlabel("")
    plt.title("Share of verified solutions with no violations")
    out = FIG_DIR / "feasibility_bars"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def scatter_raw_vs_modes(df: pd.DataFrame, case_focus: str = "matpower/case300"):
    """
    For a given case, plot scatter of runtime RAW (x) vs runtime of several other modes (y).
    Points below the diagonal indicate a speed-up over RAW.
    """
    g = df[df["case_folder"] == case_focus].copy()
    base = g[g["mode"].str.upper().str.startswith("RAW")][
        ["instance_name", "runtime_sec"]
    ].rename(columns={"runtime_sec": "rt_raw"})
    others = sorted(
        [m for m in g["mode"].unique() if not str(m).upper().startswith("RAW")]
    )
    if not others:
        return
    # choose up to 4 most common other modes
    freq = g[g["mode"].isin(others)].groupby("mode").size().sort_values(ascending=False)
    modes_pick = list(freq.index[:4])
    n = len(modes_pick)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(6.0 * cols, 4.6 * rows), squeeze=False)
    for i, mode in enumerate(modes_pick):
        r = i // cols
        c = i % cols
        ax = axs[r, c]
        m = g[g["mode"] == mode][["instance_name", "runtime_sec"]].rename(
            columns={"runtime_sec": "rt_m"}
        )
        M = base.merge(m, on="instance_name")
        if M.empty:
            ax.axis("off")
            continue
        ax.scatter(
            M["rt_raw"],
            M["rt_m"],
            s=22,
            alpha=0.8,
            edgecolor="k",
            linewidths=0.2,
            color="#4c72b0",
        )
        lim = max(M["rt_raw"].max(), M["rt_m"].max())
        ax.plot([0, lim], [0, lim], ls="--", c="grey", lw=1)
        ax.set_xlabel("RAW runtime [s]")
        ax.set_ylabel(f"{mode} runtime [s]")
        ax.set_title(f"{case_focus} — RAW vs {mode}")
        ax.grid(alpha=0.25)
        _set_log_if_needed(ax, M["rt_raw"], axis="x", ratio_thr=10.0)
        _set_log_if_needed(ax, M["rt_m"], axis="y", ratio_thr=10.0)
    # cleanup empty axes
    for j in range(i + 1, rows * cols):
        r = j // cols
        c = j % cols
        axs[r, c].axis("off")
    out = FIG_DIR / f"scatter_raw_vs_modes_{case_focus.replace('/', '_')}"
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def fastest_share_bars(fastest: pd.DataFrame):
    """
    Bar chart of fastest-mode share per case.
    """
    if fastest.empty:
        return
    plt.figure(figsize=(7.4, 4.8))
    sns.barplot(
        data=fastest, x="case_folder", y="fastest_share", hue="mode", palette="tab20"
    )
    plt.ylim(0, 1.0)
    plt.ylabel("Fastest share")
    plt.xlabel("Case")
    plt.title("Share of instances where mode is fastest")
    plt.legend(ncol=2, fontsize=8)
    out = FIG_DIR / "fastest_share_bars"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def runtime_violin_by_mode(df: pd.DataFrame):
    """
    Overall runtime distribution across all cases by mode (violin plot).
    """
    d = df.dropna(subset=["runtime_sec"]).copy()
    if d.empty:
        return
    plt.figure(figsize=(7.2, 4.8))
    sns.violinplot(
        data=d,
        x="mode",
        y="runtime_sec",
        hue="mode",
        dodge=False,
        legend=False,
        density_norm="width",
        inner="quartile",
        palette="Set2",
        cut=0,
    )
    ax = plt.gca()
    _set_log_if_needed(ax, d["runtime_sec"], axis="y", ratio_thr=10.0)
    plt.ylabel("Runtime [s]")
    plt.xlabel("")
    plt.title("Runtime distribution by mode (all cases)")
    out = FIG_DIR / "runtime_violin_by_mode"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


# --------------- NEW PLOTS per task ---------------


def bar_speed_ratio_by_mode(mode_speed: pd.DataFrame):
    """
    Bar Plot: Средний speedup по модам
    - Uses runtime_ratio = mode_runtime / raw_runtime (values < 1 => faster).
    - Bars show mean and error bars show std across paired instances.
    - Modes highlighted with ratio < 0.2 and success_rate_strict > 0.9.
    """
    d = mode_speed.copy()
    required = {
        "mode",
        "mean_runtime_ratio",
        "std_runtime_ratio",
        "N",
        "success_rate_strict",
    }
    if not required.issubset(set(d.columns)):
        return
    d = d.sort_values("mean_runtime_ratio")
    plt.figure(figsize=(8.2, 4.8))
    ax = sns.barplot(
        data=d,
        x="mode",
        y="mean_runtime_ratio",
        color="#4c72b0",
        ci=None,
    )
    # Add error bars (std)
    for i, r in d.reset_index(drop=True).iterrows():
        y = r["mean_runtime_ratio"]
        e = r["std_runtime_ratio"]
        if np.isfinite(y) and np.isfinite(e):
            ax.errorbar(i, y, yerr=e, color="k", capsize=3, lw=0.8)
    # reference line at 1
    plt.axhline(1.0, ls="--", color="gray", lw=1)
    # highlight recommended modes
    for i, r in d.reset_index(drop=True).iterrows():
        if (
            (r["mean_runtime_ratio"] < 0.2)
            and (r["success_rate_strict"] is not None)
            and (r["success_rate_strict"] > 0.9)
        ):
            ax.text(
                i,
                r["mean_runtime_ratio"] + 0.02,
                "★",
                ha="center",
                va="bottom",
                color="#cc5a49",
                fontsize=12,
            )
    plt.ylabel("Mean runtime ratio vs RAW (lower is better)")
    plt.xlabel("")
    plt.title("Средний speedup по модам (ratio < 1 => быстрее)")
    plt.xticks(rotation=20, ha="right")
    out = FIG_DIR / "bar_speed_ratio_by_mode"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def scatter_runtime_vs_mipgap(df: pd.DataFrame):
    """
    Scatter: Runtime vs MIP Gap (точность)
    - hue by mode
    - size by success_rate (mode-level feasible rate)
    - add regression line (overall)
    """
    d = df.copy()
    d["mip_gap"] = pd.to_numeric(d["mip_gap"], errors="coerce")
    d = d.dropna(subset=["runtime_sec", "mip_gap"])
    if d.empty:
        return
    # success rate per mode (feasible OK)
    succ = (
        d.assign(OK=(d["violations"].astype(str) == "OK").astype(float))
        .groupby("mode")["OK"]
        .mean()
        .rename("success_rate")
        .reset_index()
    )
    d = d.merge(succ, on="mode", how="left")
    plt.figure(figsize=(7.4, 5.2))
    sns.scatterplot(
        data=d,
        x="runtime_sec",
        y="mip_gap",
        hue="mode",
        size="success_rate",
        sizes=(20, 140),
        alpha=0.7,
        edgecolor="k",
        linewidth=0.2,
    )
    # Regression line (overall)
    try:
        sns.regplot(
            data=d,
            x="runtime_sec",
            y="mip_gap",
            scatter=False,
            color="black",
            line_kws={"lw": 1.5, "alpha": 0.5},
        )
    except Exception:
        pass
    plt.xlabel("Runtime [s]")
    plt.ylabel("MIP gap")
    plt.title("Trade-off: Runtime vs Accuracy (MIP gap)")
    plt.grid(alpha=0.25)
    out = FIG_DIR / "scatter_runtime_vs_mipgap"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def scatter_runtime_vs_memory(df: pd.DataFrame):
    """
    Scatter: Runtime vs Memory (OPTIMAL only)
    """
    d = df.copy()
    d = d[d["status"].isin(["OPTIMAL"])]
    d = d.dropna(subset=["runtime_sec", "peak_memory_gb"])
    if d.empty:
        return
    plt.figure(figsize=(7.0, 5.0))
    sns.scatterplot(
        data=d,
        x="runtime_sec",
        y="peak_memory_gb",
        hue="mode",
        alpha=0.9,
        edgecolor="k",
        linewidth=0.25,
    )
    plt.xlabel("Runtime [s]")
    plt.ylabel("Peak memory [GB]")
    plt.title("Runtime vs Memory (OPTIMAL only)")
    plt.grid(alpha=0.25)
    out = FIG_DIR / "scatter_runtime_vs_memory"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def box_runtime_by_group_flags(df: pd.DataFrame):
    """
    Box Plot: runtime distribution by group flags (e.g., with_GNN, with_LAZY).
    We will show a grid of 3x2 plots comparing True vs False for each flag.
    """
    flags = [
        "with_LAZY",
        "with_PRUNE",
        "with_GNN",
        "with_COMMIT",
        "with_GRU",
        "with_BANDIT",
    ]
    d = df.dropna(subset=["runtime_sec"]).copy()
    if d.empty:
        return
    n = len(flags)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(6.0 * cols, 4.6 * rows), squeeze=False)
    for i, flg in enumerate(flags):
        r = i // cols
        c = i % cols
        ax = axs[r, c]
        if flg not in d.columns:
            ax.axis("off")
            continue
        tmp = d.copy()
        tmp["flag"] = np.where(tmp[flg], "True", "False")
        if tmp["flag"].nunique() < 2:
            ax.axis("off")
            continue
        sns.boxplot(
            data=tmp,
            x="flag",
            y="runtime_sec",
            palette=["#8ecae6", "#f4a261"],
            ax=ax,
        )
        ax.set_title(f"Runtime by {flg}")
        ax.set_xlabel("")
        ax.set_ylabel("Runtime [s]")
        _set_log_if_needed(ax, tmp["runtime_sec"], axis="y", ratio_thr=10.0)
        ax.grid(alpha=0.25)
    # empty cells
    for j in range(i + 1, rows * cols):
        r = j // cols
        c = j % cols
        axs[r, c].axis("off")
    out = FIG_DIR / "box_runtime_by_group_flags"
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def line_prune_tau_runtime_mip(df: pd.DataFrame):
    """
    Line Plot: Runtime and MIP Gap vs Tau (for PRUNE).
    x = tau, y1 = runtime_sec (median), hue = with_GNN
    twin y-axis for mip_gap (median).
    """
    d = df[df["mode"].str.startswith("WARM+PRUNE")].copy()
    if d.empty:
        return
    d["tau"] = d["mode"].apply(_parse_tau)
    d["with_GNN"] = d["mode"].str.contains("GNN", case=False, na=False)
    grp = (
        d.groupby(["with_GNN", "tau"])
        .agg(
            runtime_median=("runtime_sec", "median"),
            mip_gap_median=(
                "mip_gap",
                lambda x: pd.to_numeric(x, errors="coerce").median(),
            ),
        )
        .reset_index()
    )
    grp = grp.dropna(subset=["tau"])
    if grp.empty:
        return
    plt.figure(figsize=(7.0, 4.8))
    ax1 = plt.gca()
    sns.lineplot(
        data=grp,
        x="tau",
        y="runtime_median",
        hue="with_GNN",
        marker="o",
        palette={True: "#2a9d8f", False: "#264653"},
        ax=ax1,
    )
    ax1.set_xlabel("tau")
    ax1.set_ylabel("Runtime [s]")
    ax1.grid(alpha=0.25)
    # twin axis for MIP gap
    ax2 = ax1.twinx()
    for gn, gg in grp.groupby("with_GNN"):
        ax2.plot(
            gg["tau"],
            gg["mip_gap_median"],
            ls="--",
            marker="s",
            color="#e76f51" if gn else "#f4a261",
            label=f"MIP gap (with_GNN={gn})",
            alpha=0.8,
        )
    ax2.set_ylabel("MIP gap (median)")
    # Build legend manually
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(
        handles1,
        ["runtime (with_GNN=False)", "runtime (with_GNN=True)"],
        loc="upper left",
        fontsize=8,
    )
    out = FIG_DIR / "line_prune_tau_runtime_mip"
    plt.title("WARM+PRUNE: Runtime and MIP gap vs tau")
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def pareto_front_runtime_accuracy(df: pd.DataFrame):
    """
    Pareto Front: Cumulative front across modes for runtime and accuracy.
    - Aggregate per mode across all cases: runtime_median, accuracy = 1 - median(mip_gap).
    - Sort by runtime and plot cumulative max of accuracy.
    """
    d = df.copy()
    d["mip_gap"] = pd.to_numeric(d["mip_gap"], errors="coerce")
    grp = (
        d.groupby("mode")
        .agg(
            runtime_median=("runtime_sec", "median"),
            mip_gap_median=("mip_gap", "median"),
        )
        .reset_index()
    )
    grp["accuracy"] = 1.0 - grp["mip_gap_median"]
    grp = grp.sort_values("runtime_median")
    if grp.empty:
        return
    # cum-max accuracy
    acc = grp["accuracy"].fillna(0.0).values
    cum_max = np.maximum.accumulate(acc)
    plt.figure(figsize=(7.2, 4.6))
    plt.step(grp["runtime_median"], cum_max, where="post", color="#3b5b92", lw=2)
    # scatter points and annotate
    plt.scatter(grp["runtime_median"], grp["accuracy"], color="#cc5a49", s=30, zorder=3)
    for _, r in grp.iterrows():
        plt.annotate(
            r["mode"], (r["runtime_median"], r["accuracy"]), fontsize=8, alpha=0.8
        )
    plt.xlabel("Runtime median [s]")
    plt.ylabel("Cumulative max accuracy (1 - mip_gap)")
    plt.title("Pareto front across modes")
    plt.grid(alpha=0.25)
    out = FIG_DIR / "pareto_front_runtime_accuracy"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def heatmap_mode_corrs(df: pd.DataFrame):
    """
    Heatmap: Correlations per mode between runtime and other metrics.
    Columns:
      - corr(runtime, mip_gap)
      - corr(runtime, peak_memory_gb)
      - corr(runtime, nodes)
      - corr(runtime, num_constrs_root)
    Rows: modes
    """
    d = df.copy()
    d["mip_gap"] = pd.to_numeric(d["mip_gap"], errors="coerce")
    cols = ["mip_gap", "peak_memory_gb", "nodes", "num_constrs_root"]
    rows = []
    for mode, g in d.groupby("mode"):
        vals = {}
        vals["mode"] = mode
        try:
            rt = pd.to_numeric(g["runtime_sec"], errors="coerce")
            for c in cols:
                cc = pd.to_numeric(g[c], errors="coerce")
                m = pd.concat([rt, cc], axis=1).dropna()
                corr = m.corr().iloc[0, 1] if len(m) >= 2 else np.nan
                vals[f"corr_runtime_{c}"] = corr
        except Exception:
            for c in cols:
                vals[f"corr_runtime_{c}"] = np.nan
        rows.append(vals)
    H = pd.DataFrame(rows).set_index("mode")
    if H.empty:
        return
    plt.figure(figsize=(8.8, 6.2))
    sns.heatmap(
        H,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        cbar_kws={"label": "correlation"},
    )
    plt.title("Correlation of runtime with other metrics by mode")
    out = FIG_DIR / "heatmap_mode_corrs"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def time_breakdown_stacked(df: pd.DataFrame):
    """
    Stacked Bar: Breakdown времени (root + branch) if columns available:
      - phase_root_sec, phase_branch_sec per row (not provided by default).
    Will skip silently if missing.
    """
    if not {"phase_root_sec", "phase_branch_sec", "mode"}.issubset(set(df.columns)):
        # Not available in current logs
        return
    d = df.dropna(subset=["phase_root_sec", "phase_branch_sec"]).copy()
    if d.empty:
        return
    grp = (
        d.groupby("mode")[["phase_root_sec", "phase_branch_sec"]].median().reset_index()
    )
    grp = grp.sort_values("phase_root_sec")
    plt.figure(figsize=(7.0, 4.6))
    plt.bar(grp["mode"], grp["phase_root_sec"], label="root", color="#457b9d")
    plt.bar(
        grp["mode"],
        grp["phase_branch_sec"],
        bottom=grp["phase_root_sec"],
        label="branch",
        color="#e76f51",
    )
    plt.ylabel("Time [s] (median)")
    plt.xlabel("Mode")
    plt.title("Time breakdown (root + branch) by mode (median)")
    plt.legend()
    plt.xticks(rotation=15, ha="right")
    out = FIG_DIR / "time_breakdown_stacked"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def radar_modes_top5(df: pd.DataFrame):
    """
    Radar Chart: Multi-score (speed, accuracy, memory, success) for top-5 modes.
    - speed score: normalized (higher better) using inverse of runtime median
    - accuracy: 1 - mip_gap_median
    - memory: inverse of mem_gb_median
    - success: feasible_rate (violations == OK)
    """
    s = _read_csv_required(SUMMARY_EXT).copy()
    # Aggregate across cases: take medians of medians
    grp = (
        s.groupby("mode")
        .agg(
            rt=("runtime_median", "median"),
            mip=("mip_gap_median", "median"),
            mem=("mem_gb_median", "median"),
            feas=("feasible_rate", "median"),
        )
        .reset_index()
    )
    if grp.empty:
        return

    # scores: larger is better
    def _norm_inv(x):
        x = pd.to_numeric(x, errors="coerce")
        if x.isna().all():
            return x
        inv = 1.0 / x.replace(0, np.nan)
        # normalize 0..1
        mn, mx = inv.min(), inv.max()
        return (inv - mn) / (mx - mn) if mx > mn else inv * 0.0

    speed_score = _norm_inv(grp["rt"])
    mem_score = _norm_inv(grp["mem"])
    acc_score = grp["mip"]
    acc_score = (
        1.0
        - (acc_score - np.nanmin(acc_score))
        / (np.nanmax(acc_score) - np.nanmin(acc_score))
        if np.nanmax(acc_score) > np.nanmin(acc_score)
        else acc_score * 0.0
    )
    succ_score = grp["feas"]
    # normalize succ 0..1
    mn, mx = np.nanmin(succ_score), np.nanmax(succ_score)
    succ_score = (succ_score - mn) / (mx - mn) if mx > mn else succ_score * 0.0

    grp["score"] = (
        0.35 * speed_score + 0.25 * acc_score + 0.20 * mem_score + 0.20 * succ_score
    )
    top = grp.sort_values("score", ascending=False).head(5).reset_index(drop=True)
    if top.empty:
        return
    # Radar
    categories = ["Speed", "Accuracy", "Memory", "Success"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(7.2, 7.2))
    ax = plt.subplot(111, polar=True)
    for _, r in top.iterrows():
        vals = [
            float(speed_score[grp["mode"] == r["mode"]]),
            float(acc_score[grp["mode"] == r["mode"]]),
            float(mem_score[grp["mode"] == r["mode"]]),
            float(succ_score[grp["mode"] == r["mode"]]),
        ]
        vals += vals[:1]
        ax.plot(angles, vals, label=r["mode"], linewidth=2, alpha=0.85)
        ax.fill(angles, vals, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.0)
    plt.title("Top-5 modes — multi-criterion radar")
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    out = FIG_DIR / "radar_modes_top5"
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"))
    plt.savefig(out.with_suffix(".png"))
    plt.close()


def main():
    _ensure()
    df = _read_csv_required(MERGED)

    # Core plots already present
    runtime_cdfs(df)
    runtime_cdf_grid(df)
    nodes_boxplot_ablation(df, "matpower/case118")
    pareto_constraints_vs_runtime(df, "matpower/case118")
    heatmap_pruned_ratio(df, sorted(df["case_folder"].unique()))
    scalability(df)
    memory_distribution(df)
    residual_distribution_ood(df)

    # Additional plots
    pairs = _read_csv_required(PAIRS)
    speedup_violin(pairs)
    slope_runtime(df, "matpower/case118")
    eff = _read_csv_required(EFFECTS_WLZ)
    effects_forest(eff)
    status_stacked(df)
    feasibility_bars(df)
    scatter_raw_vs_modes(df, "matpower/case300")
    runtime_violin_by_mode(df)

    # NEW per task
    mode_speed = _read_csv_required(MODE_SPEED)
    bar_speed_ratio_by_mode(mode_speed)
    scatter_runtime_vs_mipgap(df)
    scatter_runtime_vs_memory(df)
    box_runtime_by_group_flags(df)
    line_prune_tau_runtime_mip(df)
    pareto_front_runtime_accuracy(df)
    heatmap_mode_corrs(df)
    time_breakdown_stacked(df)  # will skip if columns missing
    radar_modes_top5(df)

    fastest = _read_csv_required(FASTEST)
    fastest_share_bars(fastest)

    print(f"Figures saved under {FIG_DIR}")


if __name__ == "__main__":
    main()
