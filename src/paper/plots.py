"""
Improved publication-quality plots from existing merged results (no re-run).

Inputs:
  - results/merged_results.csv     (from analysis.py)
  - results/pairs.csv              (paired vs RAW)
  - results/summary_extended.csv   (extended per-case summaries)
  - results/effects_wlz_vs_raw.csv (effect sizes WARM+LAZY vs RAW)

Outputs (.pdf and .png) under results/figures:
  - runtime_cdf_grid.{pdf,png}
  - runtime_cdf_<case>.{pdf,png}
  - nodes_boxplot_ablation_<case>.{pdf,png}
  - pareto_constraints_vs_runtime_<case>.{pdf,png}     [now: clearer trade-off curve]
  - heatmap_pruned_ratio.{pdf,png}                     [more robust; fills missing data]
  - scaling_runtime_vs_size.{pdf,png}                  [now: two subplots + trend]
  - memory_distribution.{pdf,png}                      [cleaner, no deprecation warnings]
  - residual_distribution_ood.{pdf,png}                [now: ECDF (log) across modes]
  - speedup_violin.{pdf,png}                           [no deprecation warnings]
  - slope_runtime_<case>.{pdf,png}                     [log y, sorted and compact]
  - effects_forest_wlz.{pdf,png}
  - status_stacked.{pdf,png}
  - feasibility_bars.{pdf,png}
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
SUMMARY_EXT = Path("results") / "summary_extended.csv"
EFFECTS_WLZ = Path("results") / "effects_wlz_vs_raw.csv"

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
    # Fix seaborn deprecation: pass hue and disable dodge/legend to get same appearance
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
    # Overlay small swarm points
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
    """
    Improved: plot trade-off curve per tau:
      x = median constraint ratio vs RAW
      y = median speed-up vs RAW
    Points are connected in ascending tau and labeled with tau.
    """
    g = df[
        (df["case_folder"] == case_focus) & (df["mode"].str.startswith("WARM+PRUNE"))
    ].copy()
    if g.empty:
        return
    g["tau"] = g["mode"].apply(_parse_tau)

    # Baseline RAW medians for normalization
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
        # Constraint ratio: prefer direct (kept / total) if present, else root vs RAW root
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
        # Speed-up vs RAW
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
    """
    More robust heatmap: fills missing tau levels across cases with NaN,
    then heatmaps median constraint ratio. Keeps filename for compatibility.
    """
    series_list = []
    all_taus = set()
    # Collect per-case series indexed by tau
    for case in cases:
        g = df[
            (df["case_folder"] == case) & (df["mode"].str.startswith("WARM+PRUNE"))
        ].copy()
        if g.empty:
            continue
        g["tau"] = g["mode"].apply(_parse_tau)
        # Resolve ratio
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

    # Union over all taus to align columns
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
    """
    Improved: two subplots (RAW vs WARM+LAZY), scatter + simple trend,
    log scales when needed.
    """
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
        # Trend line (simple median in bins)
        try:
            bins = pd.qcut(g["num_constrs_root"], q=min(8, len(g)), duplicates="drop")
            trend = g.groupby(bins)["runtime_sec"].median().reset_index()
            # Use midpoints for x
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
    """
    Cleaner: violin with quartiles, proper seaborn API for palette (hue=mode),
    optional log y if spread is large.
    """
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
        density_norm="width",  # replaces 'scale="width"'
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
    """
    Reworked: ECDF of max verification residuals (log x) across modes (RAW vs WARM+LAZY).
    Shows that both methods meet tight tolerances, more informative than boxplots near zero.
    """
    if "max_constraint_residual" not in df.columns:
        return
    d = df[df["mode"].isin(["RAW", "WARM+LAZY"])].copy()
    d["max_constraint_residual"] = pd.to_numeric(
        d["max_constraint_residual"], errors="coerce"
    )
    d = d.dropna(subset=["max_constraint_residual"])
    if d.empty:
        return
    # Residuals should be >=0; filter non-positives to avoid log issues in x-axis when plotting ECDF
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
    # Distribution of speedups vs RAW across all cases/modes
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
        density_norm="width",  # replaces 'scale="width"'
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
    # Slopegraph RAW→WARM+LAZY per-instance; improve readability on wide spreads.
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
    # Sort by raw runtime; if too many instances, show top 80 by raw runtime to keep readable
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
    # Forest-style plot for WLZ vs RAW effect sizes per case
    if eff.empty:
        return
    e = eff.copy()
    e["case"] = e["case_folder"].apply(lambda s: s.split("/")[-1])
    e = e.sort_values("HL_runtime_delta")
    plt.figure(figsize=(6.8, 4.8))
    y = np.arange(len(e))
    # Marker: HL median delta (runtime); color by sign
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
    # Stacked bar of status distribution per mode (across cases)
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
    # Share of 'violations' == 'OK' by mode
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


def main():
    _ensure()
    df = _read_csv_required(MERGED)
    # Core plots
    runtime_cdfs(df)
    runtime_cdf_grid(df)
    nodes_boxplot_ablation(df, "matpower/case118")
    pareto_constraints_vs_runtime(df, "matpower/case118")
    heatmap_pruned_ratio(df, sorted(df["case_folder"].unique()))
    scalability(df)
    memory_distribution(df)
    residual_distribution_ood(df)
    # Additional advanced plots
    pairs = _read_csv_required(PAIRS)
    speedup_violin(pairs)
    slope_runtime(df, "matpower/case118")
    eff = _read_csv_required(EFFECTS_WLZ)
    effects_forest(eff)
    status_stacked(df)
    feasibility_bars(df)
    print(f"Figures saved under {FIG_DIR}")


if __name__ == "__main__":
    main()
