"""
Minimal, publication-focused plotting.

Generates only clear PNG figures:
  - runtime_cdf_grid.png
  - bar_speed_ratio_by_mode.png
  - line_prune_tau_runtime_mip.png
  - status_stacked.png
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = Path("results") / "figures"
MERGED = Path("results") / "merged_results.csv"
MODE_SPEED = Path("results") / "mode_speed_stats.csv"
RAW_BASELINE_MODE = "RAW"
HEURISTIC_MODES = {"SHRINK+LAZY"}

# Keep visuals simple and readable in papers.
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


def _mode_label(mode: str) -> str:
    m = str(mode)
    return f"{m}$^\\dagger$" if _mode_exactness(m) == "heuristic" else m


def _save_png(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)


def _parse_tau(mode: str) -> float:
    mt = re.search(r"(?:PRUNE|LPSCREEN)-([0-9]+(?:\.[0-9]+)?)", str(mode).upper())
    if mt:
        return float(mt.group(1))
    return np.nan


def runtime_cdf_grid(df: pd.DataFrame, max_cols: int = 3) -> str | None:
    cases = sorted(df["case_folder"].dropna().astype(str).unique())
    if not cases:
        return None
    n = len(cases)
    cols = max_cols
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.0 * rows), squeeze=False)
    i = -1
    for i, case in enumerate(cases):
        ax = axs[i // cols, i % cols]
        g = df[df["case_folder"] == case]
        raw = g[g["mode"].str.upper() == RAW_BASELINE_MODE]["runtime_sec"].dropna().to_numpy()
        wlz = g[g["mode"] == "WARM+LAZY"]["runtime_sec"].dropna().to_numpy()
        if len(raw) == 0 or len(wlz) == 0:
            ax.axis("off")
            continue
        x1 = np.sort(raw)
        y1 = np.arange(1, len(x1) + 1) / len(x1)
        x2 = np.sort(wlz)
        y2 = np.arange(1, len(x2) + 1) / len(x2)
        ax.plot(x1, y1, label="RAW", lw=2.0, color="#355070")
        ax.plot(x2, y2, label="WARM+LAZY", lw=2.0, color="#d1495b", linestyle="--")
        ax.set_title(case.split("/")[-1])
        ax.set_xlabel("Runtime [s]")
        ax.set_ylabel("CDF")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend()
    for j in range(i + 1, rows * cols):
        axs[j // cols, j % cols].axis("off")
    _save_png(fig, "runtime_cdf_grid")
    return "runtime_cdf_grid.png"


def bar_speed_ratio_by_mode(mode_speed: pd.DataFrame) -> str | None:
    req = {"mode", "mean_runtime_ratio", "std_runtime_ratio", "success_rate_strict"}
    if not req.issubset(set(mode_speed.columns)):
        return None

    d = mode_speed.copy()
    d = d.dropna(subset=["mean_runtime_ratio"]).copy()
    if d.empty:
        return None
    d = d[d["mode"].astype(str).str.upper() != RAW_BASELINE_MODE].copy()
    if d.empty:
        return None
    d["label"] = d["mode"].astype(str).map(_mode_label)
    d = d.sort_values("mean_runtime_ratio")

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    sns.barplot(
        data=d,
        x="label",
        y="mean_runtime_ratio",
        color="#4c78a8",
        errorbar=None,
        ax=ax,
    )
    for i, r in d.reset_index(drop=True).iterrows():
        y = float(r["mean_runtime_ratio"])
        e = float(pd.to_numeric(r.get("std_runtime_ratio", np.nan), errors="coerce"))
        if np.isfinite(y) and np.isfinite(e):
            ax.errorbar(i, y, yerr=e, color="black", capsize=3, lw=0.9)
    ax.axhline(1.0, ls="--", color="gray", lw=1)
    ax.set_ylabel("Runtime ratio vs RAW (lower is better)")
    ax.set_xlabel("Mode")
    ax.set_title("Mode Comparison by End-to-End Runtime")
    # Vertical labels as requested.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    if d["mode"].astype(str).str.upper().isin(HEURISTIC_MODES).any():
        fig.text(
            0.01,
            0.01,
            r"$^\dagger$ heuristic mode",
            fontsize=8,
            ha="left",
        )
    _save_png(fig, "bar_speed_ratio_by_mode")
    return "bar_speed_ratio_by_mode.png"


def line_prune_tau_runtime_mip(df: pd.DataFrame) -> str | None:
    d = df[
        df["mode"].astype(str).str.startswith("WARM+PRUNE")
        & ~df["mode"].astype(str).str.contains("LAZY", case=False, na=False)
    ].copy()
    if d.empty:
        return None
    d["tau"] = d["mode"].map(_parse_tau)
    d["with_GNN"] = d["mode"].astype(str).str.contains("GNN", case=False, na=False)
    grp = (
        d.groupby(["with_GNN", "tau"], dropna=True)
        .agg(
            runtime_median=("runtime_sec", "median"),
            mip_gap_median=("mip_gap", lambda x: pd.to_numeric(x, errors="coerce").median()),
        )
        .reset_index()
    )
    grp = grp.dropna(subset=["tau"])
    if grp.empty:
        return None

    fig, ax1 = plt.subplots(figsize=(7.6, 4.8))
    sns.lineplot(
        data=grp,
        x="tau",
        y="runtime_median",
        hue="with_GNN",
        marker="o",
        palette={False: "#355070", True: "#2a9d8f"},
        ax=ax1,
    )
    ax1.set_xlabel("tau")
    ax1.set_ylabel("Runtime median [s]")
    ax1.set_title("PRUNE Tau Sweep: Runtime and MIP Gap")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    for use_gnn, gg in grp.groupby("with_GNN"):
        ax2.plot(
            gg["tau"],
            gg["mip_gap_median"],
            ls="--",
            marker="s",
            lw=1.2,
            color="#d1495b" if bool(use_gnn) else "#f4a261",
            alpha=0.85,
        )
    ax2.set_ylabel("MIP gap median")
    _save_png(fig, "line_prune_tau_runtime_mip")
    return "line_prune_tau_runtime_mip.png"


def status_stacked(df: pd.DataFrame) -> str | None:
    preferred = [
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
    present = [m for m in preferred if m in set(df["mode"].astype(str))]
    if not present:
        return None
    g = df[df["mode"].isin(present)].copy()
    if g.empty:
        return None
    status = g["status"].astype(str).str.upper()
    g = g.assign(
        s_opt=(status == "OPTIMAL").astype(float),
        s_inf=status.isin(["INFEASIBLE", "INF_OR_UNBD", "INFEASIBLE_OR_UNBOUNDED"]).astype(
            float
        ),
    )
    agg = (
        g.groupby("mode")
        .agg(optimal=("s_opt", "mean"), infeasible=("s_inf", "mean"))
        .reindex(present)
        .fillna(0.0)
    )
    agg["other"] = 1.0 - agg["optimal"] - agg["infeasible"]
    agg["label"] = [ _mode_label(m) for m in agg.index ]

    x = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.bar(x, agg["optimal"], label="Optimal", color="#2a9d8f")
    ax.bar(x, agg["infeasible"], bottom=agg["optimal"], label="Infeasible", color="#e76f51")
    ax.bar(
        x,
        agg["other"],
        bottom=agg["optimal"] + agg["infeasible"],
        label="Other",
        color="#9aa0a6",
    )
    ax.set_xticks(x)
    # Vertical labels as requested.
    ax.set_xticklabels(agg["label"], rotation=90, ha="center")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Share")
    ax.set_xlabel("Mode")
    ax.set_title("Termination Status by Mode")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save_png(fig, "status_stacked")
    return "status_stacked.png"


def main() -> None:
    _ensure()
    df = _normalize_runtime_frame(_read_csv_required(MERGED))
    mode_speed = _read_csv_required(MODE_SPEED)

    generated = set()
    for fn in (
        runtime_cdf_grid,
        lambda x: bar_speed_ratio_by_mode(mode_speed),
        line_prune_tau_runtime_mip,
        status_stacked,
    ):
        out = fn(df)
        if out:
            generated.add(out)

    _cleanup_figure_dir(generated)
    print(f"Figures saved under {FIG_DIR}")
    print("Generated:", ", ".join(sorted(generated)) if generated else "<none>")


if __name__ == "__main__":
    main()
