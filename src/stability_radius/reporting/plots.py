from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _auto_figsize(
    n_items: int,
    base_w: float = 10.0,
    per_item: float = 0.15,
    max_w: float = 40.0,
    h: float = 4.5,
) -> Tuple[float, float]:
    # Scale width with number of items; cap to max_w to avoid giant files
    width = min(max(base_w, base_w + per_item * max(n_items - 40, 0)), max_w)
    return (width, h)


def _savefig(path: Path, dpi: int = 200):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_sorted_radii_bars(df: pd.DataFrame, out_path: Path, kind: str = "sigma"):
    """
    kind='sigma' -> radius_sigma; kind='l2' -> radius_l2_bal
    Autosizes figures for readability on large cases.
    """
    if kind == "sigma":
        col = "radius_sigma"
        title = "Lines sorted by σ-radius (smaller = riskier)"
        color = "tab:orange"
    else:
        col = "radius_l2_bal"
        title = "Lines sorted by L2_bal radius (smaller = riskier)"
        color = "tab:blue"

    df_sorted = df.sort_values(col).reset_index(drop=True)
    x = np.arange(len(df_sorted))
    fig_w, fig_h = _auto_figsize(len(df_sorted))
    plt.figure(figsize=(fig_w, fig_h))
    plt.bar(x, df_sorted[col].values, color=color)
    plt.xticks(x, df_sorted["line"].values, rotation=90, fontsize=8)
    plt.ylabel(col)
    plt.title(title)
    _savefig(out_path)


def plot_pred_vs_empirical(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(6.5, 5))
    plt.scatter(df["pred_overload"], df["overload_freq"], c="tab:green", alpha=0.8)
    plt.xlabel("Predicted overload probability")
    plt.ylabel("Empirical overload frequency")
    plt.title("Predicted vs empirical overload (linear MC)")
    plt.grid(True, alpha=0.3)
    _savefig(out_path)


def plot_line_sigma_bars(df: pd.DataFrame, out_path: Path):
    df_sorted = df.sort_values("sigma_line").reset_index(drop=True)
    x = np.arange(len(df_sorted))
    fig_w, fig_h = _auto_figsize(len(df_sorted))
    plt.figure(figsize=(fig_w, fig_h))
    plt.bar(x, df_sorted["sigma_line"].values, color="tab:purple")
    plt.xticks(x, df_sorted["line"].values, rotation=90, fontsize=8)
    plt.ylabel("Std of flow Δf (σ_ℓ)")
    plt.title("Per-line standard deviation under covariance")
    _savefig(out_path)


def plot_bus_variance(df_nodes: pd.DataFrame, out_path: Path):
    df_sorted = df_nodes.sort_values("Sigma_diag", ascending=False).reset_index(
        drop=True
    )
    x = np.arange(len(df_sorted))
    fig_w, fig_h = _auto_figsize(len(df_sorted))
    plt.figure(figsize=(fig_w, fig_h))
    plt.bar(x, df_sorted["Sigma_diag"].values, color="tab:red")
    plt.xticks(x, df_sorted["bus"].values, rotation=90, fontsize=8)
    plt.ylabel("Var(δp_i) [Σ_ii]")
    plt.title("Per-bus injection variance (empirical)")
    _savefig(out_path)


def plot_line_utilization(df: pd.DataFrame, out_path: Path):
    """
    Bar plot of utilization |f0| / capacity sorted high→low.
    """
    if not {"flow0", "capacity", "line"}.issubset(df.columns):
        return
    util = (df["flow0"].abs() / df["capacity"]).clip(0, None)
    df2 = df.assign(utilization=util).sort_values("utilization", ascending=False)
    x = np.arange(len(df2))
    fig_w, fig_h = _auto_figsize(len(df2))
    plt.figure(figsize=(fig_w, fig_h))
    plt.bar(x, df2["utilization"].values, color="tab:brown")
    plt.xticks(x, df2["line"].values, rotation=90, fontsize=8)
    plt.ylabel("|flow0| / capacity")
    plt.title("Line utilization at baseline")
    _savefig(out_path)


def plot_n1_vs_balanced_scatter(df: pd.DataFrame, out_path: Path):
    """
    Scatter r_l2_n1 (y) vs r_l2_bal (x). Adds y=x reference line and annotates up to 10 worst outliers.
    Requires columns: radius_l2_bal, radius_l2_n1, line
    """
    if not {"radius_l2_bal", "radius_l2_n1"}.issubset(df.columns):
        return
    x = df["radius_l2_bal"].values
    y = df["radius_l2_n1"].values
    lines = df["line"].values

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(x, y, alpha=0.7)
    lim = max(np.nanmax(x), np.nanmax(y)) * 1.05 if len(x) else 1.0
    plt.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5, label="y = x")
    plt.xlabel("Balanced L2 radius")
    plt.ylabel("N-1 effective L2 radius")
    plt.title("N-1 vs Balanced L2 radii (per line)")
    plt.legend()

    # Annotate top outliers where N-1 << balanced (large drop)
    drop = x - y
    idx = np.argsort(-drop)[:10]
    for i in idx:
        if np.isfinite(x[i]) and np.isfinite(y[i]):
            plt.annotate(lines[i], (x[i], y[i]), fontsize=8, alpha=0.8)

    _savefig(out_path)


# New generic and PTDF/LODF plots


def plot_bar_by_column(
    df: pd.DataFrame,
    value_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
    ascending: bool = False,
    color: str = "tab:gray",
):
    if value_col not in df.columns:
        return
    df_sorted = df.sort_values(value_col, ascending=ascending).reset_index(drop=True)
    x = np.arange(len(df_sorted))
    fig_w, fig_h = _auto_figsize(len(df_sorted))
    plt.figure(figsize=(fig_w, fig_h))
    plt.bar(x, df_sorted[value_col].values, color=color)
    plt.xticks(x, df_sorted["line"].values, rotation=90, fontsize=8)
    plt.ylabel(ylabel)
    plt.title(title)
    _savefig(out_path)


def plot_ptdf_bars(df: pd.DataFrame, out_path: Path, metric: str = "ptdf_absmax"):
    """
    Plot PTDF-to-slack metric per line as bars (largest first).
    Requires columns: 'line' and one of {'ptdf_absmax', 'ptdf_l2_slack'}.
    """
    labels = {
        "ptdf_absmax": (
            "Max |PTDF| to slack (per line)",
            "max |PTDF_slack|",
            "tab:olive",
        ),
        "ptdf_l2_slack": (
            "L2 norm of PTDF-to-slack (per line)",
            "||PTDF_slack||_2",
            "tab:olive",
        ),
    }
    if metric not in labels:
        metric = "ptdf_absmax"
    title, ylabel, color = labels[metric]
    plot_bar_by_column(
        df, metric, out_path, title, ylabel, ascending=False, color=color
    )


def plot_lodf_bars(df: pd.DataFrame, out_path: Path, metric: str = "lodf_absmax"):
    """
    Plot LODF vulnerability metric per line as bars (largest first).
    Requires columns: 'line' and one of {'lodf_absmax', 'lodf_mean_abs'}.
    """
    labels = {
        "lodf_absmax": (
            "Max |LODF| over outages (per line)",
            "max_k |LODF_{m,k}|",
            "tab:pink",
        ),
        "lodf_mean_abs": (
            "Mean |LODF| over outages (per line)",
            "mean_k |LODF_{m,k}|",
            "tab:pink",
        ),
    }
    if metric not in labels:
        metric = "lodf_absmax"
    title, ylabel, color = labels[metric]
    plot_bar_by_column(
        df, metric, out_path, title, ylabel, ascending=False, color=color
    )


def plot_scatter_xy(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    diag: bool = False,
):
    """
    Generic scatter plot. If diag=True, draws y=x line (for comparable axes).
    """
    if not {x_col, y_col}.issubset(df.columns):
        return
    x = df[x_col].values
    y = df[y_col].values
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(x, y, alpha=0.7)
    if diag:
        lim = max(np.nanmax(x), np.nanmax(y)) * 1.05 if len(x) else 1.0
        plt.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5)
    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel or y_col)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    _savefig(out_path)


def plot_n1_drop_bars(df: pd.DataFrame, out_path: Path):
    """
    Plot bars for N-1 drop: (radius_l2_bal - radius_l2_n1), sorted descending.
    Requires columns: radius_l2_bal, radius_l2_n1
    """
    if not {"radius_l2_bal", "radius_l2_n1"}.issubset(df.columns):
        return
    df2 = df.copy()
    df2["n1_drop"] = df2["radius_l2_bal"] - df2["radius_l2_n1"]
    df2 = df2.sort_values("n1_drop", ascending=False).reset_index(drop=True)
    x = np.arange(len(df2))
    fig_w, fig_h = _auto_figsize(len(df2))
    plt.figure(figsize=(fig_w, fig_h))
    plt.bar(x, df2["n1_drop"].values, color="tab:cyan")
    plt.xticks(x, df2["line"].values, rotation=90, fontsize=8)
    plt.ylabel("r_bal - r_n1 (MW)")
    plt.title("N-1 drop in balanced L2 radius (larger = more sensitive)")
    _savefig(out_path)
