from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_scatter_radius_vs_drop(
    df: pd.DataFrame, radius_col: str, drop_col: str, out_path: Path, title: str
):
    plt.figure(figsize=(6.5, 5.5))
    x = df[radius_col].values
    y = df[drop_col].values
    plt.scatter(x, y, alpha=0.75)
    plt.xlabel(radius_col)
    plt.ylabel(drop_col)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    _savefig(out_path)


def _auto_fig_width(
    n: int, min_w: float = 12.0, per_bar: float = 0.25, max_w: float = 40.0
) -> float:
    # scale figure width with number of bars; clamp to avoid huge images
    return float(min(max(min_w, per_bar * max(n, 1)), max_w))


def plot_bars(
    df: pd.DataFrame,
    value_col: str,
    out_path: Path,
    ascending: bool = False,
    title: str = "",
):
    # Show all rows, sorted by value_col
    df2 = df.sort_values(value_col, ascending=ascending)
    n = len(df2)
    plt.figure(figsize=(_auto_fig_width(n), 4.5))
    x = np.arange(n)
    plt.bar(x, df2[value_col].values, color="tab:blue")
    plt.xticks(x, df2["line_key"].values, rotation=90, fontsize=8)
    plt.ylabel(value_col)
    if title:
        plt.title(title)
    _savefig(out_path)


def plot_radius_sorted(df: pd.DataFrame, radius_col: str, out_path: Path):
    df2 = df.sort_values(radius_col, ascending=True)
    n = len(df2)
    plt.figure(figsize=(_auto_fig_width(n), 4.5))
    x = np.arange(n)
    plt.bar(x, df2[radius_col].values, color="tab:orange")
    plt.xticks(x, df2["line_key"].values, rotation=90, fontsize=8)
    plt.ylabel(radius_col)
    plt.title(f"Lines sorted by {radius_col} (smaller = safer to drop constraints)")
    _savefig(out_path)
