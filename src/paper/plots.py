"""
Unified publication plots.

Concept:
  - small/medium cases: show all tested modes
  - large cases: show only the fast/relevant modes that were actually run
  - keep one consistent ordering and one consistent color per mode
  - use speedup vs RAW on small/medium cases, and runtime + verified success on large cases

Generated PNG figures:
  - bar_speed_ratio_by_mode.png
  - Speedup_general.png
  - heatmap_speedup_by_case_mode.png
  - ecdf.png
  - large_cases_runtime_selected.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FIG_DIR = Path("results") / "figures"
SUMMARY = Path("results") / "summary.csv"
SUMMARY_EXT = Path("results") / "summary_extended.csv"
MERGED = Path("results") / "merged_results.csv"
PAIRS = Path("results") / "pairs.csv"

RAW_BASELINE_MODE = "RAW"
HEURISTIC_MODES = {"SHRINK+LAZY"}
LARGE_CASE_TAGS = {"case118", "case300", "case1354pegase"}
ECDF_TOP_N = 10
LARGE_MAX_MODES = 8

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


def _load_summary() -> pd.DataFrame:
    if SUMMARY_EXT.is_file():
        return pd.read_csv(SUMMARY_EXT)
    return _read_csv_required(SUMMARY)


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


def _case_tag(case_folder: str) -> str:
    return str(case_folder).split("/")[-1]


def _case_sort_key(case_folder: str) -> tuple[int, str]:
    tag = _case_tag(case_folder)
    order = {
        "case14": 14,
        "case30": 30,
        "case57": 57,
        "case89pegase": 89,
        "case118": 118,
        "case300": 300,
        "case1354pegase": 1354,
    }
    return (order.get(tag, 10_000), tag)


def _is_large_case(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda x: _case_tag(x) in LARGE_CASE_TAGS)


def _mode_exactness(mode: str) -> str:
    return "heuristic" if str(mode).upper() in HEURISTIC_MODES else "exact"


def _mode_label(mode: str) -> str:
    m = str(mode)
    return f"{m}$^\\dagger$" if _mode_exactness(m) == "heuristic" else m


def _verified_ok_mask(df: pd.DataFrame) -> pd.Series:
    if "feasible_ok" not in df.columns:
        return pd.Series(True, index=df.index)
    vals = df["feasible_ok"].astype(str).str.strip().str.upper()
    return vals.isin({"OK", "TRUE", "1", "YES"})


def _strict_success_mask(df: pd.DataFrame) -> pd.Series:
    status_ok = df["status"].astype(str).str.upper().eq("OPTIMAL")
    return status_ok & _verified_ok_mask(df)


def _mode_color_map(mode_order: list[str]) -> dict[str, tuple[float, float, float]]:
    if not mode_order:
        return {}
    palette = sns.color_palette("husl", n_colors=len(mode_order))
    return {mode: palette[i] for i, mode in enumerate(mode_order)}


def _outline_color(strict_success: float) -> str:
    if not np.isfinite(strict_success):
        return "#444444"
    if strict_success < 0.85:
        return "#b00020"
    if strict_success < 0.95:
        return "#d17a00"
    return "#222222"


def _line_style(mode: str, strict_success: float) -> str:
    if _mode_exactness(mode) == "heuristic":
        return "-."
    if not np.isfinite(strict_success):
        return "-"
    if strict_success < 0.85:
        return ":"
    if strict_success < 0.95:
        return "--"
    return "-"


def _build_small_mode_stats(
    merged: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    merged_small = merged.loc[~_is_large_case(merged["case_folder"])].copy()
    pairs_small = pairs.loc[~_is_large_case(pairs["case_folder"])].copy()

    pairs_small["runtime_speedup"] = pd.to_numeric(
        pairs_small["runtime_speedup"], errors="coerce"
    )
    pairs_small = pairs_small[np.isfinite(pairs_small["runtime_speedup"])].copy()

    speed = (
        pairs_small.groupby("mode", as_index=False)
        .agg(
            mean_speedup=("runtime_speedup", "mean"),
            median_speedup=("runtime_speedup", "median"),
            N_pairs=("runtime_speedup", "size"),
        )
        .copy()
    )

    merged_small["strict_success"] = _strict_success_mask(merged_small).astype(float)
    qual = (
        merged_small.groupby("mode", as_index=False)
        .agg(
            strict_success=("strict_success", "mean"),
            N_rows=("strict_success", "size"),
        )
        .copy()
    )

    out = speed.merge(qual, on="mode", how="outer")
    if RAW_BASELINE_MODE not in set(out["mode"].astype(str)):
        raw_rows = merged_small.loc[merged_small["mode"].astype(str) == RAW_BASELINE_MODE]
        if not raw_rows.empty:
            out = pd.concat(
                [
                    out,
                    pd.DataFrame(
                        [
                            {
                                "mode": RAW_BASELINE_MODE,
                                "mean_speedup": 1.0,
                                "median_speedup": 1.0,
                                "N_pairs": len(raw_rows),
                                "strict_success": float(
                                    _strict_success_mask(raw_rows).mean()
                                ),
                                "N_rows": len(raw_rows),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    out["method_exactness"] = out["mode"].astype(str).map(_mode_exactness)
    out["mean_speedup"] = pd.to_numeric(out["mean_speedup"], errors="coerce")
    out["strict_success"] = pd.to_numeric(out["strict_success"], errors="coerce")
    out["N_rows"] = pd.to_numeric(out["N_rows"], errors="coerce")
    out = out.dropna(subset=["mean_speedup"]).copy()
    out = out.sort_values(
        ["mean_speedup", "strict_success", "mode"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return out


def _large_focus_modes(
    merged: pd.DataFrame,
    global_order: list[str],
) -> list[str]:
    large = merged.loc[_is_large_case(merged["case_folder"])].copy()
    if large.empty:
        return []

    d = large.copy()
    d["strict_success"] = _strict_success_mask(d).astype(float)
    agg = (
        d.groupby("mode", as_index=False)
        .agg(
            runtime_median=("runtime_sec", "median"),
            strict_success=("strict_success", "mean"),
            N=("strict_success", "size"),
        )
        .copy()
    )
    agg["runtime_median"] = pd.to_numeric(agg["runtime_median"], errors="coerce")
    agg = agg.dropna(subset=["runtime_median"])
    if agg.empty:
        return []

    present = set(agg["mode"].astype(str))
    ordered = [m for m in global_order if m in present]
    if not ordered:
        ordered = (
            agg.sort_values(["runtime_median", "strict_success", "mode"])
            ["mode"]
            .astype(str)
            .tolist()
        )
    if len(ordered) <= LARGE_MAX_MODES:
        return ordered

    keep: list[str] = []
    for must in [RAW_BASELINE_MODE, "WARM+LAZY"]:
        if must in present and must not in keep:
            keep.append(must)

    ranked = (
        agg.sort_values(["runtime_median", "strict_success", "mode"])
        ["mode"]
        .astype(str)
        .tolist()
    )
    for mode in ranked:
        if mode not in keep:
            keep.append(mode)
        if len(keep) >= LARGE_MAX_MODES:
            break
    return keep[:LARGE_MAX_MODES]


def _save_png(fig: plt.Figure, name: str, aliases: list[str] | None = None) -> list[str]:
    fig.tight_layout()
    names = [name] + list(aliases or [])
    for nm in names:
        fig.savefig(FIG_DIR / f"{nm}.png")
    plt.close(fig)
    return [f"{nm}.png" for nm in names]


def bar_speed_ratio_by_mode(
    small_mode_stats: pd.DataFrame,
    mode_order: list[str],
    color_map: dict[str, tuple[float, float, float]],
) -> list[str] | None:
    if small_mode_stats.empty or not mode_order:
        return None

    d = small_mode_stats.set_index("mode").reindex(mode_order).reset_index()
    width = max(13.0, 0.42 * len(d) + 2.5)
    fig, ax = plt.subplots(figsize=(width, 6.6))

    x = np.arange(len(d))
    colors = [color_map.get(m, "#4c78a8") for m in d["mode"].astype(str)]
    edges = [_outline_color(float(v)) for v in d["strict_success"].to_numpy(dtype=float)]
    alphas = [
        0.75 if _mode_exactness(str(m)) == "heuristic" else 0.95
        for m in d["mode"].astype(str)
    ]
    bars = ax.bar(
        x,
        d["mean_speedup"],
        color=colors,
        edgecolor=edges,
        linewidth=1.1,
    )
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    ax.axhline(1.0, ls="--", color="gray", lw=1)
    ax.set_ylabel("Mean speedup vs RAW (higher is better)")
    ax.set_xlabel("Mode")
    ax.set_title("Small/Medium Cases: All Modes Ranked by Mean Speedup")
    ax.set_xticks(x)
    ax.set_xticklabels([_mode_label(m) for m in d["mode"].astype(str)], rotation=90, ha="center")
    ax.tick_params(axis="x", labelsize=7 if len(d) > 18 else 8)
    ax.grid(axis="y", alpha=0.25)

    legend_items = [
        Patch(facecolor="#999999", edgecolor="#222222", label="strict success >= 95%"),
        Patch(facecolor="#999999", edgecolor="#d17a00", label="strict success 85-95%"),
        Patch(facecolor="#999999", edgecolor="#b00020", label="strict success < 85%"),
        Patch(facecolor="#999999", edgecolor="#222222", alpha=0.75, label=r"$^\dagger$ heuristic"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, ncols=2)
    return _save_png(fig, "bar_speed_ratio_by_mode", aliases=["Speedup_general"])


def heatmap_speedup_by_case_mode(
    summary: pd.DataFrame,
    mode_order: list[str],
) -> list[str] | None:
    req = {"case_folder", "mode", "speedup_vs_raw"}
    if not req.issubset(summary.columns):
        return None

    d = summary.loc[~_is_large_case(summary["case_folder"])].copy()
    d["speedup_vs_raw"] = pd.to_numeric(d["speedup_vs_raw"], errors="coerce")
    d = d.dropna(subset=["speedup_vs_raw"]).copy()
    if d.empty:
        return None

    case_order = sorted(d["case_folder"].astype(str).unique(), key=_case_sort_key)
    pivot = d.pivot_table(
        index="mode",
        columns="case_folder",
        values="speedup_vs_raw",
        aggfunc="first",
    ).reindex(index=mode_order, columns=case_order)
    pivot = pivot.dropna(how="all")
    if pivot.empty:
        return None

    fig_w = max(7.2, 1.5 * len(case_order) + 3.2)
    fig_h = max(9.5, 0.33 * len(pivot.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        linewidths=0.25,
        linecolor="white",
        mask=pivot.isna(),
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Speedup vs RAW (higher is better)"},
        ax=ax,
    )
    ax.set_title("Small/Medium Cases: Speedup by Case and Mode")
    ax.set_xlabel("Case")
    ax.set_ylabel("Mode")
    ax.set_xticklabels([_case_tag(c) for c in pivot.columns], rotation=0)
    ax.set_yticklabels([_mode_label(m) for m in pivot.index], rotation=0)
    ax.tick_params(axis="y", labelsize=7 if len(pivot.index) > 18 else 8)
    return _save_png(fig, "heatmap_speedup_by_case_mode")


def ecdf_speedup(
    pairs: pd.DataFrame,
    small_mode_stats: pd.DataFrame,
    mode_order: list[str],
    color_map: dict[str, tuple[float, float, float]],
) -> list[str] | None:
    req = {"case_folder", "mode", "runtime_speedup"}
    if not req.issubset(pairs.columns):
        return None

    d = pairs.loc[~_is_large_case(pairs["case_folder"])].copy()
    d["runtime_speedup"] = pd.to_numeric(d["runtime_speedup"], errors="coerce")
    d = d[np.isfinite(d["runtime_speedup"]) & (d["runtime_speedup"] > 0)].copy()
    if d.empty:
        return None

    small_stats = small_mode_stats.set_index("mode")
    top_modes = [
        m
        for m in mode_order
        if m in set(d["mode"].astype(str))
    ][:ECDF_TOP_N]
    if not top_modes:
        return None

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    for mode in top_modes:
        x = np.sort(
            d.loc[d["mode"].astype(str) == mode, "runtime_speedup"].to_numpy(dtype=float)
        )
        if x.size == 0:
            continue
        y = np.arange(1, x.size + 1, dtype=float) / float(x.size)
        strict = float(small_stats.loc[mode, "strict_success"]) if mode in small_stats.index else np.nan
        ax.step(
            x,
            y,
            where="post",
            lw=2.1,
            color=color_map.get(mode, "#4c78a8"),
            linestyle=_line_style(mode, strict),
            label=_mode_label(mode),
        )

    ax.axvline(1.0, ls="--", color="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(left=max(0.1, float(d["runtime_speedup"].min()) * 0.9))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Per-instance speedup vs RAW (higher is better, log scale)")
    ax.set_ylabel("Empirical cumulative share")
    ax.set_title("Small/Medium Cases: ECDF of the Top Ranked Modes")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, ncols=2 if len(top_modes) > 5 else 1)

    fig.text(
        0.01,
        0.01,
        "Solid = strict success >= 95%, dashed = 85-95%, dotted = < 85%, dash-dot = heuristic.",
        fontsize=8,
        ha="left",
    )
    return _save_png(fig, "ecdf")


def large_cases_runtime_selected(
    merged: pd.DataFrame,
    mode_order: list[str],
    color_map: dict[str, tuple[float, float, float]],
) -> list[str] | None:
    large = merged.loc[_is_large_case(merged["case_folder"])].copy()
    if large.empty:
        return None

    large["strict_success"] = _strict_success_mask(large).astype(float)
    agg = (
        large.groupby(["case_folder", "mode"], as_index=False)
        .agg(
            runtime_median=("runtime_sec", "median"),
            strict_success=("strict_success", "mean"),
            N=("strict_success", "size"),
        )
        .copy()
    )
    agg["runtime_median"] = pd.to_numeric(agg["runtime_median"], errors="coerce")
    agg["strict_success"] = pd.to_numeric(agg["strict_success"], errors="coerce")
    agg = agg.dropna(subset=["runtime_median"])
    if agg.empty:
        return None

    focus_modes = _large_focus_modes(large, mode_order)
    agg = agg[agg["mode"].astype(str).isin(focus_modes)].copy()
    if agg.empty:
        return None

    case_order = sorted(agg["case_folder"].astype(str).unique(), key=_case_sort_key)
    x = np.arange(len(case_order), dtype=float)
    width = 0.11 if len(focus_modes) >= 6 else 0.14
    fig_w = max(9.0, 2.2 * len(case_order) + 0.9 * len(focus_modes))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(fig_w, 8.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.2]},
    )
    ax_rt, ax_ok = axes

    for idx, mode in enumerate(focus_modes):
        subset = agg.loc[agg["mode"].astype(str) == mode].copy()
        rt_vals: list[float] = []
        ok_vals: list[float] = []
        for case in case_order:
            row = subset.loc[subset["case_folder"].astype(str) == case]
            if row.empty:
                rt_vals.append(np.nan)
                ok_vals.append(np.nan)
            else:
                rt_vals.append(float(row["runtime_median"].iloc[0]))
                ok_vals.append(float(row["strict_success"].iloc[0]))

        offsets = x + (idx - (len(focus_modes) - 1) / 2.0) * width
        color = color_map.get(mode, "#4c78a8")
        edge = _outline_color(float(np.nanmean(ok_vals)) if np.isfinite(np.nanmean(ok_vals)) else np.nan)
        bars_rt = ax_rt.bar(
            offsets,
            rt_vals,
            width=width,
            color=color,
            edgecolor=edge,
            linewidth=1.0,
            alpha=0.75 if _mode_exactness(mode) == "heuristic" else 0.95,
            label=_mode_label(mode),
        )
        bars_ok = ax_ok.bar(
            offsets,
            ok_vals,
            width=width,
            color=color,
            edgecolor=edge,
            linewidth=1.0,
            alpha=0.75 if _mode_exactness(mode) == "heuristic" else 0.95,
        )
        for bar, ok in zip(bars_ok, ok_vals):
            if np.isfinite(ok) and ok < 0.95:
                bar.set_edgecolor(_outline_color(ok))

    ax_rt.set_yscale("log")
    ax_rt.set_ylabel("Median runtime [s]")
    ax_rt.set_title("Large Cases: Runtime of the Fast Selected Modes")
    ax_rt.grid(axis="y", alpha=0.25, which="both")
    ax_rt.legend(fontsize=8, ncols=2 if len(focus_modes) > 4 else 1)

    ax_ok.set_ylim(0.0, 1.05)
    ax_ok.axhline(1.0, ls="--", color="gray", lw=1)
    ax_ok.axhline(0.9, ls=":", color="gray", lw=1)
    ax_ok.set_ylabel("Strict success")
    ax_ok.set_xlabel("Large case")
    ax_ok.set_title("Large Cases: Verified-Quality of the Same Mode Set")
    ax_ok.grid(axis="y", alpha=0.25)
    ax_ok.set_xticks(x)
    ax_ok.set_xticklabels([_case_tag(c) for c in case_order], rotation=0)

    missing_large = [
        tag for tag in sorted(LARGE_CASE_TAGS, key=lambda x: _case_sort_key(x))
        if tag not in {_case_tag(c) for c in case_order}
    ]
    footer = "Orange/red outlines mark lower strict success."
    if missing_large:
        footer += f" No plotted rows for: {', '.join(missing_large)}."
    fig.text(0.01, 0.01, footer, fontsize=8, ha="left")
    return _save_png(fig, "large_cases_runtime_selected")


def main() -> None:
    _ensure()
    summary = _load_summary()
    merged = _normalize_runtime_frame(_read_csv_required(MERGED))
    pairs = _read_csv_required(PAIRS)

    small_mode_stats = _build_small_mode_stats(merged, pairs)
    mode_order = small_mode_stats["mode"].astype(str).tolist()
    color_map = _mode_color_map(mode_order)

    generated: set[str] = set()
    outputs = [
        bar_speed_ratio_by_mode(small_mode_stats, mode_order, color_map),
        heatmap_speedup_by_case_mode(summary, mode_order),
        ecdf_speedup(pairs, small_mode_stats, mode_order, color_map),
        large_cases_runtime_selected(merged, mode_order, color_map),
    ]

    for out in outputs:
        if not out:
            continue
        if isinstance(out, list):
            generated.update(out)
        else:
            generated.add(out)

    _cleanup_figure_dir(generated)
    print(f"Figures saved under {FIG_DIR}")
    print("Generated:", ", ".join(sorted(generated)) if generated else "<none>")


if __name__ == "__main__":
    main()
