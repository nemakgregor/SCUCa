from pathlib import Path
import pandas as pd


def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}g}"
    except Exception:
        return str(x)


def write_case_markdown(
    out_path: Path,
    case_name: str,
    n_buses: int,
    n_lines: int,
    n_scenarios: int,
    time_info: str,
    balance_mode: str,
    slack_name: str,
    brief_stats: dict,
    df_top: pd.DataFrame,
    plots: dict,
    tables: dict,
):
    """
    Generate a concise report.md with links to plots and tables.
    brief_stats = {
      "min_r_sigma": float,
      "median_r_sigma": float,
      "mean_pred_over": float,
      "mean_emp_over": float,
    }
    plots: keys to Path-like strings
    tables: keys to Path-like strings
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build "reason" column if not present
    if "reason" not in df_top.columns:
        reason = []
        for _, row in df_top.iterrows():
            util = abs(row.get("flow0", 0.0)) / max(row.get("capacity", 1.0), 1e-9)
            small_margin = row.get("margin", 0.0) < 0.15 * row.get("capacity", 1.0)
            high_var = (
                row.get("sigma_line", 0.0) > df_top["sigma_line"].median()
                if "sigma_line" in df_top
                else False
            )
            high_sens = (
                row.get("g_norm_bal", 0.0) > df_top["g_norm_bal"].median()
                if "g_norm_bal" in df_top
                else False
            )
            if util > 0.85 or small_margin:
                reason.append("margin-limited")
            elif high_var:
                reason.append("variance-driven")
            elif high_sens:
                reason.append("sensitivity-driven")
            else:
                reason.append("mixed")
        df_top = df_top.copy()
        df_top["reason"] = reason

    lines = []
    lines.append(f"# Stability Radius Report — {case_name}")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Buses: {n_buses}, Lines: {n_lines}")
    lines.append(f"- Scenarios: {n_scenarios}, time: {time_info}")
    lines.append(f"- Balance: {balance_mode}, Slack bus: {slack_name}")
    lines.append("")
    lines.append("## Key metrics")
    lines.append(f"- min σ-radius: {_fmt(brief_stats.get('min_r_sigma', 'n/a'))}")
    lines.append(f"- median σ-radius: {_fmt(brief_stats.get('median_r_sigma', 'n/a'))}")
    lines.append(
        f"- mean predicted overload prob: {_fmt(brief_stats.get('mean_pred_over', 'n/a'))}"
    )
    lines.append(
        f"- mean empirical overload freq: {_fmt(brief_stats.get('mean_emp_over', 'n/a'))}"
    )
    lines.append("")
    lines.append("## Plots")
    for label, p in plots.items():
        lines.append(f"- {label}: {p}")
    lines.append("")
    lines.append("## Tables")
    for label, p in tables.items():
        lines.append(f"- {label}: {p}")
    lines.append("")
    lines.append("## Top risky lines (by σ-radius)")
    cols_keep = [
        "line",
        "capacity",
        "flow0",
        "utilization",
        "margin",
        "radius_sigma",
        "radius_l2_bal",
        "radius_l2_n1",
        "sigma_line",
        "pred_overload",
        "overload_freq",
        "g_norm_bal",
        "g_norm_pre",
        "reason",
    ]
    cols = [c for c in cols_keep if c in df_top.columns]
    # Markdown table
    lines.append(df_top[cols].to_markdown(index=False))
    content = "\n".join(lines)
    out_path.write_text(content, encoding="utf-8")
