import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from src.stability_radius.pipeline.case_batch import (
    load_instance_from_case_dir,
    collect_injection_samples,
    compute_baseline_from_samples,
    build_baseline_graph_from_pbar,
    compute_empirical_covariance,
    default_agc_from_pbar,
)
from src.stability_radius.metrics.radius import (
    compute_line_radii,
    predict_overload_probability,
    make_balance_map,
)
from src.stability_radius.powerflow.dc import (
    get_nodes,
    get_edges,
    compute_H_full,
    dc_flows,
    compute_ptdf_to_slack_matrix,
)
from src.stability_radius.metrics.n1 import compute_lodf
from src.stability_radius.sim.monte_carlo import monte_carlo_overload_linear
from src.stability_radius.reporting.tables import write_csv, write_topk_csv
from src.stability_radius.reporting.plots import (
    plot_sorted_radii_bars,
    plot_pred_vs_empirical,
    plot_line_sigma_bars,
    plot_bus_variance,
    plot_line_utilization,
    plot_n1_vs_balanced_scatter,
    plot_ptdf_bars,
    plot_lodf_bars,
    plot_scatter_xy,
    plot_n1_drop_bars,
)
from src.stability_radius.reporting.report_md import write_case_markdown


def parse_custom_participation(a_str: str, nodes: List[str]) -> np.ndarray:
    parts = [p.strip() for p in a_str.split(",") if p.strip()]
    mapping = {}
    for part in parts:
        name, val = part.split(":")
        mapping[name.strip()] = float(val)
    return np.array([mapping.get(n, 0.0) for n in nodes], dtype=float)


def main():
    parser = argparse.ArgumentParser(
        description="Build baseline from case folder (365 subcases), compute stability radii, tables, plots, and a report.md"
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Folder with 365 JSON(.gz) daily scenarios (e.g., src/data/input/matpower/case14)",
    )
    parser.add_argument(
        "--balance",
        choices=["orth", "slack", "agc", "custom"],
        default="agc",
        help="Balance mapping P mode",
    )
    parser.add_argument(
        "--a",
        type=str,
        default="",
        help='Participation vector for balance=custom, e.g., "Bus1:0.5,Bus2:0.5"',
    )
    parser.add_argument(
        "--use-all-times",
        action="store_true",
        help="Use all time steps from each scenario (recommended). If not set, uses t=0 only.",
    )
    parser.add_argument(
        "--t", type=int, default=0, help="Time index to use when not --use-all-times"
    )
    parser.add_argument(
        "--n-sim",
        type=int,
        default=2000,
        help="Monte Carlo samples for empirical overload validation",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: src/stability_radius/outputs/<case-name>)",
    )
    parser.add_argument(
        "--n1",
        action="store_true",
        help="Compute N-1 effective radii (balanced L2, σ if Σ provided)",
    )
    args = parser.parse_args()

    case_name = args.case_dir.name
    out_root = Path("src/stability_radius/outputs")
    out_dir = args.out if args.out is not None else (out_root / case_name)
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load all scenarios from case-dir
    inst = load_instance_from_case_dir(args.case_dir)
    scenarios = inst.scenarios
    if not scenarios:
        raise SystemExit(f"No scenarios found in {args.case_dir}")

    # Collect samples p(t) across scenarios and times
    nodes, P_samples = collect_injection_samples(
        scenarios,
        use_all_times=args.use_all_times,
        t_select=args.t,
    )
    n = len(nodes)
    S = P_samples.shape[1]
    time_info = "all times" if args.use_all_times else f"t={args.t}"
    print(f"Collected {S} injection samples over {n} buses ({time_info}).")

    # Baseline p̄ and baseline graph Ḡ
    pbar = compute_baseline_from_samples(P_samples)
    G_base, slack_name = build_baseline_graph_from_pbar(scenarios[0], nodes, pbar)
    print(f"Baseline graph built. Slack: {slack_name}")

    # Participation vector for P (if needed)
    if args.balance == "custom":
        if not args.a.strip():
            raise SystemExit("balance=custom requires --a 'Bus:val,...'")
        a_vec = parse_custom_participation(args.a, nodes)
    elif args.balance == "agc":
        a_vec = default_agc_from_pbar(nodes, pbar)
    else:
        a_vec = None

    # Empirical covariance Σ̂ (pre-balance) from δp = p - p̄
    Sigma_hat = compute_empirical_covariance(P_samples, pbar)
    print("Empirical covariance Σ computed from data.")

    # Radii and predicted overloads
    radii = compute_line_radii(
        G_base,
        balance_mode=args.balance,
        a=a_vec,
        slack_name=slack_name,
        Sigma=Sigma_hat,
    )
    p_pred = predict_overload_probability(
        G_base,
        balance_mode=args.balance,
        a=a_vec,
        slack_name=slack_name,
        Sigma=Sigma_hat,
    )

    # Monte Carlo validation (linearized propagation)
    freq = monte_carlo_overload_linear(
        G_base,
        slack_name=slack_name,
        n_sim=args.n_sim,
        Sigma=Sigma_hat,
        balance_mode=args.balance,
        a=a_vec,
        disturb_mode="all",
    )

    # Base flows and margins
    edges = get_edges(G_base)
    flows0 = dc_flows(G_base, get_nodes(G_base), get_nodes(G_base).index(slack_name))
    caps = np.array([float(G_base[u][v]["capacity"]) for (u, v) in edges], dtype=float)
    f0 = np.array([flows0[e] for e in edges], dtype=float)
    margins = np.maximum(caps - np.abs(f0), 0.0)
    utilization = np.abs(f0) / np.maximum(caps, 1e-9)

    # PTDF-to-slack metrics (per line)
    slack_idx = get_nodes(G_base).index(slack_name)
    H_full = compute_H_full(G_base, get_nodes(G_base), slack_idx)
    P_map = make_balance_map(n, mode=args.balance, a=a_vec, slack_idx=slack_idx)
    P_orth = make_balance_map(n, mode="orth", slack_idx=slack_idx)
    Gmat_pre = H_full @ P_map
    Gmat_bal = H_full @ P_orth
    var_lines = np.einsum("ij,jk,ik->i", Gmat_pre, Sigma_hat, Gmat_pre, optimize=True)
    std_lines = np.sqrt(np.clip(var_lines, 0.0, None))
    gnorm_pre = np.linalg.norm(Gmat_pre, axis=1)
    gnorm_bal = np.linalg.norm(Gmat_bal, axis=1)

    # Efficient PTDF (i->slack) matrix and aggregates
    PTS = compute_ptdf_to_slack_matrix(G_base, get_nodes(G_base), slack_idx)  # (m, n)
    ptdf_absmax = np.max(np.abs(PTS), axis=1)
    ptdf_l2 = np.linalg.norm(PTS, axis=1)

    # LODF aggregates per line
    lodf = compute_lodf(G_base, slack_name=slack_name)
    lodf_absmax = []
    lodf_mean_abs = []
    for m in edges:
        vals = [abs(lodf.get((m, k), 0.0)) for k in edges if k != m]
        if vals:
            lodf_absmax.append(max(vals))
            lodf_mean_abs.append(float(np.mean(vals)))
        else:
            lodf_absmax.append(0.0)
            lodf_mean_abs.append(0.0)

    # Assemble DataFrame
    def edge_label(e: Tuple[str, str]) -> str:
        return f"{e[0]}-{e[1]}"

    df = (
        pd.DataFrame(
            {
                "line": [edge_label(e) for e in edges],
                "capacity": caps,
                "flow0": f0,
                "utilization": utilization,
                "margin": margins,
                "radius_l2_bal": [radii["l2"][e] for e in edges],
                "radius_l2_pre": [radii["l2_pre"][e] for e in edges],
                "radius_sigma": [radii["sigma"][e] for e in edges],
                "sigma_line": std_lines,
                "pred_overload": p_pred,
                "overload_freq": [freq[e] for e in edges],
                "g_norm_pre": gnorm_pre,
                "g_norm_bal": gnorm_bal,
                "ptdf_absmax": ptdf_absmax,
                "ptdf_l2_slack": ptdf_l2,
                "lodf_absmax": lodf_absmax,
                "lodf_mean_abs": lodf_mean_abs,
            }
        )
        .sort_values("radius_sigma")
        .reset_index(drop=True)
    )

    # Optional N-1 radii
    df["radius_l2_n1"] = np.nan
    if args.n1:
        try:
            from src.stability_radius.metrics.n1 import effective_radii_n1

            r_n1 = effective_radii_n1(
                G_base,
                balance_mode=args.balance,
                a=a_vec,
                slack_name=slack_name,
                Sigma=Sigma_hat,
            )
            # Join to df by line label
            r_n1_series = pd.Series({edge_label(k): v for k, v in r_n1["l2"].items()})
            df["radius_l2_n1"] = df["line"].map(r_n1_series)
            # Also write raw N-1 table
            df_n1 = pd.DataFrame(
                {
                    "line": [edge_label(e) for e in r_n1["l2"].keys()],
                    "radius_l2_n1": list(r_n1["l2"].values()),
                }
            ).sort_values("radius_l2_n1")
            write_csv(df_n1, tables_dir / "n1_radii.csv")
        except Exception as e:
            print(f"[warn] N-1 computation failed: {e}")

    # Save main tables
    write_csv(df, tables_dir / "lines_radii.csv")
    write_topk_csv(
        df,
        tables_dir / "lines_top20_by_sigma.csv",
        by="radius_sigma",
        k=20,
        ascending=True,
    )

    # Baseline node table
    df_nodes = pd.DataFrame(
        {
            "bus": nodes,
            "pbar": pbar,
            "generation_base": np.maximum(pbar, 0.0),
            "load_base": np.maximum(-pbar, 0.0),
            "Sigma_diag": np.diag(Sigma_hat),
        }
    )
    write_csv(df_nodes, tables_dir / "baseline_nodes.csv")

    # Plots (autosized for large cases)
    plot_sorted_radii_bars(df, plots_dir / "sorted_r_sigma.png", kind="sigma")
    plot_sorted_radii_bars(df, plots_dir / "sorted_r_l2.png", kind="l2")
    plot_pred_vs_empirical(df, plots_dir / "pred_vs_empirical.png")
    plot_line_sigma_bars(df, plots_dir / "line_std.png")
    plot_bus_variance(df_nodes, plots_dir / "bus_variance.png")
    plot_line_utilization(df, plots_dir / "line_utilization.png")
    # New PTDF/LODF plots
    plot_ptdf_bars(df, plots_dir / "ptdf_absmax.png", metric="ptdf_absmax")
    plot_ptdf_bars(df, plots_dir / "ptdf_l2.png", metric="ptdf_l2_slack")
    plot_lodf_bars(df, plots_dir / "lodf_absmax.png", metric="lodf_absmax")
    # Competitive scatters
    plot_scatter_xy(
        df,
        "radius_sigma",
        "margin",
        plots_dir / "sigma_vs_margin.png",
        xlabel="σ-radius (std)",
        ylabel="Margin (MW)",
        title="σ-radius vs margin",
    )
    plot_scatter_xy(
        df,
        "radius_l2_bal",
        "margin",
        plots_dir / "l2_vs_margin.png",
        xlabel="L2_bal radius (MW)",
        ylabel="Margin (MW)",
        title="Balanced L2 radius vs margin",
    )
    plot_scatter_xy(
        df,
        "radius_sigma",
        "ptdf_absmax",
        plots_dir / "sigma_vs_ptdf.png",
        xlabel="σ-radius (std)",
        ylabel="Max |PTDF| to slack",
        title="σ-radius vs PTDF sensitivity",
    )
    plot_scatter_xy(
        df,
        "radius_l2_bal",
        "lodf_absmax",
        plots_dir / "l2_vs_lodf.png",
        xlabel="L2_bal radius (MW)",
        ylabel="Max |LODF|",
        title="L2_bal radius vs LODF vulnerability",
    )
    if args.n1 and "radius_l2_n1" in df.columns:
        plot_n1_vs_balanced_scatter(
            df.dropna(subset=["radius_l2_n1"]), plots_dir / "n1_vs_balanced.png"
        )
        plot_n1_drop_bars(df.dropna(subset=["radius_l2_n1"]), plots_dir / "n1_drop.png")

    # Brief terminal report
    min_r_sigma = float(df["radius_sigma"].min())
    median_r_sigma = float(df["radius_sigma"].median())
    mean_pred_over = float(df["pred_overload"].mean())
    mean_emp_over = float(df["overload_freq"].mean())
    print("\n=== Brief Report ===")
    print(f"Case={case_name} | Slack={slack_name} | Balance={args.balance}")
    print(
        f"Buses={n}, Lines={len(edges)}, Scenarios={len(scenarios)}, Using={time_info}"
    )
    print(f"σ-radius: min={min_r_sigma:.4g}, median={median_r_sigma:.4g}")
    print(
        f"Overload: predicted mean={mean_pred_over:.4f}, empirical mean={mean_emp_over:.4f}"
    )
    print("Top-5 risky lines by σ-radius:")
    print(
        df[["line", "radius_sigma", "pred_overload", "overload_freq"]]
        .head(5)
        .to_string(index=False)
    )

    # Markdown report
    plots_dict = {
        "Sorted σ-radius": str((plots_dir / "sorted_r_sigma.png").as_posix()),
        "Sorted L2 radius": str((plots_dir / "sorted_r_l2.png").as_posix()),
        "Pred vs Empirical": str((plots_dir / "pred_vs_empirical.png").as_posix()),
        "Line σ (std)": str((plots_dir / "line_std.png").as_posix()),
        "Bus variance": str((plots_dir / "bus_variance.png").as_posix()),
        "Line utilization": str((plots_dir / "line_utilization.png").as_posix()),
        "PTDF max |PTDF| to slack": str((plots_dir / "ptdf_absmax.png").as_posix()),
        "PTDF L2 to slack": str((plots_dir / "ptdf_l2.png").as_posix()),
        "LODF max |LODF|": str((plots_dir / "lodf_absmax.png").as_posix()),
        "σ-radius vs margin": str((plots_dir / "sigma_vs_margin.png").as_posix()),
        "L2 radius vs margin": str((plots_dir / "l2_vs_margin.png").as_posix()),
        "σ-radius vs PTDF": str((plots_dir / "sigma_vs_ptdf.png").as_posix()),
        "L2 vs LODF": str((plots_dir / "l2_vs_lodf.png").as_posix()),
    }
    if args.n1:
        plots_dict["N-1 vs Balanced (scatter)"] = str(
            (plots_dir / "n1_vs_balanced.png").as_posix()
        )
        plots_dict["N-1 drop (bars)"] = str((plots_dir / "n1_drop.png").as_posix())
    tables_dict = {
        "All lines (radii + stats)": str((tables_dir / "lines_radii.csv").as_posix()),
        "Top-20 lines (by σ-radius)": str(
            (tables_dir / "lines_top20_by_sigma.csv").as_posix()
        ),
        "Baseline nodes": str((tables_dir / "baseline_nodes.csv").as_posix()),
    }
    if args.n1:
        tables_dict["N-1 radii"] = str((tables_dir / "n1_radii.csv").as_posix())

    df_top = df.nsmallest(20, "radius_sigma").copy()
    # Add a 'reason' column for the report
    reason = []
    for _, row in df_top.iterrows():
        util = abs(row["flow0"]) / max(row["capacity"], 1e-9)
        small_margin = row["margin"] < 0.15 * row["capacity"]
        high_var = row["sigma_line"] > df["sigma_line"].median()
        high_sens = row["g_norm_bal"] > df["g_norm_bal"].median()
        if util > 0.85 or small_margin:
            reason.append("margin-limited")
        elif high_var:
            reason.append("variance-driven")
        elif high_sens:
            reason.append("sensitivity-driven")
        else:
            reason.append("mixed")
    df_top["reason"] = reason

    write_case_markdown(
        out_dir / "report.md",
        case_name=case_name,
        n_buses=n,
        n_lines=len(edges),
        n_scenarios=len(scenarios),
        time_info=time_info,
        balance_mode=args.balance,
        slack_name=slack_name,
        brief_stats={
            "min_r_sigma": min_r_sigma,
            "median_r_sigma": median_r_sigma,
            "mean_pred_over": mean_pred_over,
            "mean_emp_over": mean_emp_over,
        },
        df_top=df_top,
        plots=plots_dict,
        tables=tables_dict,
    )

    print(f"\nSaved tables -> {tables_dir}")
    print(f"Saved plots  -> {plots_dir}")
    print(f"Report       -> {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
