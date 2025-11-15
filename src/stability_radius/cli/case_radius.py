"""
CLI: Compute stability radii for a Matpower/UC case folder (e.g., case14) by:
  - loading scenario(s) from JSON/.json.gz using your existing read_data
  - converting the scenario at a given time index t to a NetworkX graph
  - computing radii and (optionally) Monte Carlo overload frequencies
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_preparation.read_data import _read as read_generic
from src.stability_radius.io.uc_scenario import (
    scenario_to_nx_graph,
    agc_vector_from_scenario,
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
    compute_ptdf_to_slack_matrix,
)
from src.stability_radius.metrics.n1 import compute_lodf
from src.stability_radius.sim.monte_carlo import (
    build_disturbance_mask,
    monte_carlo_overload_linear,
    calibrate_sigma_for_target,
)
from src.stability_radius.reporting.plots import (
    plot_sorted_radii_bars,
    plot_pred_vs_empirical,
    plot_n1_vs_balanced_scatter,
    plot_ptdf_bars,
    plot_lodf_bars,
    plot_scatter_xy,
)
from src.stability_radius.reporting.tables import write_csv


def find_case_files(case_dir: Path) -> List[Path]:
    files: List[Path] = []
    files.extend(sorted(case_dir.glob("*.json.gz")))
    files.extend(sorted(case_dir.glob("*.json")))
    return files


def make_participation_vector(
    mode: str, nodes: List[str], a_str: str, scenario, t: int
):
    """
    Build participation vector depending on CLI options.
    """
    if mode == "custom":
        if not a_str.strip():
            raise SystemExit("balance=custom requires --a")
        parts = [p.strip() for p in a_str.split(",") if p.strip()]
        mapping = {}
        for part in parts:
            name, val = part.split(":")
            mapping[name.strip()] = float(val)
        a = np.array([mapping.get(n, 0.0) for n in nodes], dtype=float)
        return a

    if mode == "agc":
        return agc_vector_from_scenario(scenario, nodes, t=t)

    # Not used for orth or slack
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Stability radii for UC/Matpower cases"
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Path to a case folder (e.g., src/data/input/matpower/case14)",
    )
    parser.add_argument("--scenario-idx", type=int, default=0, help="Scenario index")
    parser.add_argument("--time", type=int, default=0, help="Time index t")
    parser.add_argument(
        "--balance",
        choices=["orth", "slack", "custom", "agc"],
        default="agc",
        help="Balance mode",
    )
    parser.add_argument(
        "--a",
        type=str,
        default="",
        help='Participation vector for balance=custom, e.g., "Bus1:0.5,Bus4:0.5"',
    )
    parser.add_argument(
        "--disturb",
        choices=["all", "load", "gen"],
        default="load",
        help="Which buses receive random disturbances (for Σ and MC)",
    )
    parser.add_argument("--sigma", type=float, default=None, help="Noise scale σ")
    parser.add_argument(
        "--auto-sigma",
        action="store_true",
        help="Auto-calibrate σ to target mean overload probability",
    )
    parser.add_argument(
        "--target", type=float, default=0.1, help="Target mean overload probability"
    )
    parser.add_argument("--n-sim", type=int, default=0, help="MC samples (0 to skip)")
    parser.add_argument("--n1", action="store_true", help="Compute N-1 effective radii")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: src/stability_radius/outputs/<case-name>)",
    )
    args = parser.parse_args()

    case_files = find_case_files(args.case_dir)
    if not case_files:
        raise SystemExit(
            f"No JSON files found in {args.case_dir}. Expected *.json or *.json.gz"
        )

    # Load as stochastic instance if multiple files
    inst = read_generic([str(p) for p in case_files], quiet=True)
    if args.scenario_idx < 0 or args.scenario_idx >= len(inst.scenarios):
        raise SystemExit(
            f"Scenario index {args.scenario_idx} out of range (0..{len(inst.scenarios) - 1})"
        )
    scenario = inst.scenarios[args.scenario_idx]
    t = args.time

    # Default output folder by case name
    case_name = args.case_dir.name
    out_root = Path("src/stability_radius/outputs")
    out_dir = args.out if args.out is not None else (out_root / case_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Convert to graph and pick slack
    G, slack_name = scenario_to_nx_graph(scenario, t=t)
    nodes = sorted(G.nodes)

    # Participation vector
    a_vec = make_participation_vector(args.balance, nodes, args.a, scenario, t)

    # Build Σ from σ and mask (or auto calibrate σ)
    mask = build_disturbance_mask(G, nodes, mode=args.disturb).astype(float)
    sigma = args.sigma
    if args.auto_sigma:
        sigma = calibrate_sigma_for_target(
            G,
            balance_mode=args.balance,
            a=a_vec,
            disturb_mode=args.disturb,
            target=args.target,
            slack_name=slack_name,
        )
        print(
            f"[auto-sigma] calibrated σ = {sigma:.4g} to hit target {args.target:.3f}"
        )
    if sigma is None:
        sigma = 0.3
    Sigma = (sigma**2) * np.diag(mask)

    # Radii
    radii = compute_line_radii(
        G,
        balance_mode=args.balance,
        a=a_vec,
        slack_name=slack_name,
        Sigma=Sigma,
    )

    # Predicted overload
    p_pred = predict_overload_probability(
        G,
        balance_mode=args.balance,
        a=a_vec,
        slack_name=slack_name,
        Sigma=Sigma,
    )

    # Optional N-1
    if args.n1:
        try:
            from src.stability_radius.metrics.n1 import effective_radii_n1

            r_n1 = effective_radii_n1(
                G,
                balance_mode=args.balance,
                a=a_vec,
                slack_name=slack_name,
                Sigma=Sigma,
            )
        except Exception as e:
            r_n1 = None
            print(f"[warn] N-1 radii failed: {e}")
    else:
        r_n1 = None

    # PTDF-to-slack metrics
    slack_idx = nodes.index(slack_name)
    H_full = compute_H_full(G, nodes, slack_idx)
    PTS = compute_ptdf_to_slack_matrix(G, nodes, slack_idx)
    ptdf_absmax = np.max(np.abs(PTS), axis=1)
    ptdf_l2 = np.linalg.norm(PTS, axis=1)

    # LODF aggregates
    edges = [(u, v) for (u, v) in radii["l2"].keys()]
    lodf = compute_lodf(G, slack_name=slack_name)
    lodf_absmax = []
    lodf_mean_abs = []
    for m in edges:
        vals = [abs(lodf.get((m, k), 0.0)) for k in edges if k != m]
        lodf_absmax.append(max(vals) if vals else 0.0)
        lodf_mean_abs.append(float(np.mean(vals)) if vals else 0.0)

    # Assemble DF
    edge_labels = [f"{u}-{v}" for (u, v) in edges]
    df = pd.DataFrame(
        {
            "line": edge_labels,
            "radius_l2_bal": [radii["l2"][e] for e in edges],
            "radius_l2_pre": [radii.get("l2_pre", {}).get(e, np.nan) for e in edges],
            "radius_sigma": [radii.get("sigma", {}).get(e, np.nan) for e in edges],
            "pred_overload": list(p_pred),
            "ptdf_absmax": ptdf_absmax,
            "ptdf_l2_slack": ptdf_l2,
            "lodf_absmax": lodf_absmax,
            "lodf_mean_abs": lodf_mean_abs,
        }
    ).sort_values("radius_l2_bal")

    # Optional MC
    if args.n_sim and args.n_sim > 0:
        freq = monte_carlo_overload_linear(
            G,
            slack_name=slack_name,
            n_sim=args.n_sim,
            Sigma=Sigma,
            balance_mode=args.balance,
            a=a_vec,
            disturb_mode=args.disturb,
        )
        df["overload_freq"] = [freq[e] for e in edges]
    else:
        df["overload_freq"] = np.nan

    # Add N-1 to DF if present
    if r_n1 is not None:
        ser_n1 = pd.Series({f"{u}-{v}": val for (u, v), val in r_n1["l2"].items()})
        df["radius_l2_n1"] = df["line"].map(ser_n1)

    # Save table
    write_csv(df, tables_dir / "case_radii.csv")

    # Brief summary
    min_r_sigma = (
        float(np.nanmin(df["radius_sigma"].values))
        if "radius_sigma" in df
        else float("nan")
    )
    median_r_sigma = (
        float(np.nanmedian(df["radius_sigma"].values))
        if "radius_sigma" in df
        else float("nan")
    )
    mean_pred_over = float(np.nanmean(df["pred_overload"].values))
    mean_emp_over = (
        float(np.nanmean(df["overload_freq"].values))
        if "overload_freq" in df
        else float("nan")
    )

    print(
        f"\nCase: {case_name} | scenario={scenario.name} | t={t} | Slack={slack_name}"
    )
    print(f"Balance={args.balance}, σ={sigma:.4g}, disturb={args.disturb}")
    print(f"σ-radius: min={min_r_sigma:.4g}, median={median_r_sigma:.4g}")
    print(
        f"Overload: predicted mean={mean_pred_over:.4f}, empirical mean={mean_emp_over:.4f}"
    )
    print("Top-5 risky lines (by σ-radius):")
    if "radius_sigma" in df.columns:
        print(
            df.sort_values("radius_sigma")[
                ["line", "radius_sigma", "pred_overload", "overload_freq"]
            ]
            .head(5)
            .to_string(index=False)
        )
    else:
        print(df.head(5).to_string(index=False))

    # Plots
    plot_sorted_radii_bars(df, plots_dir / "sorted_r_sigma.png", kind="sigma")
    plot_sorted_radii_bars(df, plots_dir / "sorted_r_l2.png", kind="l2")
    plot_pred_vs_empirical(df, plots_dir / "pred_vs_empirical.png")
    if r_n1 is not None and "radius_l2_n1" in df.columns:
        plot_n1_vs_balanced_scatter(
            df.dropna(subset=["radius_l2_n1"]), plots_dir / "n1_vs_balanced.png"
        )
    # New PTDF/LODF and competitive plots
    plot_ptdf_bars(df, plots_dir / "ptdf_absmax.png", metric="ptdf_absmax")
    plot_ptdf_bars(df, plots_dir / "ptdf_l2.png", metric="ptdf_l2_slack")
    plot_lodf_bars(df, plots_dir / "lodf_absmax.png", metric="lodf_absmax")
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

    print(f"\nSaved tables -> {tables_dir}")
    print(f"Saved plots  -> {plots_dir}")


if __name__ == "__main__":
    main()
