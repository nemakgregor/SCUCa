"""
CLI: compute radii and validate against Monte Carlo overloads.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

try:
    from sklearn.metrics import roc_auc_score

    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

from ..data.grids import GRID_FACTORY
from ..metrics.radius import (
    compute_line_radii,
    make_balance_map,
    predict_overload_probability,
)
from ..sim.monte_carlo import (
    monte_carlo_overload_linear,
    build_disturbance_mask,
    calibrate_sigma_for_target,
)


def evaluate_grid(
    G,
    name: str,
    sigma: float | None,
    n_sim: int,
    balance_mode: str,
    a_vec: np.ndarray | None,
    disturb_mode: str,
    Sigma: np.ndarray | None,
    do_n1: bool,
    out_dir: Path,
    make_plots: bool,
    auto_sigma: bool,
    target_overload: float,
):
    nodes = sorted(G.nodes)
    n = len(nodes)

    # Disturbance mask and covariance
    mask = build_disturbance_mask(G, nodes, mode=disturb_mode)
    if Sigma is None:
        # If auto-sigma: calibrate using predicted overloads (analytic)
        if auto_sigma:
            # Prepare ingredients for calibration
            P = make_balance_map(
                n,
                mode=balance_mode,
                a=a_vec,
                slack_idx=nodes.index(next(nm for nm in nodes if "Slack" in nm)),
            )
            sigma = calibrate_sigma_for_target(
                G,
                balance_mode=balance_mode,
                a=a_vec,
                disturb_mode=disturb_mode,
                target=target_overload,
            )
            print(
                f"[auto-sigma] calibrated σ = {sigma:.4g} for target mean overload {target_overload:.3f}"
            )
        else:
            if sigma is None:
                sigma = 0.3
        # Build Σ = σ^2 diag(mask)
        Sigma = (sigma**2) * np.diag(mask.astype(float))
    else:
        # If Σ is provided explicitly, ignore sigma and mask
        pass

    # Radii (includes: l2_bal=default 'l2', l2_pre, sigma)
    radii = compute_line_radii(
        G,
        balance_mode=balance_mode,
        a=a_vec,
        metric_M=None,
        Sigma=Sigma,
    )

    # MC frequency via linearized propagation
    freq = monte_carlo_overload_linear(
        G,
        n_sim=n_sim,
        Sigma=Sigma,
        balance_mode=balance_mode,
        a=a_vec,
        disturb_mode=disturb_mode,
    )

    # Predicted overload probability from σ-radius
    # Same linear response as used in MC (G = H P)
    # predict_overload_probability returns array aligned to get_edges(G)
    p_pred = predict_overload_probability(
        G,
        balance_mode=balance_mode,
        a=a_vec,
        Sigma=Sigma,
    )

    # Optional N-1 effective radii (uses l2_bal inside)
    if do_n1:
        from ..metrics.n1 import effective_radii_n1

        radii_n1 = effective_radii_n1(G, balance_mode=balance_mode, a=a_vec)
    else:
        radii_n1 = None

    # Assemble DataFrame
    l2_bal = pd.Series(radii["l2"])  # balanced L2 is default "l2"
    df = (
        pd.DataFrame(
            {
                "line": [f"{u}-{v}" for (u, v) in l2_bal.index],
                "radius_l2_bal": l2_bal.values,
                "overload_freq": [freq[e] for e in l2_bal.index],
                "pred_overload": p_pred,
            }
        )
        .sort_values("radius_l2_bal")
        .reset_index(drop=True)
    )

    if "l2_pre" in radii:
        df["radius_l2_pre"] = pd.Series(radii["l2_pre"]).values

    if "sigma" in radii:
        rs = pd.Series(radii["sigma"])
        df["radius_sigma"] = rs.values

    # Correlations
    # Prefer σ-radius for correlation with frequency if available
    if "radius_sigma" in df.columns:
        r_rank = df["radius_sigma"].rank(ascending=True)
        label_r = "r_sigma"
    else:
        r_rank = df["radius_l2_bal"].rank(ascending=True)
        label_r = "r_l2_bal"

    f_rank = df["overload_freq"].rank(ascending=False)
    rho, pval = st.spearmanr(r_rank, f_rank)

    print(
        f"\n=== {name} | N={n_sim}, balance={balance_mode}, disturb={disturb_mode} ===\n"
        f"Spearman ρ ({label_r} vs freq) = {rho:.3f} (p = {pval:.3g})"
    )

    # AUC metrics
    if HAVE_SKLEARN:
        y_true = (df["overload_freq"] > 0.05).astype(int)
        # Smaller radius => higher risk
        if "radius_sigma" in df.columns:
            y_score = -df["radius_sigma"]
            auc_sigma = roc_auc_score(y_true, y_score)
            print(f"ROC-AUC (σ-radius) = {auc_sigma:.3f}")
        y_score_bal = -df["radius_l2_bal"]
        auc_bal = roc_auc_score(y_true, y_score_bal)
        print(f"ROC-AUC (L2_bal) = {auc_bal:.3f}")

    # Calibration sanity check: mean predicted vs empirical
    print(
        f"Mean predicted overload = {df['pred_overload'].mean():.4f}, "
        f"Mean empirical overload = {df['overload_freq'].mean():.4f}"
    )
    print(df)

    # Plots
    if make_plots:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Balanced L2 radius vs overload
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["radius_l2_bal"], df["overload_freq"], label="L2_bal radius")
        ax.set_xlabel("Balanced L2 stability radius (MW)")
        ax.set_ylabel("Overload frequency")
        ax.set_title(f"{name}: L2_bal radius vs overload")
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_dir / f"{name}_l2bal_vs_overload.png")
        plt.close(fig)

        # σ-radius vs overload (if present)
        if "radius_sigma" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(
                df["radius_sigma"],
                df["overload_freq"],
                color="tab:orange",
                label="σ-radius",
            )
            ax.set_xlabel("Probabilistic σ-radius (std)")
            ax.set_ylabel("Overload frequency")
            ax.set_title(f"{name}: σ-radius vs overload")
            ax.legend()
            plt.tight_layout()
            fig.savefig(out_dir / f"{name}_sigma_vs_overload.png")
            plt.close(fig)

        # Predicted vs empirical
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["pred_overload"], df["overload_freq"], color="tab:green")
        ax.set_xlabel("Predicted overload probability")
        ax.set_ylabel("Empirical overload frequency")
        ax.set_title(f"{name}: predicted vs empirical overload")
        plt.tight_layout()
        fig.savefig(out_dir / f"{name}_pred_vs_empirical.png")
        plt.close(fig)

        print(f"Plots saved → {str(out_dir)}")

    if do_n1 and radii_n1 is not None:
        df_n1 = pd.DataFrame(
            {
                "line": [f"{u}-{v}" for (u, v) in radii_n1["l2"].keys()],
                "radius_l2_n1": list(radii_n1["l2"].values()),
            }
        ).sort_values("radius_l2_n1")
        print("\nN-1 effective L2_bal radii:")
        print(df_n1)

    return df


def parse_a_vector(a_str: str, nodes: List[str]) -> np.ndarray:
    """
    Parse "Bus1:0.4,Bus2:0.6" into a vector aligned with nodes (sorted).
    Missing entries are zero. Not normalized (handled in balance map).
    """
    parts = [p.strip() for p in a_str.split(",") if p.strip()]
    mapping = {}
    for part in parts:
        name, val = part.split(":")
        mapping[name.strip()] = float(val)
    a = np.array([mapping.get(n, 0.0) for n in nodes], dtype=float)
    return a


def default_agc_participation(G, nodes: List[str]) -> np.ndarray:
    """
    Default AGC participation vector: proportional to positive generation at buses.
    """
    a = np.array([max(G.nodes[n]["generation"], 0.0) for n in nodes], dtype=float)
    if np.sum(a) <= 0:
        # Fallback to uniform over all nodes if no generators found
        a = np.ones(len(nodes), dtype=float)
    return a


def main():
    parser = argparse.ArgumentParser(
        description="Stability radii + Monte Carlo validation (DC)"
    )
    parser.add_argument(
        "--grids",
        nargs="+",
        default=["default", "radial", "meshed"],
        choices=list(GRID_FACTORY.keys()),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Noise scale if Sigma not provided (overridden by --auto-sigma)",
    )
    parser.add_argument("--n-sim", type=int, default=3000, help="Monte Carlo samples")
    parser.add_argument(
        "--balance",
        choices=["orth", "slack", "custom", "agc"],
        default="agc",
        help="Balance mode",
    )
    parser.add_argument(
        "--disturb",
        choices=["all", "load", "gen"],
        default="load",
        help="Which buses receive random disturbances",
    )
    parser.add_argument(
        "--a",
        type=str,
        default="",
        help='Participation vector, e.g., "Bus1:0.5,Bus4:0.5" (used with balance=custom or agc override)',
    )
    parser.add_argument("--n1", action="store_true", help="Compute N-1 effective radii")
    parser.add_argument("--plot", action="store_true", help="Make scatter plots")
    parser.add_argument(
        "--out", type=Path, default=Path("plots"), help="Output dir for plots"
    )
    parser.add_argument(
        "--auto-sigma",
        action="store_true",
        help="Auto-calibrate σ to hit target overload frequency (ignores --sigma)",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.1,
        help="Target mean overload frequency for --auto-sigma",
    )
    args = parser.parse_args()

    for gname in args.grids:
        G = GRID_FACTORY[gname]()
        nodes = sorted(G.nodes)
        # Participation vector
        a_vec = None
        if args.balance == "custom":
            if not args.a:
                raise SystemExit("balance=custom requires --a")
            a_vec = parse_a_vector(args.a, nodes)
        elif args.balance == "agc":
            a_vec = default_agc_participation(G, nodes)
            # Optional override via --a
            if args.a:
                a_vec = parse_a_vector(args.a, nodes)

        # Σ will be built in evaluate_grid from sigma and disturb mode unless provided explicitly
        Sigma = None

        evaluate_grid(
            G,
            name=gname,
            sigma=args.sigma,
            n_sim=args.n_sim,
            balance_mode=args.balance,
            a_vec=a_vec,
            disturb_mode=args.disturb,
            Sigma=Sigma,
            do_n1=args.n1,
            out_dir=args.out,
            make_plots=args.plot,
            auto_sigma=args.auto_sigma,
            target_overload=args.target,
        )


if __name__ == "__main__":
    main()
