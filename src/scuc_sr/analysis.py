from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.stability_radius.io.uc_scenario import (
    scenario_to_nx_graph,
    agc_vector_from_scenario,
)
from src.stability_radius.metrics.radius import compute_line_radii
from src.stability_radius.powerflow.dc import get_nodes


def _edge_key(u: str, v: str) -> str:
    a, b = sorted((str(u), str(v)))
    return f"{a}-{b}"


def radii_for_scenario(
    scenario,
    *,
    t: int = 0,
    balance: str = "agc",
    sigma_sr: float = 0.3,
) -> Dict[str, Dict[str, float]]:
    G, slack = scenario_to_nx_graph(scenario, t=t)
    nodes = get_nodes(G)
    a_vec = agc_vector_from_scenario(scenario, nodes, t=t) if balance == "agc" else None

    Sigma = None
    if sigma_sr and float(sigma_sr) > 0:
        n = len(nodes)
        Sigma = (float(sigma_sr) ** 2) * np.eye(n, dtype=float)

    radii = compute_line_radii(
        G,
        balance_mode=balance,
        a=a_vec,
        slack_name=slack,
        Sigma=Sigma,
    )
    out: Dict[str, Dict[str, float]] = {}
    for k, d in radii.items():
        out[k] = {_edge_key(u, v): float(val) for (u, v), val in d.items()}
    return out


def aggregate_radii_over_instances(
    scenarios: List, *, t: int = 0, balance: str = "agc", sigma_sr: float = 0.3
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate radii across scenarios. Returns both mean (legacy) and min:
      - 'l2'        -> mean balanced L2
      - 'sigma'     -> mean σ-radius (if computed)
      - 'l2_pre'    -> mean pre-image L2 (legacy)
      - 'l2_min'    -> min balanced L2
      - 'sigma_min' -> min σ-radius (if computed)
    """
    sums: Dict[str, Dict[str, float]] = {"l2": {}, "sigma": {}, "l2_pre": {}}
    counts: Dict[str, Dict[str, int]] = {"l2": {}, "sigma": {}, "l2_pre": {}}
    mins: Dict[str, Dict[str, float]] = {"l2_min": {}, "sigma_min": {}}

    for sc in scenarios:
        rd = radii_for_scenario(sc, t=t, balance=balance, sigma_sr=sigma_sr)
        for k, d in rd.items():
            for ek, v in d.items():
                sums.setdefault(k, {})
                counts.setdefault(k, {})
                sums[k][ek] = sums[k].get(ek, 0.0) + float(v)
                counts[k][ek] = counts[k].get(ek, 0) + 1
                if k == "l2":
                    mins["l2_min"][ek] = min(
                        mins["l2_min"].get(ek, float("inf")), float(v)
                    )
                if k == "sigma":
                    mins["sigma_min"][ek] = min(
                        mins["sigma_min"].get(ek, float("inf")), float(v)
                    )

    out: Dict[str, Dict[str, float]] = {}
    for k in ("l2", "sigma", "l2_pre"):
        out[k] = {}
        for ek, s in sums.get(k, {}).items():
            c = max(1, counts[k].get(ek, 1))
            out[k][ek] = s / c

    # Add minima (if any were computed)
    for k in ("l2_min", "sigma_min"):
        if mins.get(k):
            out[k] = mins[k]
    return out


def build_joined_df(
    agg_counts: Dict[str, Dict[str, int]],
    agg_radii: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Join aggregated counts and radii by line_key ('u-v'); include mean and min radii.
    """
    keys = sorted(
        set().union(
            agg_counts.get("prune_events_calls", {}).keys(),
            agg_counts.get("prune_events_pruned", {}).keys(),
            agg_counts.get("lazy_events_potential", {}).keys(),
            agg_counts.get("lazy_events_dropped", {}).keys(),
            agg_counts.get("prune_line_constraints_added", {}).keys(),
            agg_counts.get("lazy_line_constraints_added", {}).keys(),
            agg_radii.get("l2", {}).keys(),
            agg_radii.get("sigma", {}).keys(),
            agg_radii.get("l2_min", {}).keys(),
            agg_radii.get("sigma_min", {}).keys(),
        )
    )

    rows = []
    for ek in keys:
        pe = int(agg_counts.get("prune_events_calls", {}).get(ek, 0))
        pp = int(agg_counts.get("prune_events_pruned", {}).get(ek, 0))
        lp = int(agg_counts.get("lazy_events_potential", {}).get(ek, 0))
        ld = int(agg_counts.get("lazy_events_dropped", {}).get(ek, 0))
        pr_added = int(agg_counts.get("prune_line_constraints_added", {}).get(ek, 0))
        la_added = int(agg_counts.get("lazy_line_constraints_added", {}).get(ek, 0))

        r_l2_mean = float(agg_radii.get("l2", {}).get(ek, np.nan))
        r_sig_mean = float(agg_radii.get("sigma", {}).get(ek, np.nan))
        r_l2_min = float(agg_radii.get("l2_min", {}).get(ek, np.nan))
        r_sig_min = float(agg_radii.get("sigma_min", {}).get(ek, np.nan))

        prune_ratio = (pp / pe) if pe > 0 else np.nan
        lazy_ratio = (ld / lp) if lp > 0 else np.nan

        rows.append(
            {
                "line_key": ek,
                "prune_events_calls": pe,
                "prune_events_pruned": pp,
                "prune_drop_ratio": prune_ratio,
                "lazy_events_potential": lp,
                "lazy_events_dropped": ld,
                "lazy_drop_ratio": lazy_ratio,
                "prune_line_constraints_added": pr_added,
                "lazy_line_constraints_added": la_added,
                "radius_l2_bal_mean": r_l2_mean,
                "radius_sigma_mean": r_sig_mean,
                "radius_l2_bal_min": r_l2_min,
                "radius_sigma_min": r_sig_min,
            }
        )

    df = pd.DataFrame(rows)
    return df
