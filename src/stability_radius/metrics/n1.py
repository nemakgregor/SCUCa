"""
N-1 effective radii and LODF helpers.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from ..powerflow.dc import (
    get_edges,
    get_nodes,
    compute_H_full,
    dc_flows,
    compute_ptdf_dict,
)
from .radius import (
    make_balance_map,
    compute_margins,
    minimal_norm_radius_l2_balanced,
    minimal_norm_radius_metric,
    probabilistic_sigma_radius,
)


def compute_lodf(
    G: nx.Graph, slack_name: str | None = None
) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
    """
    LODF mapping: LODF[(m, k)] = flow change on line m per pre-outage flow on k.
    Uses PTDF with the canonical formula:
      LODF_{m,k} = PTDF_{m,(u_k→v_k)} / (1 - PTDF_{k,(u_k→v_k)})
    Lines m and k are oriented as in G.edges().
    """
    nodes = get_nodes(G)
    if slack_name is None:
        slack_name = next(n for n in nodes if "Slack" in n)
    slack_idx = nodes.index(slack_name)

    edges = get_edges(G)
    ptdf = compute_ptdf_dict(G, nodes, slack_idx)
    lodf: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float] = {}

    for k in edges:
        uk, vk = k
        denom = 1.0 - ptdf[(uk, vk)][k]
        if abs(denom) < 1e-10:
            continue
        for m in edges:
            if m == k:
                continue
            lodf[(m, k)] = ptdf[(uk, vk)][m] / denom
    return lodf


def effective_radii_n1(
    G: nx.Graph,
    balance_mode: str = "orth",
    a: np.ndarray | None = None,
    slack_name: str | None = None,
    metric_M: np.ndarray | None = None,
    Sigma: np.ndarray | None = None,
) -> Dict[str, Dict[Tuple[str, str], float]]:
    """
    For each single-line outage, recompute H_full and margins and take the minimum radius.
    Returns dicts keyed like compute_line_radii but for N-1 "effective" radii.
    """
    nodes = get_nodes(G)
    if slack_name is None:
        slack_name = next(n for n in nodes if "Slack" in n)
    slack_idx = nodes.index(slack_name)
    n = len(nodes)

    P = make_balance_map(n, mode=balance_mode, a=a, slack_idx=slack_idx)
    edges = get_edges(G)

    # Initialize with +inf; we'll take min across contingencies
    r_l2 = {e: float("inf") for e in edges}
    r_M = {e: float("inf") for e in edges} if metric_M is not None else None
    r_S = {e: float("inf") for e in edges} if Sigma is not None else None

    for k in edges:
        Gk = G.copy()
        if not Gk.has_edge(*k):
            continue
        Gk.remove_edge(*k)
        try:
            Hk = compute_H_full(Gk, nodes, slack_idx)
            flows_k = dc_flows(Gk, nodes, slack_idx)
        except Exception:
            continue

        margins_k, edges_k = compute_margins(Gk, flows_k)

        # Balanced L2
        rk_l2 = minimal_norm_radius_l2_balanced(Hk, margins_k)
        for e, val in zip(edges_k, rk_l2):
            r_l2[e] = min(r_l2[e], val)

        # Metric
        if metric_M is not None:
            rk_M = minimal_norm_radius_metric(Hk, P, margins_k, metric_M)
            for e, val in zip(edges_k, rk_M):
                r_M[e] = min(r_M[e], val)

        # Sigma
        if Sigma is not None:
            rk_S = probabilistic_sigma_radius(Hk, P, margins_k, Sigma)
            for e, val in zip(edges_k, rk_S):
                r_S[e] = min(r_S[e], val)

    out = {"l2": r_l2}
    if r_M is not None:
        out["metric"] = r_M
    if r_S is not None:
        out["sigma"] = r_S
    return out
