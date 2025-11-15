"""
DC power-flow utilities: B-matrix, PTDF, H matrix, and flows
All line orientations follow nx.Graph edges as-inserted.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import scipy.linalg as la


def get_nodes(G: nx.Graph) -> List[str]:
    """Stable list of nodes; sorted for reproducibility."""
    return sorted(G.nodes)


def get_edges(G: nx.Graph) -> List[Tuple[str, str]]:
    """Stable list of oriented edges as stored by NetworkX iteration."""
    return [(u, v) for (u, v) in G.edges]


def node_index(nodes: List[str]) -> Dict[str, int]:
    return {n: i for i, n in enumerate(nodes)}


def build_B(G: nx.Graph, nodes: List[str]) -> np.ndarray:
    """Build nodal susceptance matrix B (DC)."""
    n = len(nodes)
    B = np.zeros((n, n), dtype=float)
    idx = node_index(nodes)
    for u, v, dat in G.edges(data=True):
        b = 1.0 / float(dat["reactance"])
        i, j = idx[u], idx[v]
        B[i, i] += b
        B[j, j] += b
        B[i, j] -= b
        B[j, i] -= b
    return B


def incidence_rows(
    G: nx.Graph, nodes: List[str], edges: List[Tuple[str, str]]
) -> np.ndarray:
    """
    Build E with rows r: e_u^T - e_v^T for oriented edge (u, v).
    Shape: (m, n).
    """
    n = len(nodes)
    E = np.zeros((len(edges), n), dtype=float)
    idx = node_index(nodes)
    for r, (u, v) in enumerate(edges):
        E[r, idx[u]] = 1.0
        E[r, idx[v]] = -1.0
    return E


def line_susceptances(G: nx.Graph, edges: List[Tuple[str, str]]) -> np.ndarray:
    """Vector b_ℓ for each oriented edge."""
    b = np.array([1.0 / float(G[u][v]["reactance"]) for (u, v) in edges], dtype=float)
    return b


def k_matrix(G: nx.Graph, nodes: List[str], edges: List[Tuple[str, str]]) -> np.ndarray:
    """
    K maps bus angles θ to line flows: f = K θ, with K = diag(b) @ E.
    Shape: (m, n).
    """
    E = incidence_rows(G, nodes, edges)
    b = line_susceptances(G, edges)
    return np.diag(b) @ E


def _selector_non_slack(n: int, slack_idx: int) -> np.ndarray:
    """
    Selection matrix R that picks non-slack components: p_ns = R p_full.
    Shape: (n-1, n).
    """
    rows = [i for i in range(n) if i != slack_idx]
    R = np.zeros((n - 1, n), dtype=float)
    for r, i in enumerate(rows):
        R[r, i] = 1.0
    return R


def compute_theta(
    G: nx.Graph, nodes: List[str], slack_idx: int, p_full: np.ndarray | None = None
) -> np.ndarray:
    """
    Solve B θ = p with θ_slack = 0.
    If p_full is None, p_full is taken from node (generation - load).
    Returns full θ with slack angle at 0. Shape: (n,).
    """
    n = len(nodes)
    if p_full is None:
        p_full = np.array(
            [G.nodes[n]["generation"] - G.nodes[n]["load"] for n in nodes], dtype=float
        )

    # Form reduced system
    B = build_B(G, nodes)
    R = _selector_non_slack(n, slack_idx)  # (n-1, n)
    B_red = B[
        np.ix_(
            [i for i in range(n) if i != slack_idx],
            [i for i in range(n) if i != slack_idx],
        )
    ]
    p_ns = R @ p_full
    theta_ns = la.solve(B_red, p_ns, assume_a="sym")  # SPD
    theta = np.zeros(n, dtype=float)
    theta[[i for i in range(n) if i != slack_idx]] = theta_ns
    return theta


def dc_flows(
    G: nx.Graph, nodes: List[str], slack_idx: int, p_full: np.ndarray | None = None
) -> Dict[Tuple[str, str], float]:
    """Compute DC line flows f_ℓ = b_ℓ (θ_u - θ_v) for oriented edges."""
    edges = get_edges(G)
    theta = compute_theta(G, nodes, slack_idx, p_full=p_full)
    idx = node_index(nodes)
    flows = {
        (u, v): (1.0 / float(G[u][v]["reactance"])) * (theta[idx[u]] - theta[idx[v]])
        for (u, v) in edges
    }
    return flows


def compute_H_full(G: nx.Graph, nodes: List[str], slack_idx: int) -> np.ndarray:
    """
    Build the full sensitivity H_full mapping balanced bus injections to line flows:
    f = H_full p_balanced, where 1^T p_balanced = 0.
    Internally uses a slack to invert B_red, but the result is valid for any balanced p.
    Shape: (m, n) with rows in the same order as get_edges(G).
    """
    n = len(nodes)
    edges = get_edges(G)
    K = k_matrix(G, nodes, edges)  # (m, n)
    B = build_B(G, nodes)
    non_slack = [i for i in range(n) if i != slack_idx]
    B_red = B[np.ix_(non_slack, non_slack)]
    B_red_inv = la.inv(B_red)
    R = _selector_non_slack(n, slack_idx)  # (n-1, n)
    # f = K [θ_ns; 0] = K R^T θ_ns = K R^T B_red^{-1} p_ns = K R^T B_red^{-1} R p_balanced
    H_full = K @ R.T @ B_red_inv @ R  # (m, n)
    return H_full


def compute_ptdf_dict(
    G: nx.Graph, nodes: List[str], slack_idx: int
) -> Dict[Tuple[str, str], Dict[Tuple[str, str], float]]:
    """
    PTDF in reduced space: for every (i→j) and line ℓ=(u,v),
    PTDF_{ℓ,(i→j)} = b_ℓ (e_u-e_v)^T B_red^{-1} (e_i - e_j).
    Returned as nested dict: ptdf[(i,j)][(u,v)].
    """
    n = len(nodes)
    edges = get_edges(G)
    idx = node_index(nodes)
    non_slack = [k for k in range(n) if k != slack_idx]
    B = build_B(G, nodes)
    B_red = B[np.ix_(non_slack, non_slack)]
    B_red_inv = la.inv(B_red)

    # Build X (n x n) with reduced inverse embedded; slack row/col are zeros
    X = np.zeros((n, n), dtype=float)
    for a, i in enumerate(non_slack):
        for b, j in enumerate(non_slack):
            X[i, j] = B_red_inv[a, b]

    ptdf: Dict[Tuple[str, str], Dict[Tuple[str, str], float]] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            key = (nodes[i], nodes[j])
            # For injection at i and withdrawal at j
            ptdf[key] = {}
            for u, v in edges:
                b_uv = 1.0 / float(G[u][v]["reactance"])
                ptdf[key][(u, v)] = b_uv * (
                    X[idx[u], i] - X[idx[u], j] - X[idx[v], i] + X[idx[v], j]
                )
    return ptdf


def compute_ptdf_to_slack_matrix(
    G: nx.Graph, nodes: List[str], slack_idx: int
) -> np.ndarray:
    """
    Efficient PTDF-to-slack matrix:
      PTDF_slack[:, i] = PTDF for unit injection at bus i and withdrawal at slack.
    Using H_full and columns v_i = e_i - e_slack:
      PTDF_slack = H_full @ (I - e_s 1^T)
    The 'slack' column is zero.
    Shape: (m, n).
    """
    H_full = compute_H_full(G, nodes, slack_idx)
    n = len(nodes)
    I = np.eye(n)
    e_s = np.zeros((n, 1))
    e_s[slack_idx, 0] = 1.0
    V = I - (e_s @ np.ones((1, n)))
    return H_full @ V
