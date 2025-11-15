"""
Stability radii: minimal L2, metric-weighted, and probabilistic σ-radius.
Supports orthogonal balance projection and generator participation map P_a.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import scipy.linalg as la
import scipy.stats as stats
import networkx as nx

from ..powerflow.dc import get_edges, get_nodes, compute_H_full, dc_flows


def orth_projector(n: int) -> np.ndarray:
    """Orthogonal projector onto the balance hyperplane {δ: 1^T δ = 0}."""
    I = np.eye(n, dtype=float)
    one = np.ones((n, 1), dtype=float)
    return I - (one @ one.T) / float(n)


def make_balance_map(
    n: int,
    mode: str = "orth",
    a: np.ndarray | None = None,
    slack_idx: int | None = None,
) -> np.ndarray:
    """
    Build balance mapping P:
    - mode="orth":  P = I - 11^T / n         (orthogonal projector onto {δ: 1^T δ = 0})
    - mode="slack": P = I - e_s 1^T          (slack-only compensation at index s)
    - mode="custom"/"agc": P = I - a 1^T / (1^T a)  (distributed compensation via participation a)
      a must have positive sum; not required to sum to 1 (we normalize internally).
    """
    I = np.eye(n, dtype=float)
    one = np.ones((n, 1), dtype=float)

    if mode == "orth":
        return orth_projector(n)

    if mode == "slack":
        if slack_idx is None:
            raise ValueError("mode='slack' requires slack_idx")
        e = np.zeros((n, 1))
        e[slack_idx, 0] = 1.0
        return I - (e @ one.T)

    if mode in ("custom", "agc"):
        if a is None:
            raise ValueError("mode='custom' or 'agc' requires a participation vector a")
        a = np.asarray(a, dtype=float).reshape(-1, 1)
        s = float(np.sum(a))
        if s <= 0:
            raise ValueError("Participation vector a must have positive sum")
        return I - (a @ one.T) / s

    raise ValueError(f"Unknown balance mode: {mode}")


def compute_margins(
    G: nx.Graph, flows: Dict[Tuple[str, str], float]
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    edges = get_edges(G)
    margin = np.array(
        [float(G[u][v]["capacity"]) - abs(flows[(u, v)]) for (u, v) in edges],
        dtype=float,
    )
    # Negative margins -> set to zero (already overloaded)
    margin = np.maximum(margin, 0.0)
    return margin, edges


def minimal_norm_radius_l2_balanced(
    H_full: np.ndarray, margins: np.ndarray
) -> np.ndarray:
    """
    Balanced L2 minimal-norm radius (default):
    r_ℓ = margin_ℓ / ||row_ℓ(H_full P_orth)||_2,
    where P_orth is the orthogonal projector onto {δ: 1^T δ = 0}.
    This is the minimal norm of balanced-injection change δp_bal to hit the limit.
    """
    n = H_full.shape[1]
    P_orth = orth_projector(n)
    Gbal = H_full @ P_orth
    g_norms = np.linalg.norm(Gbal, axis=1)
    g_norms = np.where(g_norms < 1e-12, np.inf, g_norms)
    r = margins / g_norms
    return np.maximum(r, 0.0)


def minimal_norm_radius_l2_preimage(
    H_full: np.ndarray, P: np.ndarray, margins: np.ndarray
) -> np.ndarray:
    """
    Pre-image L2 minimal-norm radius (legacy):
    For each line ℓ, r_ℓ = margin_ℓ / ||g_ℓ||_2, where g_ℓ = row_ℓ(H_full P).
    This minimizes ||δp|| before balancing (δp_bal = P δp).
    """
    Gmat = H_full @ P  # (m, n)
    g_norms = np.linalg.norm(Gmat, axis=1)
    g_norms = np.where(g_norms < 1e-12, np.inf, g_norms)
    r = margins / g_norms
    return np.maximum(r, 0.0)


def minimal_norm_radius_metric(
    H_full: np.ndarray, P: np.ndarray, margins: np.ndarray, M: np.ndarray
) -> np.ndarray:
    """
    Metric-weighted minimal-norm radius with positive-definite M in the pre-image space:
    r_ℓ = margin_ℓ / sqrt(g_ℓ M^{-1} g_ℓ^T), where g_ℓ = row_ℓ(H_full P).
    """
    n = P.shape[0]
    if M.shape != (n, n):
        raise ValueError("M must be (n x n) where n is the number of buses")
    M_inv = la.pinvh(M)
    Gmat = H_full @ P  # (m, n)
    q = np.einsum("ij,jk,ik->i", Gmat, M_inv, Gmat, optimize=True)
    q = np.where(q < 1e-16, np.inf, q)
    r = margins / np.sqrt(q)
    return np.maximum(r, 0.0)


def probabilistic_sigma_radius(
    H_full: np.ndarray, P: np.ndarray, margins: np.ndarray, Sigma: np.ndarray
) -> np.ndarray:
    """
    σ-radius under Gaussian δp ~ N(0, Σ), with flow sensitivity g_ℓ = row_ℓ(H_full P):
    r_ℓ^(Σ) = margin_ℓ / sqrt(g_ℓ Σ g_ℓ^T)
    This is dimensionless (# of standard deviations to the limit).
    """
    n = P.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError("Sigma must be (n x n)")
    Gmat = H_full @ P
    var = np.einsum("ij,jk,ik->i", Gmat, Sigma, Gmat, optimize=True)
    var = np.where(var < 1e-16, np.inf, var)
    r = margins / np.sqrt(var)
    return np.maximum(r, 0.0)


def predict_overload_probability(
    G: nx.Graph,
    balance_mode: str = "orth",
    a: np.ndarray | None = None,
    slack_name: str | None = None,
    Sigma: np.ndarray | None = None,
) -> np.ndarray:
    """
    Predict two-sided overload probability per line under Gaussian perturbations,
    correctly accounting for nonzero base flow f0:

      Given Δf_ℓ ~ N(0, σ_ℓ^2) with σ_ℓ^2 = g_ℓ Σ g_ℓ^T and base flow f0_ℓ, capacity c_ℓ,
      the overload event is |f0_ℓ + Δf_ℓ| > c_ℓ. Its probability is

        p_ℓ = Q((c_ℓ - |f0_ℓ|)/σ_ℓ) + Q((c_ℓ + |f0_ℓ|)/σ_ℓ),

      where Q(x) = 1 - Φ(x). The symmetric 2(1-Φ(m_ℓ/σ_ℓ)) is valid only if f0_ℓ = 0.

    Returns a vector aligned with get_edges(G).
    """
    nodes = get_nodes(G)
    if slack_name is None:
        slack_name = next(n for n in nodes if "Slack" in n)
    slack_idx = nodes.index(slack_name)

    H_full = compute_H_full(G, nodes, slack_idx)
    flows0 = dc_flows(G, nodes, slack_idx)
    margins, edges = compute_margins(G, flows0)

    n = len(nodes)
    if Sigma is None:
        Sigma = np.eye(n, dtype=float)
    P = make_balance_map(n, mode=balance_mode, a=a, slack_idx=slack_idx)

    # Sensitivity in pre-image space and variance of Δf
    Gmat = H_full @ P  # (m, n)
    var = np.einsum("ij,jk,ik->i", Gmat, Sigma, Gmat, optimize=True)
    var = np.clip(var, 0.0, None)
    std = np.sqrt(var)
    std = np.where(std < 1e-12, np.inf, std)  # avoid divide-by-zero

    caps = np.array([float(G[u][v]["capacity"]) for (u, v) in edges], dtype=float)
    f0 = np.array([flows0[e] for e in edges], dtype=float)
    abs_f0 = np.abs(f0)

    # Two asymmetric thresholds with nonzero base flow
    t1 = np.maximum(caps - abs_f0, 0.0)  # "near" side (equals margins)
    t2 = caps + abs_f0  # "far" side

    r1 = t1 / std
    r2 = t2 / std

    # p = Q(r1) + Q(r2), with Q=1-Φ (use survival function for stability)
    p = stats.norm.sf(r1) + stats.norm.sf(r2)
    return np.clip(p, 0.0, 1.0)


def compute_line_radii(
    G: nx.Graph,
    balance_mode: str = "orth",
    a: np.ndarray | None = None,
    slack_name: str | None = None,
    metric_M: np.ndarray | None = None,
    Sigma: np.ndarray | None = None,
) -> Dict[str, Dict[Tuple[str, str], float]]:
    """
    High-level convenience:
    - builds H_full
    - selects P or P_a
    - computes base flows and margins
    - returns radii as dicts:
      {
        "l2": balanced L2 radius,
        "l2_pre": pre-image L2 radius (legacy; optional),
        "metric": metric-weighted pre-image radius (optional),
        "sigma": σ-radius (optional)
      }
    """
    nodes = get_nodes(G)
    if slack_name is None:
        slack_name = next(n for n in nodes if "Slack" in n)
    slack_idx = nodes.index(slack_name)

    H_full = compute_H_full(G, nodes, slack_idx)
    flows0 = dc_flows(G, nodes, slack_idx)
    margins, edges = compute_margins(G, flows0)

    n = len(nodes)
    if balance_mode in ("custom", "agc") and a is None:
        raise ValueError("balance_mode='custom'/'agc' requires participation vector a")
    P = make_balance_map(n, mode=balance_mode, a=a, slack_idx=slack_idx)

    out: Dict[str, Dict[Tuple[str, str], float]] = {}

    # Balanced L2 (default)
    r_l2_bal = minimal_norm_radius_l2_balanced(H_full, margins)
    out["l2"] = dict(zip(edges, r_l2_bal))

    # Pre-image L2 (legacy)
    r_l2_pre = minimal_norm_radius_l2_preimage(H_full, P, margins)
    out["l2_pre"] = dict(zip(edges, r_l2_pre))

    # Metric-weighted (if provided)
    if metric_M is not None:
        r_M = minimal_norm_radius_metric(H_full, P, margins, metric_M)
        out["metric"] = dict(zip(edges, r_M))

    # Probabilistic σ-radius (if Sigma provided)
    if Sigma is not None:
        r_sig = probabilistic_sigma_radius(H_full, P, margins, Sigma)
        out["sigma"] = dict(zip(edges, r_sig))

    return out
