"""
Monte Carlo overload tester aligned with balance projection and covariance Σ.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import networkx as nx

from ..powerflow.dc import get_nodes, get_edges, compute_H_full, dc_flows
from ..metrics.radius import make_balance_map, predict_overload_probability


def build_disturbance_mask(
    G: nx.Graph, nodes: List[str], mode: str = "all"
) -> np.ndarray:
    """
    Build a 0/1 mask over buses:
    - 'all':  all buses disturbed
    - 'load': buses with load > 0 disturbed
    - 'gen':  buses with generation > 0 disturbed
    """
    mode = mode.lower()
    if mode not in ("all", "load", "gen"):
        raise ValueError(f"Unknown disturb mode: {mode}")
    if mode == "all":
        return np.ones(len(nodes), dtype=int)
    if mode == "load":
        return np.array([1 if G.nodes[n]["load"] > 0 else 0 for n in nodes], dtype=int)
    if mode == "gen":
        return np.array(
            [1 if G.nodes[n]["generation"] > 0 else 0 for n in nodes], dtype=int
        )


def monte_carlo_overload_linear(
    G: nx.Graph,
    slack_name: str | None = None,
    n_sim: int = 3000,
    Sigma: np.ndarray | None = None,
    sigma: float | None = None,
    balance_mode: str = "orth",
    a: np.ndarray | None = None,
    disturb_mode: str = "all",
    seed: int | None = 7,
) -> Dict[Tuple[str, str], float]:
    """
    Linearized MC: draw δp ~ N(0, Σ) or Σ=σ^2 diag(mask), balance via P/P_a, and propagate using H_full.
    Returns empirical overload frequency per line.
    """
    rng = np.random.default_rng(seed)
    nodes = get_nodes(G)
    if slack_name is None:
        slack_name = next(n for n in nodes if "Slack" in n)
    slack_idx = nodes.index(slack_name)
    n = len(nodes)

    edges = get_edges(G)
    flows0 = dc_flows(G, nodes, slack_idx)
    caps = np.array([float(G[u][v]["capacity"]) for (u, v) in edges], dtype=float)

    H_full = compute_H_full(G, nodes, slack_idx)
    P = make_balance_map(n, mode=balance_mode, a=a, slack_idx=slack_idx)

    if Sigma is None:
        if sigma is None:
            sigma = 0.3
        mask = build_disturbance_mask(G, nodes, mode=disturb_mode).astype(float)
        Sigma = (sigma**2) * np.diag(mask)
    # Base flow vector
    f0 = np.array([flows0[e] for e in edges], dtype=float)
    counts = np.zeros(len(edges), dtype=int)

    # Draw using Cholesky or eig
    # For PSD Σ (mask may make it singular) fall back to eigen sampling
    try:
        L = np.linalg.cholesky(Sigma)
        chol_ok = True
    except np.linalg.LinAlgError:
        chol_ok = False
        w, V = np.linalg.eigh(Sigma)
        w = np.clip(w, 0.0, None)

    for _ in range(n_sim):
        if chol_ok:
            z = rng.normal(size=n)
            dp = L @ z
        else:
            z = rng.normal(size=n)
            dp = V @ (np.sqrt(w) * (V.T @ z))
        dp_bal = P @ dp
        f = f0 + H_full @ dp_bal
        counts += (np.abs(f) > caps).astype(int)

    freq = {e: counts[i] / float(n_sim) for i, e in enumerate(edges)}
    return freq


def monte_carlo_overload_dc(
    G: nx.Graph,
    slack_name: str | None = None,
    n_sim: int = 3000,
    Sigma: np.ndarray | None = None,
    sigma: float | None = None,
    balance_mode: str = "orth",
    a: np.ndarray | None = None,
    disturb_mode: str = "all",
    seed: int | None = 7,
) -> Dict[Tuple[str, str], float]:
    """
    Full DC MC: draw δp ~ N(0, Σ or σ^2 diag(mask)), balance via P/P_a to form p', solve DC PF.
    This re-solves DC per scenario; use for validation of linearization.
    """
    rng = np.random.default_rng(seed)
    nodes = get_nodes(G)
    if slack_name is None:
        slack_name = next(n for n in nodes if "Slack" in n)
    slack_idx = nodes.index(slack_name)
    n = len(nodes)

    edges = get_edges(G)
    caps = np.array([float(G[u][v]["capacity"]) for (u, v) in edges], dtype=float)
    P = make_balance_map(n, mode=balance_mode, a=a, slack_idx=slack_idx)

    if Sigma is None:
        if sigma is None:
            sigma = 0.3
        mask = build_disturbance_mask(G, nodes, mode=disturb_mode).astype(float)
        Sigma = (sigma**2) * np.diag(mask)

    # Precompute sampling factors
    try:
        L = np.linalg.cholesky(Sigma)
        chol_ok = True
    except np.linalg.LinAlgError:
        chol_ok = False
        w, V = np.linalg.eigh(Sigma)
        w = np.clip(w, 0.0, None)

    p0 = np.array(
        [G.nodes[k]["generation"] - G.nodes[k]["load"] for k in nodes], dtype=float
    )
    freq = np.zeros(len(edges), dtype=int)

    for _ in range(n_sim):
        if chol_ok:
            z = rng.normal(size=n)
            dp = L @ z
        else:
            z = rng.normal(size=n)
            dp = V @ (np.sqrt(w) * (V.T @ z))
        dp_bal = P @ dp
        p = p0 + dp_bal
        flows = dc_flows(G, nodes, slack_idx, p_full=p)
        f = np.array([flows[e] for e in edges], dtype=float)
        freq += (np.abs(f) > caps).astype(int)

    return {e: freq[i] / float(n_sim) for i, e in enumerate(edges)}


def calibrate_sigma_for_target(
    G: nx.Graph,
    balance_mode: str = "orth",
    a: np.ndarray | None = None,
    disturb_mode: str = "all",
    target: float = 0.1,
    tol: float = 1e-3,
    max_iter: int = 40,
    sigma_bounds: Tuple[float, float] = (1e-4, 1e3),
    slack_name: str | None = None,
) -> float:
    """
    Calibrate σ (assuming Σ = σ^2 diag(mask)) so that the mean predicted overload
    probability across lines is approximately target.
    Uses bisection on σ. Returns σ.
    """
    nodes = get_nodes(G)
    # Precompute mask
    mask = build_disturbance_mask(G, nodes, mode=disturb_mode).astype(float)

    def mean_pred(s):
        Sigma = (s**2) * np.diag(mask)
        p = predict_overload_probability(
            G, balance_mode=balance_mode, a=a, Sigma=Sigma, slack_name=slack_name
        )
        return float(np.mean(p))

    lo, hi = sigma_bounds
    f_lo = mean_pred(lo)
    f_hi = mean_pred(hi)

    # Expand hi if needed
    it_expand = 0
    while f_hi < target and it_expand < 10:
        hi *= 2.0
        f_hi = mean_pred(hi)
        it_expand += 1

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = mean_pred(mid)
        if abs(f_mid - target) <= tol:
            return mid
        if f_mid < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
