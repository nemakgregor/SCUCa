from typing import List, Tuple
from pathlib import Path

import numpy as np

from src.data_preparation.read_data import _read as read_generic
from src.data_preparation.data_structure import UnitCommitmentScenario
from src.stability_radius.io.uc_scenario import (
    compute_bus_injections,
    _infer_line_capacity,  # reuse from module
    _reactance_from_susceptance,  # reuse from module
)


def load_instance_from_case_dir(case_dir: Path):
    files = []
    files.extend(sorted(case_dir.glob("*.json.gz")))
    files.extend(sorted(case_dir.glob("*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON(.gz) files in {case_dir}")
    inst = read_generic([str(p) for p in files], quiet=True)
    return inst


def collect_injection_samples(
    scenarios: List[UnitCommitmentScenario],
    use_all_times: bool = True,
    t_select: int = 0,
) -> Tuple[List[str], np.ndarray]:
    """
    Return (nodes, P_samples) where P_samples is (n x S) with S = total samples.
    Each sample is p(t) = gen(t) - load(t) at bus level, without slack rebalancing.
    """
    # Determine canonical node ordering from the first scenario (sorted by bus name)
    first = scenarios[0]
    nodes = sorted([b.name for b in first.buses])
    n = len(nodes)
    samples = []

    for sc in scenarios:
        T = sc.time
        times = range(T) if use_all_times else [min(max(0, t_select), T - 1)]
        for t in times:
            names, p, _, _ = compute_bus_injections(sc, t)
            if names != nodes:
                raise ValueError(
                    "Bus sets/order differ across scenarios; ensure consistent case folder."
                )
            samples.append(p.reshape(n, 1))

    if not samples:
        raise ValueError("No samples collected from scenarios/time.")
    P_samples = np.hstack(samples)  # (n, S)
    return nodes, P_samples


def compute_baseline_from_samples(P_samples: np.ndarray) -> np.ndarray:
    """
    p̄ = mean of p samples across all scenarios and time steps.
    """
    return np.mean(P_samples, axis=1)


def _pick_slack_from_pbar(nodes: List[str], pbar: np.ndarray) -> str:
    # Choose bus with maximum positive injection; fallback to first
    idx = int(np.argmax(pbar))
    if pbar[idx] > 0:
        return nodes[idx]
    return nodes[0]


def build_baseline_graph_from_pbar(
    template_scenario: UnitCommitmentScenario,
    nodes: List[str],
    pbar: np.ndarray,
):
    """
    Build NetworkX graph with node attributes generation/load s.t. generation-load = p̄.
    Lines (reactance, capacity) are taken from the template_scenario (t=0).
    """
    import networkx as nx

    G = nx.Graph()
    # Nodes
    for i, name in enumerate(nodes):
        gen = float(max(pbar[i], 0.0))
        load = float(max(-pbar[i], 0.0))
        G.add_node(name, generation=gen, load=load)

    # Lines from template scenario at t=0
    for line in template_scenario.lines:
        u = line.source.name
        v = line.target.name
        x = _reactance_from_susceptance(getattr(line, "susceptance", 0.0))
        cap = _infer_line_capacity(line, t=0)
        G.add_edge(u, v, reactance=float(x), capacity=float(cap), name=line.name)

    slack_name = _pick_slack_from_pbar(nodes, pbar)
    return G, slack_name


def compute_empirical_covariance(P_samples: np.ndarray, pbar: np.ndarray) -> np.ndarray:
    """
    Σ̂ (pre-balance) from δp = p - p̄.
    """
    n, S = P_samples.shape
    D = P_samples - pbar.reshape(n, 1)
    # Center (should be nearly zero already, but do it robustly)
    mu = np.mean(D, axis=1, keepdims=True)
    Dc = D - mu
    # Sample covariance
    if S <= 1:
        return np.zeros((n, n), dtype=float)
    Sigma = (Dc @ Dc.T) / float(S - 1)
    return Sigma


def default_agc_from_pbar(nodes: List[str], pbar: np.ndarray) -> np.ndarray:
    a = np.maximum(pbar, 0.0)
    if np.sum(a) <= 0:
        a = np.ones_like(a)
    return a
