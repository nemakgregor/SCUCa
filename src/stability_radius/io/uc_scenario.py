from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from src.data_preparation.data_structure import (
    UnitCommitmentScenario,
    Bus,
    ThermalUnit,
    ProfiledUnit,
    StorageUnit,
    TransmissionLine,
    PriceSensitiveLoad,
)


# Small utility to read a time-series value safely
def _val_at(series: Sequence, t: int, default: float = 0.0) -> float:
    if series is None:
        return float(default)
    try:
        if len(series) == 0:
            return float(default)
        if t < 0:
            t = 0
        if t >= len(series):
            return float(series[-1])
        return float(series[t])
    except Exception:
        try:
            return float(series)
        except Exception:
            return float(default)


def _infer_line_capacity(
    line: TransmissionLine, t: int, fallback: float = 1e6
) -> float:
    cap = _val_at(getattr(line, "normal_limit", None), t, None)
    if cap is None or cap <= 0:
        cap = _val_at(getattr(line, "emergency_limit", None), t, None)
    if cap is None or cap <= 0:
        cap = fallback
    return float(cap)


def _reactance_from_susceptance(susceptance: float) -> float:
    try:
        b = float(susceptance)
        if b > 1e-12:
            return 1.0 / b
    except Exception:
        pass
    # Fallback reactance
    return 0.1


def _sum_bus_psl(bus: Bus, t: int) -> float:
    # Sum price-sensitive loads attached to this bus (if any)
    total = 0.0
    try:
        for psl in bus.price_sensitive_loads:
            total += _val_at(getattr(psl, "demand", None), t, 0.0)
    except Exception:
        pass
    return total


def _sum_bus_profiled(bus: Bus, t: int) -> float:
    # Profiled units often have min == max == scheduled output.
    total = 0.0
    try:
        for pu in bus.profiled_units:
            mp = _val_at(getattr(pu, "min_power", None), t, 0.0)
            mx = _val_at(getattr(pu, "max_power", None), t, mp)
            # Use min_power when available (often equals actual output)
            g = mp if mp > 0 else mx
            total += max(0.0, g)
    except Exception:
        pass
    return total


def _sum_bus_thermal(bus: Bus, t: int) -> float:
    """
    Baseline thermal generation:
      - if initial_power present: use it (clamped to [min, max])
      - elif must_run[t] > 0: use min_power[t]
      - else: 0
    """
    total = 0.0
    try:
        for tu in bus.thermal_units:
            minp = _val_at(getattr(tu, "min_power", None), t, 0.0)
            maxp = _val_at(getattr(tu, "max_power", None), t, minp)
            must = _val_at(getattr(tu, "must_run", None), t, 0.0)

            p = 0.0
            ip = getattr(tu, "initial_power", None)
            if ip is not None:
                p = float(ip)
                if maxp >= minp:
                    p = min(maxp, max(minp, p))
            else:
                # If must-run at t, use min_power
                if must and must > 0.5:
                    p = max(0.0, minp)
                else:
                    p = 0.0
            total += max(0.0, p)
    except Exception:
        pass
    return total


def _sum_bus_storage(bus: Bus, t: int) -> float:
    # For base injection we ignore storage by default (no net contribution).
    return 0.0


def _pick_slack_bus(
    bus_names: List[str],
    generation_by_bus: Dict[str, float],
    bus_index_order: Dict[str, int],
) -> str:
    if not bus_names:
        raise ValueError("No buses available to pick a slack bus")
    # Prefer the bus with maximum positive generation; fallback to smallest index
    best = None
    best_g = -1.0
    for b in bus_names:
        g = generation_by_bus.get(b, 0.0)
        if g > best_g:
            best_g = g
            best = b
    if best is not None and best_g > 0.0:
        return best
    # Fallback: lowest index
    return min(bus_names, key=lambda x: bus_index_order.get(x, 10**9))


def scenario_to_nx_graph(
    scenario: UnitCommitmentScenario,
    t: int = 0,
    choose_slack: Optional[str] = None,
) -> Tuple[nx.Graph, str]:
    """
    Convert a UnitCommitmentScenario at time index t into a NetworkX graph compatible with
    stability_radius power-flow and metrics modules.

    Returns (G, slack_bus_name).
    Node attributes: generation (MW), load (MW).
    Edge attributes: reactance (p.u. or consistent), capacity (MW).
    """
    G = nx.Graph()

    # Buses
    bus_index_order = {b.name: b.index for b in scenario.buses}
    bus_names = [b.name for b in scenario.buses]

    generation_by_bus: Dict[str, float] = {}
    load_by_bus: Dict[str, float] = {}

    for bus in scenario.buses:
        load_val = _val_at(bus.load, t, 0.0) + _sum_bus_psl(bus, t)
        gen_val = (
            _sum_bus_thermal(bus, t)
            + _sum_bus_profiled(bus, t)
            + _sum_bus_storage(bus, t)
        )
        generation_by_bus[bus.name] = gen_val
        load_by_bus[bus.name] = load_val

    # Slack selection and balance
    if choose_slack is None:
        slack_name = _pick_slack_bus(bus_names, generation_by_bus, bus_index_order)
    else:
        slack_name = choose_slack

    total_gen = sum(generation_by_bus.values())
    total_load = sum(load_by_bus.values())
    imbalance = total_load - total_gen  # if positive => need more gen at slack

    generation_by_bus[slack_name] = generation_by_bus.get(slack_name, 0.0) + max(
        0.0, imbalance
    )
    # If generation exceeds load overall (negative imbalance), model as dummy load at slack:
    if imbalance < 0:
        load_by_bus[slack_name] = load_by_bus.get(slack_name, 0.0) + (-imbalance)

    # Create nodes
    for name in bus_names:
        G.add_node(
            name,
            generation=float(generation_by_bus.get(name, 0.0)),
            load=float(load_by_bus.get(name, 0.0)),
        )

    # Lines
    for line in scenario.lines:
        u = line.source.name
        v = line.target.name
        x = _reactance_from_susceptance(getattr(line, "susceptance", 0.0))
        cap = _infer_line_capacity(line, t)
        # Ensure no duplicate attribute addition merges multiple lines (case14 typically has no parallel lines)
        G.add_edge(u, v, reactance=float(x), capacity=float(cap), name=line.name)

    return G, slack_name


def agc_vector_from_scenario(
    scenario: UnitCommitmentScenario, nodes: List[str], t: int = 0
) -> np.ndarray:
    """
    Default AGC participation vector: proportional to positive generation at buses
    after the baseline construction used in scenario_to_nx_graph (without the slack correction).
    """
    # Build generation without slack correction
    gen_map: Dict[str, float] = {}
    bus_by_name = {b.name: b for b in scenario.buses}

    for name in nodes:
        bus = bus_by_name.get(name)
        if bus is None:
            gen_map[name] = 0.0
            continue
        g = (
            _sum_bus_thermal(bus, t)
            + _sum_bus_profiled(bus, t)
            + _sum_bus_storage(bus, t)
        )
        gen_map[name] = max(0.0, g)

    a = np.array([gen_map.get(n, 0.0) for n in nodes], dtype=float)
    if np.sum(a) <= 0:
        a = np.ones(len(nodes), dtype=float)
    return a


def compute_bus_injections(
    scenario: UnitCommitmentScenario, t: int = 0
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-bus injections without slack balancing:
      p_i(t) = generation_i(t) - load_i(t).
    Returns (nodes_sorted, p(t), gen(t), load(t)), all aligned to nodes_sorted.
    """
    bus_by_name = {b.name: b for b in scenario.buses}
    nodes = sorted(bus_by_name.keys())
    gen = np.zeros(len(nodes), dtype=float)
    load = np.zeros(len(nodes), dtype=float)
    for i, name in enumerate(nodes):
        bus = bus_by_name[name]
        g = (
            _sum_bus_thermal(bus, t)
            + _sum_bus_profiled(bus, t)
            + _sum_bus_storage(bus, t)
        )
        l = _val_at(bus.load, t, 0.0) + _sum_bus_psl(bus, t)
        gen[i] = float(g)
        load[i] = float(l)
    p = gen - load
    return nodes, p, gen, load