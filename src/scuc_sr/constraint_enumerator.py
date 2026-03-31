"""
Optimized Enumerator:
Pre-computes topology matrices ONCE instead of re-doing it for every time step.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from src.stability_radius.io.uc_scenario import scenario_to_nx_graph, _val_at
from src.stability_radius.metrics.n1 import compute_lodf
from src.stability_radius.powerflow.dc import compute_ptdf_to_slack_matrix


@dataclass
class ConstraintEvent:
    kind: str
    line_name: str
    out_name: str
    t: int
    coeff: float
    F_em: float


def enumerate_potential_constraints(
    scenario, lodf_tol: float = 1e-4, isf_tol: float = 1e-8
) -> List[ConstraintEvent]:
    """
    Builds the N-1 constraint set.
    Efficiently loops Time(Inner) to leverage constant topology.
    """
    events = []
    lines = list(scenario.lines or [])
    contingencies = list(scenario.contingencies or [])
    T = int(scenario.time)

    # 1. Build Topology / Matrices (using t=0 geometry)
    G, _ = scenario_to_nx_graph(scenario, t=0)

    ordered_buses = sorted(scenario.buses, key=lambda b: int(b.index))
    nodes = [b.name for b in ordered_buses]

    # Determine slack consistent with optimization
    ref_bus_1b = getattr(scenario, "ptdf_ref_bus_index", ordered_buses[0].index)
    slack_name = next(
        (b.name for b in ordered_buses if b.index == ref_bus_1b), ordered_buses[0].name
    )
    slack_idx = nodes.index(slack_name)

    # Compute Matrices
    try:
        lodf_dict = compute_lodf(G, slack_name=slack_name)
    except:
        lodf_dict = {}

    try:
        PTS = compute_ptdf_to_slack_matrix(G, nodes, slack_idx)
        col_by_bus = {n: i for i, n in enumerate(nodes)}
    except:
        PTS = None
        col_by_bus = {}

    # Map: line_name -> (u,v) and row index
    row_by_line = {}
    for r, (u, v, dat) in enumerate(G.edges(data=True)):
        if "name" in dat:
            row_by_line[str(dat["name"])] = r

    line_uv = {ln.name: (ln.source.name, ln.target.name) for ln in lines}

    # 2. Enumerate Line Outages
    for cont in contingencies:
        if not cont.lines:
            continue
        for out_line in cont.lines:
            out_key = line_uv.get(out_line.name)
            if not out_key:
                continue

            for mon_line in lines:
                if mon_line.name == out_line.name:
                    continue
                mon_key = line_uv.get(mon_line.name)

                alpha = float(lodf_dict.get((mon_key, out_key), 0.0))
                if abs(alpha) < lodf_tol:
                    continue

                # Add for all time steps
                for t in range(T):
                    # Handle missing limits safely
                    fem = _val_at(mon_line.emergency_limit, t, 1e9)
                    if fem <= 0:
                        fem = _val_at(mon_line.normal_limit, t, 1e9)

                    events.append(
                        ConstraintEvent(
                            "line", mon_line.name, out_line.name, t, alpha, float(fem)
                        )
                    )

    # 3. Enumerate Gen Outages
    if PTS is not None:
        for cont in contingencies:
            if not cont.units:
                continue
            for gen in cont.units:
                j = col_by_bus.get(gen.bus.name)
                if j is None:
                    continue

                for mon_line in lines:
                    r = row_by_line.get(mon_line.name)
                    if r is None:
                        continue

                    beta = float(PTS[r, j])
                    if abs(beta) < isf_tol:
                        continue

                    for t in range(T):
                        fem = _val_at(mon_line.emergency_limit, t, 1e9)
                        if fem <= 0:
                            fem = _val_at(mon_line.normal_limit, t, 1e9)

                        events.append(
                            ConstraintEvent(
                                "gen", mon_line.name, gen.name, t, beta, float(fem)
                            )
                        )

    return events


def group_events_by_line(
    events: List[ConstraintEvent],
) -> Dict[str, List[ConstraintEvent]]:
    d = {}
    for e in events:
        d.setdefault(e.line_name, []).append(e)
    return d
