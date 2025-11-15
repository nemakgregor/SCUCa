"""
Small demo grids for testing and plots.
"""

from __future__ import annotations

import networkx as nx


def create_default_5bus() -> nx.Graph:
    G = nx.Graph()
    buses = {
        "Bus1": (150, 50),
        "Bus2": (0, 120),
        "Bus3": (0, 80),
        "Bus4": (100, 10),
        "SlackBus": (0, 0),
    }
    for n, (g, l) in buses.items():
        G.add_node(n, generation=g, load=l)
    net = sum(g - l for g, l in buses.values())
    G.nodes["SlackBus"]["generation"] = -net
    edges = [
        ("Bus1", "Bus2", 0.10, 100),
        ("Bus1", "Bus3", 0.125, 80),
        ("Bus2", "Bus4", 0.08, 120),
        ("Bus3", "Bus4", 0.08, 120),
        ("SlackBus", "Bus1", 0.05, 200),
        ("SlackBus", "Bus4", 0.05, 200),
    ]
    for u, v, x, c in edges:
        G.add_edge(u, v, reactance=x, capacity=c)
    return G


def create_radial_6bus() -> nx.Graph:
    G = nx.Graph()
    for i in range(6):
        G.add_node(f"B{i + 1}", generation=0, load=50)
    G.nodes["B1"]["generation"] = 300
    G.add_node("SlackBus", generation=-50, load=0)
    edges = [
        ("B1", "B2", 0.12, 120),
        ("B2", "B3", 0.10, 100),
        ("B3", "B4", 0.10, 80),
        ("B4", "B5", 0.08, 80),
        ("B5", "B6", 0.08, 60),
        ("B1", "SlackBus", 0.05, 200),
    ]
    for u, v, x, c in edges:
        G.add_edge(u, v, reactance=x, capacity=c)
    return G


def create_meshed_8bus() -> nx.Graph:
    G = nx.Graph()
    data = {
        "N1": (100, 20),
        "N2": (0, 40),
        "N3": (0, 60),
        "N4": (50, 10),
        "N5": (0, 40),
        "N6": (0, 70),
        "N7": (0, 40),
        "SlackBus": (0, 0),
    }
    for n, (g, l) in data.items():
        G.add_node(n, generation=g, load=l)
    net = sum(g - l for g, l in data.values())
    G.nodes["SlackBus"]["generation"] = -net
    edges = [
        ("N1", "N2", 0.15, 120),
        ("N2", "N3", 0.12, 100),
        ("N3", "N4", 0.10, 100),
        ("N4", "N5", 0.10, 100),
        ("N5", "N6", 0.08, 100),
        ("N6", "N7", 0.08, 100),
        ("N7", "N1", 0.15, 120),
        ("N2", "N6", 0.12, 90),
        ("SlackBus", "N1", 0.05, 250),
        ("SlackBus", "N4", 0.05, 250),
    ]
    for u, v, x, c in edges:
        G.add_edge(u, v, reactance=x, capacity=c)
    return G


GRID_FACTORY = {
    "default": create_default_5bus,
    "radial": create_radial_6bus,
    "meshed": create_meshed_8bus,
}
