import numpy as np
import matplotlib.pyplot as plt

import json
import networkx as nx

def build_grid_graph(file):

    buses = list(file["Buses"].keys())
    bus_idx = {b: i for i, b in enumerate(buses)}

    G = nx.Graph()

    for lid, line in file["Transmission lines"].items():
        u = bus_idx[line["Source bus"]]
        v = bus_idx[line["Target bus"]]
        G.add_edge(u, v)

    return G, bus_idx


def plot_grid_graph(G, bus_idx, seed = 42):

    pos = nx.spring_layout(G, seed=seed)

    nx.draw_networkx_nodes(G, pos, node_size=300)
    nx.draw_networkx_edges(G, pos, width=1.2)
    inv_idx = {i: b for b, i in bus_idx.items()}
    
    nx.draw_networkx_labels(G, pos, labels=inv_idx, font_size=8)
    plt.title(f"Case {len(bus_idx)}")
    plt
    plt.show()
