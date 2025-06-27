import numpy as np
import matplotlib.pyplot as plt

import json
import networkx as nx

def build_grid_graph(path_grid, limit):
    with open(path_grid) as f:
        grid = json.load(f)

    buses = list(grid["Buses"].keys())
    bus_idx = {b: i for i, b in enumerate(buses)}

    G = nx.Graph()
    edges_order = []
    edge_attr_list = []

    for lid, line in grid["Transmission lines"].items():
        u = bus_idx[line["Source bus"]]
        v = bus_idx[line["Target bus"]]
        G.add_edge(u, v)
        edges_order.append((u, v))
        edge_attr_list.append([
            line["Reactance (ohms)"],
            line["Susceptance (S)"],
            limit
        ])

    edge_attr_static = np.asarray(edge_attr_list)
    return G, bus_idx, edges_order, edge_attr_static


def plot_grid_graph(G, bus_idx, seed = 42):

    pos = nx.spring_layout(G, seed=seed)

    nx.draw_networkx_nodes(G, pos, node_size=300)
    nx.draw_networkx_edges(G, pos, width=1.2)
    inv_idx = {i: b for b, i in bus_idx.items()}
    
    nx.draw_networkx_labels(G, pos, labels=inv_idx, font_size=8)
    plt.title(f"Case {len(bus_idx)}")
    plt
    plt.show()
