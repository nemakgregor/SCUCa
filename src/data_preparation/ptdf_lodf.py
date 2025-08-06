import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from .data_structure import UnitCommitmentScenario


def compute_ptdf_lodf(scenario: UnitCommitmentScenario) -> None:
    """
    Compute PTDF and LODF matrices from network topology and susceptance.
    Assumes reference bus index 0 (0-based).
    """
    bus_indices = {bus.name: bus.index - 1 for bus in scenario.buses}  # 0-based indices
    n_buses = len(scenario.buses)
    n_lines = len(scenario.lines)
    line_indices = {line.name: line.index - 1 for line in scenario.lines}  # 0-based

    # Build graph
    G = nx.Graph()
    for line in scenario.lines:
        i = bus_indices[line.source.name]
        j = bus_indices[line.target.name]
        b = line.susceptance
        G.add_edge(i, j, susceptance=b, line_idx=line_indices[line.name])

    # Build B matrix
    B = np.zeros((n_buses, n_buses))
    for u, v, data in G.edges(data=True):
        b = data["susceptance"]
        B[u, v] = -b
        B[v, u] = -b
        B[u, u] += b
        B[v, v] += b

    ref = 0  # 0-based reference bus
    B_red = np.delete(np.delete(B, ref, axis=0), ref, axis=1)
    try:
        X = np.linalg.inv(B_red)
    except np.linalg.LinAlgError:
        raise ValueError("Singular matrix: Network may be disconnected")

    # Compute PTDF (n_lines x (n_buses - 1))
    ptdf = np.zeros((n_lines, n_buses - 1))
    for line in scenario.lines:
        l_idx = line_indices[line.name]
        i = bus_indices[line.source.name]
        j = bus_indices[line.target.name]
        x_l = 1 / line.susceptance
        i_red = i - 1 if i > ref else -1
        j_red = j - 1 if j > ref else -1
        row = np.zeros(n_buses - 1)
        if i_red >= 0:
            row += X[i_red, :] / x_l
        if j_red >= 0:
            row -= X[j_red, :] / x_l
        ptdf[l_idx] = row

    # Extend PTDF to include ref bus column (all 0)
    ptdf_with_ref = np.zeros((n_lines, n_buses))
    ptdf_with_ref[:, ref + 1 :] = ptdf[:, ref:]  # Shift columns for ref=0

    # Compute LODF (n_lines x n_lines)
    lodf = np.zeros((n_lines, n_lines))
    for c_line in scenario.lines:
        c_idx = line_indices[c_line.name]
        m = bus_indices[c_line.source.name]
        n = bus_indices[c_line.target.name]
        ptdf_c_m = ptdf_with_ref[c_idx, m]
        ptdf_c_n = ptdf_with_ref[c_idx, n]
        ptdf_c_mn = ptdf_c_m - ptdf_c_n
        denom = 1 - ptdf_c_mn
        if abs(denom) < 1e-6:
            continue  # Skip self-outage or singular cases
        for l_idx in range(n_lines):
            ptdf_l_m = ptdf_with_ref[l_idx, m]
            ptdf_l_n = ptdf_with_ref[l_idx, n]
            ptdf_l_mn = ptdf_l_m - ptdf_l_n
            lodf[l_idx, c_idx] = ptdf_l_mn / denom

    scenario.isf = csr_matrix(ptdf)
    scenario.lodf = csr_matrix(lodf)
