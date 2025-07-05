import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def compute_dc_power_flow(data):
    buses = data["Buses"]
    lines = data["Transmission lines"]
    gens  = data["Generators"]

    bus_ids = list(buses.keys())
    bus_idx = {bus: i for i, bus in enumerate(bus_ids)}
    n = len(bus_ids)
    m = len(lines)

    rows, cols, vals, x_vals = [], [], [], []

    for iline, line in enumerate(lines.values()):
        i = bus_idx[line["Source bus"]]
        j = bus_idx[line["Target bus"]]
        x = line["Reactance (ohms)"]

        rows += [iline, iline]
        cols += [i, j]
        vals += [1, -1]
        x_vals.append(x)

    A = coo_matrix((vals, (rows, cols)), shape=(m, n)).tocsc()
    B_diag = np.array([1 / x for x in x_vals])
    B = A.T @ coo_matrix((B_diag, (range(m), range(m))), shape=(m, m)) @ A

    P = np.zeros(n)
    for g in gens.values():
        b = bus_idx[g["Bus"]]
        P[b] += g["Initial power (MW)"]

    for b_id, b in buses.items():
        b_idx = bus_idx[b_id]
        load = b.get("Load (MW)", 0.0)
        if isinstance(load, list):
            load = load[0]
        P[b_idx] -= load

    B_red = B[1:, 1:]
    P_red = P[1:]

    theta = np.zeros(n)
    theta[1:] = spsolve(B_red, P_red)

    flows = []
    for line in lines.values():
        i = bus_idx[line["Source bus"]]
        j = bus_idx[line["Target bus"]]
        x = line["Reactance (ohms)"]
        flows.append((theta[i] - theta[j]) / x)

    return np.abs(np.array(flows))
