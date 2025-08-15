import numpy as np
from scipy.sparse import csr_matrix
from .data_structure import UnitCommitmentScenario


def compute_ptdf_lodf(scenario: UnitCommitmentScenario) -> None:
    """
    Compute PTDF (ISF) and LODF matrices from network topology and line susceptances.

    Conventions
    - scenario.buses have 1-based indices (b.index ∈ {1..N}).
    - A reference bus is chosen (default: first bus in scenario.buses by its index).
    - ISF shape is (n_lines, n_buses - 1): columns correspond to non-reference buses.
    - LODF shape is (n_lines, n_lines).
    - Orientation follows the line's source->target direction.
    """
    buses = scenario.buses
    lines = scenario.lines
    n_buses = len(buses)
    n_lines = len(lines)

    if n_buses == 0 or n_lines == 0:
        scenario.isf = csr_matrix((0, 0), dtype=float)
        scenario.lodf = csr_matrix((0, 0), dtype=float)
        return

    bus_indices_1b = sorted([b.index for b in buses])
    pos_by_bus1b = {bidx: (i0) for i0, bidx in enumerate(bus_indices_1b)}
    ref_bus_1b = bus_indices_1b[0]
    ref_pos = pos_by_bus1b[ref_bus_1b]

    B = np.zeros((n_buses, n_buses), dtype=float)
    for ln in lines:
        i0 = pos_by_bus1b[ln.source.index]
        j0 = pos_by_bus1b[ln.target.index]
        b = float(ln.susceptance)
        B[i0, i0] += b
        B[j0, j0] += b
        B[i0, j0] -= b
        B[j0, i0] -= b

    keep = [k for k in range(n_buses) if k != ref_pos]
    B_red = B[np.ix_(keep, keep)]

    try:
        X = np.linalg.inv(B_red)
    except np.linalg.LinAlgError:
        X = np.linalg.pinv(B_red)

    non_ref_positions = [k for k in range(n_buses) if k != ref_pos]
    col_by_pos = {pos: c for c, pos in enumerate(non_ref_positions)}

    isf = np.zeros((n_lines, n_buses - 1), dtype=float)
    for l_idx, ln in enumerate(lines):
        i0 = pos_by_bus1b[ln.source.index]
        j0 = pos_by_bus1b[ln.target.index]
        b_l = float(ln.susceptance)
        row = np.zeros(n_buses - 1, dtype=float)
        if i0 != ref_pos:
            row += b_l * X[col_by_pos[i0], :]
        if j0 != ref_pos:
            row -= b_l * X[col_by_pos[j0], :]
        isf[l_idx, :] = row

    isf_full = np.zeros((n_lines, n_buses), dtype=float)
    for c, pos in enumerate(non_ref_positions):
        isf_full[:, pos] = isf[:, c]

    lodf = np.zeros((n_lines, n_lines), dtype=float)
    for c_idx, c_line in enumerate(lines):
        mpos = pos_by_bus1b[c_line.source.index]
        npos = pos_by_bus1b[c_line.target.index]
        ptdf_c_mn = isf_full[c_idx, mpos] - isf_full[c_idx, npos]
        denom = 1.0 - ptdf_c_mn
        if abs(denom) < 1e-9:
            continue
        for l_idx in range(n_lines):
            ptdf_l_mn = isf_full[l_idx, mpos] - isf_full[l_idx, npos]
            lodf[l_idx, c_idx] = ptdf_l_mn / denom

    # Set diagonal to -1 (outaged line post-flow is zero)
    np.fill_diagonal(lodf, -1.0)

    # Numerics: drop extremely small entries
    tol = 1e-10
    isf[np.abs(isf) < tol] = 0.0
    lodf[np.abs(lodf) < tol] = 0.0

    scenario.isf = csr_matrix(isf)  # shape (n_lines, n_buses - 1)
    scenario.lodf = csr_matrix(lodf)  # shape (n_lines, n_lines)
    scenario.__dict__["ptdf_ref_bus_index"] = ref_bus_1b
