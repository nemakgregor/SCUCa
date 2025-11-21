import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from scipy.sparse.linalg import splu
from .data_structure import UnitCommitmentScenario
from typing import Dict

# Simple in-process cache keyed by grid signature
_PTDF_LODF_CACHE = {}


def _grid_signature(scenario: UnitCommitmentScenario) -> tuple:
    buses = scenario.buses
    lines = scenario.lines
    bus_idx = tuple(sorted(int(b.index) for b in buses))
    edge_sig = []
    for ln in lines:
        u = int(ln.source.index)
        v = int(ln.target.index)
        a, b = (u, v) if u <= v else (v, u)
        edge_sig.append((a, b, round(float(ln.susceptance), 12)))
    return (bus_idx, tuple(sorted(edge_sig)))


def compute_ptdf_lodf(scenario: UnitCommitmentScenario) -> None:
    """
    Compute PTDF (ISF) and LODF matrices from network topology and line susceptances.
    - Reference bus = smallest bus index.
    - ISF shape: (n_lines, n_buses - 1) (non-reference buses columns; bus order ascending by index).
    - LODF shape: (n_lines, n_lines).
    Implementation uses sparse LU factorization (no explicit inverse).
    """
    buses = scenario.buses
    lines = scenario.lines
    n_buses = len(buses)
    n_lines = len(lines)

    if n_buses == 0 or n_lines == 0:
        scenario.isf = csr_matrix((0, 0), dtype=float)
        scenario.lodf = csr_matrix((0, 0), dtype=float)
        return

    sig = _grid_signature(scenario)
    cached = _PTDF_LODF_CACHE.get(sig)
    if cached is not None:
        isf, lodf, ref_bus_1b = cached
        scenario.isf = isf
        scenario.lodf = lodf
        scenario.__dict__["ptdf_ref_bus_index"] = ref_bus_1b
        return

    # Bus ordering and reference bus consistent with optimization model
    bus_indices_1b = sorted([int(b.index) for b in buses])
    pos_by_bus1b = {bidx: i0 for i0, bidx in enumerate(bus_indices_1b)}
    ref_bus_1b = bus_indices_1b[0]
    ref_pos = pos_by_bus1b[ref_bus_1b]

    # Build sparse B (n_buses x n_buses)
    rows = []
    cols = []
    data = []
    # Accumulate contributions
    diag_accum = [0.0] * n_buses
    for ln in lines:
        i0 = pos_by_bus1b[int(ln.source.index)]
        j0 = pos_by_bus1b[int(ln.target.index)]
        b = float(ln.susceptance)
        diag_accum[i0] += b
        diag_accum[j0] += b
        # Off-diagonal -b
        rows.extend([i0, j0])
        cols.extend([j0, i0])
        data.extend([-b, -b])
    # Diagonal entries
    for i in range(n_buses):
        if diag_accum[i] != 0.0:
            rows.append(i)
            cols.append(i)
            data.append(diag_accum[i])

    B = coo_matrix((data, (rows, cols)), shape=(n_buses, n_buses)).tocsr()

    # Reduced B (remove ref bus)
    keep = [k for k in range(n_buses) if k != ref_pos]
    B_red = B[keep, :][:, keep].tocsc()

    # Sparse LU factorization
    try:
        lu = splu(B_red)

        def solve(rhs):
            return lu.solve(rhs)
    except Exception:
        # Fallback: dense solve (small systems)
        B_red_dense = B_red.toarray()

        def solve(rhs):
            return np.linalg.solve(B_red_dense, rhs)

    # Map non-ref positions
    non_ref_positions = [k for k in range(n_buses) if k != ref_pos]
    col_by_pos = {pos: c for c, pos in enumerate(non_ref_positions)}

    # Precompute inverse rows (via solves) lazily
    inv_rows_cache: Dict[int, np.ndarray] = {}

    def _inverse_row(i_pos: int) -> np.ndarray:
        """
        Return row i of B_red^{-1} (as a 1D array of length n_buses-1),
        using symmetry (row == solve for e_i because B is SPD).
        """
        col = col_by_pos[i_pos]
        if col in inv_rows_cache:
            return inv_rows_cache[col]
        e = np.zeros(len(non_ref_positions), dtype=float)
        e[col] = 1.0
        y = solve(e)
        inv_rows_cache[col] = np.asarray(y).reshape(-1)
        return inv_rows_cache[col]

    # ISF rows
    isf = np.zeros((n_lines, n_buses - 1), dtype=float)
    for l_idx, ln in enumerate(lines):
        i0 = pos_by_bus1b[int(ln.source.index)]
        j0 = pos_by_bus1b[int(ln.target.index)]
        b_l = float(ln.susceptance)

        row_vec = np.zeros(n_buses - 1, dtype=float)
        if i0 != ref_pos:
            row_vec += b_l * _inverse_row(i0)
        if j0 != ref_pos:
            row_vec -= b_l * _inverse_row(j0)
        isf[l_idx, :] = row_vec

    # Expand ISF to full-bus columns for LODF computation
    isf_full = np.zeros((n_lines, n_buses), dtype=float)
    for c, pos in enumerate(non_ref_positions):
        isf_full[:, pos] = isf[:, c]

    # LODF via standard formula: PTDF_m(m,n) with reference bus fixed
    lodf = np.zeros((n_lines, n_lines), dtype=float)
    for c_idx, c_line in enumerate(lines):
        mpos = pos_by_bus1b[int(c_line.source.index)]
        npos = pos_by_bus1b[int(c_line.target.index)]
        ptdf_c_mn = isf_full[c_idx, mpos] - isf_full[c_idx, npos]
        denom = 1.0 - ptdf_c_mn
        if abs(denom) < 1e-9:
            continue
        for l_idx in range(n_lines):
            ptdf_l_mn = isf_full[l_idx, mpos] - isf_full[l_idx, npos]
            lodf[l_idx, c_idx] = ptdf_l_mn / denom

    np.fill_diagonal(lodf, -1.0)

    # Drop tiny values for sparsity
    tol = 1e-10
    isf[np.abs(isf) < tol] = 0.0
    lodf[np.abs(lodf) < tol] = 0.0

    isf_csr = csr_matrix(isf)
    lodf_csr = csr_matrix(lodf)
    scenario.isf = isf_csr
    scenario.lodf = lodf_csr
    scenario.__dict__["ptdf_ref_bus_index"] = ref_bus_1b

    _PTDF_LODF_CACHE[sig] = (isf_csr, lodf_csr, ref_bus_1b)
