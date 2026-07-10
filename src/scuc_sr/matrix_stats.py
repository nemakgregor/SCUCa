from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import gurobipy as gp


@dataclass
class _RangeStats:
    """Simple container for coefficient statistics."""

    num_entries: int
    abs_min: float
    abs_max: float
    order_min: int
    order_max: int
    counts_by_order: Dict[int, int]


def _log10_order(value: float) -> int:
    """Return floor(log10(|value|)) for value > 0, else a sentinel."""
    if value <= 0.0:
        return -999
    return int(math.floor(math.log10(value)))


def _stats_from_values(values) -> _RangeStats:
    """
    Build _RangeStats from an iterable of positive values.
    Zero entries must already be filtered out.
    """
    num = 0
    vmin = float("inf")
    vmax = 0.0
    counts: Counter[int] = Counter()

    for v in values:
        if v <= 0.0 or not math.isfinite(v):
            continue
        num += 1
        if v < vmin:
            vmin = v
        if v > vmax:
            vmax = v
        o = _log10_order(v)
        counts[o] += 1

    if num == 0:
        return _RangeStats(
            num_entries=0,
            abs_min=0.0,
            abs_max=0.0,
            order_min=0,
            order_max=0,
            counts_by_order={},
        )

    order_min = min(counts.keys())
    order_max = max(counts.keys())
    return _RangeStats(
        num_entries=num,
        abs_min=vmin,
        abs_max=vmax,
        order_min=order_min,
        order_max=order_max,
        counts_by_order=dict(counts),
    )


def collect_matrix_statistics(model: gp.Model) -> Dict[str, Any]:
    """
    Scan the linear part of the model and compute coefficient distributions.

    Returns a dict with:
      - matrix: stats over all nonzero A_ij (constraint matrix)
      - objective: stats over nonzero Obj coefficients
      - rhs: stats over nonzero RHS entries
    """
    # --- Matrix coefficients (A) ---
    constrs = list(model.getConstrs())
    nnz_values = []
    rows_nonzero = 0

    for c in constrs:
        row = model.getRow(c)
        nz = row.size()
        if nz == 0:
            continue
        rows_nonzero += 1
        for k in range(nz):
            coef = float(row.getCoeff(k))
            if coef == 0.0:
                continue
            v = abs(coef)
            if v > 0.0 and math.isfinite(v):
                nnz_values.append(v)

    matrix_stats = _stats_from_values(nnz_values)
    matrix_payload = {
        "num_rows": int(len(constrs)),
        "rows_nonzero": int(rows_nonzero),
        "nnz": int(matrix_stats.num_entries),
        "abs_min": float(matrix_stats.abs_min),
        "abs_max": float(matrix_stats.abs_max),
        "order_min": int(matrix_stats.order_min),
        "order_max": int(matrix_stats.order_max),
        "counts_by_order": {
            str(k): int(v) for k, v in sorted(matrix_stats.counts_by_order.items())
        },
    }

    # --- Objective coefficients ---
    obj_values = []
    for v in model.getVars():
        coef = abs(float(v.Obj))
        if coef > 0.0 and math.isfinite(coef):
            obj_values.append(coef)
    obj_stats = _stats_from_values(obj_values)
    obj_payload = {
        "num_nonzero": int(obj_stats.num_entries),
        "abs_min": float(obj_stats.abs_min),
        "abs_max": float(obj_stats.abs_max),
        "order_min": int(obj_stats.order_min),
        "order_max": int(obj_stats.order_max),
        "counts_by_order": {
            str(k): int(v) for k, v in sorted(obj_stats.counts_by_order.items())
        },
    }

    # --- RHS values ---
    rhs_values = []
    for c in constrs:
        rhs = abs(float(c.RHS))
        if rhs > 0.0 and math.isfinite(rhs):
            rhs_values.append(rhs)
    rhs_stats = _stats_from_values(rhs_values)
    rhs_payload = {
        "num_nonzero": int(rhs_stats.num_entries),
        "abs_min": float(rhs_stats.abs_min),
        "abs_max": float(rhs_stats.abs_max),
        "order_min": int(rhs_stats.order_min),
        "order_max": int(rhs_stats.order_max),
        "counts_by_order": {
            str(k): int(v) for k, v in sorted(rhs_stats.counts_by_order.items())
        },
    }

    return {
        "matrix": matrix_payload,
        "objective": obj_payload,
        "rhs": rhs_payload,
    }


def write_matrix_statistics(
    model: gp.Model, out_path: Path, meta: Dict[str, Any] | None = None
) -> Path:
    """
    Compute matrix/objective/RHS statistics for *model* and write them to *out_path*
    as pretty-printed JSON.

    meta can be used to store additional context (instance name, mode, status...).
    """
    stats = collect_matrix_statistics(model)
    payload: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_name": getattr(model, "ModelName", ""),
        "scenario_name": getattr(model, "_scenario_name", None),
        "meta": meta or {},
        "stats": stats,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
