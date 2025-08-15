"""ID: O-301 — Total cost with reserve, startup, and line penalties.

This module assembles the SCUC objective from modular term expressions:
  - Minimum-output (no-load) cost
  - Energy (segment) cost
  - Startup cost
  - Reserve shortfall penalty
  - Base-case line overflow penalty
  - Contingency overflow penalty
"""

import logging
import gurobipy as gp
from typing import Sequence, Optional

from src.optimization_model.solver.scuc.objectives import (
    minimum_output_cost as oc_min,
    segment_power_cost as oc_seg,
    startup_cost as oc_su,
    reserve_shortfall_penalty as oc_res,
    base_overflow_penalty as oc_flow,
    contingency_overflow_penalty as oc_cflow,
)

logger = logging.getLogger(__name__)

# Penalty multiplier for contingency slacks vs. base-case overflow
_CONTINGENCY_PENALTY_FACTOR = 10.0


def set_objective(
    model: gp.Model,
    generators: Sequence,
    commit,
    seg_power,
    time_periods: range,
    reserves: Optional[Sequence] = None,
    lines: Optional[Sequence] = None,
) -> None:
    # Build separate expressions
    expr_min = oc_min.build_expression(model, generators, commit, time_periods)
    expr_seg = oc_seg.build_expression(model, generators, seg_power, time_periods)

    startup_vars = getattr(model, "startup", None)
    expr_su = gp.LinExpr()
    if startup_vars is not None:
        expr_su = oc_su.build_expression(model, generators, startup_vars, time_periods)

    expr_res = gp.LinExpr()
    if reserves:
        expr_res = oc_res.build_expression(model, reserves, time_periods)

    expr_flow = gp.LinExpr()
    expr_cflow = gp.LinExpr()
    if lines:
        expr_flow = oc_flow.build_expression(model, lines, time_periods)
        expr_cflow = oc_cflow.build_expression(
            model,
            lines,
            time_periods,
            contingency_penalty_factor=_CONTINGENCY_PENALTY_FACTOR,
        )

    # Sum expressions
    obj = gp.LinExpr()
    obj += expr_min
    obj += expr_seg
    obj += expr_su
    obj += expr_res
    obj += expr_flow
    obj += expr_cflow

    # Set the objective
    model.setObjective(obj, gp.GRB.MINIMIZE)
    try:
        model._objective_name = "TotalCostWithReserveStartupAndLinePenalty"
    except Exception:
        pass

    logger.info("Obj(O-301): objective assembled from modular terms.")
