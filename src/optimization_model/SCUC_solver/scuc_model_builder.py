"""Simple SCUC with segmented costs and commitment.

This file orchestrates model building by delegating:
  - data preparation (total load)
  - variable creation (commitment, segment power, reserve, startup/shutdown, line flows)
  - constraints (commitment fixing, linking, power balance, reserve, initial conditions/ramping, line flow PTDF, line limits, min up/down, line/generator-contingency limits)
  - objective (production cost + reserve shortfall penalty + line overflow penalty)
"""

from __future__ import annotations

import gurobipy as gp
import logging
from typing import Optional, Callable

from src.data_preparation.data_structure import UnitCommitmentScenario
from src.optimization_model.solver.scuc.data.load import compute_total_load
from src.optimization_model.solver.scuc import (
    vars as scuc_vars,
    constraints as scuc_cons,
    objectives as scuc_obj,
)

logger = logging.getLogger(__name__)


def build_model(
    scenario: UnitCommitmentScenario,
    *,
    contingency_filter: Optional[Callable] = None,
) -> gp.Model:
    """Build a segmented SCUC model with commitment, startup/shutdown, reserves, and line constraints.

    Parameters
    ----------
    scenario : UnitCommitmentScenario
    contingency_filter : Optional[Callable]
        Optional predicate(kind, line_l, out_obj, t, coeff, F_em) -> bool.
        If provided, contingency constraints (+/- pairs) for which this returns False are pruned.
    """
    model = gp.Model("SCUC_Segmented")

    units = scenario.thermal_units
    time_periods = range(scenario.time)

    # Data preparation
    total_load = compute_total_load(scenario.buses, scenario.time)

    # Scenario summary
    logger.info(
        "SCUC build: units=%d, buses=%d, lines=%d, reserves=%d, contingencies=%d, T=%d, dt(min)=%d",
        len(scenario.thermal_units),
        len(scenario.buses),
        len(scenario.lines),
        len(scenario.reserves),
        len(scenario.contingencies),
        scenario.time,
        scenario.time_step,
    )
    if scenario.lines:
        try:
            isf = scenario.isf
            lodf = scenario.lodf
            ref_bus = getattr(scenario, "ptdf_ref_bus_index", scenario.buses[0].index)
            logger.info(
                "PTDF/ISF shape=%s, nnz=%d; LODF shape=%s, nnz=%d; ref_bus_index(1-based)=%s",
                getattr(isf, "shape", None),
                getattr(isf, "nnz", None),
                getattr(lodf, "shape", None),
                getattr(lodf, "nnz", None),
                ref_bus,
            )
        except Exception:
            pass

    # Variables

    gen_commit = scuc_vars.commitment.add_variables(model, units, time_periods)
    gen_segment_power = scuc_vars.segment_power.add_variables(
        model, units, time_periods
    )
    gen_startup, gen_shutdown = scuc_vars.startup_shutdown.add_variables(
        model, units, time_periods
    )

    reserve = None
    reserve_shortfall = None
    if scenario.reserves:
        reserve, reserve_shortfall = scuc_vars.reserve.add_variables(
            model, scenario.reserves, units, time_periods
        )

    line_flow = None
    line_over_pos = None
    line_over_neg = None
    if scenario.lines:
        line_flow, line_over_pos, line_over_neg = scuc_vars.line_flow.add_variables(
            model, scenario.lines, time_periods
        )

    # Shared contingency overflow slacks (per line/time)
    cont_over_pos = None
    cont_over_neg = None
    if scenario.lines and scenario.contingencies:
        from src.optimization_model.solver.scuc.vars import (
            contingency_overflow as covars,
        )

        cont_over_pos, cont_over_neg = covars.add_variables(
            model, scenario.lines, time_periods
        )

    # Constraints
    scuc_cons.commitment_fixing.add_constraints(model, units, gen_commit, time_periods)

    scuc_cons.linking.add_constraints(
        model, units, gen_commit, gen_segment_power, time_periods
    )

    scuc_cons.power_balance_segmented.add_constraints(
        model, total_load, units, gen_commit, gen_segment_power, time_periods
    )

    scuc_cons.initial_conditions.add_constraints(
        model,
        units,
        gen_commit,
        gen_segment_power,
        gen_startup,
        gen_shutdown,
        time_periods,
    )

    scuc_cons.min_up_down.add_constraints(
        model, units, gen_commit, gen_startup, gen_shutdown, time_periods
    )

    if scenario.reserves:
        scuc_cons.reserve.add_constraints(
            model,
            scenario.reserves,
            gen_commit,
            gen_segment_power,
            reserve,
            time_periods,
        )

        scuc_cons.reserve_requirement.add_constraints(
            model, scenario.reserves, reserve, reserve_shortfall, time_periods
        )

    if scenario.lines:
        scuc_cons.line_flow_ptdf.add_constraints(
            model, scenario, gen_commit, gen_segment_power, line_flow, time_periods
        )

        scuc_cons.line_limits.add_constraints(
            model, scenario.lines, line_flow, line_over_pos, line_over_neg, time_periods
        )

        # Line and generator contingency limits with shared slacks
        if scenario.contingencies:
            scuc_cons.contingencies.add_constraints(
                model,
                scenario,
                gen_commit,
                gen_segment_power,
                line_flow,
                time_periods,
                cont_over_pos,
                cont_over_neg,
                filter_predicate=contingency_filter,
            )

    # Objective with reserve penalty and line overflow penalty (base case and contingencies)
    scuc_obj.power_cost_segmented.set_objective(
        model,
        units,
        gen_commit,
        gen_segment_power,
        time_periods,
        scenario.reserves,
        scenario.lines,
    )

    # Expose key data for downstream tooling (solution dump, verification)
    model.__dict__["_total_load"] = total_load
    model.__dict__["_scenario_name"] = scenario.name or "scenario"

    return model
