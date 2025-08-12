"""Simple Economic Dispatch with segmented costs and commitment.

This file now only orchestrates the model building by delegating:
  - data preparation (total load)
  - variable creation (commitment, segment power)
  - constraints (commitment fixing, linking, power balance)
  - objective (production cost)

All logic lives under:
  src/optimization_model/solver/economic_dispatch/
"""

from __future__ import annotations

import gurobipy as gp

from src.data_preparation.data_structure import UnitCommitmentScenario
from src.optimization_model.solver.economic_dispatch.data.load import compute_total_load
from src.optimization_model.solver.economic_dispatch import (
    vars as ed_vars,
    constraints as ed_cons,
    objectives as ed_obj,
)


def build_model(scenario: UnitCommitmentScenario) -> gp.Model:
    """Build a segmented ED model with commitment using modular components."""
    model = gp.Model("SimpleEconomicDispatch_Segmented")

    units = scenario.thermal_units
    time_periods = range(scenario.time)

    # Data preparation
    total_load = compute_total_load(scenario.buses, scenario.time)

    # Variables
    commit = ed_vars.commitment.add_variables(model, units, time_periods)
    seg_power = ed_vars.segment_power.add_variables(model, units, time_periods)

    # Keep attributes used by downstream code (e.g., optimizer pretty print)
    model.__dict__["_commit"] = commit
    model.__dict__["_seg_power"] = seg_power

    # Constraints
    ed_cons.commitment_fixing.add_constraints(model, units, commit, time_periods)
    ed_cons.linking.add_constraints(model, units, commit, seg_power, time_periods)
    ed_cons.power_balance_segmented.add_constraints(
        model, total_load, units, commit, seg_power, time_periods
    )

    # Objective
    ed_obj.power_cost_segmented.set_objective(
        model, units, commit, seg_power, time_periods
    )

    return model
