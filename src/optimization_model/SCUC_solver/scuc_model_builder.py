"""Simple SCUC with segmented costs and commitment.

This file now only orchestrates the model building by delegating:
  - data preparation (total load)
  - variable creation (commitment, segment power, reserve)
  - constraints (commitment fixing, linking, power balance, reserve)
  - objective (production cost + reserve shortfall penalty)

All logic lives under:
  src/optimization_model/solver/economic_dispatch/
"""

import gurobipy as gp

from src.data_preparation.data_structure import UnitCommitmentScenario
from src.optimization_model.solver.scuc.data.load import compute_total_load
from src.optimization_model.solver.scuc import (
    vars as scuc_vars,
    constraints as scuc_cons,
    objectives as scuc_obj,
)


def build_model(scenario: UnitCommitmentScenario) -> gp.Model:
    """Build a segmented ED model with commitment using modular components."""
    model = gp.Model("SimpleEconomicDispatch_Segmented")

    units = scenario.thermal_units
    time_periods = range(scenario.time)

    # Data preparation
    total_load = compute_total_load(scenario.buses, scenario.time)

    # Variables
    gen_commit = scuc_vars.commitment.add_variables(model, units, time_periods)
    gen_segment_power = scuc_vars.segment_power.add_variables(
        model, units, time_periods
    )

    # Reserve variables
    reserve = None
    reserve_shortfall = None
    if scenario.reserves:
        reserve, reserve_shortfall = scuc_vars.reserve.add_variables(
            model, scenario.reserves, units, time_periods
        )

    # Constraints
    scuc_cons.commitment_fixing.add_constraints(model, units, gen_commit, time_periods)
    scuc_cons.linking.add_constraints(
        model, units, gen_commit, gen_segment_power, time_periods
    )
    scuc_cons.power_balance_segmented.add_constraints(
        model, total_load, units, gen_commit, gen_segment_power, time_periods
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

    # Objective with reserve penalty
    scuc_obj.power_cost_segmented.set_objective(
        model, units, gen_commit, gen_segment_power, time_periods, scenario.reserves
    )

    return model
