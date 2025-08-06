import gurobipy as gp
import sys
import os

from src.optimization_model.solver import objectives, constraints, vars

from src.data_preparation.data_structure import UnitCommitmentScenario


def build_model(scenario: UnitCommitmentScenario) -> gp.Model:
    """Build the optimization model for the given scenario."""

    model = gp.Model("simple_economic_dispatch")

    # --------------------------- index sets ----------------------------------
    units = scenario.thermal_units  # list of Unit objects
    time_periods = range(scenario.time)  # horizon length already in hours

    # --------------------------- variables -----------------------------------
    gen_power = vars.generators_power.add_variables(model, units, time_periods)

    # --------------------------- constraints ---------------------------------
    constraints.power_balance.add_constraints(
        model, scenario.buses, gen_power, units, time_periods
    )

    # --------------------------- objective -----------------------------------
    objectives.power_cost.set_objective(model, units, gen_power, time_periods)

    return model
