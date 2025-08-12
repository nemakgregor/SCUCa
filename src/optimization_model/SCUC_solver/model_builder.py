import gurobipy as gp

from src.optimization_model.solver.economic_dispatch import (
    objectives,
    constraints,
    vars,
)
from src.data_preparation.data_structure import UnitCommitmentScenario


def build_model(scenario: UnitCommitmentScenario) -> gp.Model:
    """Build the optimization model for the given scenario."""

    model = gp.Model("EconomicDispatch")

    time_periods = range(scenario.time)  # horizon length already in hours
    generators = scenario.thermal_units
    buses = scenario.buses

    total_load = [sum(bus.load[t] for bus in buses) for t in time_periods]
    # --------------------------- index sets ----------------------------------
    units = scenario.thermal_units  # list of Unit objects
    

    # --------------------------- variables -----------------------------------
    gen_power = vars.generators_power.add_variables(model, units, time_periods)

    # --------------------------- constraints ---------------------------------
    constraints.power_balance.add_constraints(
        model, scenario.buses, gen_power, units, time_periods
    )

    # --------------------------- objective -----------------------------------
    objectives.power_cost.set_objective(model, units, gen_power, time_periods)

    return model
