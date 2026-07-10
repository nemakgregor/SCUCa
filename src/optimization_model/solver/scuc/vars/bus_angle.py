"""ID: V-210 — Bus voltage angle variables (DC power flow).

theta[b,t] ∈ ℝ for each bus b and time period t.
"""

import logging
from typing import Sequence

import gurobipy as gp
from gurobipy import GRB

logger = logging.getLogger(__name__)


def add_variables(model: gp.Model, buses: Sequence, time_periods: range) -> gp.tupledict:
    idx = [(bus.name, t) for bus in buses for t in time_periods]
    theta = model.addVars(idx, lb=-GRB.INFINITY, name="bus_angle")
    model.__dict__["bus_angle"] = theta
    logger.info("Vars(V-210): bus_angle added=%d (buses=%d, T=%d)", len(idx), len(buses), len(time_periods))
    return theta

