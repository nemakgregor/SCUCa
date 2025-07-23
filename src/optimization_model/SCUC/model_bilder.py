import gurobipy as gp
from gurobipy import GRB
import sys
import os

# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.optimization_model.solver.vars.test_vars import add_variables
from src.optimization_model.solver.objectives.test_objective import set_objective
from src.optimization_model.solver.constraints.test_constraints import add_constraints


def build_model():
    model = gp.Model("simple_lp")
    x, y = add_variables(model)
    set_objective(model, x, y)
    add_constraints(model, x, y)
    return model
