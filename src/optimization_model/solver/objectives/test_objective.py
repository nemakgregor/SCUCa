import gurobipy as gp
from gurobipy import GRB


def set_objective(model, x, y):
    model.setObjective(3 * x + 4 * y, GRB.MAXIMIZE)
