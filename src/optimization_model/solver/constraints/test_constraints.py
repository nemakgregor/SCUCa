import gurobipy as gp
from gurobipy import GRB


def add_constraints(model, x, y):
    model.addConstr(x + 2 * y <= 40, "c1")
    model.addConstr(2 * x + y <= 50, "c2")
