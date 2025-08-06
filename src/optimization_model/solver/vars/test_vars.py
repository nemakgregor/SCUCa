import gurobipy as gp
from gurobipy import GRB


def add_variables(model):
    x = model.addVar(vtype=GRB.CONTINUOUS, name="x")
    y = model.addVar(vtype=GRB.CONTINUOUS, name="y")
    return x, y
