import gurobipy as gp
from gurobipy import GRB

import sys
import os

# Imports adjusted to use relative imports for package structure
from ...data_preparation.read_data import read_benchmark
from .model_builder import build_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    # if model.status == GRB.OPTIMAL:
    #     x = model.getVarByName("x")
    #     y = model.getVarByName("y")
    #     print(f"Optimal solution found:")
    #     print(f"x = {x.X}")
    #     print(f"y = {y.X}")
    #     print(f"Objective value = {model.ObjVal}")
    # else:
    #     print("No optimal solution found.")

    SAMPLE = "test/case14"
    logger.info(f"→ Loading sample instance '{SAMPLE}' …")
    inst = read_benchmark(SAMPLE, quiet=False)
    sc = inst.deterministic
    logger.info(
        f"Loaded scenario '{sc.name}' with "
        f"{len(sc.thermal_units)} thermal units, "
        f"{len(sc.buses)} buses, "
        f"{len(sc.lines)} lines, horizon {sc.time} steps of "
        f"{sc.time_step} minutes."
    )

    logger.info("→ Building optimization model …")
    model = build_model(sc)

    logger.info("→ Optimizing model …")
    model.optimize()

    if model.status == GRB.OPTIMAL:
        x = model.getVarByName("x")
        y = model.getVarByName("y")
        logger.info(f"Optimal solution found:")
        logger.info(f"Objective value = {model.ObjVal}")
    else:
        logger.info("No optimal solution found.")
