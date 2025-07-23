from optimization_model.SCUC.model_builder import build_model
import gurobipy as gp
from gurobipy import GRB

from src.data_preparation.read_data import read_benchmark

if __name__ == "__main__":
    # model = build_model()
    # model.optimize()

    # if model.status == GRB.OPTIMAL:
    #     x = model.getVarByName("x")
    #     y = model.getVarByName("y")
    #     print(f"Optimal solution found:")
    #     print(f"x = {x.X}")
    #     print(f"y = {y.X}")
    #     print(f"Objective value = {model.ObjVal}")
    # else:
    #     print("No optimal solution found.")

    SAMPLE = "matpower/case300/2017-06-24"
    print(f"→ Loading sample instance '{SAMPLE}' …")
    inst = read_benchmark(SAMPLE, quiet=False)
    sc = inst.deterministic
    print(
        f"Loaded scenario '{sc.name}' with "
        f"{len(sc.thermal_units)} thermal units, "
        f"{len(sc.buses)} buses, "
        f"{len(sc.lines)} lines, horizon {sc.time} steps of "
        f"{sc.time_step} minutes."
    )
