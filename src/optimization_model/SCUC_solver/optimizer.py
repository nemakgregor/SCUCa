import logging
from datetime import datetime
from gurobipy import GRB

from src.data_preparation.read_data import read_benchmark
from src.optimization_model.SCUC_solver.scuc_model_builder import build_model
from src.optimization_model.helpers.save_solution import save_solution_to_log
from src.optimization_model.helpers.verify_solution import verify_solution_to_log
from src.optimization_model.helpers.run_utils import allocate_run_id, make_log_filename
from src.data_preparation.params import DataParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _status_str(code: int) -> str:
    mapping = DataParams.SOLVER_STATUS_STR
    return mapping.get(code, f"STATUS_{code}")


if __name__ == "__main__":
    SAMPLE = "test/case14"
    # SAMPLE = "matpower/case300/2017-06-24"
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

    # Optional numeric setting to reduce sensitivity to small coefficients
    try:
        model.Params.NumericFocus = 1
    except Exception:
        pass

    logger.info("→ Optimizing model …")
    try:
        model.Params.OutputFlag = 1
        model.Params.MIPGap = 0.05
        model.Params.TimeLimit = 600
    except Exception:
        pass
    model.optimize()

    logger.info(f"Solver status: {_status_str(model.Status)}")
    if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        run_id = allocate_run_id(sc.name)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sol_fname = make_log_filename(
            kind="solution", scenario=sc.name, run_id=run_id, ts=ts
        )
        ver_fname = make_log_filename(
            kind="verify", scenario=sc.name, run_id=run_id, ts=ts
        )

        sol_path = save_solution_to_log(sc, model, filename=sol_fname)
        logger.info(f"Full solution saved to: {sol_path}")

        ver_path = verify_solution_to_log(sc, model, filename=ver_fname)
        logger.info(f"Verification report saved to: {ver_path}")
    else:
        logger.info("No feasible solution found.")
