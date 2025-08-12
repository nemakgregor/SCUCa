import logging
from gurobipy import GRB

from src.data_preparation.read_data import read_benchmark
from src.optimization_model.SCUC_solver.scuc_model_builder import build_model
from src.optimization_model.SCUC_solver.restore_solution import restore_solution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _status_str(code: int) -> str:
    mapping = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
    }
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

    logger.info("→ Optimizing model …")
    model.optimize()

    logger.info(f"Solver status: {_status_str(model.Status)}")
    if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        sol = restore_solution(sc, model)
        logger.info(f"Objective value = {sol['objective']:.6f}")

        T = sc.time
        gens = sc.thermal_units
        # Per-generator details
        for g in gens:
            gsol = sol["generators"][g.name]
            for t in range(T):
                segs = gsol["segment_power"][t]
                logger.info(
                    f"t={t:02d} gen={g.name} "
                    f"commit={gsol['commit'][t]} "
                    f"min_part={gsol['min_power_output'][t]:.3f} "
                    f"segments_total={sum(segs):.3f} "
                    f"total_power={gsol['total_power'][t]:.3f}"
                )
                if segs:
                    for s, val in enumerate(segs):
                        logger.info(f"        segment[{s}]={val:.3f}")

        # System balance summary
        for t in range(T):
            load_t = sol["system"]["load"][t]
            prod_t = sol["system"]["total_production"][t]
            logger.info(
                f"t={t:02d}  system_load={load_t:.3f}  "
                f"system_production={prod_t:.3f}  slack={prod_t - load_t:.6f}"
            )

        # Reserve summary
        if "reserves" in sol and sol["reserves"]:
            for r_name, rsol in sol["reserves"].items():
                for t in range(T):
                    prov = rsol["total_provided"][t]
                    req = rsol["requirement"][t]
                    short = rsol["shortfall"][t]
                    logger.info(
                        f"t={t:02d} reserve={r_name} total_provided={prov:.3f} "
                        f"requirement={req:.3f} shortfall={short:.3f}"
                    )

    else:
        logger.info("No feasible solution found.")
