import gurobipy as gp
from gurobipy import GRB

from src.data_preparation.read_data import read_benchmark
from src.optimization_model.SCUC_solver.ed_model_builder import build_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    if model.status == GRB.OPTIMAL:
        logger.info("Optimal solution found.")
        logger.info(f"Objective value = {model.ObjVal:.4f}")

        # Optional: print commitment and total generation per time step
        gens = sc.thermal_units
        T = range(sc.time)

        for t in T:
            load_t = sum(b.load[t] for b in sc.buses)
            prod_t = 0.0
            rows = []
            for g in gens:
                u = model._commit[g.name, t].X
                # min power if on
                p = u * g.min_power[t]
                # incremental segments
                seg_sum = 0.0
                for s in range(len(g.segments) if g.segments else 0):
                    seg_sum += model._seg_power[g.name, t, s].X
                p += seg_sum
                prod_t += p
                rows.append((g.name, int(round(u)), p))
            logger.info(f"t={t}  load={load_t:.3f}, production={prod_t:.3f}, slack={prod_t - load_t:.6f}")
            for r in rows:
                logger.info(f"    {r[0]:>20s}  u={r[1]}  p={r[2]:.3f}")

    else:
        logger.info("No optimal solution found.")