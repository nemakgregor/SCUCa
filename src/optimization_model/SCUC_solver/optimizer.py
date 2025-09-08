import argparse
import argparse
import logging
from datetime import datetime
from typing import List, Optional, Tuple

from typing import List, Optional, Tuple

from gurobipy import GRB

from src.data_preparation.read_data import read_benchmark
from src.optimization_model.SCUC_solver.scuc_model_builder import build_model
from src.optimization_model.helpers.save_solution import save_solution_to_log
from src.optimization_model.helpers.verify_solution import verify_solution_to_log
from src.optimization_model.helpers.run_utils import allocate_run_id, make_log_filename
from src.optimization_model.helpers.save_json_solution import (
    save_solution_as_json,
    compute_output_path,
)
from src.optimization_model.helpers.save_json_solution import (
    save_solution_as_json,
    compute_output_path,
)
from src.data_preparation.params import DataParams

# Warm-start provider and instance listers
from src.ml_models.warm_start import WarmStartProvider
from src.optimization_model.SCUC_solver.solve_instances import (
    list_remote_instances,
    list_local_cached_instances,
)

# Warm-start provider and instance listers
from src.ml_models.warm_start import WarmStartProvider
from src.optimization_model.SCUC_solver.solve_instances import (
    list_remote_instances,
    list_local_cached_instances,
)

logging.basicConfig(level=logging.INFO)

logging.getLogger("gurobipy").setLevel(logging.WARNING)
logging.getLogger("gurobipy").propagate = False


logging.getLogger("gurobipy").setLevel(logging.WARNING)
logging.getLogger("gurobipy").propagate = False

logger = logging.getLogger(__name__)

# Default remote listing scope (kept internal; no CLI noise)
_DEFAULT_ROOTS = ["matpower", "test"]
_DEFAULT_MAX_DEPTH = 4


def _status_str(code: int) -> str:
    mapping = DataParams.SOLVER_STATUS_STR
    return mapping.get(code, f"STATUS_{code}")


def _solve_one(
    name: str,
    time_limit: int,
    mip_gap: float,
    use_warm_start: bool,
    require_pretrained: bool,
    skip_existing: bool,
    warm_cache: dict,
    warm_start_mode: str,
    warm_use_train_db: bool,
    save_logs: bool,
) -> Tuple[bool, Optional[str]]:
    """
    Solve one dataset by name; optionally apply a warm start if available.
    Returns (success, output_json_path_if_any).
    """
    out_json_path = compute_output_path(name)
    if skip_existing and out_json_path.is_file():
        logger.info("Skipping existing solution: %s", out_json_path)
        return True, str(out_json_path)

    inst = read_benchmark(name, quiet=True)
    sc = inst.deterministic
    logger.info(
        "Build '%s' (units=%d, buses=%d, lines=%d, T=%d, dt=%dmin)",
        name,
        len(sc.thermal_units),
        len(sc.buses),
        len(sc.lines),
        sc.time,
        sc.time_step,
    )

    # Warm-start prepare (only if explicitly requested)
    wsp = None
    if use_warm_start:
        cf = "/".join(name.strip().split("/")[:2])
        wsp = warm_cache.get(cf)
        if wsp is None:
            wsp = WarmStartProvider(case_folder=cf)
            trained, cov = wsp.ensure_trained(
                cf, allow_build_if_missing=not require_pretrained
            )
            if not trained:
                logger.info(
                    "Warm-start: no trained index for %s (coverage=%.3f; require_pretrained=%s)",
                    cf,
                    cov,
                    require_pretrained,
                )
                wsp = None
            else:
                warm_cache[cf] = wsp
                logger.info(
                    "Warm-start: using pre-trained index for %s (coverage=%.3f)",
                    cf,
                    cov,
                )

    # If trained index present, write warm file for this instance
    if wsp is not None:
        try:
            warm_path = wsp.generate_and_save_warm_start(
                name, use_train_index_only=warm_use_train_db, exclude_self=True
            )
            if warm_path:
                logger.info("Warm-start file generated: %s", warm_path)
            else:
                logger.info(
                    "Warm-start not generated for '%s' (no neighbor or missing features).",
                    name,
                )
        except Exception as e:
            logger.warning("Warm-start generation failed for '%s': %s", name, e)

    # Build model
    model = build_model(sc)

    # Apply warm start if available
    if wsp is not None:
        try:
            # map CLI term 'fixed' to internal 'repair'
            mode = warm_start_mode.strip().lower()
            if mode == "fixed":
                mode = "repair"
            assigned = wsp.apply_warm_start_to_model(model, sc, name, mode=mode)
            if assigned > 0:
                logger.info(
                    "Warm-start applied (%s): Start set on %d variable(s).",
                    mode,
                    assigned,
                )
            else:
                logger.info("No warm-start variables set.")
        except Exception as e:
            logger.warning("Failed to apply warm-start to model: %s", e)

    # Solver params
    try:
        model.Params.OutputFlag = 1
        model.Params.MIPGap = mip_gap
        model.Params.TimeLimit = time_limit
        model.Params.NumericFocus = 1
    except Exception:
        pass

    # Optimize
    # Optimize
    model.optimize()
    logger.info("Solver status: %s", _status_str(model.Status))
    logger.info("Solver status: %s", _status_str(model.Status))

    # Save JSON solution (used as training data and for later analysis)
    # Save JSON solution (used as training data and for later analysis)
    if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        try:
            out_path = save_solution_as_json(sc, model, instance_name=name)
            logger.info("Saved JSON solution: %s", out_path)
        except Exception as e:
            logger.warning("Failed to save JSON solution for '%s': %s", name, e)

        if save_logs:
            # Also write human-readable logs and verification
            run_id = allocate_run_id(sc.name)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                sol_fname = make_log_filename(
                    kind="solution", scenario=sc.name, run_id=run_id, ts=ts
                )
                sol_path = save_solution_to_log(sc, model, filename=sol_fname)
                logger.info("Full solution saved to: %s", sol_path)
            except Exception:
                pass
            try:
                ver_fname = make_log_filename(
                    kind="verify", scenario=sc.name, run_id=run_id, ts=ts
                )
                ver_path = verify_solution_to_log(sc, model, filename=ver_fname)
                logger.info("Verification report saved to: %s", ver_path)
            except Exception:
                pass
        return True, str(out_json_path)
    else:
        return False, None


def main():
    ap = argparse.ArgumentParser(
        description="Solve UnitCommitment.jl SCUC instances. Defaults keep everything OFF (no warm start, no logs)."
    )
    # What to solve
    ap.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Explicit dataset names (e.g., matpower/case14/2017-01-01).",
    )
    ap.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Tokens to include in dataset name (e.g., case57). If omitted, --instances is required.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Solve at most N instances (0 => unlimited)",
    )

    # Solver controls
    ap.add_argument("--time-limit", type=int, default=600, help="Gurobi time limit [s]")
    ap.add_argument("--mip-gap", type=float, default=0.05, help="Gurobi MIP gap")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Do not re-solve instances that already have a JSON solution.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List instances and exit (do not solve)",
    )

    # Warm start (all OFF by default)
    ap.add_argument(
        "--warm",
        dest="use_warm_start",
        action="store_true",
        default=False,
        help="Use warm start (generate from pre-trained DB if available).",
    )
    ap.add_argument(
        "--require-pretrained",
        action="store_true",
        default=False,
        help="Only use pre-trained warm-start index (do not auto-build from outputs).",
    )
    ap.add_argument(
        "--warm-mode",
        choices=["fixed", "commit-only", "as-is"],
        default="fixed",
        help="Warm-start application mode. 'fixed' = repaired for feasibility (recommended).",
    )
    ap.add_argument(
        "--warm-use-train-db",
        action="store_true",
        default=False,
        help="Restrict neighbor search to the training split of the index.",
    )

    # Logs (OFF by default)
    ap.add_argument(
        "--save-logs",
        action="store_true",
        default=False,
        help="Write human-readable solution and verification logs to src/data/logs",
    )

    args = ap.parse_args()

    # Build instance list
    instances: List[str] = []
    if args.instances:
        instances = args.instances
    elif args.include:
        logger.info("Listing remote instances from: %s", DataParams.INSTANCES_URL)
        instances = list_remote_instances(
            include_filters=args.include,
            roots=_DEFAULT_ROOTS,
            max_depth=_DEFAULT_MAX_DEPTH,
        )
        if not instances:
            logger.warning("Remote listing returned none. Falling back to local cache.")
            instances = list_local_cached_instances(include_filters=args.include)
        if not instances:
            logger.error("No instances found for include tokens: %s", args.include)
            return
    else:
        logger.error("Please pass either --instances or --include filters.")
        return

    if args.dry_run:
        for n in instances:
            print(n)
        return

    # Solve
    warm_cache = {}  # case_folder -> WarmStartProvider (loaded index)
    limit = args.limit if args.limit and args.limit > 0 else None
    solved = 0
    total = 0
    for name in instances:
        if limit is not None and total >= limit:
            break
        total += 1
        logger.info("[%d/%s] Solving %s", total, str(limit) if limit else "-", name)
        ok, out_json = _solve_one(
            name=name,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            use_warm_start=args.use_warm_start,
            require_pretrained=args.require_pretrained,
            skip_existing=args.skip_existing,
            warm_cache=warm_cache,
            warm_start_mode=args.warm_mode,
            warm_use_train_db=args.warm_use_train_db,
            save_logs=args.save_logs,
        )
        if ok:
            solved += 1

    logger.info("Done. Solved %d/%d", solved, total)


if __name__ == "__main__":
    main()
