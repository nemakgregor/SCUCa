import argparse
import logging
from datetime import datetime
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
from src.data_preparation.params import DataParams

# Warm-start provider and instance listers
from src.ml_models.warm_start import WarmStartProvider
from src.optimization_model.SCUC_solver.solve_instances import (
    list_remote_instances,
    list_local_cached_instances,
)

logging.basicConfig(level=logging.INFO)

logging.getLogger("gurobipy").setLevel(logging.WARNING)
logging.getLogger("gurobipy").propagate = False

logger = logging.getLogger(__name__)


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
) -> Tuple[bool, Optional[str]]:
    """
    Solve one dataset by name; optionally apply a warm start if available.
    Returns (success, output_json_path_if_any).
    """
    out_json_path = compute_output_path(name)
    if skip_existing and out_json_path.is_file():
        logger.info("Skipping existing solution: %s", out_json_path)

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

    # Warm-start prepare
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
            warm_path = wsp.generate_and_save_warm_start(name)
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
            assigned = wsp.apply_warm_start_to_model(
                model, sc, name, mode=warm_start_mode
            )
            if assigned > 0:
                logger.info(
                    "Warm-start applied (%s): Start set on %d variable(s).",
                    warm_start_mode,
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
    model.optimize()
    logger.info("Solver status: %s", _status_str(model.Status))

    # Save JSON solution (used as training data and for later analysis)
    if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        try:
            out_path = save_solution_as_json(sc, model, instance_name=name)
            logger.info("Saved JSON solution: %s", out_path)
        except Exception as e:
            logger.warning("Failed to save JSON solution for '%s': %s", name, e)

        # Also write human-readable logs (optional)
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
        description="Solve selected instances with optional ML warm-start."
    )
    ap.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Explicit dataset names (e.g., matpower/case14/2017-01-01). If omitted, we'll list --include filters.",
    )
    ap.add_argument(
        "--include",
        nargs="*",
        default=["case57"],
        help="Tokens to include in dataset name",
    )
    ap.add_argument(
        "--roots",
        nargs="*",
        default=["matpower", "test"],
        help="Top-level folders to search remotely",
    )
    ap.add_argument("--max-depth", type=int, default=4, help="Depth for remote listing")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Solve at most N instances (0 => unlimited)",
    )
    ap.add_argument("--time-limit", type=int, default=600, help="Gurobi time limit [s]")
    ap.add_argument("--mip-gap", type=float, default=0.05, help="Gurobi MIP gap")
    ap.add_argument(
        "--use-warm-start",
        action="store_true",
        default=True,
        help="Enable warm start application",
    )
    ap.add_argument(
        "--no-warm-start",
        dest="use_warm_start",
        action="store_false",
        help="Disable warm start",
    )
    ap.add_argument(
        "--require-pretrained",
        action="store_true",
        help="Only use pre-trained indexes (do not build in-memory if missing)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip instances that already have an output JSON",
    )
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument(
        "--dry-run", action="store_true", help="List instances and exit (do not solve)"
    )
    ap.add_argument(
        "--warm-start-mode",
        choices=["repair", "commit-only", "as-is"],
        default="repair",
        help="How to apply warm start: 'repair' (default), 'commit-only', or 'as-is'.",
    )
    args = ap.parse_args()

    # Build instance list
    instances: List[str]
    if args.instances:
        instances = args.instances
    else:
        logger.info("Listing remote instances from: %s", DataParams.INSTANCES_URL)
        instances = list_remote_instances(
            include_filters=args.include, roots=args.roots, max_depth=args.max_depth
        )
        if not instances:
            logger.warning("Remote listing returned none. Falling back to local cache.")
            instances = list_local_cached_instances(include_filters=args.include)
        if not instances:
            logger.error("No instances found.")
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
            warm_start_mode=args.warm_start_mode,
        )
        if ok:
            solved += 1

    logger.info("Done. Solved %d/%d", solved, total)


if __name__ == "__main__":
    main()
