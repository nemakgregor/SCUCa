import argparse
import logging
import re
from urllib.parse import urljoin, urlparse
from typing import Iterable, List, Set, Tuple, Optional

import requests
from gurobipy import GRB

from src.data_preparation.params import DataParams
from src.data_preparation.read_data import read_benchmark
from src.optimization_model.SCUC_solver.scuc_model_builder import build_model
from src.optimization_model.helpers.save_json_solution import (
    save_solution_as_json,
    compute_output_path,
)
from src.optimization_model.helpers.perf_logger import PerfCSVLogger

logger = logging.getLogger(__name__)
# NOTE:
# Do NOT call logging.basicConfig() at import time. This module is imported by others
# (e.g., src/paper/experiments.py). Configuring the root logger here causes duplicate
# logs and interferes with caller-controlled logging.


def _fetch_links(url: str, timeout: int = 30) -> List[str]:
    """
    Fetch an HTML index page and return list of href-links (as strings).
    Non-HTML responses return empty list.
    """
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        text = r.text
    except Exception as e:
        logger.warning("Failed to fetch listing '%s': %s", url, e)
        return []

    # crude anchor parser; sufficient for simple directory index listings
    hrefs = re.findall(r'href=[\'"]([^\'"]+)[\'"]', text, flags=re.IGNORECASE)
    return hrefs


def _normalize_href(base_url: str, href: str) -> Optional[str]:
    """
    Make href absolute and canonical. Ignore fragments and query.
    """
    if not href or href.startswith("#"):
        return None
    abs_url = urljoin(base_url if base_url.endswith("/") else base_url + "/", href)
    return abs_url


def list_remote_instances(
    include_filters: Optional[List[str]] = None,
    roots: Optional[List[str]] = None,
    max_depth: int = 4,
) -> List[str]:
    """
    Recursively list remote .json.gz instances and return dataset names
    relative to DataParams.INSTANCES_URL without extension.

    Example return values:
      - 'test/case14'
      - 'matpower/case30/2017-06-24'
      - 'matpower/case57/2017-01-01'
    """
    base = DataParams.INSTANCES_URL.rstrip("/") + "/"
    root_urls: List[str] = []
    if roots:
        for r in roots:
            r = str(r).strip().strip("/\\")
            root_urls.append(urljoin(base, r + "/"))
    else:
        root_urls = [base]

    visited: Set[str] = set()
    results: List[str] = []

    base_path = urlparse(base).path

    def _walk(url: str, depth: int):
        if url in visited or depth > max_depth:
            return
        visited.add(url)
        links = _fetch_links(url)
        if not links:
            return

        for href in links:
            abs_url = _normalize_href(url, href)
            if abs_url is None:
                continue
            if abs_url in visited:
                continue
            # Only traverse under the same host/base
            if not abs_url.startswith(base):
                continue

            if abs_url.endswith("/"):
                _walk(abs_url, depth + 1)
            elif abs_url.lower().endswith(".json.gz"):
                # Derive dataset name relative to base path without extension
                path = urlparse(abs_url).path
                if not path.startswith(base_path):
                    continue
                rel = path[len(base_path) :].lstrip("/")
                if rel.endswith(".json.gz"):
                    rel = rel[: -len(".json.gz")]
                dataset_name = rel.replace("\\", "/")
                # Apply include filters (if any)
                if include_filters:
                    if not any(tok in dataset_name for tok in include_filters):
                        continue
                results.append(dataset_name)

    for ru in root_urls:
        _walk(ru, depth=0)

    # Make unique and sorted for reproducibility
    return sorted(set(results))


def list_local_cached_instances(
    include_filters: Optional[List[str]] = None,
) -> List[str]:
    """
    Fallback: list locally cached .json.gz under src/data/input and return dataset
    names without extension, filtered by include_filters if provided.
    """
    base = DataParams._CACHE.resolve()
    out: List[str] = []
    for p in base.rglob("*.json.gz"):
        try:
            rel = p.resolve().relative_to(base).as_posix()
        except Exception:
            continue
        if rel.endswith(".json.gz"):
            rel = rel[: -len(".json.gz")]
        if include_filters:
            if not any(tok in rel for tok in include_filters):
                continue
        out.append(rel)
    return sorted(set(out))


def solve_instance(
    name: str,
    time_limit: int,
    mip_gap: float,
    skip_existing: bool = True,
    perf_logger: Optional[PerfCSVLogger] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Solve a single instance by name (e.g., 'matpower/case30/2017-06-24').

    Returns:
      (success, output_json_path, error_message_if_any)
    """
    try:
        out_json_path = compute_output_path(name)
        if skip_existing and out_json_path.is_file():
            logger.info("Skipping existing solution: %s", out_json_path)
            return True, str(out_json_path), None

        inst = read_benchmark(name, quiet=True)
        sc = inst.deterministic
        logger.info(
            "Building model: '%s' (units=%d, lines=%d, T=%d, dt=%dmin)",
            name,
            len(sc.thermal_units),
            len(sc.lines),
            sc.time,
            sc.time_step,
        )
        model = build_model(sc)

        try:
            model.Params.OutputFlag = 1
            model.Params.MIPGap = mip_gap
            model.Params.TimeLimit = time_limit
            model.Params.NumericFocus = 1
        except Exception:
            pass

        model.optimize()
        status = model.Status

        # Always log performance for this attempt (even if infeasible or interrupted)
        if perf_logger is not None:
            try:
                csv_path = perf_logger.append_result(name, sc, model)
                logger.info("Appended performance row to: %s", csv_path)
            except Exception as e:
                logger.warning("Failed to append performance CSV row: %s", e)

        if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            out_path = save_solution_as_json(sc, model, instance_name=name)
            logger.info("Saved JSON solution: %s", out_path)
            return True, str(out_path), None
        else:
            msg = f"Unsolved: status={status}"
            logger.warning("%s for '%s'", msg, name)
            return False, None, msg
    except Exception as e:
        emsg = f"Exception while solving '{name}': {e}"
        logger.exception(emsg)
        return False, None, emsg


def _append_to_log(filename: str, lines: Iterable[str]) -> None:
    path = DataParams._LOGS / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for ln in lines:
            fh.write(ln.rstrip("\n") + "\n")


def main():
    # Configure logging ONLY when this module is invoked as a script.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Batch solve UnitCommitment.jl instances and save JSON solutions."
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=["case57", "case30", "case14"],
        help="Filter tokens to include (e.g., case57 case30 case14)",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=["matpower", "test"],
        help="Top-level folders to search under the base URL",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum recursion depth for remote listing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of instances to solve (0 => unlimited)",
    )
    parser.add_argument(
        "--time-limit", type=int, default=600, help="Gurobi time limit in seconds"
    )
    parser.add_argument("--mip-gap", type=float, default=0.05, help="Gurobi MIP gap")
    parser.add_argument(
        "--option",
        type=str,
        default="basic",
        help="Technique/option tag placed into CSV filename (e.g., basic, warm_start, redundant_constraints)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip instances that already have an output JSON",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Do not skip existing outputs",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List instances but do not solve"
    )

    args = parser.parse_args()

    # 1) List remote; if empty, fall back to locally cached
    logger.info("Listing remote instances from: %s", DataParams.INSTANCES_URL)
    instances = list_remote_instances(
        include_filters=args.include, roots=args.roots, max_depth=args.max_depth
    )
    if not instances:
        logger.warning(
            "Remote listing returned no instances. Falling back to local cache."
        )
        instances = list_local_cached_instances(include_filters=args.include)

    if not instances:
        logger.error("No instances found matching filters %s", args.include)
        return

    logger.info("Found %d instance(s) to consider.", len(instances))

    if args.dry_run:
        for n in instances:
            print(n)
        return

    # Create a performance CSV logger for this technique
    perf_logger = PerfCSVLogger(
        technique=args.option, base_output_dir=DataParams._OUTPUT
    )

    # 2) Solve
    unsolved_log_name = "unsolved_instances.log"
    solved_log_name = "solved_instances.log"
    solved_count = 0
    unsolved_count = 0

    limit = args.limit if args.limit and args.limit > 0 else None
    count = 0

    for name in instances:
        if limit is not None and count >= limit:
            break
        count += 1
        logger.info("[%d/%s] Solving %s", count, str(limit) if limit else "-", name)
        ok, out_json, err = solve_instance(
            name=name,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            skip_existing=args.skip_existing,
            perf_logger=perf_logger,
        )
        if ok:
            solved_count += 1
            _append_to_log(solved_log_name, [f"{name} -> {out_json}"])
        else:
            unsolved_count += 1
            _append_to_log(unsolved_log_name, [f"{name} : {err}"])

    logger.info("Done. Solved=%d, Unsolved=%d", solved_count, unsolved_count)
    if unsolved_count > 0:
        logger.info("See unsolved log: %s", DataParams._LOGS / unsolved_log_name)
    logger.info("Solved items log: %s", DataParams._LOGS / solved_log_name)


if __name__ == "__main__":
    main()
