from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.data_preparation.params import DataParams
from src.optimization_model.SCUC_solver.solve_instances import (
    list_remote_instances,
    list_local_cached_instances,
)


def case_tag(case_folder: str) -> str:
    t = case_folder.strip().strip("/\\").replace("\\", "/")
    return "".join(ch if ch.isalnum() else "_" for ch in t).strip("_").lower()


def ensure_case_cached(
    case_folder: str, max_download: Optional[int] = None
) -> List[str]:
    cached = list_local_cached_instances(include_filters=[case_folder])
    if cached:
        return sorted(set(cached))
    rem = list_remote_instances(
        include_filters=[case_folder], roots=["matpower", "test"], max_depth=4
    )
    if max_download and max_download > 0:
        rem = rem[:max_download]
    return sorted(set(rem))


def results_root() -> Path:
    p = Path("src") / "scuc_sr" / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def case_results_dir(case_folder: str) -> Path:
    d = results_root() / case_tag(case_folder)
    d.mkdir(parents=True, exist_ok=True)
    (d / "plots").mkdir(parents=True, exist_ok=True)
    (d / "per-instance").mkdir(parents=True, exist_ok=True)
    return d


def undirected_key(u: str, v: str) -> Tuple[str, str]:
    return tuple(sorted((u, v)))


def edge_key_str(u: str, v: str) -> str:
    """Alphabetical 'u-v' key for undirected edges."""
    a, b = sorted((str(u), str(v)))
    return f"{a}-{b}"


def map_line_to_edge_key(line) -> Tuple[str, str]:
    """Kept for compatibility (tuple return)."""
    return undirected_key(line.source.name, line.target.name)


def aggregate_counts_per_line(
    instances_results: List[Dict],
) -> Dict[str, Dict[str, int]]:
    """
    Aggregate per-line counts across instances (keys use 'u-v' strings).
    instances_results[i] must contain:
      line_name_to_edge_key: dict[line_name] -> 'u-v' (undirected)
      prune: {calls_by_line, pruned_by_line, line_added_by_line}
      lazy : {line_added_by_line}
    Returns dict keyed by 'u-v' with fields:
      - prune_events_calls, prune_events_pruned
      - prune_line_constraints_added, lazy_line_constraints_added
      - prune_plus_lazy_line_constraints_added
      (legacy lazy_events_potential/dropped removed for speed)
    """
    agg: Dict[str, Dict[str, int]] = {}
    for item in instances_results:
        name_to_key = item.get("line_name_to_edge_key", {})
        pr = item.get("prune", {})
        la = item.get("lazy", {})
        calls = pr.get("calls_by_line", {}) or {}
        pruned = pr.get("pruned_by_line", {}) or {}
        prune_added = pr.get("line_added_by_line", {}) or {}
        lazy_added = la.get("line_added_by_line", {}) or {}

        all_lines = set().union(
            calls.keys(),
            pruned.keys(),
            prune_added.keys(),
            lazy_added.keys(),
        )
        for lname in all_lines:
            key = name_to_key.get(lname)
            if not key:
                continue
            rec = agg.setdefault(
                key,
                {
                    "prune_events_calls": 0,
                    "prune_events_pruned": 0,
                    "prune_line_constraints_added": 0,
                    "lazy_line_constraints_added": 0,
                    "prune_plus_lazy_line_constraints_added": 0,
                },
            )
            rec["prune_events_calls"] += int(calls.get(lname, 0))
            rec["prune_events_pruned"] += int(pruned.get(lname, 0))
            pa = int(prune_added.get(lname, 0))
            laa = int(lazy_added.get(lname, 0))
            rec["prune_line_constraints_added"] += pa
            rec["lazy_line_constraints_added"] += laa
            rec["prune_plus_lazy_line_constraints_added"] += pa + laa
    return agg
