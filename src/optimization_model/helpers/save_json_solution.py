import json
from pathlib import Path
from typing import Optional, Dict, Any

from gurobipy import Model
from src.data_preparation.data_structure import UnitCommitmentScenario
from src.optimization_model.helpers.restore_solution import restore_solution
from src.data_preparation.params import DataParams


def _instance_name_to_output_path(
    instance_name: str, base_dir: Optional[Path] = None
) -> Path:
    """
    Map an instance name like:
        'matpower/case300/2017-06-24'
    to output path:
        <base_dir>/matpower/case300/2017-06-24.json

    base_dir defaults to DataParams._OUTPUT.
    """
    base = base_dir or DataParams._OUTPUT
    rel = instance_name.strip().strip("/\\").replace("\\", "/")
    return base / rel / ".."  # placeholder to simplify next line


def compute_output_path(instance_name: str, base_dir: Optional[Path] = None) -> Path:
    """
    Compute output JSON path mirroring the input hierarchy, with .json extension.
    """
    base = base_dir or DataParams._OUTPUT
    rel = instance_name.strip().strip("/\\").replace("\\", "/")
    # If rel contains multiple segments, parent directories will be created.
    # Replace trailing ".json" or ".json.gz" if mistakenly passed with extension.
    if rel.endswith(".json.gz"):
        rel = rel[: -len(".json.gz")]
    elif rel.endswith(".json"):
        rel = rel[: -len(".json")]
    out_path = base / rel
    # ensure the filename ends with .json (not a directory)
    if out_path.suffix != ".json":
        out_path = out_path.with_suffix(".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def save_solution_as_json(
    scenario: UnitCommitmentScenario,
    model: Model,
    instance_name: str,
    out_base_dir: Optional[Path] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a SCUC solution as JSON in src/data/output mirroring the input hierarchy.

    Example mapping:
      input:  src/data/input/matpower/case300/2017-06-24.json.gz
      output: src/data/output/matpower/case300/2017-06-24.json

    Returns the path to the written JSON file.
    """
    out_path = compute_output_path(instance_name, base_dir=out_base_dir)

    sol = restore_solution(scenario, model)

    # Minimal metadata to facilitate ML usage and traceability
    meta: Dict[str, Any] = {
        "instance_name": instance_name,
        "scenario_name": scenario.name,
        "time_steps": scenario.time,
        "time_step_min": scenario.time_step,
    }
    if extra_meta:
        meta.update(extra_meta)

    payload: Dict[str, Any] = {
        "meta": meta,
        "objective": sol.get("objective"),
        "status": sol.get("status"),
        "system": sol.get("system"),
        "generators": sol.get("generators"),
        "reserves": sol.get("reserves", {}),
        "network": sol.get("network", {}),
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
