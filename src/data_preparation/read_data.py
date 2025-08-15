from typing import Union, Sequence, Optional
import gzip
import json
import re
from pathlib import Path

from src.data_preparation.params import DataParams
from src.data_preparation.download_data import download
from src.data_preparation.data_structure import (
    UnitCommitmentInstance,
    UnitCommitmentScenario,
)
from src.data_preparation.utils import (
    from_json,
    repair_scenario_names_and_probabilities,
    migrate,
)


def _sanitize_identifier(s: str) -> str:
    """
    Turn an instance name like 'matpower/case300/2017-06-24'
    into a filesystem/log-friendly id: 'matpower_case300_2017-06-24'.
    """
    s = s.strip().strip("/\\")
    s = s.replace("\\", "/")
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")


def read_benchmark(name: str, *, quiet: bool = False) -> UnitCommitmentInstance:
    """
    Download (if necessary) a benchmark instance and load it.

    Example
    -------
    inst = read_benchmark("matpower/case3375wp/2017-02-01")
    """
    gz_name = f"{name}.json.gz"
    local_path = DataParams._CACHE / gz_name
    url = f"{DataParams.INSTANCES_URL}/{gz_name}"

    if not local_path.is_file():
        if not quiet:
            print(f"Downloading  {url}")
        download(url, local_path)

    instance = _read(str(local_path), scenario_id_hint=name)

    print(f"→ Loaded instance '{name}' with {len(instance.scenarios)} scenarios.")
    print("Path to instance:", local_path)

    return instance


def _read(
    path_or_paths: Union[str, Sequence[str]],
    scenario_id_hint: Optional[str] = None,
) -> UnitCommitmentInstance:
    """
    Generic loader.  Accepts:
      • single path (JSON or JSON.GZ) ➜ deterministic instance
      • list / tuple of paths           ➜ stochastic instance

    scenario_id_hint:
      When a single path is passed (deterministic case), use this hint to
      label the scenario name in a log-friendly way, so logs clearly show
      which dataset was solved.
    """
    if isinstance(path_or_paths, (list, tuple)):
        scenarios = [_read_scenario(p) for p in path_or_paths if isinstance(p, str)]
        repair_scenario_names_and_probabilities(scenarios, list(path_or_paths))
    else:
        scenarios = [_read_scenario(path_or_paths)]
        # Name scenario using the original "name" hint if available; otherwise
        # fall back to a sanitized path-based id.
        if scenario_id_hint:
            scenarios[0].name = _sanitize_identifier(scenario_id_hint)
        else:
            try:
                rel = (
                    Path(path_or_paths)
                    .resolve()
                    .relative_to(DataParams._CACHE.resolve())
                )
                base = rel.as_posix()
                if base.endswith(".json.gz"):
                    base = base[: -len(".json.gz")]
                elif base.endswith(".json"):
                    base = base[: -len(".json")]
                scenarios[0].name = _sanitize_identifier(base)
            except Exception:
                scenarios[0].name = "scenario"
        scenarios[0].probability = 1.0

    return UnitCommitmentInstance(time=scenarios[0].time, scenarios=scenarios)


def _read_scenario(path: str) -> UnitCommitmentScenario:
    raw = _read_json(path)
    migrate(raw)
    return from_json(raw)


def _read_json(path: str) -> dict:
    """Open JSON or JSON.GZ transparently."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
