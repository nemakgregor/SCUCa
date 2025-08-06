# src/data_preparation/read_data.py
from typing import Union, Sequence
import gzip
import json

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

    instance = _read(str(local_path))

    print(f"→ Loaded instance '{name}' with {len(instance.scenarios)} scenarios.")
    print("Path to instance:", local_path)

    return instance


def _read(path_or_paths: Union[str, Sequence[str]]) -> UnitCommitmentInstance:
    """
    Generic loader.  Accepts:
      • single path (JSON or JSON.GZ) ➜ deterministic instance
      • list / tuple of paths           ➜ stochastic instance
    """
    if isinstance(path_or_paths, (list, tuple)):
        scenarios = [_read_scenario(p) for p in path_or_paths if isinstance(p, str)]
        repair_scenario_names_and_probabilities(scenarios, list(path_or_paths))
    else:
        scenarios = [_read_scenario(path_or_paths)]
        scenarios[0].name = "s1"
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
