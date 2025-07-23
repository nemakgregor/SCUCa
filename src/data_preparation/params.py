from pathlib import Path


class DataParams:
    INSTANCES_URL = "https://axavier.org/UnitCommitment.jl/0.4/instances"
    _CACHE = Path(__file__).resolve().parent.parent / "data" / "input"
    _CACHE.mkdir(parents=True, exist_ok=True)
