from typing import Any, List
from pathlib import Path
from instances.instance_structure import UnitCommitmentScenario


def migrate(json_: dict) -> None:
    """
    Bring legacy (< 0.4) files up to date:

        * v0.2 → v0.3:  restructure reserves & generator flags
        * v0.3 → v0.4:  ensure every generator has `"Type": "Thermal"`
    """
    params = json_.get("Parameters", {})
    ver_raw = params.get("Version")
    if ver_raw is None:
        raise ValueError(
            "Input file has no Parameters['Version'] entry – please add it "
            '(e.g. {"Parameters": {"Version": "0.3"}}).'
        )

    ver = _parse_version(ver_raw)
    if ver < (0, 3):
        _migrate_to_v03(json_)
    if ver < (0, 4):
        _migrate_to_v04(json_)


def _parse_version(v):
    """Return (major, minor) tuple; treat malformed strings as (0, 0)."""
    try:
        return tuple(int(x) for x in str(v).split(".")[:2])
    except Exception:
        return (0, 0)


def _migrate_to_v03(json_: dict) -> None:
    """Match Julia’s _migrate_to_v03: create r1 spinning reserve, map flags."""
    reserves = json_.get("Reserves")
    if reserves and "Spinning (MW)" in reserves:
        amount = reserves["Spinning (MW)"]
        # Replace the old flat field with the new nested structure
        json_["Reserves"] = {
            "r1": {
                "Type": "spinning",
                "Amount (MW)": amount,
            }
        }
        # Any generator that set the legacy boolean now becomes eligible for r1
        for gen in json_.get("Generators", {}).values():
            if gen.get("Provides spinning reserves?") is True:
                gen["Reserve eligibility"] = ["r1"]


def _migrate_to_v04(json_: dict) -> None:
    """Match Julia’s _migrate_to_v04: default missing types to Thermal."""
    for gen in json_.get("Generators", {}).values():
        gen.setdefault("Type", "Thermal")


def scalar(val: Any, default: Any = None) -> Any:
    """
    Replicates Julia's scalar(x; default) helper, modified to handle scalars or lists.
    Returns a list if the input is a list, otherwise returns the scalar value or default.
    """
    if val is None:
        return default
    if isinstance(val, list):
        return val
    return [val] if isinstance(default, list) else val


def repair(scenario: UnitCommitmentScenario) -> None:
    """
    Julia's repair! performs several tasks:
      • fills commitment_status for must-run units
      • clamps initial conditions
      • builds ISF/LODF if missing
    Here we implement minimal sanity checks.
    """
    for tu in scenario.thermal_units:
        # ensure commitment_status consistent with must_run
        for t, mr in enumerate(tu.must_run):
            if mr is True:
                tu.commitment_status[t] = True


def repair_scenario_names_and_probabilities(
    scenarios: List["UnitCommitmentScenario"], paths: List[str]
) -> None:
    """Normalize names and probabilities so they sum to 1."""
    total = sum(sc.probability for sc in scenarios)
    for sc, p in zip(scenarios, paths):
        if not sc.name:
            sc.name = Path(p).stem.split(".")[0]
        sc.probability /= total
