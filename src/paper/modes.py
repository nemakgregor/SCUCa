from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

try:
    from src.paper.experiment_spec import (
        MODE_CATALOG_MEDLARGE_FULL,
        MODE_CATALOG_SMALL,
        mode_exactness_from_id,
    )
except Exception:
    MODE_CATALOG_SMALL = []
    MODE_CATALOG_MEDLARGE_FULL = []

    def mode_exactness_from_id(mode_id: object) -> str:
        mode = str(mode_id or "").strip().upper()
        if (
            mode == "SHRINK_LAZY"
            or mode.startswith("STREDUCE")
            or "PRUNE" in mode
            or "LPSCREEN" in mode
            or "GNN" in mode
            or mode == "WARM_LAZY_ML_STACK"
        ):
            return "heuristic"
        return "exact"


RAW_BASELINE_MODE = "RAW"


_STATIC_ALIASES = {
    "RAW+GNN": "RAW_GNN",
    "RAW+COMMIT": "RAW_COMMIT_HINTS",
    "WARM+HINTS": "WARM_BRANCH_HINTS",
    "WARM+COMMIT": "WARM_COMMIT_HINTS",
    "WARM+HINTS+COMMIT": "WARM_COMMIT_HINTS",
    "WARM+GRU": "WARM_GRU",
    "WARM+HINTS+GRU": "WARM_GRU",
    "WARM+HINTS+COMMIT+GRU": "WARM_GRU",
    "WARM+LAZY": "WARM_LAZY",
    "WARM+LAZY+HINTS": "WARM_LAZY_BRANCH_HINTS",
    "WARM+LAZY+BANDIT": "WARM_LAZY_BANDIT",
    "WARM+LAZY+COMMIT": "WARM_LAZY_COMMIT_HINTS",
    "WARM+LAZY+GRU": "WARM_LAZY_GRU",
    "WARM+LAZY+COMMIT+GRU": "WARM_LAZY_COMMIT_GRU",
    "WARM+LAZY+K128": "WARM_LAZY_TOPK128",
    "WARM+PRUNE-0.10": "WARM_PRUNE_T010",
    "WARM+PRUNE-0.20": "WARM_PRUNE_T020",
    "WARM+PRUNE-0.30": "WARM_PRUNE_T030",
    "WARM+PRUNE-0.50": "WARM_PRUNE_T050",
    "WARM+PRUNE-0.80": "WARM_PRUNE_T080",
    "WARM+LPSCREEN-0.10": "WARM_LPSCREEN_T010",
    "WARM+LPSCREEN-0.20": "WARM_LPSCREEN_T020",
    "WARM+LPSCREEN-0.30": "WARM_LPSCREEN_T030",
    "WARM+LPSCREEN-0.50": "WARM_LPSCREEN_T050",
    "WARM+LPSCREEN-0.80": "WARM_LPSCREEN_T080",
}
_DISPLAY_ALIASES: dict[str, str] = {}


def _normalized_alias_key(text: object) -> str:
    return str(text or "").strip().upper().replace("-", "_").replace("+", "_")


def canonical_mode(mode: object) -> str:
    raw = str(mode or "").strip()
    if not raw:
        return raw
    upper = _normalized_alias_key(raw)
    aliases = {**_STATIC_ALIASES, **_DISPLAY_ALIASES}
    if raw in aliases:
        return aliases[raw]
    if upper in aliases:
        return aliases[upper]
    return upper


def mode_display(mode: object) -> str:
    m = canonical_mode(mode)
    tau_map = {
        "T010": "0.10",
        "T020": "0.20",
        "T030": "0.30",
        "T050": "0.50",
        "T080": "0.80",
    }
    if m == "RAW":
        return "RAW"
    if m == "RAW_GNN":
        return "RAW+GNN"
    if m == "RAW_COMMIT_HINTS":
        return "RAW+COMMIT"
    if m == "WARM_BRANCH_HINTS":
        return "WARM+HINTS"
    if m == "WARM_COMMIT_HINTS":
        return "WARM+COMMIT"
    if m == "WARM_GRU":
        return "WARM+GRU"
    if m == "LAZY_ALL":
        return "LAZY"
    if m == "LAZY_TOPK128":
        return "LAZY+K128"
    if m == "LAZY_BANDIT":
        return "LAZY+BANDIT"
    if m == "LAZY_COMMIT_HINTS":
        return "LAZY+COMMIT"
    if m == "LAZY_GNN_T060":
        return "LAZY+GNN-0.60"
    if m == "LAZY_GNN_T080":
        return "LAZY+GNN-0.80"
    if m == "LAZY_COMMIT_GNN_T060":
        return "LAZY+COMMIT+GNN-0.60"
    if m == "WARM_LAZY":
        return "WARM+LAZY"
    if m == "WARM_LAZY_TOPK128":
        return "WARM+LAZY+K128"
    if m == "WARM_LAZY_BANDIT":
        return "WARM+LAZY+BANDIT"
    if m == "WARM_LAZY_COMMIT_HINTS":
        return "WARM+LAZY+COMMIT"
    if m == "WARM_LAZY_GRU":
        return "WARM+LAZY+GRU"
    if m == "WARM_LAZY_BRANCH_HINTS":
        return "WARM+LAZY+HINTS"
    if m == "WARM_LAZY_GNN_T060":
        return "WARM+LAZY+GNN-0.60"
    if m == "WARM_LAZY_COMMIT_GNN_T060":
        return "WARM+LAZY+COMMIT+GNN-0.60"
    if m == "WARM_LAZY_COMMIT_GRU":
        return "WARM+LAZY+COMMIT+GRU"
    if m == "WARM_LAZY_ML_STACK":
        return "WARM+LAZY+ML-STACK"
    for prefix in ("WARM_PRUNE_LAZY", "WARM_LPSCREEN_LAZY", "WARM_PRUNE", "WARM_LPSCREEN"):
        if m.startswith(prefix):
            suffix = m.rsplit("_", 1)[-1]
            tau = tau_map.get(suffix, suffix)
            name = prefix.replace("_", "+")
            return f"{name}-{tau}"
    if m == "WARM_SR_LAZY":
        return "WARM+SR+LAZY"
    if m.startswith("ACTIVESET"):
        return m.replace("_", "+")
    if m.startswith("WARM_ACTIVESET"):
        return m.replace("_", "+")
    if m.startswith("SHRINK"):
        return m.replace("_", "+")
    if m.startswith("STREDUCE"):
        return m.replace("_", "+")
    return m.replace("_", "+")


def mode_exactness(mode: object) -> str:
    """Return whether a mode is intended to be full-MILP-equivalent.

    `heuristic` means the mode changes the feasible region or masks the
    contingency event set before solve, so paper claims must rely on the
    independent checker rather than on method exactness.
    """
    m = canonical_mode(mode)
    return mode_exactness_from_id(m)


def _catalog_order() -> list[str]:
    ordered = OrderedDict()
    for spec in list(MODE_CATALOG_SMALL) + list(MODE_CATALOG_MEDLARGE_FULL):
        mode_id = canonical_mode(getattr(spec, "mode_id", ""))
        if mode_id:
            ordered.setdefault(mode_id, None)
    return list(ordered.keys())


CANONICAL_MODE_ORDER = _catalog_order()


def _build_display_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for mode_id in CANONICAL_MODE_ORDER:
        display = mode_display(mode_id)
        aliases[display] = mode_id
        aliases[_normalized_alias_key(display)] = mode_id
        aliases[_normalized_alias_key(mode_id)] = mode_id
    return aliases


_DISPLAY_ALIASES.update(_build_display_aliases())


def ordered_modes(present_modes: Iterable[object] = ()) -> list[str]:
    present = [canonical_mode(m) for m in present_modes if str(m or "").strip()]
    ordered = [m for m in CANONICAL_MODE_ORDER if m]
    for mode in present:
        if mode and mode not in ordered:
            ordered.append(mode)
    return ordered
