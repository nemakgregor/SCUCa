from __future__ import annotations

from typing import Dict
import threading

_STATS_ATTR = "_constraint_stats"
_STATS_LOCK = threading.Lock()


def _ensure_stats_dict(model) -> Dict[str, int]:
    stats = getattr(model, _STATS_ATTR, None)
    if stats is None or not isinstance(stats, dict):
        stats = {}
        setattr(model, _STATS_ATTR, stats)
    return stats


def record_constraint_stat(model, key: str, count: int) -> None:
    if model is None or not key:
        return
    try:
        value = int(count)
    except Exception:
        return
    if value <= 0:
        return
    with _STATS_LOCK:
        stats = _ensure_stats_dict(model)
        stats[key] = stats.get(key, 0) + value


def get_constraint_stats(model) -> Dict[str, int]:
    stats = getattr(model, _STATS_ATTR, None)
    if isinstance(stats, dict):
        return dict(stats)
    return {}


def format_constraint_stats(stats: Dict[str, int]) -> str:
    if not stats:
        return ""
    return ", ".join(f"{name}={count}" for name, count in sorted(stats.items()))
