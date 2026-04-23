from __future__ import annotations

import math
from typing import Iterable, List


FEATURE_VERSION = 2
SUMMARY_DIM = 4
SUMMARY_WEIGHT = 1.0


def _as_float_list(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values]


def zscore_profile(values: Iterable[float]) -> List[float]:
    vec = _as_float_list(values)
    if not vec:
        return []
    mean = sum(vec) / len(vec)
    var = sum((x - mean) ** 2 for x in vec) / max(1, len(vec) - 1)
    std = math.sqrt(var) if var > 0 else 1.0
    return [(x - mean) / std for x in vec]


def load_summary_features(values: Iterable[float]) -> List[float]:
    vec = _as_float_list(values)
    if not vec:
        return [0.0] * SUMMARY_DIM
    mean = sum(vec) / len(vec)
    peak = max(vec)
    energy = sum(vec)
    var = sum((x - mean) ** 2 for x in vec) / max(1, len(vec) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    return [mean, peak, energy, std]


def build_load_feature_vector(values: Iterable[float]) -> List[float]:
    vec = _as_float_list(values)
    if not vec:
        return []
    return zscore_profile(vec) + load_summary_features(vec)


def _pad_profile(vec: List[float], target_len: int) -> List[float]:
    if len(vec) >= target_len:
        return vec[:target_len]
    if not vec:
        return [0.0] * target_len
    return vec + [vec[-1]] * (target_len - len(vec))


def _split_features(vec: List[float]) -> tuple[List[float], List[float]]:
    if len(vec) <= SUMMARY_DIM:
        return list(vec), []
    return list(vec[:-SUMMARY_DIM]), list(vec[-SUMMARY_DIM:])


def load_feature_distance(
    left: Iterable[float],
    right: Iterable[float],
    *,
    summary_weight: float = SUMMARY_WEIGHT,
) -> float:
    lvec = _as_float_list(left)
    rvec = _as_float_list(right)
    if not lvec and not rvec:
        return 0.0
    if not lvec or not rvec:
        other = lvec or rvec
        return math.sqrt(sum(v * v for v in other))

    lprof, lsum = _split_features(lvec)
    rprof, rsum = _split_features(rvec)

    plen = max(len(lprof), len(rprof))
    lprof = _pad_profile(lprof, plen)
    rprof = _pad_profile(rprof, plen)
    d_profile_sq = sum((a - b) ** 2 for a, b in zip(lprof, rprof))

    d_summary_sq = 0.0
    for a, b in zip(lsum, rsum):
        scale = max(abs(a), abs(b), 1.0)
        d_summary_sq += ((a - b) / scale) ** 2

    return math.sqrt(d_profile_sq + (summary_weight**2) * d_summary_sq)
