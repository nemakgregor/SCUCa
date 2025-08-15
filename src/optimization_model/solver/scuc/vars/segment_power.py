"""ID: V-202 — Segment power variables pseg[gen,t,s] >= 0."""

import logging
import gurobipy as gp
from typing import Sequence

logger = logging.getLogger(__name__)


def add_variables(model: gp.Model, generators: Sequence, time_periods: range):
    """Add segment power variables to the model.

    Returns
    -------
    Var
        Gurobi Var dict keyed by (gen_name, t, s)
    """
    idx = []
    total_segments = 0
    for gen in generators:
        n_segments = len(gen.segments) if gen.segments else 0
        total_segments += n_segments
        for t in time_periods:
            for s in range(n_segments):
                idx.append((gen.name, t, s))

    gen_segment_power = model.addVars(idx, lb=0.0, name="gen_segment_power")
    model.__dict__["gen_segment_power"] = gen_segment_power
    logger.info(
        "Vars(V-202): segment power pseg[g,t,s] added=%d (gens=%d, total_segments=%d, T=%d)",
        len(idx),
        len(generators),
        total_segments,
        len(time_periods),
    )
    return gen_segment_power
