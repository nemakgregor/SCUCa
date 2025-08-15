"""ID: V-210/V-211 — Post-contingency redispatch variables.

- V-210: delta_up[c, g, t] >= 0  (upward deployment of reserves under contingency c)
- V-211: delta_down[c, g, t] >= 0 (downward curtailment of above-min energy under contingency c)

These variables are bounded in constraints/contingencies.py by:
- delta_up[c,g,t] <= sum_k r[k,g,t]
- delta_down[c,g,t] <= sum_s pseg[g,t,s]
and set to 0 for outaged units.
"""

import logging
from typing import Sequence, Tuple
import gurobipy as gp

logger = logging.getLogger(__name__)


def add_variables(
    model: gp.Model,
    contingencies: Sequence,
    generators: Sequence,
    time_periods: range,
) -> Tuple[gp.tupledict, gp.tupledict]:
    idx = [
        (c.name, g.name, t)
        for c in contingencies
        for g in generators
        for t in time_periods
    ]
    before = model.NumVars
    delta_up = model.addVars(idx, lb=0.0, name="cont_delta_up")
    delta_dn = model.addVars(idx, lb=0.0, name="cont_delta_down")

    model.__dict__["cont_delta_up"] = delta_up
    model.__dict__["cont_delta_down"] = delta_dn

    logger.info(
        "Vars(V-210/V-211): contingency redispatch added=%d (conts=%d, gens=%d, T=%d)",
        model.NumVars - before,
        len(contingencies),
        len(generators),
        len(time_periods),
    )
    return delta_up, delta_dn
