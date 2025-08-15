"""ID: V-203/V-204 — Reserve variables.

- V-203: Reserve provision variables r[k,g,t] >= 0
- V-204: Reserve shortfall variables s[k,t] >= 0
"""

import logging
from typing import Sequence
import gurobipy as gp

logger = logging.getLogger(__name__)


def add_variables(
    model: gp.Model,
    reserves: Sequence,
    generators: Sequence,  # kept for signature consistency; not used directly
    time_periods: range,
):
    """
    Create:
      - reserve provision variables r[k,g,t] for each reserve product k and eligible generator g
      - shortfall variables s[k,t] for each reserve product k

    Returns
    -------
    (Var, Var)
        (reserve, shortfall) Gurobi Var dicts keyed by (k, g, t) and (k, t)
    """
    prov_idx = []
    eligible_count = 0
    for r in reserves:
        for g in r.thermal_units:
            eligible_count += 1
            for t in time_periods:
                prov_idx.append((r.name, g.name, t))
    before = model.NumVars
    reserve = model.addVars(prov_idx, lb=0.0, name="r")

    short_idx = [(r.name, t) for r in reserves for t in time_periods]
    shortfall = model.addVars(short_idx, lb=0.0, name="r_shortfall")
    model.__dict__["reserve"] = reserve
    model.__dict__["reserve_shortfall"] = shortfall

    logger.info(
        "Vars(V-203/V-204): reserve r[k,g,t] added=%d (eligible_pairs=%d*T), shortfall s[k,t] added=%d",
        len(prov_idx),
        eligible_count,
        len(short_idx),
    )
    return reserve, shortfall
