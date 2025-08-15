"""ID: C-104 — Reserve headroom linking (shared across products).

For each generator gen and time t:
    sum_s pseg[gen,t,s] + sum_k r[k,gen,t] <= (max[gen,t] - min[gen,t]) * u[gen,t]

This shares the above-minimum headroom across all reserve products,
preventing double counting of headroom.
"""

import logging
from typing import Sequence
import gurobipy as gp

logger = logging.getLogger(__name__)


def add_constraints(
    model: gp.Model,
    reserves: Sequence,
    commit,
    seg_power,
    reserve,
    time_periods: range,
) -> None:
    """
    Shared headroom constraint across all reserve products per generator and time.
    """
    eligible_by_gen = {}
    for r in reserves:
        for g in r.thermal_units:
            eligible_by_gen.setdefault(g.name, []).append(r)

    n = 0
    for gen_name, rlist in eligible_by_gen.items():
        gen_obj = None
        for r in reserves:
            for g in r.thermal_units:
                if g.name == gen_name:
                    gen_obj = g
                    break
            if gen_obj is not None:
                break
        if gen_obj is None:
            continue

        n_segments = len(gen_obj.segments) if gen_obj.segments else 0
        for t in time_periods:
            energy_above_min = (
                gp.quicksum(seg_power[gen_obj.name, t, s] for s in range(n_segments))
                if n_segments > 0
                else 0.0
            )
            total_reserve = gp.quicksum(reserve[r.name, gen_obj.name, t] for r in rlist)
            headroom_coeff = float(gen_obj.max_power[t]) - float(gen_obj.min_power[t])

            model.addConstr(
                energy_above_min + total_reserve
                <= headroom_coeff * commit[gen_obj.name, t],
                name=f"reserve_headroom_shared[{gen_obj.name},{t}]",
            )
            n += 1
    logger.info("Cons(C-104): reserve headroom linking added=%d", n)
