"""Fix commitment variables to known statuses when provided by data."""

import gurobipy as gp
from typing import Sequence


def add_constraints(
    model: gp.Model,
    generators: Sequence,
    commit,
    time_periods: range,
) -> None:
    for g in generators:
        for t in time_periods:
            status = g.commitment_status[t] if g.commitment_status else None
            if status is True:
                model.addConstr(
                    commit[g.name, t] == 1, name=f"fix_commit_on[{g.name},{t}]"
                )
            elif status is False:
                model.addConstr(
                    commit[g.name, t] == 0, name=f"fix_commit_off[{g.name},{t}]"
                )
