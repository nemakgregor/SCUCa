import gurobipy as gp
from typing import Sequence


def build_expression(
    model: gp.Model, reserves: Sequence, time_periods: range
) -> gp.LinExpr:
    """
    Reserve shortfall penalty:
      sum_{k,t} shortfall[k,t] * shortfall_penalty_k
    """
    expr = gp.LinExpr()
    shortfall_vars = getattr(model, "reserve_shortfall", None)
    if reserves and shortfall_vars is not None:
        for r in reserves:
            penalty = float(r.shortfall_penalty)
            for t in time_periods:
                expr += shortfall_vars[r.name, t] * penalty
    return expr
