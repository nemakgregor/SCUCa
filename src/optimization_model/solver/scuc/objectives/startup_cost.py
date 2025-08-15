import gurobipy as gp
from typing import Sequence


def _startup_cost_for_gen(g) -> float:
    """
    Chosen policy: 'hot start' cost = minimum of provided startup categories.
    If no categories, return 0.0.

    Rationale:
      - We currently model a single startup indicator per period (no downtime-dependent category selection).
      - Using the 'hot' (lowest) cost keeps consistency and avoids over-charging.
      - Can be extended later to downtime-dependent categories if needed.
    """
    try:
        if g.startup_categories:
            return float(min(cat.cost for cat in g.startup_categories))
    except Exception:
        pass
    return 0.0


def build_expression(
    model: gp.Model, generators: Sequence, startup, time_periods: range
) -> gp.LinExpr:
    """
    Sum of startup costs:
      sum_{g,t} startup[g,t] * StartupCost_g

    StartupCost_g is computed with _startup_cost_for_gen (currently 'hot' cost).
    """
    expr = gp.LinExpr()
    for g in generators:
        s_cost = _startup_cost_for_gen(g)
        if s_cost == 0.0:
            continue
        for t in time_periods:
            expr += startup[g.name, t] * s_cost
    return expr
