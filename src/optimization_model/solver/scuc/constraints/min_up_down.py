"""ID: C-110/C-111/C-112 — Minimum up/down time constraints.

Implements minimum up and down times using startup/shutdown indicators (v, w):

Sliding-window constraints (classic UC formulation):
  - Minimum up-time L_u:
        sum_{k=t-L_u+1}^t v[g,k] <= u[g,t]
  - Minimum down-time L_d:
        sum_{k=t-L_d+1}^t w[g,k] <= 1 - u[g,t]

Initial-condition enforcement at the horizon boundary:
  Let s = initial_status (in steps); s > 0 means the unit has been ON for s steps,
  s < 0 means the unit has been OFF for |s| steps. Then, if s < L_u and the unit is
  ON initially, it must stay ON for (L_u - s) steps from t=0. Similarly, if s < L_d
  and the unit is OFF initially, it must stay OFF for (L_d - s) steps from t=0:

      if s > 0:  for t=0..(L_u - s - 1):  u[g,t] == 1
      if s < 0:  for t=0..(L_d - |s| - 1):  u[g,t] == 0
"""

import logging
from typing import Sequence
import gurobipy as gp

from .log_utils import record_constraint_stat

logger = logging.getLogger(__name__)


def _int_or_zero(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def add_constraints(
    model: gp.Model,
    generators: Sequence,
    commit,
    startup,
    shutdown,
    time_periods: range,
) -> None:
    T = len(time_periods)
    n_min_up = 0
    n_min_down = 0
    n_init_on = 0
    n_init_off = 0

    for g in generators:
        Lu = _int_or_zero(getattr(g, "min_up", 0))
        Ld = _int_or_zero(getattr(g, "min_down", 0))

        if Lu and Lu > 0:
            for t in time_periods:
                start_k = max(0, t - Lu + 1)
                if start_k <= t:
                    lhs = gp.quicksum(startup[g.name, k] for k in range(start_k, t + 1))
                    model.addConstr(
                        lhs <= commit[g.name, t],
                        name=f"min_up_window[{g.name},{t}]",
                    )
                    n_min_up += 1

        if Ld and Ld > 0:
            for t in time_periods:
                start_k = max(0, t - Ld + 1)
                if start_k <= t:
                    lhs = gp.quicksum(
                        shutdown[g.name, k] for k in range(start_k, t + 1)
                    )
                    model.addConstr(
                        lhs <= 1 - commit[g.name, t],
                        name=f"min_down_window[{g.name},{t}]",
                    )
                    n_min_down += 1

    for g in generators:
        Lu = _int_or_zero(getattr(g, "min_up", 0))
        Ld = _int_or_zero(getattr(g, "min_down", 0))
        s = getattr(g, "initial_status", None)
        if s is None:
            continue

        if s > 0 and Lu > 0:
            remaining_on = max(0, Lu - int(s))
            for t in range(min(remaining_on, T)):
                model.addConstr(
                    commit[g.name, t] == 1, name=f"min_up_initial_enforce[{g.name},{t}]"
                )
                n_init_on += 1

        if s < 0 and Ld > 0:
            s_off = -int(s)
            remaining_off = max(0, Ld - s_off)
            for t in range(min(remaining_off, T)):
                model.addConstr(
                    commit[g.name, t] == 0,
                    name=f"min_down_initial_enforce[{g.name},{t}]",
                )
                n_init_off += 1

    logger.info(
        "Cons(C-110/111/112): min_up=%d, min_down=%d, initial_on=%d, initial_off=%d, total=%d",
        n_min_up,
        n_min_down,
        n_init_on,
        n_init_off,
        n_min_up + n_min_down + n_init_on + n_init_off,
    )
    record_constraint_stat(model, "C-110_min_up", n_min_up)
    record_constraint_stat(model, "C-111_min_down", n_min_down)
    record_constraint_stat(model, "C-112_initial_on", n_init_on)
    record_constraint_stat(model, "C-112_initial_off", n_init_off)
    record_constraint_stat(
        model,
        "C-110_112_total",
        n_min_up + n_min_down + n_init_on + n_init_off,
    )
