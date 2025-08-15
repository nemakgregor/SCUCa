"""ID: C-106/C-107 — Initial status and ramping with startup/shutdown limits.

Adds constraints that leverage:
  - Initial status (h)      -> to define previous commitment u_prev at t=0
  - Initial power (MW)      -> to define previous power p_prev at t=0
  - Startup limit (MW)      -> ramp up extra capability on startup
  - Shutdown limit (MW)     -> ramp down extra capability on shutdown

Define startup/shutdown indicators exactly:
  u[gen,t] - u_prev = v[gen,t] - w[gen,t]
  v[gen,t] + w[gen,t] <= 1

Ramping with startup/shutdown limits:
  p[gen,t] - p[gen,t-1] <= RU[gen] * u[gen,t-1] + SU[gen] * v[gen,t]
  p[gen,t-1] - p[gen,t] <= RD[gen] * u[gen,t]   + SD[gen] * w[gen,t]

For t=0, we use u_prev = u0 (from initial status) and p_prev = p0 (initial power).
"""

import logging
from typing import Sequence
import gurobipy as gp

logger = logging.getLogger(__name__)


def _total_power_expr(gen, t: int, commit, seg_power) -> gp.LinExpr:
    """
    Build a linear expression for total power of generator gen at time t:
      p[gen,t] = u[gen,t] * min_power[gen,t] + sum_s pseg[gen,t,s]
    """
    expr = commit[gen.name, t] * float(gen.min_power[t])
    n_segments = len(gen.segments) if gen.segments else 0
    if n_segments > 0:
        expr += gp.quicksum(seg_power[gen.name, t, s] for s in range(n_segments))
    return expr


def _initial_u(gen) -> float:
    """Return 1.0 if initial_status > 0 else 0.0 (None or <=0 => off)."""
    s = gen.initial_status
    try:
        return 1.0 if (s is not None and s > 0) else 0.0
    except Exception:
        return 0.0


def _initial_p(gen) -> float:
    """Return initial power if provided, else 0.0."""
    try:
        return float(gen.initial_power) if gen.initial_power is not None else 0.0
    except Exception:
        return 0.0


def add_constraints(
    model: gp.Model,
    generators: Sequence,
    commit,
    seg_power,
    startup,
    shutdown,
    time_periods: range,
) -> None:
    n_def = 0
    n_excl = 0
    n_ramp = 0

    # C-106: startup/shutdown definition + exclusivity
    for gen in generators:
        u0 = _initial_u(gen)
        for t in time_periods:
            if t == 0:
                model.addConstr(
                    commit[gen.name, t] - u0
                    == startup[gen.name, t] - shutdown[gen.name, t],
                    name=f"startstop_def[{gen.name},{t}]",
                )
            else:
                model.addConstr(
                    commit[gen.name, t] - commit[gen.name, t - 1]
                    == startup[gen.name, t] - shutdown[gen.name, t],
                    name=f"startstop_def[{gen.name},{t}]",
                )
            n_def += 1
            model.addConstr(
                startup[gen.name, t] + shutdown[gen.name, t] <= 1,
                name=f"startstop_exclusive[{gen.name},{t}]",
            )
            n_excl += 1

    # C-107: ramping with startup/shutdown limits (includes t=0 with initial conditions)
    for gen in generators:
        ru = float(gen.ramp_up)
        rd = float(gen.ramp_down)
        su = float(gen.startup_limit)
        sd = float(gen.shutdown_limit)

        p0 = _initial_p(gen)
        u0 = _initial_u(gen)

        for t in time_periods:
            p_t = _total_power_expr(gen, t, commit, seg_power)

            if t == 0:
                model.addConstr(
                    p_t - p0 <= ru * u0 + su * startup[gen.name, t],
                    name=f"ramp_up_init[{gen.name},{t}]",
                )
                model.addConstr(
                    p0 - p_t <= rd * commit[gen.name, t] + sd * shutdown[gen.name, t],
                    name=f"ramp_down_init[{gen.name},{t}]",
                )
                n_ramp += 2
            else:
                p_prev = _total_power_expr(gen, t - 1, commit, seg_power)
                model.addConstr(
                    p_t - p_prev
                    <= ru * commit[gen.name, t - 1] + su * startup[gen.name, t],
                    name=f"ramp_up[{gen.name},{t}]",
                )
                model.addConstr(
                    p_prev - p_t
                    <= rd * commit[gen.name, t] + sd * shutdown[gen.name, t],
                    name=f"ramp_down[{gen.name},{t}]",
                )
                n_ramp += 2

    logger.info(
        "Cons(C-106/C-107): start/stop def=%d, exclusivity=%d, ramping=%d (total=%d)",
        n_def,
        n_excl,
        n_ramp,
        n_def + n_excl + n_ramp,
    )
