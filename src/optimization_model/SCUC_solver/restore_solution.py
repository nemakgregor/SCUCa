from typing import Dict, List, Any
from gurobipy import Model, GRB
from src.data_preparation.data_structure import UnitCommitmentScenario


def restore_solution(scenario: UnitCommitmentScenario, model: Model) -> Dict[str, Any]:
    """
    Restore a full SCUC solution from a solved Gurobi model into Python structures.

    Returns a nested dict with:
      - objective: float
      - status: str
      - generators: dict[name] with
          commit: List[int] (0/1 over time)
          min_power_output: List[float]  (u * min_power)
          segment_power: List[List[float]] (per t: per-segment power)
          total_power: List[float]  (min + sum(segments))
      - reserves: dict[reserve_name] with
          requirement: List[float]
          shortfall: List[float]
          provided_by_gen: dict[gen_name] -> List[float]
          total_provided: List[float]
      - system:
          load: List[float]
          total_production: List[float]
    """
    T = scenario.time
    gens = scenario.thermal_units

    # Helper to convert solver status to string
    def _status_str(code: int) -> str:
        mapping = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.TIME_LIMIT: "TIME_LIMIT",
        }
        return mapping.get(code, f"STATUS_{code}")

    sol: Dict[str, Any] = {
        "objective": float(getattr(model, "ObjVal", float("nan"))),
        "status": _status_str(getattr(model, "Status", -1)),
        "generators": {},
        "system": {
            "load": [sum(b.load[t] for b in scenario.buses) for t in range(T)],
            "total_production": [0.0 for _ in range(T)],
        },
    }

    # Generator commitments and power by segments
    commit_vars = getattr(model, "commit", None)
    seg_vars = getattr(model, "gen_segment_power", None)

    for g in gens:
        Sg = len(g.segments) if g.segments else 0
        g_commit: List[int] = []
        g_min_output: List[float] = []
        g_seg_power: List[List[float]] = []
        g_total_power: List[float] = []

        for t in range(T):
            u = 0.0
            if commit_vars is not None:
                u = commit_vars[g.name, t].X
            u_int = int(round(u))
            g_commit.append(u_int)

            min_out = u * float(g.min_power[t])
            g_min_output.append(min_out)

            segs_t: List[float] = []
            if Sg > 0 and seg_vars is not None:
                for s in range(Sg):
                    segs_t.append(float(seg_vars[g.name, t, s].X))
            g_seg_power.append(segs_t)

            total = min_out + (sum(segs_t) if segs_t else 0.0)
            g_total_power.append(total)

        sol["generators"][g.name] = {
            "commit": g_commit,
            "min_power_output": g_min_output,
            "segment_power": g_seg_power,
            "total_power": g_total_power,
        }

        # accumulate system production
        for t in range(T):
            sol["system"]["total_production"][t] += sol["generators"][g.name][
                "total_power"
            ][t]

    # Reserves (if present)
    reserve_defs = scenario.reserves
    if reserve_defs:
        reserve_vars = getattr(model, "reserve", None)
        shortfall_vars = getattr(model, "reserve_shortfall", None)
        reserves_out: Dict[str, Any] = {}

        for r in reserve_defs:
            req = [float(r.amount[t]) for t in range(T)]
            short = [0.0 for _ in range(T)]
            provided_by_gen: Dict[str, List[float]] = {}
            total_provided = [0.0 for _ in range(T)]

            # Shortfall per t
            if shortfall_vars is not None:
                for t in range(T):
                    short[t] = float(shortfall_vars[r.name, t].X)

            # Provision by eligible generators
            if reserve_vars is not None:
                for g in r.thermal_units:
                    gvals: List[float] = []
                    for t in range(T):
                        gvals.append(float(reserve_vars[r.name, g.name, t].X))
                        total_provided[t] += gvals[-1]
                    provided_by_gen[g.name] = gvals

            reserves_out[r.name] = {
                "requirement": req,
                "shortfall": short,
                "provided_by_gen": provided_by_gen,
                "total_provided": total_provided,
            }

        sol["reserves"] = reserves_out

    return sol
