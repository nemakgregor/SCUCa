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
          initial_status_steps: int or None
          initial_u: int (0/1)
          initial_power: float
          commit: List[int] (0/1 over time)
          startup: List[int] (0/1 over time)
          shutdown: List[int] (0/1 over time)
          min_power_output: List[float]  (u * min_power)
          segment_power: List[List[float]] (per t: per-segment power)
          total_power: List[float]  (min + sum(segments))
      - reserves: dict[reserve_name] with
          requirement: List[float]
          shortfall: List[float]
          provided_by_gen: dict[gen_name] -> List[float]
          total_provided: List[float]
      - network:
          lines: dict[line_name] with
            source: str
            target: str
            flow: List[float]
            limit: List[float]  (normal limit)
            overflow_pos: List[float]
            overflow_neg: List[float]
            penalty: List[float] (per-time penalty $/MW)
      - system:
          load: List[float]
          total_production: List[float]
    """
    T = scenario.time
    generators = scenario.thermal_units

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

    solution: Dict[str, Any] = {
        "objective": float(getattr(model, "ObjVal", float("nan"))),
        "status": _status_str(getattr(model, "Status", -1)),
        "generators": {},
        "system": {
            "load": [sum(b.load[t] for b in scenario.buses) for t in range(T)],
            "total_production": [0.0 for _ in range(T)],
        },
    }

    # Model attributes created by SCUC builders
    commit_vars = getattr(model, "commit", None)
    seg_vars = getattr(model, "gen_segment_power", None)
    startup_vars = getattr(model, "startup", None)
    shutdown_vars = getattr(model, "shutdown", None)

    for gen in generators:
        n_segments = len(gen.segments) if gen.segments else 0
        gen_commit_list: List[int] = []
        gen_startup_list: List[int] = []
        gen_shutdown_list: List[int] = []
        gen_min_output_list: List[float] = []
        gen_segment_power_list: List[List[float]] = []
        gen_total_power_list: List[float] = []

        initial_status_steps = (
            gen.initial_status if gen.initial_status is not None else None
        )
        initial_u = (
            1 if (gen.initial_status is not None and gen.initial_status > 0) else 0
        )
        initial_power = (
            float(gen.initial_power) if gen.initial_power is not None else 0.0
        )

        for t in range(T):
            u_val = 0.0
            if commit_vars is not None:
                u_val = commit_vars[gen.name, t].X
            gen_commit_list.append(int(round(u_val)))

            st = startup_vars[gen.name, t].X if startup_vars is not None else 0.0
            sh = shutdown_vars[gen.name, t].X if shutdown_vars is not None else 0.0
            gen_startup_list.append(int(round(st)))
            gen_shutdown_list.append(int(round(sh)))

            min_output = u_val * float(gen.min_power[t])
            gen_min_output_list.append(min_output)

            segs_t: List[float] = []
            if n_segments > 0 and seg_vars is not None:
                for s in range(n_segments):
                    segs_t.append(float(seg_vars[gen.name, t, s].X))
            gen_segment_power_list.append(segs_t)

            total = min_output + (sum(segs_t) if segs_t else 0.0)
            gen_total_power_list.append(total)

        solution["generators"][gen.name] = {
            "initial_status_steps": initial_status_steps,
            "initial_u": initial_u,
            "initial_power": initial_power,
            "commit": gen_commit_list,
            "startup": gen_startup_list,
            "shutdown": gen_shutdown_list,
            "min_power_output": gen_min_output_list,
            "segment_power": gen_segment_power_list,
            "total_power": gen_total_power_list,
        }

        # accumulate system production
        for t in range(T):
            solution["system"]["total_production"][t] += solution["generators"][
                gen.name
            ]["total_power"][t]

    # Reserves (if present)
    reserve_defs = scenario.reserves
    if reserve_defs:
        reserve_vars = getattr(model, "reserve", None)
        shortfall_vars = getattr(model, "reserve_shortfall", None)
        reserves_out: Dict[str, Any] = {}

        for reserve_def in reserve_defs:
            req = [float(reserve_def.amount[t]) for t in range(T)]
            short = [0.0 for _ in range(T)]
            provided_by_gen: Dict[str, List[float]] = {}
            total_provided = [0.0 for _ in range(T)]

            # Shortfall per t
            if shortfall_vars is not None:
                for t in range(T):
                    short[t] = float(shortfall_vars[reserve_def.name, t].X)

            # Provision by eligible generators
            if reserve_vars is not None:
                for gen in reserve_def.thermal_units:
                    gvals: List[float] = []
                    for t in range(T):
                        gvals.append(
                            float(reserve_vars[reserve_def.name, gen.name, t].X)
                        )
                        total_provided[t] += gvals[-1]
                    provided_by_gen[gen.name] = gvals

            reserves_out[reserve_def.name] = {
                "requirement": req,
                "shortfall": short,
                "provided_by_gen": provided_by_gen,
                "total_provided": total_provided,
            }

        solution["reserves"] = reserves_out

    # Network: line flows and overflows
    lines = scenario.lines
    flow_vars = getattr(model, "line_flow", None)
    ovp_vars = getattr(model, "line_overflow_pos", None)
    ovn_vars = getattr(model, "line_overflow_neg", None)
    if lines:
        net_out: Dict[str, Any] = {"lines": {}}
        for line in lines:
            flow = [0.0 for _ in range(T)]
            ovp = [0.0 for _ in range(T)]
            ovn = [0.0 for _ in range(T)]
            limit = [float(line.normal_limit[t]) for t in range(T)]
            penalty = [float(line.flow_penalty[t]) for t in range(T)]
            if flow_vars is not None:
                for t in range(T):
                    flow[t] = float(flow_vars[line.name, t].X)
            if ovp_vars is not None:
                for t in range(T):
                    ovp[t] = float(ovp_vars[line.name, t].X)
            if ovn_vars is not None:
                for t in range(T):
                    ovn[t] = float(ovn_vars[line.name, t].X)
            net_out["lines"][line.name] = {
                "source": line.source.name,
                "target": line.target.name,
                "flow": flow,
                "limit": limit,
                "overflow_pos": ovp,
                "overflow_neg": ovn,
                "penalty": penalty,
            }
        solution["network"] = net_out

    return solution
