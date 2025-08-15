from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from src.data_preparation.data_structure import UnitCommitmentScenario, ThermalUnit
from src.optimization_model.helpers.restore_solution import restore_solution
from src.data_preparation.params import DataParams


def _fmt_float(x: float, nd: int = 6) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _generator_segment_costs(
    gen: ThermalUnit, t: int, seg_prod_t: List[float]
) -> List[float]:
    """Compute cost per segment produced at time t."""
    costs: List[float] = []
    n_segments = len(gen.segments) if gen.segments else 0
    for s in range(n_segments):
        c = float(gen.segments[s].cost[t]) * float(seg_prod_t[s])
        costs.append(c)
    return costs


def _avg_incremental_cost_at_t(
    gens: List[ThermalUnit], t: int, seg_prod: Dict[str, List[List[float]]]
) -> Optional[float]:
    """Weighted average marginal cost at time t across all produced segments."""
    total_mw = 0.0
    total_cost = 0.0
    for gen in gens:
        n_segments = len(gen.segments) if gen.segments else 0
        if n_segments == 0:
            continue
        segs_t = seg_prod[gen.name][t]
        for s in range(n_segments):
            mw = float(segs_t[s])
            if mw > 0:
                price = float(gen.segments[s].cost[t])
                total_mw += mw
                total_cost += price * mw
    if total_mw <= 0:
        return None
    return total_cost / total_mw


def _startup_cost_for_gen(gen: ThermalUnit) -> float:
    """Hot-start policy: min of startup category costs; 0 if none."""
    try:
        if gen.startup_categories:
            return float(min(cat.cost for cat in gen.startup_categories))
    except Exception:
        pass
    return 0.0


def save_solution_to_log(
    scenario: UnitCommitmentScenario,
    model,
    out_dir: Optional[Path] = None,
    filename: Optional[str] = None,
) -> Path:
    """
    Write a detailed solution log with improved readability.

    Contents
    - Solver/instance summary
    - System load and production by time
    - Reserve details by product and time (requirement, provided, shortfall)
    - Line flows by time (flow, limit, overflow+, overflow-)
    - Generator-by-generator details per time (aligned columns):
        * initial status and initial power
        * commitment, startup/shutdown, min/max/headroom
        * produced min power, segment-by-segment: produced, capacity, price, cost
        * energy totals, reserve provided per product, remaining headroom
        * generator total cost components (min, segments, startup)
    - Objective breakdown (incl. reserve penalty, base and contingency line overflow penalty, startup)
    - Simple feasibility checks (energy balance)
    """
    out_dir = out_dir or DataParams._LOGS
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        scen_name = scenario.name or "scenario"
        filename = f"SCUC_{scen_name}_run001_{ts}.solution.log"
    out_path = out_dir / filename

    # Restore solution in structured form
    sol = restore_solution(scenario, model)

    T = scenario.time
    generators = scenario.thermal_units
    buses = scenario.buses

    # Unpack for convenience
    sys_load = sol["system"]["load"]  # length T
    sys_prod = sol["system"]["total_production"]  # length T

    gen_sol = sol["generators"]  # dict[gen_name] -> dict with lists over time

    # Precompute costs per generator and time
    gen_min_cost: Dict[str, List[float]] = {}
    gen_segs_cost: Dict[str, List[List[float]]] = {}
    gen_startup_cost: Dict[str, List[float]] = {}
    gen_total_cost: Dict[str, List[float]] = {}

    startup_vars = getattr(model, "startup", None)

    for gen in generators:
        gsol = gen_sol[gen.name]
        min_cost_t: List[float] = []
        segs_cost_t: List[List[float]] = []
        startup_cost_t: List[float] = []
        total_cost_t: List[float] = []
        s_cost_param = _startup_cost_for_gen(gen)
        for t in range(T):
            # cost for min power if on
            minc = float(gsol["commit"][t]) * float(gen.min_power_cost[t])
            min_cost_t.append(minc)
            # per-segment costs
            seg_costs = _generator_segment_costs(gen, t, gsol["segment_power"][t])
            segs_cost_t.append(seg_costs)
            # startup cost this period
            st_cost = 0.0
            if startup_vars is not None and s_cost_param > 0.0:
                try:
                    st = float(startup_vars[gen.name, t].X)
                except Exception:
                    st = 0.0
                st_cost = st * s_cost_param
            startup_cost_t.append(st_cost)

            total_cost_t.append(minc + sum(seg_costs) + st_cost)

        gen_min_cost[gen.name] = min_cost_t
        gen_segs_cost[gen.name] = segs_cost_t
        gen_startup_cost[gen.name] = startup_cost_t
        gen_total_cost[gen.name] = total_cost_t

    # Average incremental energy cost at each time (over produced segments)
    from typing import Optional as _OptionalTypeHint  # local alias

    avg_incr_cost_t: List[_OptionalTypeHint[float]] = []
    seg_prod_by_gen = {
        gen.name: gen_sol[gen.name]["segment_power"] for gen in generators
    }
    for t in range(T):
        avg_incr_cost_t.append(
            _avg_incremental_cost_at_t(generators, t, seg_prod_by_gen)
        )

    # Reserve unpack
    reserves_sol = sol.get("reserves", {})
    reserve_names = list(reserves_sol.keys())

    # Network unpack
    network_sol = sol.get("network", {})
    line_sol = network_sol.get("lines", {}) if network_sol else {}
    line_names_sorted = list(line_sol.keys())

    # Compose log text with aligned columns
    lines: List[str] = []
    lines.append("===== SCUC Solution Report =====")
    lines.append(f"Scenario         : {scenario.name}")
    lines.append(f"Time steps       : {T} (step = {scenario.time_step} min)")
    lines.append(
        f"Counts           : {len(generators)} thermal units, {len(buses)} buses, {len(scenario.lines)} lines, {len(reserve_names)} reserve products"
    )
    lines.append(f"Solver status    : {sol['status']}")
    # Objective
    obj_val = sol.get("objective", float("nan"))
    try:
        lines.append(f"Objective value  : {obj_val:.6f}")
    except Exception:
        lines.append(f"Objective value  : {obj_val}")

    # MIP gap etc. if present
    try:
        mip_gap = getattr(model, "MIPGap", None)
        if mip_gap is not None:
            lines.append(f"MIP gap          : {mip_gap:.6f}")
    except Exception:
        pass
    lines.append("")

    # System load and production
    lines.append("== System load and production by time ==")
    header = f"{'t':>3}  {'load(MW)':>12}  {'production(MW)':>16}  {'balance(MW)':>13}  {'avg_incremental_cost($/MW)':>28}"
    lines.append(header)
    for t in range(T):
        bal = float(sys_prod[t]) - float(sys_load[t])
        aic = avg_incr_cost_t[t]
        aic_str = _fmt_float(aic, 6) if aic is not None else "n/a"
        lines.append(
            f"{t:>3}  {_fmt_float(sys_load[t], 4):>12}  {_fmt_float(sys_prod[t], 4):>16}  {_fmt_float(bal, 4):>13}  {aic_str:>28}"
        )
    lines.append("")

    # Reserve details
    if reserve_names:
        lines.append("== Reserves by product and time ==")
        for rname in reserve_names:
            rsol = reserves_sol[rname]
            lines.append(f"[Reserve '{rname}']")
            lines.append(
                f"{'t':>3}  {'requirement(MW)':>18}  {'provided(MW)':>14}  {'shortfall(MW)':>14}"
            )
            for t in range(T):
                lines.append(
                    f"{t:>3}  {_fmt_float(rsol['requirement'][t], 4):>18}  {_fmt_float(rsol['total_provided'][t], 4):>14}  {_fmt_float(rsol['shortfall'][t], 4):>14}"
                )
            lines.append("")
    else:
        lines.append("== Reserves: none defined in this scenario ==")
        lines.append("")

    # Line flows
    if line_names_sorted:
        lines.append("== Line flows (base case) by time ==")
        for lname in line_names_sorted:
            lsol = line_sol[lname]
            src = lsol.get("source", "?")
            tgt = lsol.get("target", "?")
            lines.append(f"[Line '{lname}' {src}->{tgt}]")
            lines.append(
                f"{'t':>3}  {'flow(MW)':>12}  {'limit(MW)':>12}  {'overflow+(MW)':>15}  {'overflow-(MW)':>15}"
            )
            for t in range(T):
                lines.append(
                    f"{t:>3}  {_fmt_float(lsol['flow'][t], 4):>12}  {_fmt_float(lsol['limit'][t], 4):>12}  {_fmt_float(lsol['overflow_pos'][t], 4):>15}  {_fmt_float(lsol['overflow_neg'][t], 4):>15}"
                )
            lines.append("")
    else:
        lines.append("== Lines: none defined in this scenario ==")
        lines.append("")

    # Generator details: per generator, per time, include segments and prices
    lines.append("== Generator details (per time step) ==")
    for gen in generators:
        gsol = gen_sol[gen.name]
        lines.append(f"[Generator '{gen.name}' | Bus='{gen.bus.name}']")
        # Initial conditions
        lines.append(
            f"   initial: status_steps={gsol['initial_status_steps']}, u0={gsol['initial_u']}, p0={_fmt_float(gsol['initial_power'], 4)}"
        )
        n_segments = len(gen.segments) if gen.segments else 0

        # Table header for time rows
        lines.append(
            f"{'t':>3}  {'commit':>6}  {'start':>6}  {'stop':>6}  "
            f"{'min(MW)':>10}  {'max(MW)':>10}  {'min_out(MW)':>13}  {'energy>min(MW)':>16}  {'total(MW)':>11}"
        )
        for t in range(T):
            committed = int(gsol["commit"][t])
            started = int(gsol["startup"][t])
            stopped = int(gsol["shutdown"][t])
            min_p = float(gen.min_power[t])
            max_p = float(gen.max_power[t])
            min_out = float(gsol["min_power_output"][t])
            segs_prod = gsol["segment_power"][t]
            energy_above_min = sum(segs_prod) if segs_prod else 0.0
            total_power = float(gsol["total_power"][t])

            lines.append(
                f"{t:>3}  {committed:>6}  {started:>6}  {stopped:>6}  "
                f"{_fmt_float(min_p, 4):>10}  {_fmt_float(max_p, 4):>10}  {_fmt_float(min_out, 4):>13}  {_fmt_float(energy_above_min, 4):>16}  {_fmt_float(total_power, 4):>11}"
            )

            # Headroom and reserves per product
            headroom_if_on = (max_p - min_p) if committed == 1 else 0.0
            reserves_by_product: Dict[str, float] = {}
            for rname in reserve_names:
                provided_map = reserves_sol[rname]["provided_by_gen"]
                reserves_by_product[rname] = float(
                    provided_map.get(gen.name, [0.0] * T)[t]
                    if isinstance(provided_map.get(gen.name), list)
                    else 0.0
                )
            total_reserve_t = sum(reserves_by_product.values())
            remaining_headroom = headroom_if_on - energy_above_min - total_reserve_t

            lines.append(
                f"       headroom_if_on={_fmt_float(headroom_if_on, 4)}, reserve_total={_fmt_float(total_reserve_t, 4)}, remaining_headroom={_fmt_float(remaining_headroom, 4)}"
            )

            # Segment details
            if n_segments > 0:
                segs_costs = _generator_segment_costs(gen, t, segs_prod)
                lines.append(
                    "       segments: idx, produced(MW), capacity(MW), price($/MW), cost($)"
                )
                for s in range(n_segments):
                    capacity = float(gen.segments[s].amount[t])
                    price = float(gen.segments[s].cost[t])
                    produced = float(segs_prod[s])
                    scost = float(segs_costs[s])
                    lines.append(
                        f"         {s:02d}, {_fmt_float(produced, 4)}, {_fmt_float(capacity, 4)}, {_fmt_float(price, 4)}, {_fmt_float(scost, 4)}"
                    )
            else:
                lines.append("       segments: none")

            # Reserve breakdown per product
            if reserve_names:
                parts = ", ".join(
                    [
                        f"{rn}={_fmt_float(reserves_by_product[rn], 4)}"
                        for rn in reserve_names
                    ]
                )
                lines.append(f"       reserves_by_product: {parts}")

            # Costs
            min_c = float(gen_min_cost[gen.name][t])
            seg_c = (
                sum(_generator_segment_costs(gen, t, segs_prod))
                if n_segments > 0
                else 0.0
            )
            su_cost = float(gen_startup_cost[gen.name][t])
            tot_c = min_c + seg_c + su_cost
            lines.append(
                f"       cost: min_out_cost={_fmt_float(min_c, 4)} | segments_cost={_fmt_float(seg_c, 4)} | startup_cost={_fmt_float(su_cost, 4)} | generator_total_cost={_fmt_float(tot_c, 4)}"
            )
        lines.append("")  # blank line between generators

    # Objective breakdown
    lines.append("== Objective breakdown ==")
    total_min = sum(sum(gen_min_cost[gen.name]) for gen in generators)
    total_seg = sum(sum(map(sum, gen_segs_cost[gen.name])) for gen in generators)
    total_su = sum(sum(gen_startup_cost[gen.name]) for gen in generators)

    total_res_shortfall_penalty = 0.0
    if reserve_names:
        for r in scenario.reserves:
            rname = r.name
            rsol = reserves_sol.get(rname)
            if not rsol:
                continue
            penalty = float(r.shortfall_penalty)
            total_res_shortfall_penalty += sum(
                penalty * float(x) for x in rsol["shortfall"]
            )

    total_line_penalty = 0.0
    if line_names_sorted:
        for lname in line_names_sorted:
            lsol = line_sol[lname]
            for t in range(T):
                total_line_penalty += float(lsol["penalty"][t]) * (
                    float(lsol["overflow_pos"][t]) + float(lsol["overflow_neg"][t])
                )

    # Contingency penalties (if any)
    cont_ovp = getattr(model, "contingency_overflow_pos", None)
    cont_ovn = getattr(model, "contingency_overflow_neg", None)
    CONT_FACTOR = 10.0
    total_cont_penalty = 0.0
    if line_names_sorted and cont_ovp is not None and cont_ovn is not None:
        for lname in line_names_sorted:
            line = next((L for L in scenario.lines if L.name == lname), None)
            if line is None:
                continue
            for t in range(T):
                pen = float(line.flow_penalty[t]) * CONT_FACTOR
                try:
                    total_cont_penalty += pen * (
                        float(cont_ovp[lname, t].X) + float(cont_ovn[lname, t].X)
                    )
                except Exception:
                    pass

    lines.append(f"Total min-output cost               = {_fmt_float(total_min, 4)}")
    lines.append(f"Total segment (energy) cost         = {_fmt_float(total_seg, 4)}")
    lines.append(f"Total startup cost                  = {_fmt_float(total_su, 4)}")
    if reserve_names:
        lines.append(
            f"Total reserve shortfall penalty     = {_fmt_float(total_res_shortfall_penalty, 4)}"
        )
    if line_names_sorted:
        lines.append(
            f"Total line overflow penalty         = {_fmt_float(total_line_penalty, 4)}"
        )
        lines.append(
            f"Total contingency overflow penalty  = {_fmt_float(total_cont_penalty, 4)}"
        )
    try:
        lines.append(f"Reported objective (solver)         = {_fmt_float(obj_val, 4)}")
        lines.append(
            f"Sum of above components             = {_fmt_float(total_min + total_seg + total_su + total_res_shortfall_penalty + total_line_penalty + total_cont_penalty, 4)}"
        )
    except Exception:
        lines.append(f"Reported objective (solver)         = {obj_val}")
        lines.append(
            f"Sum of above components             = {total_min + total_seg + total_su + total_res_shortfall_penalty + total_line_penalty + total_cont_penalty}"
        )
    lines.append("")

    # Feasibility checks
    lines.append("== Checks ==")
    max_balance_abs = max(
        abs(float(sys_prod[t]) - float(sys_load[t])) for t in range(T)
    )
    lines.append(
        f"Max |production - load| over time = {_fmt_float(max_balance_abs, 6)}"
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
