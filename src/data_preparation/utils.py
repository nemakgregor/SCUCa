# src/data_preparation/utils.py

from pathlib import Path
from typing import List, Dict

import numpy as np
from scipy import sparse
import logging

from src.data_preparation.data_structure import (
    UnitCommitmentScenario,
    ThermalUnit,
    ProfiledUnit,
    StorageUnit,
    PriceSensitiveLoad,
    TransmissionLine,
    Contingency,
    Bus,
    Reserve,
    CostSegment,
    StartupCategory,
)

from .ptdf_lodf import compute_ptdf_lodf

from .params import DataParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ts(x, T, default=None):
    return _timeseries(x, T, default=default)


def _timeseries(val, T: int, *, default=None):
    """
    * if val is missing ➜ default
    * if val is array  ➜ keep
    * if val is scalar ➜ replicate T times
    """
    if val is None:
        return default if default is not None else [None] * T
    if isinstance(val, list):
        if len(val) != T:
            raise ValueError(f"Time-series length {len(val)} does not match T={T}")
        return val
    return [val] * T


def _scalar(val, default=None):
    return default if val is None else val


def _parse_version(v):
    """Return (major, minor) tuple; treat malformed strings as (0, 0)."""
    try:
        return tuple(int(x) for x in str(v).split(".")[:2])
    except Exception:
        return (0, 0)


def migrate(json_: dict) -> None:
    """
    Bring legacy (< 0.4) files up to date:
        * v0.2 → v0.3:  restructure reserves & generator flags
        * v0.3 → v0.4:  ensure every generator has `"Type": "Thermal"`
    """
    params = json_.get("Parameters", {})
    ver_raw = params.get("Version")
    if ver_raw is None:
        raise ValueError(
            "Input file has no Parameters['Version'] entry – please add it "
            '(e.g. {"Parameters": {"Version": "0.3"}}).'
        )

    ver = _parse_version(ver_raw)
    if ver < (0, 3):
        _migrate_to_v03(json_)
    if ver < (0, 4):
        _migrate_to_v04(json_)


def _migrate_to_v03(json_: dict) -> None:
    """Match Julia’s _migrate_to_v03: create r1 spinning reserve, map flags."""
    reserves = json_.get("Reserves")
    if reserves and "Spinning (MW)" in reserves:
        amount = reserves["Spinning (MW)"]
        # Replace the old flat field with the new nested structure
        json_["Reserves"] = {
            "r1": {
                "Type": "spinning",
                "Amount (MW)": amount,
            }
        }
        # Any generator that set the legacy boolean now becomes eligible for r1
        for gen in json_.get("Generators", {}).values():
            if gen.get("Provides spinning reserves?") is True:
                gen["Reserve eligibility"] = ["r1"]


def _migrate_to_v04(json_: dict) -> None:
    """Match Julia’s _migrate_to_v04: default missing types to Thermal."""
    for gen in json_.get("Generators", {}).values():
        gen.setdefault("Type", "Thermal")


def from_json(j: dict) -> UnitCommitmentScenario:
    # -- Time grid ---------------------------------------------------------- #
    par = j["Parameters"]
    time_horizon = (
        par.get("Time horizon (min)")
        or par.get("Time (h)")
        or par.get("Time horizon (h)")
    )
    if time_horizon is None:
        raise ValueError("Missing parameter: Time horizon")
    if "Time (h)" in par or "Time horizon (h)" in par:
        time_horizon *= 60  # convert hours → minutes

    time_horizon = int(time_horizon)
    time_step = int(_scalar(par.get("Time step (min)"), default=DataParams.TIMESTEP))
    if 60 % time_step or time_horizon % time_step:
        raise ValueError("Time step must divide 60 and the horizon")

    time_multiplier = 60 // time_step
    T = time_horizon // time_step

    # ---------------------------------------------------------------------- #
    #  Look-up tables                                                        #
    # ---------------------------------------------------------------------- #
    buses: List[Bus] = []
    lines: List[TransmissionLine] = []
    thermal_units: List[ThermalUnit] = []
    profiled_units: List[ProfiledUnit] = []
    storage_units: List[StorageUnit] = []
    reserves: List[Reserve] = []
    contingencies: List[Contingency] = []
    loads: List[PriceSensitiveLoad] = []

    name_to_bus: Dict[str, Bus] = {}
    name_to_line: Dict[str, TransmissionLine] = {}
    name_to_unit: Dict[str, ThermalUnit] = {}  # Only thermal for contingencies
    name_to_reserve: Dict[str, Reserve] = {}

    # logger.debug("Parsing JSON data structure")
    # for key, value in j.items():
    #     if isinstance(value, dict):
    #         logger.debug(f" - {key}: dict with {len(value)} keys")
    #         for subkey in list(value.keys()):
    #             logger.debug(f"    - {subkey}")
    #     elif isinstance(value, list):
    #         logger.debug(f" - {key}: list of length {len(value)}")
    #     else:
    #         logger.debug(f" - {key}: {value}")

    # ---------------------------------------------------------------------- #
    #  Helper to make sure each list has length T                            #
    # ---------------------------------------------------------------------- #

    def ts(x, *, default=None):
        return _timeseries(x, T, default=default)

    # ---------------------------------------------------------------------- #
    #  Penalties                                                             #
    # ---------------------------------------------------------------------- #
    power_balance_penalty = ts(
        par.get("Power balance penalty ($/MW)"),
        default=[DataParams.BALANCE_PENALTY] * T,
    )

    # ---------------------------------------------------------------------- #
    #  Buses                                                                 #
    # ---------------------------------------------------------------------- #
    for idx, (bname, bdict) in enumerate(j.get("Buses", {}).items(), start=1):
        if "Load (MW)" not in bdict:
            raise ValueError(f"Bus '{bname}' missing 'Load (MW)'")
        bus = Bus(
            name=bname,
            index=idx,
            load=ts(bdict.get("Load (MW)"), default=[DataParams.LOAD] * T),
        )
        name_to_bus[bname] = bus
        buses.append(bus)

    # ---------------------------------------------------------------------- #
    #  Reserves                                                              #
    # ---------------------------------------------------------------------- #
    if "Reserves" in j:
        for rname, rdict in j["Reserves"].items():
            if "Amount (MW)" not in rdict:
                raise ValueError(f"Reserve '{rname}' missing 'Amount (MW)'")
            r_type = rdict.get("Type", "spinning").lower()
            if r_type != "spinning":
                raise ValueError(f"Unsupported reserve type '{r_type}' for '{rname}'")
            r = Reserve(
                name=rname,
                type=r_type,
                amount=ts(rdict.get("Amount (MW)"), default=[DataParams.RESERVS] * T),
                thermal_units=[],
                shortfall_penalty=_scalar(
                    rdict.get("Shortfall penalty ($/MW)"),
                    default=DataParams.SHORTFALL_PENALTY,
                ),
            )
            name_to_reserve[rname] = r
            reserves.append(r)

    # ---------------------------------------------------------------------- #
    #  Generators                                                            #
    # ---------------------------------------------------------------------- #
    for uname, udict in j.get("Generators", {}).items():
        utype = udict.get("Type", "Thermal")
        bus_name = udict["Bus"]
        if bus_name not in name_to_bus:
            raise ValueError(f"Unknown bus '{bus_name}' for generator '{uname}'")
        bus = name_to_bus[bus_name]

        if utype.lower() == "thermal":
            # Production cost curve validation and parsing
            if (
                "Production cost curve (MW)" not in udict
                or "Production cost curve ($)" not in udict
            ):
                raise ValueError(f"Generator '{uname}' missing production cost curve")
            curve_mw_list = udict["Production cost curve (MW)"]
            curve_cost_list = udict["Production cost curve ($)"]

            K = len(curve_mw_list)
            if len(curve_cost_list) != K:
                raise ValueError(
                    f"Generator '{uname}' production cost curve lengths mismatch (MW: {K}, $: {len(curve_cost_list)})"
                )
            if K < 1:
                raise ValueError(f"Generator '{uname}' has no break-points (K=0)")

            if K == 1:
                # Add [0, 0] to represent off state, making K=2
                logger.warning(
                    f"Generator '{uname}' has only one break-point (K=1). Adding [0, 0] to represent off state."
                )
                curve_mw_list = [0] + curve_mw_list  # [0, 100] for g6
                curve_cost_list = [0] + curve_cost_list  # [0, 10000] for g6
                K = 2  # Update K
                logger.warning(f"Updated curve MW list for '{uname}': {curve_mw_list}")
                logger.warning(f"Updated curve cost list for '{uname}': {curve_cost_list}")

            # Convert to time series arrays and stack
            curve_mw = np.column_stack([ts(curve_mw_list[k]) for k in range(K)])
            curve_cost = np.column_stack([ts(curve_cost_list[k]) for k in range(K)])

            # Validate shapes
            if curve_mw.shape != curve_cost.shape:
                raise ValueError(
                    f"Generator '{uname}' production cost curve shapes mismatch after stacking"
                )

            T = curve_mw.shape[0]  # Time horizon from the arrays
            min_power = curve_mw[:, 0].tolist()  # [0, 0, 0, 0] for g6
            max_power = curve_mw[:, -1].tolist()  # [100, 100, 100, 100] for g6
            min_power_cost = curve_cost[:, 0].tolist()  # [0, 0, 0, 0] for g6

            # Initialize segments as list of length T, each a list of CostSegment
            segments: list[list[CostSegment]] = []

            # Validate strictly increasing MW breakpoints per time step
            mw_diffs = np.diff(curve_mw, axis=1)
            if not np.all(mw_diffs > 0):
                raise ValueError(
                    f"Generator '{uname}' production cost curve MW must be strictly increasing"
                )

            # Validate non-decreasing costs
            cost_diffs = np.diff(curve_cost, axis=1)
            if not np.all(cost_diffs >= 0):
                raise ValueError(
                    f"Generator '{uname}' production cost curve $ must be non-decreasing"
                )

            # Compute amounts and marginals vectorized (shape (T, K-1))
            amounts = mw_diffs
            marginals = np.divide(
                cost_diffs,
                amounts,
                out=np.zeros_like(amounts, dtype=float),
                where=amounts != 0,
            )

            # Build segments per time step
            for t in range(T):
                t_segments: list[CostSegment] = []
                for s in range(K - 1):
                    t_segments.append(
                        CostSegment(amount=amounts[t, s], cost=marginals[t, s])
                    )
                segments.append(t_segments)

            logger.debug(f"Generator '{uname}' segments:\n {segments}\n\n")


            # Startup categories validation and parsing
            delays = udict.get("Startup delays (h)", [DataParams.STARTUP_DELAY])
            scost = udict.get("Startup costs ($)", [DataParams.STARTUP_COST])
            if len(delays) != len(scost):
                raise ValueError(f"Startup delays/costs mismatch for '{uname}'")
            startup_categories = sorted(
                [
                    StartupCategory(
                        delay_steps=int(delays[k] * time_multiplier), cost=scost[k]
                    )
                    for k in range(len(delays))
                ],
                key=lambda cat: cat.delay_steps,
            )

            # Reserve eligibility
            unit_reserves = [
                name_to_reserve[n]
                for n in udict.get("Reserve eligibility", [])
                if n in name_to_reserve
            ]

            # Initial conditions validation
            init_p = udict.get("Initial power (MW)")
            init_s = udict.get("Initial status (h)")
            if init_p is not None and init_s is None:
                raise ValueError(f"{uname} has initial power but no status")
            if init_s is not None:
                init_s = int(init_s * time_multiplier)
                if init_p > 0 and init_s <= 0:
                    raise ValueError(f"{uname} initial power >0 but status <=0")

            # Ramp and limits validation
            ramp_up = _scalar(udict.get("Ramp up limit (MW)"), DataParams.RAMP_UP)
            ramp_down = _scalar(udict.get("Ramp down limit (MW)"), DataParams.RAMP_DOWN)
            if ramp_up <= 0 or ramp_down <= 0:
                raise ValueError(f"Invalid ramp limits for '{uname}'")

            commitment_status = ts(
                udict.get("Commitment status"),
                default=[DataParams.COMMITMENT_STATUS] * T,
            )

            tu = ThermalUnit(
                name=uname,
                bus=bus,
                max_power=max_power,
                min_power=min_power,
                must_run=ts(udict.get("Must run?", [DataParams.MUST_RUN] * T)),
                min_power_cost=min_power_cost,
                segments=segments,
                min_up=int(
                    _scalar(udict.get("Minimum uptime (h)"), DataParams.MIN_UPTIME)
                    * time_multiplier
                ),
                min_down=int(
                    _scalar(udict.get("Minimum downtime (h)"), DataParams.MIN_DOWNTIME)
                    * time_multiplier
                ),
                ramp_up=ramp_up,
                ramp_down=ramp_down,
                startup_limit=_scalar(
                    udict.get("Startup limit (MW)"), DataParams.STARTUP_LIMIT
                ),
                shutdown_limit=_scalar(
                    udict.get("Shutdown limit (MW)"), DataParams.SHUTDOWN_LIMIT
                ),
                initial_status=init_s,
                initial_power=init_p,
                startup_categories=startup_categories,
                reserves=unit_reserves,
                commitment_status=commitment_status,
            )
            bus.thermal_units.append(tu)
            thermal_units.append(tu)
            name_to_unit[uname] = tu
            for r in unit_reserves:
                r.thermal_units.append(tu)

            logger.debug(
                f"ThermalUnit '{uname}':\n"
                f"  bus={bus.name}, min_power={min_power}, max_power={max_power}\n"
                f"  min_power_cost={min_power_cost}\n"
                f"  segments={segments}\n"
                f"  min_up={tu.min_up}, min_down={tu.min_down}\n"
                f"  ramp_up={ramp_up}, ramp_down={ramp_down}\n"
                f"  startup_limit={tu.startup_limit}, shutdown_limit={tu.shutdown_limit}\n"
                f"  initial_status={init_s}, initial_power={init_p}\n"
                f"  startup_categories={startup_categories}\n"
                f"  reserves={[r.name for r in unit_reserves]}\n"
                f"  commitment_status={commitment_status}\n"
            )

        elif utype.lower() == "profiled":
            pu = ProfiledUnit(
                name=uname,
                bus=bus,
                min_power=ts(
                    _scalar(udict.get("Minimum power (MW)"), DataParams.MINIMUM_POWER)
                ),
                max_power=ts(udict.get("Maximum power (MW)")),
                cost=ts(udict.get("Cost ($/MW)"), default=[DataParams.COST] * T),
            )
            bus.profiled_units.append(pu)
            profiled_units.append(pu)

        else:
            raise ValueError(f"Unit {uname} has invalid type '{utype}'")

    # ---------------------------------------------------------------------- #
    #  Lines                                                                 #
    # ---------------------------------------------------------------------- #
    if "Transmission lines" in j:
        for idx, (lname, ldict) in enumerate(j["Transmission lines"].items(), start=1):
            source_name = ldict["Source bus"]
            target_name = ldict["Target bus"]
            if source_name not in name_to_bus or target_name not in name_to_bus:
                raise ValueError(f"Unknown bus in line '{lname}'")
            source = name_to_bus[source_name]
            target = name_to_bus[target_name]
            if "Susceptance (S)" not in ldict:
                raise ValueError(f"Line '{lname}' missing susceptance")
            line = TransmissionLine(
                name=lname,
                index=idx,
                source=source,
                target=target,
                susceptance=float(ldict["Susceptance (S)"]),
                normal_limit=ts(
                    ldict.get("Normal flow limit (MW)"),
                    default=[DataParams.FLOW_LIMIT] * T,
                ),
                emergency_limit=ts(
                    ldict.get("Emergency flow limit (MW)"),
                    default=[DataParams.EMERGENCY_FLOW_LIMIT] * T,
                ),
                flow_penalty=ts(
                    ldict.get("Flow limit penalty ($/MW)"),
                    default=[DataParams.FLOW_LIMIT_PENALTY] * T,
                ),
            )
            lines.append(line)
            name_to_line[lname] = line

    # ---------------------------------------------------------------------- #
    #  Contingencies                                                         #
    # ---------------------------------------------------------------------- #
    if "Contingencies" in j:
        for cname, cdict in j["Contingencies"].items():
            affected_lines = [
                name_to_line[line]
                for line in cdict.get("Affected lines", [])
                if line in name_to_line
            ]
            affected_units = [
                name_to_unit[u]
                for u in cdict.get("Affected units", [])
                if u in name_to_unit
            ]
            cont = Contingency(name=cname, lines=affected_lines, units=affected_units)
            contingencies.append(cont)

    # ---------------------------------------------------------------------- #
    #  Price-sensitive loads                                                 #
    # ---------------------------------------------------------------------- #
    if "Price-sensitive loads" in j:
        for lname, ldict in j["Price-sensitive loads"].items():
            bus_name = ldict["Bus"]
            if bus_name not in name_to_bus:
                raise ValueError(f"Unknown bus '{bus_name}' for load '{lname}'")
            bus = name_to_bus[bus_name]
            load = PriceSensitiveLoad(
                name=lname,
                bus=bus,
                demand=ts(ldict.get("Demand (MW)"), default=[DataParams.DEMAND] * T),
                revenue=ts(
                    ldict.get("Revenue ($/MW)"), default=[DataParams.DEMAND_REVENUE] * T
                ),
            )
            loads.append(load)
            bus.price_sensitive_loads.append(load)

    # ---------------------------------------------------------------------- #
    #  Storage units                                                         #
    # ---------------------------------------------------------------------- #
    if "Storage units" in j:
        for sname, sdict in j["Storage units"].items():
            bus_name = sdict["Bus"]
            if bus_name not in name_to_bus:
                raise ValueError(f"Unknown bus '{bus_name}' for storage '{sname}'")
            bus = name_to_bus[bus_name]
            min_level = ts(
                _scalar(sdict.get("Minimum level (MWh)"), DataParams.STORAGE_MINIMUM)
            )
            max_level = ts(sdict.get("Maximum level (MWh)"))
            su = StorageUnit(
                name=sname,
                bus=bus,
                min_level=min_level,
                max_level=max_level,
                simultaneous=ts(
                    _scalar(
                        sdict.get("Allow simultaneous charging and discharging"), True
                    )
                ),
                charge_cost=ts(
                    sdict.get("Charge cost ($/MW)"),
                    default=[DataParams.STORAGE_CHARGE_COST] * T,
                ),
                discharge_cost=ts(
                    sdict.get("Discharge cost ($/MW)"),
                    default=[DataParams.STORAGE_DISCHARGE_COST] * T,
                ),
                charge_eff=ts(
                    _scalar(
                        sdict.get("Charge efficiency"), DataParams.CHARGE_EFFICIENCY
                    )
                ),
                discharge_eff=ts(
                    _scalar(
                        sdict.get("Discharge efficiency"),
                        DataParams.DISCHARGE_EFFICIENCY,
                    )
                ),
                loss_factor=ts(
                    _scalar(sdict.get("Loss factor"), DataParams.LOSS_FACTOR)
                ),
                min_charge=ts(
                    _scalar(
                        sdict.get("Minimum charge rate (MW)"),
                        DataParams.MIN_CHARGE_RATE,
                    )
                ),
                max_charge=ts(sdict.get("Maximum charge rate (MW)")),
                min_discharge=ts(
                    _scalar(
                        sdict.get("Minimum discharge rate (MW)"),
                        DataParams.MIN_DISCHARGE_RATE,
                    )
                ),
                max_discharge=ts(sdict.get("Maximum discharge rate (MW)")),
                initial_level=_scalar(
                    sdict.get("Initial level (MWh)"), DataParams.INITIAL_LEVEL
                ),
                last_min=_scalar(
                    sdict.get("Last period minimum level (MWh)"), min_level[-1]
                ),
                last_max=_scalar(
                    sdict.get("Last period maximum level (MWh)"), max_level[-1]
                ),
            )
            storage_units.append(su)
            bus.storage_units.append(su)

    # ---------------------------------------------------------------------- #
    #  Sparse matrices (zeros – replication of spzeros(Float64, …) )         #
    # ---------------------------------------------------------------------- #
    isf = sparse.csr_matrix((len(lines), len(buses) - 1 if buses else 0), dtype=float)
    lodf = sparse.csr_matrix((len(lines), len(lines)), dtype=float)

    scenario = UnitCommitmentScenario(
        name=_scalar(par.get("Scenario name"), ""),
        probability=float(_scalar(par.get("Scenario weight"), 1)),
        buses_by_name={b.name: b for b in buses},
        buses=buses,
        contingencies_by_name={c.name: c for c in contingencies},
        contingencies=contingencies,
        lines_by_name={line.name: line for line in lines},
        lines=lines,
        power_balance_penalty=power_balance_penalty,
        price_sensitive_loads_by_name={pl.name: pl for pl in loads},
        price_sensitive_loads=loads,
        reserves=reserves,
        reserves_by_name=name_to_reserve,
        time=T,
        time_step=time_step,
        thermal_units_by_name={tu.name: tu for tu in thermal_units},
        thermal_units=thermal_units,
        profiled_units_by_name={pu.name: pu for pu in profiled_units},
        profiled_units=profiled_units,
        storage_units_by_name={su.name: su for su in storage_units},
        storage_units=storage_units,
        isf=isf,
        lodf=lodf,
    )

    _repair(scenario)
    print(
        f"Parsed scenario: {len(buses)} buses, {len(thermal_units)} thermal units, {len(lines)} lines, {len(reserves)} reserves"
    )
    return scenario


def _repair(scenario: UnitCommitmentScenario) -> None:
    """
      • fills commitment_status for must-run units
      • clamps initial conditions
      • builds ISF/LODF if missing
    Here we implement minimal sanity checks.
    """
    for tu in scenario.thermal_units:
        # ensure commitment_status consistent with must_run
        for t, mr in enumerate(tu.must_run):
            if mr:
                tu.commitment_status[t] = True

    # Compute PTDF/LODF if lines present
    if scenario.lines:
        compute_ptdf_lodf(scenario)


def repair_scenario_names_and_probabilities(
    scenarios: List[UnitCommitmentScenario], paths: List[str]
) -> None:
    """Normalize names and probabilities so they sum to 1."""
    total = sum(sc.probability for sc in scenarios)
    for sc, p in zip(scenarios, paths):
        if not sc.name:
            sc.name = Path(p).stem.split(".")[0]
        sc.probability /= total
