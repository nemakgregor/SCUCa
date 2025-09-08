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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        json_["Reserves"] = {
            "r1": {
                "Type": "spinning",
                "Amount (MW)": amount,
            }
        }
        for gen in json_.get("Generators", {}).values():
            if gen.get("Provides spinning reserves?") is True:
                gen["Reserve eligibility"] = ["r1"]


def _migrate_to_v04(json_: dict) -> None:
    """Match Julia’s _migrate_to_v04: default missing types to Thermal."""
    for gen in json_.get("Generators", {}).values():
        gen.setdefault("Type", "Thermal")


def from_json(j: dict, quiet: bool = False) -> UnitCommitmentScenario:
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
    time_step = int(
        _scalar(par.get("Time step (min)"), default=DataParams.DEFAULT_TIME_STEP_MIN)
    )
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

    def ts(x, *, default=None):
        return _timeseries(x, T, default=default)

    # ---------------------------------------------------------------------- #
    #  Penalties                                                             #
    # ---------------------------------------------------------------------- #
    power_balance_penalty = ts(
        par.get("Power balance penalty ($/MW)"),
        default=[DataParams.DEFAULT_POWER_BALANCE_PENALTY_USD_PER_MW] * T,
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
            load=ts(
                bdict.get("Load (MW)"), default=[DataParams.DEFAULT_BUS_LOAD_MW] * T
            ),
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
                amount=ts(
                    rdict.get("Amount (MW)"),
                    default=[DataParams.DEFAULT_RESERVE_REQUIREMENT_MW] * T,
                ),
                thermal_units=[],
                shortfall_penalty=_scalar(
                    rdict.get("Shortfall penalty ($/MW)"),
                    default=DataParams.DEFAULT_RESERVE_SHORTFALL_PENALTY_USD_PER_MW,
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
                    f"Generator '{uname}' production cost curve lengths mismatch (MW: {len(curve_mw_list)}, $: {len(curve_cost_list)})"
                )
            if K == 0:
                raise ValueError(f"Generator '{uname}' has no break-points (K=0)")

            # Convert to time series arrays
            curve_mw = np.column_stack([ts(curve_mw_list[k]) for k in range(K)])
            curve_cost = np.column_stack([ts(curve_cost_list[k]) for k in range(K)])

            # Validate monotonicity
            if not np.all(np.diff(curve_mw, axis=1) > 0):
                raise ValueError(
                    f"Generator '{uname}' production cost curve MW must be strictly increasing"
                )
            if not np.all(np.diff(curve_cost, axis=1) >= 0):
                raise ValueError(
                    f"Generator '{uname}' production cost curve $ must be non-decreasing"
                )

            min_power = curve_mw[:, 0].tolist()
            max_power = curve_mw[:, -1].tolist()
            min_power_cost = curve_cost[:, 0].tolist()

            # Build segmented increments above minimum
            segments: List[CostSegment] = []
            if K > 1:
                for k in range(1, K):
                    amount = curve_mw[:, k] - curve_mw[:, k - 1]
                    # slope = (C_k - C_{k-1}) / (P_k - P_{k-1})
                    with np.errstate(divide="ignore", invalid="ignore"):
                        marginal = np.divide(
                            curve_cost[:, k] - curve_cost[:, k - 1],
                            amount,
                            out=np.zeros_like(amount, dtype=float),
                            where=amount != 0,
                        )
                    segments.append(
                        CostSegment(amount=amount.tolist(), cost=marginal.tolist())
                    )
            # If K == 1, there are no variable segments; only min_power_cost applies when committed.

            # Startup categories validation and parsing
            delays = udict.get(
                "Startup delays (h)", [DataParams.DEFAULT_STARTUP_DELAY_H]
            )
            scost = udict.get(
                "Startup costs ($)", [DataParams.DEFAULT_STARTUP_COST_USD]
            )
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
                if init_p and init_p > 0 and init_s <= 0:
                    raise ValueError(f"{uname} initial power >0 but status <=0")

            # Ramp and limits validation
            ramp_up = _scalar(
                udict.get("Ramp up limit (MW)"), DataParams.DEFAULT_RAMP_UP_MW
            )
            ramp_down = _scalar(
                udict.get("Ramp down limit (MW)"), DataParams.DEFAULT_RAMP_DOWN_MW
            )
            if ramp_up <= 0 or ramp_down <= 0:
                raise ValueError(f"Invalid ramp limits for '{uname}'")

            commitment_status = ts(
                udict.get("Commitment status"),
                default=[DataParams.DEFAULT_COMMITMENT_STATUS] * T,
            )

            # Robustly read startup/shutdown limits; ensure they are at least Pmin_max
            def _get_first(d, keys, default=None):
                for k in keys:
                    if k in d:
                        return d[k]
                return default

            pmin_max = float(max(min_power)) if min_power else 0.0

            su_raw = _get_first(
                udict,
                ["Startup limit (MW)", "Startup ramp limit (MW)"],
                None,
            )
            sd_raw = _get_first(
                udict,
                ["Shutdown limit (MW)", "Shutdown ramp limit (MW)"],
                None,
            )

            # Defaults: if missing, set to at least Pmin_max to guarantee feasible start/stop
            startup_limit = su_raw if su_raw is not None else pmin_max
            shutdown_limit = sd_raw if sd_raw is not None else pmin_max

            # Enforce minimum feasibility: SU, SD >= max_t Pmin
            startup_limit = max(float(startup_limit), pmin_max)
            shutdown_limit = max(float(shutdown_limit), pmin_max)

            tu = ThermalUnit(
                name=uname,
                bus=bus,
                max_power=max_power,
                min_power=min_power,
                must_run=ts(udict.get("Must run?", [DataParams.DEFAULT_MUST_RUN] * T)),
                min_power_cost=min_power_cost,
                segments=segments,
                min_up=int(
                    _scalar(
                        udict.get("Minimum uptime (h)"), DataParams.DEFAULT_MIN_UPTIME_H
                    )
                    * time_multiplier
                ),
                min_down=int(
                    _scalar(
                        udict.get("Minimum downtime (h)"),
                        DataParams.DEFAULT_MIN_DOWNTIME_H,
                    )
                    * time_multiplier
                ),
                ramp_up=ramp_up,
                ramp_down=ramp_down,
                startup_limit=startup_limit,
                shutdown_limit=shutdown_limit,
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

        elif utype.lower() == "profiled":
            pu = ProfiledUnit(
                name=uname,
                bus=bus,
                min_power=ts(
                    _scalar(
                        udict.get("Minimum power (MW)"),
                        DataParams.DEFAULT_MINIMUM_POWER_MW,
                    )
                ),
                max_power=ts(udict.get("Maximum power (MW)")),
                cost=ts(
                    udict.get("Cost ($/MW)"),
                    default=[DataParams.DEFAULT_INCREMENTAL_COST_USD_PER_MW] * T,
                ),
            )
            bus.profiled_units.append(pu)
            profiled_units.append(pu)

        else:
            raise ValueError(f"Unit {uname} has invalid type '{utype}'")

    # ---------------------------------------------------------------------- #
    #  Lines                                                                 #
    # ---------------------------------------------------------------------- #
    if "Transmission lines" in j:
        for idx, (lname, ldict) in enumerate(
            j.get("Transmission lines").items(), start=1
        ):
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
                    default=[DataParams.DEFAULT_LINE_NORMAL_LIMIT_MW] * T,
                ),
                emergency_limit=ts(
                    ldict.get("Emergency flow limit (MW)"),
                    default=[DataParams.DEFAULT_LINE_EMERGENCY_LIMIT_MW] * T,
                ),
                flow_penalty=ts(
                    ldict.get("Flow limit penalty ($/MW)"),
                    default=[DataParams.DEFAULT_LINE_FLOW_PENALTY_USD_PER_MW] * T,
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
                demand=ts(
                    ldict.get("Demand (MW)"),
                    default=[DataParams.DEFAULT_PSL_DEMAND_MW] * T,
                ),
                revenue=ts(
                    ldict.get("Revenue ($/MW)"),
                    default=[DataParams.DEFAULT_PSL_REVENUE_USD_PER_MW] * T,
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
                _scalar(
                    sdict.get("Minimum level (MWh)"),
                    DataParams.DEFAULT_STORAGE_MIN_LEVEL_MWH,
                )
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
                    default=[DataParams.DEFAULT_STORAGE_CHARGE_COST_USD_PER_MW] * T,
                ),
                discharge_cost=ts(
                    sdict.get("Discharge cost ($/MW)"),
                    default=[DataParams.DEFAULT_STORAGE_DISCHARGE_COST_USD_PER_MW] * T,
                ),
                charge_eff=ts(
                    _scalar(
                        sdict.get("Charge efficiency"),
                        DataParams.DEFAULT_STORAGE_CHARGE_EFFICIENCY,
                    )
                ),
                discharge_eff=ts(
                    _scalar(
                        sdict.get("Discharge efficiency"),
                        DataParams.DEFAULT_STORAGE_DISCHARGE_EFFICIENCY,
                    )
                ),
                loss_factor=ts(
                    _scalar(
                        sdict.get("Loss factor"), DataParams.DEFAULT_STORAGE_LOSS_FACTOR
                    )
                ),
                min_charge=ts(
                    _scalar(
                        sdict.get("Minimum charge rate (MW)"),
                        DataParams.DEFAULT_STORAGE_MIN_CHARGE_RATE_MW,
                    )
                ),
                max_charge=ts(sdict.get("Maximum charge rate (MW)"))
                if "Maximum charge rate (MW)" in sdict
                else ts(DataParams.DEFAULT_STORAGE_MAX_CHARGE_RATE_MW),
                min_discharge=ts(
                    _scalar(
                        sdict.get("Minimum discharge rate (MW)"),
                        DataParams.DEFAULT_STORAGE_MIN_DISCHARGE_RATE_MW,
                    )
                ),
                max_discharge=ts(sdict.get("Maximum discharge rate (MW)"))
                if "Maximum discharge rate (MW)" in sdict
                else ts(DataParams.DEFAULT_STORAGE_MAX_DISCHARGE_RATE_MW),
                initial_level=_scalar(
                    sdict.get("Initial level (MWh)"),
                    DataParams.DEFAULT_STORAGE_INITIAL_LEVEL_MWH,
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
    #  Sparse matrices defaults                                              #
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
    if not quiet:
        print(
            f"Parsed scenario: {len(buses)} buses, {len(thermal_units)} thermal units, {len(lines)} lines, {len(reserves)} reserves"
        )
    return scenario


def _repair(scenario: UnitCommitmentScenario) -> None:
    """
    • fills commitment_status for must-run units
    • builds ISF/LODF if lines present
    """
    for gen in scenario.thermal_units:
        for t, must_run_flag in enumerate(gen.must_run):
            if must_run_flag:
                gen.commitment_status[t] = True

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
