"""
Module for reading test cases in .JSON format.
"""

import json
import gzip

from typing import List
from instances.instance_structure import (
    UnitCommitmentScenario,
    Bus,
    ThermalUnit,
    ProfiledUnit,
    StorageUnit,
    TransmissionLine,
    Reserve,
    Contingency,
    PriceSensitiveLoad,
    CostSegment,
    StartupCategory,
)

from scipy import sparse
import numpy as np
import pandas as pd
from pandas import Series

from utils.data_parser.data_unificator import scalar, migrate, repair
from utils.defaults import DEFAULTS


def read_scenario(path: str) -> UnitCommitmentScenario:
    raw = _read_json(path)
    migrate(raw)
    return _from_json(raw)


def _read_json(path: str) -> dict:
    """Open JSON or JSON.GZ transparently."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _from_json(j: dict) -> UnitCommitmentScenario:
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
    time_step = int(scalar(par.get("Time step (min)"), default=60))
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

    name_to_bus, name_to_line, name_to_unit, name_to_reserve = ({}, {}, {}, {})

    # ---------------------------------------------------------------------- #
    #  Helper to make sure each list has length T                            #
    # ---------------------------------------------------------------------- #

    # def ts(x, *, default=None):
    #     return _timeseries(x, T, default=default)

    # def _timeseries(val, T: int, *, default=None):
    #     """
    #     Julia behaviour:
    #     * if val is missing ➜ default
    #     * if val is array  ➜ keep
    #     * if val is scalar ➜ replicate T times
    #     """
    #     if val is None:
    #         return default if default is not None else [None] * T
    #     return val if isinstance(val, list) else [val] * T

    def ts(val, default=None):
        """Convert value to time series (Series)."""
        if val is None:
            default_val = default if default is not None else [None] * T
            return Series(default_val)  # Convert to Series

        if isinstance(val, list):
            return Series(val)  # Convert list to Series
        else:
            return Series([val] * T)  # Convert single value to Series

    # ---------------------------------------------------------------------- #
    #  Penalties                                                             #
    # ---------------------------------------------------------------------- #
    power_balance_penalty = ts(
        par.get("Power balance penalty ($/MW)"), default=[1000.0] * T
    )

    # ---------------------------------------------------------------------- #
    #  Buses                                                                 #
    # ---------------------------------------------------------------------- #
    for idx, (bname, bdict) in enumerate(j["Buses"].items(), start=1):
        bus = Bus(
            name=bname,
            index=idx,
            load=ts(bdict["Load (MW)"]),
        )
        name_to_bus[bname] = bus
        buses.append(bus)

    # ---------------------------------------------------------------------- #
    #  Reserves                                                              #
    # ---------------------------------------------------------------------- #
    if "Reserves" in j:
        for rname, rdict in j["Reserves"].items():
            r = Reserve(
                name=rname,
                type=rdict["Type"].lower(),
                amount=ts(rdict["Amount (MW)"]),
                thermal_units=[],
                shortfall_penalty=scalar(
                    rdict.get("Shortfall penalty ($/MW)"), default=10
                ),
            )
            name_to_reserve[rname] = r
            reserves.append(r)

    # ---------------------------------------------------------------------- #
    #  Generators                                                            #
    # ---------------------------------------------------------------------- #
    for uname, udict in j["Generators"].items():
        utype = udict.get("Type")
        if not utype:
            raise ValueError(f"Generator {uname} missing Type")
        bus = name_to_bus[udict["Bus"]]

        if utype.lower() == "thermal":
            # Production cost curve
            curve_mw = udict["Production cost curve (MW)"]
            curve_cost = udict["Production cost curve ($)"]
            K = len(curve_mw)
            curve_mw = np.column_stack([ts(curve_mw[k]) for k in range(K)])
            curve_cost = np.column_stack([ts(curve_cost[k]) for k in range(K)])

            min_power = curve_mw[:, 0].tolist()
            max_power = curve_mw[:, -1].tolist()
            min_power_cost = curve_cost[:, 0].tolist()

            segments = []
            for k in range(1, K):
                amount = (curve_mw[:, k] - curve_mw[:, k - 1]).tolist()
                cost = (
                    (curve_cost[:, k] - curve_cost[:, k - 1])
                    / (np.maximum(amount, 1e-9))
                ).tolist()
                segments.append(CostSegment(amount, cost))

            # Startup categories
            delays = scalar(udict.get("Startup delays (h)"), default=[1])
            scost = scalar(udict.get("Startup costs ($)"), default=[0.0])
            startup_categories = [
                StartupCategory(int(delays[k] * time_multiplier), scost[k])
                for k in range(len(delays))
            ]

            # Reserve eligibility
            unit_reserves = [
                name_to_reserve[n] for n in udict.get("Reserve eligibility", [])
            ]
            # Initial conditions
            init_p = udict.get("Initial power (MW)")
            init_s = udict.get("Initial status (h)")
            if init_p is None:
                init_s = None
            elif init_s is None:
                raise ValueError(f"{uname} has power but no status")
            else:
                init_s = int(init_s * time_multiplier)

            commitment_status = scalar(
                udict.get("Commitment status"), default=[None] * T
            )

            tu = ThermalUnit(
                name=uname,
                bus=bus,
                max_power=max_power,
                min_power=min_power,
                must_run=ts(udict.get("Must run?"), default=[False] * T),
                min_power_cost=min_power_cost,
                segments=segments,
                min_up=int(
                    scalar(udict.get("Minimum uptime (h)"), 1) * time_multiplier
                ),
                min_down=int(
                    scalar(udict.get("Minimum downtime (h)"), 1) * time_multiplier
                ),
                ramp_up=scalar(udict.get("Ramp up limit (MW)"), 1e6),
                ramp_down=scalar(udict.get("Ramp down limit (MW)"), 1e6),
                startup_limit=scalar(udict.get("Startup limit (MW)"), 1e6),
                shutdown_limit=scalar(udict.get("Shutdown limit (MW)"), 1e6),
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
                min_power=ts(scalar(udict.get("Minimum power (MW)"), 0.0)),
                max_power=ts(udict["Maximum power (MW)"]),
                cost=ts(udict["Cost ($/MW)"]),
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
            line = TransmissionLine(
                name=lname,
                index=idx,
                source=name_to_bus[ldict["Source bus"]],
                target=name_to_bus[ldict["Target bus"]],
                susceptance=float(ldict["Susceptance (S)"]),
                normal_limit=ts(ldict.get("Normal flow limit (MW)"), default=[1e8] * T),
                emergency_limit=ts(
                    ldict.get("Emergency flow limit (MW)"), default=[1e8] * T
                ),
                flow_penalty=ts(
                    ldict.get("Flow limit penalty ($/MW)"), default=[5000.0] * T
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
                name_to_line[line_name] for line_name in cdict.get("Affected lines", [])
            ]
            affected_units = [
                name_to_unit[unit_name] for unit_name in cdict.get("Affected units", [])
            ]
            contingencies.append(
                Contingency(name=cname, lines=affected_lines, units=affected_units)
            )

    # ---------------------------------------------------------------------- #
    #  Price-sensitive loads                                                 #
    # ---------------------------------------------------------------------- #
    if "Price-sensitive loads" in j:
        for lname, ldict in j["Price-sensitive loads"].items():
            load = PriceSensitiveLoad(
                name=lname,
                bus=name_to_bus[ldict["Bus"]],
                demand=ts(ldict["Demand (MW)"]),
                revenue=ts(ldict["Revenue ($/MW)"]),
            )
            loads.append(load)
            load.bus.price_sensitive_loads.append(load)

    # ---------------------------------------------------------------------- #
    #  Storage units                                                         #
    # ---------------------------------------------------------------------- #
    if "Storage units" in j:
        for sname, sdict in j["Storage units"].items():
            bus = name_to_bus[sdict["Bus"]]
            min_level = ts(scalar(sdict.get("Minimum level (MWh)"), 0.0))
            max_level = ts(sdict["Maximum level (MWh)"])
            su = StorageUnit(
                name=sname,
                bus=bus,
                min_level=min_level,
                max_level=max_level,
                simultaneous=ts(
                    scalar(
                        sdict.get("Allow simultaneous charging and discharging"), True
                    )
                ),
                charge_cost=ts(sdict["Charge cost ($/MW)"]),
                discharge_cost=ts(sdict["Discharge cost ($/MW)"]),
                charge_eff=ts(scalar(sdict.get("Charge efficiency"), 1.0)),
                discharge_eff=ts(scalar(sdict.get("Discharge efficiency"), 1.0)),
                loss_factor=ts(scalar(sdict.get("Loss factor"), 0.0)),
                min_charge=ts(scalar(sdict.get("Minimum charge rate (MW)"), 0.0)),
                max_charge=ts(sdict["Maximum charge rate (MW)"]),
                min_discharge=ts(scalar(sdict.get("Minimum discharge rate (MW)"), 0.0)),
                max_discharge=ts(sdict["Maximum discharge rate (MW)"]),
                initial_level=scalar(sdict.get("Initial level (MWh)"), 0.0),
                last_min=scalar(
                    sdict.get("Last period minimum level (MWh)"), min_level[-1]
                ),
                last_max=scalar(
                    sdict.get("Last period maximum level (MWh)"), max_level[-1]
                ),
            )
            storage_units.append(su)
            bus.storage_units.append(su)

    # ---------------------------------------------------------------------- #
    #  Sparse matrices (zeros – replication of spzeros(Float64, …) )         #
    # ---------------------------------------------------------------------- #
    isf = sparse.csr_matrix((len(lines), len(buses) - 1), dtype=float)
    lodf = sparse.csr_matrix((len(lines), len(lines)), dtype=float)

    scenario = UnitCommitmentScenario(
        name=scalar(par.get("Scenario name"), ""),
        probability=float(scalar(par.get("Scenario weight"), 1)),
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

    repair(scenario)

    def serialize(obj):
        if isinstance(obj, Series):
            return obj.tolist()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return obj

    log_path = DEFAULTS.LOG_PATH
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(scenario, fh, default=serialize, indent=4)

    return scenario
