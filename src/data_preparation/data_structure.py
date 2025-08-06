# src/data_preparation/data_structure.py
from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional
import scipy.sparse as sp

Number = Union[int, float]
Series = List[Number]


@dataclass
class CostSegment:
    amount: Series
    cost: Series


@dataclass
class StartupCategory:
    delay_steps: int
    cost: float


@dataclass
class Bus:
    name: str
    index: int
    load: Series
    thermal_units: List["ThermalUnit"] = field(default_factory=list)
    price_sensitive_loads: List["PriceSensitiveLoad"] = field(default_factory=list)
    profiled_units: List["ProfiledUnit"] = field(default_factory=list)
    storage_units: List["StorageUnit"] = field(default_factory=list)


@dataclass
class Reserve:
    name: str
    type: str
    amount: Series
    thermal_units: List["ThermalUnit"]
    shortfall_penalty: float


@dataclass
class ThermalUnit:
    name: str
    bus: Bus
    max_power: Series
    min_power: Series
    must_run: Series
    min_power_cost: Series
    segments: List[CostSegment]
    min_up: int
    min_down: int
    ramp_up: float
    ramp_down: float
    startup_limit: float
    shutdown_limit: float
    initial_status: Optional[int]
    initial_power: Optional[float]
    startup_categories: List[StartupCategory]
    reserves: List[Reserve]
    commitment_status: List[Optional[bool]]


@dataclass
class ProfiledUnit:
    name: str
    bus: Bus
    min_power: Series
    max_power: Series
    cost: Series


@dataclass
class StorageUnit:
    name: str
    bus: Bus
    min_level: Series
    max_level: Series
    simultaneous: Series
    charge_cost: Series
    discharge_cost: Series
    charge_eff: Series
    discharge_eff: Series
    loss_factor: Series
    min_charge: Series
    max_charge: Series
    min_discharge: Series
    max_discharge: Series
    initial_level: float
    last_min: float
    last_max: float


@dataclass
class TransmissionLine:
    name: str
    index: int
    source: Bus
    target: Bus
    susceptance: float
    normal_limit: Series
    emergency_limit: Series
    flow_penalty: Series


@dataclass
class Contingency:
    name: str
    lines: List[TransmissionLine]
    units: List[ThermalUnit]


@dataclass
class PriceSensitiveLoad:
    name: str
    bus: Bus
    demand: Series
    revenue: Series


@dataclass
class UnitCommitmentScenario:
    name: str
    probability: float
    buses_by_name: Dict[str, Bus]
    buses: List[Bus]
    contingencies_by_name: Dict[str, Contingency]
    contingencies: List[Contingency]
    lines_by_name: Dict[str, TransmissionLine]
    lines: List[TransmissionLine]
    power_balance_penalty: Series
    price_sensitive_loads_by_name: Dict[str, PriceSensitiveLoad]
    price_sensitive_loads: List[PriceSensitiveLoad]
    reserves: List[Reserve]
    reserves_by_name: Dict[str, Reserve]
    time: int
    time_step: int
    thermal_units_by_name: Dict[str, ThermalUnit]
    thermal_units: List[ThermalUnit]
    profiled_units_by_name: Dict[str, ProfiledUnit]
    profiled_units: List[ProfiledUnit]
    storage_units_by_name: Dict[str, StorageUnit]
    storage_units: List[StorageUnit]
    isf: sp.spmatrix
    lodf: sp.spmatrix


@dataclass
class UnitCommitmentInstance:
    time: int
    scenarios: List[UnitCommitmentScenario]

    # convenient alias
    @property
    def deterministic(self) -> UnitCommitmentScenario:
        if len(self.scenarios) != 1:
            raise ValueError("Instance is stochastic; pick a scenario explicitly")
        return self.scenarios[0]
