from pathlib import Path


class DataParams:
    INSTANCES_URL = "https://axavier.org/UnitCommitment.jl/0.4/instances"
    REFERENCE_SOLUTIONS_URL = (
        "https://github.com/ANL-CEEESA/UnitCommitment.jl/tree/dev/src"
    )
    # Cache directory for downloaded instances
    _CACHE = Path(__file__).resolve().parent.parent / "data" / "input"
    _CACHE.mkdir(parents=True, exist_ok=True)

    TIMESTEP = 60  # minutes
    BALANCE_PENALTY = 1000  # $/MW

    LOAD = 100 # MW
    RESERVS = 0  # MW
    SHORTFALL_PENALTY = 100  # $/MW

    STARTUP_DELAY = 0  # hours
    STARTUP_COST = 0.0  # $
    COMMITMENT_STATUS = None  # None, "on", "off"
    MUST_RUN = False  # Whether the unit must run at least once in the horizon
    RAMP_UP = 1e6  # MW
    RAMP_DOWN = 1e6  # MW
    STARTUP_LIMIT = 1e6  # MW
    SHUTDOWN_LIMIT = 1e6  # MW
    MINIMUM_POWER = 0  # MW
    COST = 100  # $/MW
    MIN_UPTIME = 0  # hours
    MIN_DOWNTIME = 0  # hours

    FLOW_LIMIT = 1e8  # MW
    EMERGENCY_FLOW_LIMIT = 1e9  # MW
    FLOW_LIMIT_PENALTY = 5000.0  # $/MW

    DEMAND = 1000  # MW
    DEMAND_REVENUE = 1000.0  # $/MW

    STORAGE_MINIMUM = 0  # MW
    STORAGE_MAXIMUM = 1e6  # MW
    STORAGE_CHARGE_COST = COST  # $/MW
    STORAGE_DISCHARGE_COST = 0  # $/MW
    CHARGE_EFFICIENCY = 1.0  # Default charge efficiency
    DISCHARGE_EFFICIENCY = 1.0  # Default discharge efficiency
    LOSS_FACTOR = 0.0  # Default loss factor
    MIN_CHARGE_RATE = 0.0  # MW, minimum charge rate
    MAX_CHARGE_RATE = None  # MW, maximum charge rate
    MIN_DISCHARGE_RATE = 0.0  # MW, minimum discharge rate
    MAX_DISCHARGE_RATE = None  # MW, maximum discharge rate
    INITIAL_LEVEL = 0.0  # MWh, initial storage level
    LAST_PERIOD_MIN_LEVEL = 0.0  # MWh, last period minimum level
    LAST_PERIOD_MAX_LEVEL = STORAGE_MAXIMUM  # MWh, last period maximum level