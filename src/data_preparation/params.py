from pathlib import Path
from gurobipy import GRB


class DataParams:
    """
    Central place for:
      - External URLs
      - Cache/log directories
      - Solver status mapping
      - Default values used by the loader when input fields are missing

    Notes on defaults
    - These defaults describe loader behavior when a specific field is missing.
      In many places (e.g., bus load, reserve amount), the loader still enforces
      presence of key fields and will raise if missing. Defaults listed here are
      therefore either:
        • documentation of expected scales, or
        • actual fallback values when the loader allows omission.
    """

    # Remote sources
    # Base URL hosting UnitCommitment.jl benchmark JSON instances
    INSTANCES_URL = "https://axavier.org/UnitCommitment.jl/0.4/instances"
    # Reference solutions URL (informational; not used programmatically here)
    REFERENCE_SOLUTIONS_URL = (
        "https://github.com/ANL-CEEESA/UnitCommitment.jl/tree/dev/src"
    )

    # Cache directory for downloaded instances: src/data/input/<name>.json.gz
    _CACHE = Path(__file__).resolve().parent.parent / "data" / "input"
    _CACHE.mkdir(parents=True, exist_ok=True)

    # Output directory for JSON solutions mirroring input hierarchy:
    # src/data/output/<name>.json (note: without .gz)
    _OUTPUT = Path(__file__).resolve().parent.parent / "data" / "output"
    _OUTPUT.mkdir(parents=True, exist_ok=True)

    # Logs directory for solution/verification: src/data/logs
    _LOGS = Path(__file__).resolve().parent.parent / "data" / "logs"
    _LOGS.mkdir(parents=True, exist_ok=True)

    # Solver status → string (minimal mapping we print in logs)
    SOLVER_STATUS_STR = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
    }

    # Time grid defaults
    # Default time step in minutes (if "Time step (min)" is missing)
    DEFAULT_TIME_STEP_MIN = 60

    # System-level penalty
    # Penalty for power balance mismatch ($/MW); used if missing in Parameters
    DEFAULT_POWER_BALANCE_PENALTY_USD_PER_MW = 1000

    # Bus defaults
    # Default bus load if a loader path were to allow a missing entry (we still require it)
    DEFAULT_BUS_LOAD_MW = 100

    # Reserve defaults
    # Default requirement (MW) if allowed to be omitted (we still require Amount (MW))
    DEFAULT_RESERVE_REQUIREMENT_MW = 0
    # Shortfall penalty ($/MW) if missing on a reserve product
    DEFAULT_RESERVE_SHORTFALL_PENALTY_USD_PER_MW = 1000

    # Thermal generator defaults
    # Startup categories (delays in hours and cost in $) — used if Startup delays/costs missing
    DEFAULT_STARTUP_DELAY_H = 0
    DEFAULT_STARTUP_COST_USD = 0.0
    # Fixed commitment status per time step; None means "free" (not fixed)
    DEFAULT_COMMITMENT_STATUS = None
    # Must-run default flag if missing per time step
    DEFAULT_MUST_RUN = False

    # Ramp and start/stop limits (MW) if missing
    DEFAULT_RAMP_UP_MW = 100
    DEFAULT_RAMP_DOWN_MW = 100
    DEFAULT_STARTUP_RAMP_LIMIT_MW = 100
    DEFAULT_SHUTDOWN_RAMP_LIMIT_MW = 100

    # Profiled/minimum-power-style inputs (used by profiled units or when needed)
    DEFAULT_MINIMUM_POWER_MW = 0
    DEFAULT_INCREMENTAL_COST_USD_PER_MW = 100
    DEFAULT_MIN_UPTIME_H = 0
    DEFAULT_MIN_DOWNTIME_H = 0

    # Transmission line defaults (MW and penalty $/MW); used if missing
    DEFAULT_LINE_NORMAL_LIMIT_MW = 300
    DEFAULT_LINE_EMERGENCY_LIMIT_MW = 400
    DEFAULT_LINE_FLOW_PENALTY_USD_PER_MW = 500.0

    # Price-sensitive load defaults (if missing)
    DEFAULT_PSL_DEMAND_MW = 1000
    DEFAULT_PSL_REVENUE_USD_PER_MW = 1000.0

    # Storage defaults (used when omitted)
    DEFAULT_STORAGE_MIN_LEVEL_MWH = 0.0
    DEFAULT_STORAGE_MAX_LEVEL_MWH = 1e4
    DEFAULT_STORAGE_CHARGE_COST_USD_PER_MW = DEFAULT_INCREMENTAL_COST_USD_PER_MW
    DEFAULT_STORAGE_DISCHARGE_COST_USD_PER_MW = 0.0
    DEFAULT_STORAGE_CHARGE_EFFICIENCY = 1.0
    DEFAULT_STORAGE_DISCHARGE_EFFICIENCY = 1.0
    DEFAULT_STORAGE_LOSS_FACTOR = 0.0
    DEFAULT_STORAGE_MIN_CHARGE_RATE_MW = 0.0
    DEFAULT_STORAGE_MAX_CHARGE_RATE_MW = None
    DEFAULT_STORAGE_MIN_DISCHARGE_RATE_MW = 0.0
    DEFAULT_STORAGE_MAX_DISCHARGE_RATE_MW = None
    DEFAULT_STORAGE_INITIAL_LEVEL_MWH = 0.0
    DEFAULT_STORAGE_LAST_PERIOD_MIN_LEVEL_MWH = 0.0
    DEFAULT_STORAGE_LAST_PERIOD_MAX_LEVEL_MWH = DEFAULT_STORAGE_MAX_LEVEL_MWH
