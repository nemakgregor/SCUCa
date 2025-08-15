import re
from typing import Optional
from src.data_preparation.params import DataParams


def allocate_run_id(scenario: Optional[str]) -> int:
    """
    Scan the logs directory and allocate the next run id for the given scenario.
    Filenames are expected to follow: SCUC_{scenario}_runNNN_*.{solution|verify}.<ext>
    """
    out_dir = DataParams._LOGS
    out_dir.mkdir(parents=True, exist_ok=True)
    scen = (scenario or "scenario").replace("/", "_")
    pattern = re.compile(
        rf"^SCUC_{re.escape(scen)}_run(\d+)_.*\.(?:solution|verify)\.[^.]+$"
    )
    max_id = 0
    for p in out_dir.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if m:
            try:
                rid = int(m.group(1))
                if rid > max_id:
                    max_id = rid
            except Exception:
                pass
    return max_id + 1


def make_log_filename(kind: str, scenario: Optional[str], run_id: int, ts: str) -> str:
    """
    Create a standardized filename for logs.

    kind: 'solution' or 'verify'
    scenario: scenario/case name
    run_id: integer
    ts: timestamp string like YYYYmmdd_HHMMSS

    We now save:
      - solution as JSON:  SCUC_<scenario>_runNNN_<ts>.solution.json
      - verify as text:    SCUC_<scenario>_runNNN_<ts>.verify.log
    """
    scen = (scenario or "scenario").replace("/", "_")
    if kind == "solution":
        return f"SCUC_{scen}_run{run_id:03d}_{ts}.solution.log"
    else:
        return f"SCUC_{scen}_run{run_id:03d}_{ts}.verify.log"
