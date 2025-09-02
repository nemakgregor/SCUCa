import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from gurobipy import Model, GRB

from src.data_preparation.data_structure import UnitCommitmentScenario
from src.data_preparation.params import DataParams

# Try to import contingency penalty factor used by objective to compute consistent penalty contribution
try:
    from src.optimization_model.solver.scuc.objectives.power_cost_segmented import (
        _CONTINGENCY_PENALTY_FACTOR as CONTINGENCY_PENALTY_FACTOR,
    )
except Exception:
    CONTINGENCY_PENALTY_FACTOR = 10.0


def _status_str(code: int) -> str:
    mapping = DataParams.SOLVER_STATUS_STR
    return mapping.get(code, f"STATUS_{code}")


def _sanitize_token(s: str) -> str:
    s = (s or "").strip()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    # normalize repeated underscores
    t = "".join(out)
    while "__" in t:
        t = t.replace("__", "_")
    return t.strip("_").lower()


def case_folder_from_instance(instance_name: str) -> str:
    """
    For 'matpower/case300/2017-06-24' -> 'matpower/case300'
    For 'test/case14'                 -> 'test/case14'
    For 'matpower/case14'             -> 'matpower/case14'
    """
    parts = instance_name.strip().strip("/\\").replace("\\", "/").split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] if parts else "case"


def _sanitized_case_tag(case_folder: str) -> str:
    """
    'matpower/case14' -> 'matpower_case14'
    """
    return _sanitize_token(case_folder)


@dataclass
class PerfRow:
    # identity/meta
    timestamp: str
    instance_name: str
    case_folder: str
    technique: str
    run_id: int
    # solver info
    status: str
    status_code: int
    runtime_sec: float
    mip_gap: Optional[float]
    obj_val: Optional[float]
    obj_bound: Optional[float]
    nodes: Optional[float]
    # model size
    num_vars: int
    num_bin_vars: int
    num_int_vars: int
    num_constrs: int
    num_cont_vars: int
    num_nzs: Optional[int]
    # data size
    buses: int
    units: int
    lines: int
    reserves: int
    time_steps: int
    time_step_min: int
    ptdf_nnz: Optional[int]
    lodf_nnz: Optional[int]
    # penalties and counts
    startup_count: Optional[float]
    startup_cost: Optional[float]
    reserve_shortfall_mw: Optional[float]
    reserve_shortfall_penalty: Optional[float]
    base_overflow_mw: Optional[float]
    base_overflow_penalty: Optional[float]
    cont_overflow_mw: Optional[float]
    cont_overflow_penalty: Optional[float]


class PerfCSVLogger:
    """
    Create and append rows to one CSV per case folder (e.g., matpower/case14):

      src/data/output/<case_folder>/perf_<case_tag>_<technique>_<run_id>.csv

    - case_folder is typically 'matpower/case14'
    - case_tag is that folder with slashes replaced: 'matpower_case14'
    - technique is a short user label ('basic', 'warm_start', etc.)
    - run_id increments per (case, technique)
    """

    HEADER = [
        "timestamp",
        "instance_name",
        "case_folder",
        "technique",
        "run_id",
        "status",
        "status_code",
        "runtime_sec",
        "mip_gap",
        "obj_val",
        "obj_bound",
        "nodes",
        "num_vars",
        "num_bin_vars",
        "num_int_vars",
        "num_cont_vars",
        "num_constrs",
        "num_nzs",
        "buses",
        "units",
        "lines",
        "reserves",
        "time_steps",
        "time_step_min",
        "ptdf_nnz",
        "lodf_nnz",
        "startup_count",
        "startup_cost",
        "reserve_shortfall_mw",
        "reserve_shortfall_penalty",
        "base_overflow_mw",
        "base_overflow_penalty",
        "cont_overflow_mw",
        "cont_overflow_penalty",
    ]

    def __init__(self, technique: str, base_output_dir: Optional[Path] = None):
        self.technique = _sanitize_token(technique) or "basic"
        self.base_dir = base_output_dir or DataParams._OUTPUT
        self._case_to_csv: Dict[str, Path] = {}
        self._case_to_runid: Dict[str, int] = {}

    def _allocate_run_id(self, case_folder: str) -> int:
        # Scan existing perf_ files to get max id, then +1
        case_dir = self.base_dir / case_folder
        case_dir.mkdir(parents=True, exist_ok=True)
        tag = _sanitized_case_tag(case_folder)
        prefix = f"perf_{tag}_{self.technique}_"
        max_id = 0
        if case_dir.exists():
            for p in case_dir.glob(f"{prefix}*.csv"):
                name = p.name
                try:
                    # Expect suffix before .csv to be nn or nnn (digits)
                    stem = name[: -len(".csv")] if name.endswith(".csv") else name
                    run_part = stem.split("_")[-1]
                    rid = int(run_part)
                    if rid > max_id:
                        max_id = rid
                except Exception:
                    continue
        return max_id + 1

    def _get_csv_path(self, case_folder: str) -> Tuple[Path, int]:
        if case_folder in self._case_to_csv:
            return self._case_to_csv[case_folder], self._case_to_runid[case_folder]
        rid = self._allocate_run_id(case_folder)
        tag = _sanitized_case_tag(case_folder)
        filename = f"perf_{tag}_{self.technique}_{rid:02d}.csv"
        path = (self.base_dir / case_folder / filename).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write header if new file
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(self.HEADER)
        self._case_to_csv[case_folder] = path
        self._case_to_runid[case_folder] = rid
        return path, rid

    def _sum_over_vars(self, var_td, index_pairs) -> float:
        if var_td is None:
            return 0.0
        s = 0.0
        for idx in index_pairs:
            try:
                s += float(var_td[idx].X)
            except Exception:
                continue
        return s

    def _compute_penalties_and_counts(
        self, sc: UnitCommitmentScenario, model: Model
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        # Startup
        startup = getattr(model, "startup", None)
        startup_cost_total = 0.0
        startup_count_total = 0.0
        if startup is not None:
            for g in sc.thermal_units:
                # chosen policy: 'hot start' cost = min startup category cost (0 if none)
                cost = 0.0
                try:
                    if g.startup_categories:
                        cost = float(min(cat.cost for cat in g.startup_categories))
                except Exception:
                    cost = 0.0
                for t in range(sc.time):
                    try:
                        v = float(startup[g.name, t].X)
                    except Exception:
                        v = 0.0
                    startup_count_total += v
                    startup_cost_total += cost * v

        # Reserve shortfall and penalty
        shortfall_total = 0.0
        shortfall_penalty_total = 0.0
        short = getattr(model, "reserve_shortfall", None)
        if short is not None and sc.reserves:
            for r in sc.reserves:
                pen = float(r.shortfall_penalty)
                for t in range(sc.time):
                    try:
                        s = float(short[r.name, t].X)
                    except Exception:
                        s = 0.0
                    shortfall_total += s
                    shortfall_penalty_total += pen * s

        # Base overflow: sum(MW) and penalty
        base_ovp_total = 0.0
        base_ovn_total = 0.0
        base_penalty_total = 0.0
        ovp = getattr(model, "line_overflow_pos", None)
        ovn = getattr(model, "line_overflow_neg", None)
        if sc.lines and ovp is not None and ovn is not None:
            for ln in sc.lines:
                for t in range(sc.time):
                    try:
                        vpos = float(ovp[ln.name, t].X)
                    except Exception:
                        vpos = 0.0
                    try:
                        vneg = float(ovn[ln.name, t].X)
                    except Exception:
                        vneg = 0.0
                    base_ovp_total += vpos
                    base_ovn_total += vneg
                    base_penalty_total += (vpos + vneg) * float(ln.flow_penalty[t])

        # Contingency overflow: use shared slacks and same line penalty scaled by factor
        cont_ovp_total = 0.0
        cont_ovn_total = 0.0
        cont_penalty_total = 0.0
        covp = getattr(model, "contingency_overflow_pos", None)
        covn = getattr(model, "contingency_overflow_neg", None)
        if sc.lines and covp is not None and covn is not None:
            for ln in sc.lines:
                for t in range(sc.time):
                    try:
                        vpos = float(covp[ln.name, t].X)
                    except Exception:
                        vpos = 0.0
                    try:
                        vneg = float(covn[ln.name, t].X)
                    except Exception:
                        vneg = 0.0
                    cont_ovp_total += vpos
                    cont_ovn_total += vneg
                    cont_penalty_total += (
                        (vpos + vneg)
                        * float(ln.flow_penalty[t])
                        * float(CONTINGENCY_PENALTY_FACTOR)
                    )

        base_overflow_mw = base_ovp_total + base_ovn_total
        cont_overflow_mw = cont_ovp_total + cont_ovn_total
        return (
            startup_count_total,
            startup_cost_total,
            shortfall_total,
            shortfall_penalty_total,
            base_overflow_mw,
            base_penalty_total,
            cont_overflow_mw,
            cont_penalty_total,
        )

    def _make_row(
        self, instance_name: str, sc: UnitCommitmentScenario, model: Model, run_id: int
    ) -> PerfRow:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_code = getattr(model, "Status", -1)
        status = _status_str(status_code)
        runtime = getattr(model, "Runtime", None)
        mip_gap = None
        try:
            mip_gap = float(model.MIPGap)
        except Exception:
            mip_gap = None
        try:
            obj_val = float(model.ObjVal)
        except Exception:
            obj_val = None
        try:
            obj_bound = float(model.ObjBound)
        except Exception:
            obj_bound = None
        try:
            nodes = float(getattr(model, "NodeCount", 0.0))
        except Exception:
            nodes = None

        num_vars = int(getattr(model, "NumVars", 0))
        num_bin = int(getattr(model, "NumBinVars", 0))
        num_int = int(getattr(model, "NumIntVars", 0))
        num_constrs = int(getattr(model, "NumConstrs", 0))
        try:
            num_nzs = int(getattr(model, "NumNZs"))
        except Exception:
            num_nzs = None
        num_cont = num_vars - num_int  # Gurobi counts integer (incl bin) as int vars

        ptdf_nnz = None
        lodf_nnz = None
        try:
            ptdf_nnz = int(getattr(sc.isf, "nnz", 0)) if sc.isf is not None else 0
        except Exception:
            pass
        try:
            lodf_nnz = int(getattr(sc.lodf, "nnz", 0)) if sc.lodf is not None else 0
        except Exception:
            pass

        (
            startup_count,
            startup_cost,
            r_short_mw,
            r_short_penalty,
            base_ov_mw,
            base_penalty,
            cont_ov_mw,
            cont_penalty,
        ) = self._compute_penalties_and_counts(sc, model)

        case_folder = case_folder_from_instance(instance_name)

        return PerfRow(
            timestamp=now,
            instance_name=instance_name,
            case_folder=case_folder,
            technique=self.technique,
            run_id=run_id,
            status=status,
            status_code=int(status_code),
            runtime_sec=float(runtime) if runtime is not None else 0.0,
            mip_gap=mip_gap,
            obj_val=obj_val,
            obj_bound=obj_bound,
            nodes=nodes,
            num_vars=num_vars,
            num_bin_vars=num_bin,
            num_int_vars=num_int,
            num_constrs=num_constrs,
            num_cont_vars=num_cont,
            num_nzs=num_nzs,
            buses=len(sc.buses),
            units=len(sc.thermal_units),
            lines=len(sc.lines),
            reserves=len(sc.reserves) if sc.reserves else 0,
            time_steps=int(sc.time),
            time_step_min=int(sc.time_step),
            ptdf_nnz=ptdf_nnz,
            lodf_nnz=lodf_nnz,
            startup_count=startup_count,
            startup_cost=startup_cost,
            reserve_shortfall_mw=r_short_mw,
            reserve_shortfall_penalty=r_short_penalty,
            base_overflow_mw=base_ov_mw,
            base_overflow_penalty=base_penalty,
            cont_overflow_mw=cont_ov_mw,
            cont_overflow_penalty=cont_penalty,
        )

    def append_result(
        self, instance_name: str, sc: UnitCommitmentScenario, model: Model
    ) -> Path:
        """
        Build a row from model+scenario and append it to the correct per-case CSV.
        Returns the CSV path.
        """
        case_folder = case_folder_from_instance(instance_name)
        csv_path, run_id = self._get_csv_path(case_folder)
        row = self._make_row(instance_name, sc, model, run_id)

        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(
                [
                    row.timestamp,
                    row.instance_name,
                    row.case_folder,
                    row.technique,
                    row.run_id,
                    row.status,
                    row.status_code,
                    f"{row.runtime_sec:.6f}",
                    "" if row.mip_gap is None else f"{row.mip_gap:.8f}",
                    "" if row.obj_val is None else f"{row.obj_val:.6f}",
                    "" if row.obj_bound is None else f"{row.obj_bound:.6f}",
                    "" if row.nodes is None else f"{row.nodes:.0f}",
                    row.num_vars,
                    row.num_bin_vars,
                    row.num_int_vars,
                    row.num_cont_vars,
                    row.num_constrs,
                    "" if row.num_nzs is None else row.num_nzs,
                    row.buses,
                    row.units,
                    row.lines,
                    row.reserves,
                    row.time_steps,
                    row.time_step_min,
                    "" if row.ptdf_nnz is None else row.ptdf_nnz,
                    "" if row.lodf_nnz is None else row.lodf_nnz,
                    "" if row.startup_count is None else f"{row.startup_count:.4f}",
                    "" if row.startup_cost is None else f"{row.startup_cost:.6f}",
                    ""
                    if row.reserve_shortfall_mw is None
                    else f"{row.reserve_shortfall_mw:.6f}",
                    ""
                    if row.reserve_shortfall_penalty is None
                    else f"{row.reserve_shortfall_penalty:.6f}",
                    ""
                    if row.base_overflow_mw is None
                    else f"{row.base_overflow_mw:.6f}",
                    ""
                    if row.base_overflow_penalty is None
                    else f"{row.base_overflow_penalty:.6f}",
                    ""
                    if row.cont_overflow_mw is None
                    else f"{row.cont_overflow_mw:.6f}",
                    ""
                    if row.cont_overflow_penalty is None
                    else f"{row.cont_overflow_penalty:.6f}",
                ]
            )
        return csv_path
