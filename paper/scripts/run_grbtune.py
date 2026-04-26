"""Run Gurobi's built-in parameter tuner on a representative SCUC instance.

Closes reviewer comment R2-C4 ("compare against Gurobi auto-tuning").

The script builds the WARM+PRUNE-0.10 variant of a single representative date
for the requested case, writes the model to an .mps file, then calls
`model.tune()` with the requested time budget. The tuning result is compared
against the default configuration on the *same* instance so we can quote a
"tuned Gurobi" baseline in the paper.

Expected compute:
  - case118: ~30 min tuning + ~5 min verification at default 600 s TuneTimeLimit
  - case300: ~60 min tuning + ~15 min verification

Usage:
    python paper/scripts/run_grbtune.py --case matpower/case118 --tune-time 1800
    python paper/scripts/run_grbtune.py --case matpower/case300 --tune-time 3600

Output:
    paper/data/grbtune_<case>.json        # tuning record
    paper/data/grbtune_<case>.prm         # best parameter file
    paper/tables/grbtune_vs_default.csv   # aggregated result across cases
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from gurobipy import GRB

from src.data_preparation.params import DataParams
from src.data_preparation.read_data import read_benchmark
from src.optimization_model.SCUC_solver.scuc_model_builder import build_model

PAPER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PAPER_DIR / "data"
TABLES_DIR = PAPER_DIR / "tables"
DATA_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


def _pick_representative_instance(case_folder: str) -> str:
    """Pick a single locally available instance; prefer known stable dates."""
    defaults = {
        "matpower/case118": "matpower/case118/2017-01-10",
        "matpower/case300": "matpower/case300/2017-01-04",
        "matpower/case89pegase": "matpower/case89pegase/2017-01-10",
        "matpower/case1354pegase": "matpower/case1354pegase/2017-01-01",
    }
    preferred = defaults.get(case_folder)
    if preferred is not None:
        preferred_path = DataParams._CACHE / f"{preferred}.json.gz"
        if preferred_path.exists():
            return preferred

    case_dir = (DataParams._CACHE / case_folder).resolve()
    if case_dir.exists():
        candidates = sorted(case_dir.glob("*.json.gz"))
        if candidates:
            rel = candidates[0].resolve().relative_to(DataParams._CACHE.resolve()).as_posix()
            if rel.endswith(".json.gz"):
                rel = rel[: -len(".json.gz")]
            return rel

    return preferred or f"{case_folder}/2017-01-10"


def _build_and_export(instance: str, mps_path: Path):
    inst = read_benchmark(instance, quiet=True)
    sc = inst.deterministic
    model = build_model(scenario=sc, contingency_filter=None, use_lazy_contingencies=False)
    model.update()
    mps_path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(mps_path))
    return model


def _solve_with_default(model, time_limit: int) -> dict:
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 0)
    t0 = time.perf_counter()
    model.optimize()
    wall = time.perf_counter() - t0
    return {
        "runtime_sec": float(model.Runtime),
        "wall_sec": wall,
        "status": int(model.Status),
        "obj_val": float(model.ObjVal) if model.SolCount > 0 else None,
        "mip_gap": float(model.MIPGap) if model.SolCount > 0 else None,
        "node_count": int(model.NodeCount),
    }


def _solve_with_prm(mps_path: Path, prm_path: Path, time_limit: int) -> dict:
    """Re-load the model from MPS and apply the tuned parameters."""
    from gurobipy import Model, read as gread

    m = gread(str(mps_path))
    if prm_path.exists():
        m.read(str(prm_path))
    m.setParam("TimeLimit", time_limit)
    m.setParam("OutputFlag", 0)
    t0 = time.perf_counter()
    m.optimize()
    wall = time.perf_counter() - t0
    return {
        "runtime_sec": float(m.Runtime),
        "wall_sec": wall,
        "status": int(m.Status),
        "obj_val": float(m.ObjVal) if m.SolCount > 0 else None,
        "mip_gap": float(m.MIPGap) if m.SolCount > 0 else None,
        "node_count": int(m.NodeCount),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="e.g. matpower/case118")
    ap.add_argument(
        "--tune-time", type=int, default=1800, help="TuneTimeLimit in seconds"
    )
    ap.add_argument(
        "--tune-trials", type=int, default=0, help="TuneTrials (0 = solver default)"
    )
    ap.add_argument(
        "--solve-time", type=int, default=600, help="TimeLimit for default / tuned solve"
    )
    args = ap.parse_args()

    instance = _pick_representative_instance(args.case)
    case_tag = args.case.replace("/", "_")
    mps_path = DATA_DIR / f"grbtune_{case_tag}.mps"
    prm_path = DATA_DIR / f"grbtune_{case_tag}.prm"
    json_path = DATA_DIR / f"grbtune_{case_tag}.json"

    # 1) Build + export
    print(f"[grbtune] Building model for {instance}")
    model = _build_and_export(instance, mps_path)
    print(f"[grbtune] Wrote {mps_path} ({mps_path.stat().st_size / 1e6:.1f} MB)")

    # 2) Default run on the same model
    print("[grbtune] Default-parameter reference solve")
    default_stats = _solve_with_default(model, args.solve_time)

    # 3) Re-build fresh model for tuning (tune modifies param state)
    from gurobipy import read as gread
    tune_model = gread(str(mps_path))
    tune_model.setParam("TuneTimeLimit", args.tune_time)
    if args.tune_trials > 0:
        tune_model.setParam("TuneTrials", args.tune_trials)
    tune_model.setParam("TuneResults", 1)
    tune_model.setParam("OutputFlag", 1)
    print(f"[grbtune] tune() for up to {args.tune_time}s")
    t0 = time.perf_counter()
    tune_model.tune()
    tune_wall = time.perf_counter() - t0

    if tune_model.TuneResultCount > 0:
        tune_model.getTuneResult(0)
        tune_model.write(str(prm_path))
        print(f"[grbtune] Wrote {prm_path}")
        tuned_stats = _solve_with_prm(mps_path, prm_path, args.solve_time)
    else:
        tuned_stats = None
        print("[grbtune] tune() returned no candidates.")

    result = {
        "case": args.case,
        "instance": instance,
        "tune_wall_sec": tune_wall,
        "tune_time_limit": args.tune_time,
        "solve_time_limit": args.solve_time,
        "default": default_stats,
        "tuned": tuned_stats,
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[grbtune] Wrote {json_path}")

    # Aggregate CSV
    row = {
        "case": args.case,
        "instance": instance,
        "default_runtime_s": default_stats["runtime_sec"],
        "default_mip_gap": default_stats["mip_gap"],
        "tuned_runtime_s": tuned_stats["runtime_sec"] if tuned_stats else None,
        "tuned_mip_gap": tuned_stats["mip_gap"] if tuned_stats else None,
        "tune_wall_sec": tune_wall,
    }
    csv_path = TABLES_DIR / "grbtune_vs_default.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["case"] != args.case]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)
    print(f"[grbtune] Updated {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
