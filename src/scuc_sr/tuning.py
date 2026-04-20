"""
Gurobi SCUC Parameter Tuning Tool (SCUC-SR integration)

This module provides:
  - SCUCParameterTuner: utility around Gurobi's tune() API
  - build_scuc_model_for_instance: build a SCUC model for a given instance
  - run_tuning_for_instance: one-shot tuning helper (used by pipeline and CLI)

Design (this version)
---------------------
1) Base parameter configuration
   - You can specify a "base" set of Gurobi parameters, which are applied before tuning.
   - These are passed either as:
        • base_params dict to run_tuning_for_instance(), or
        • CLI flags: --base-param Name=Value (repeatable), --base-param-file params.json
   - Time-based tuning parameters (TimeLimit, MIPGap, TuneTimeLimit, TuneTrials,
     TuneMetric, TuneOutput) are controlled by the tuner and are *not* taken from
     base_params (attempts are ignored with a warning).

2) Single output artifact
   - For a given (case, instance), we now write exactly ONE file:
         <output_dir>/<case_tag>_<inst_tag>_parameters.py
   - This Python file contains:
        • CONFIG: effective tuning configuration (minus base_params),
        • BASE_PARAMS: base parameter values actually applied,
        • TUNING_RESULTS: list of dicts, one per parameter set, with:
              index, status, runtime, objective, gap, params, delta_params
          where delta_params are only those parameters that differ from the base snapshot,
        • BEST_INDEX / BEST_PARAMS,
        • apply_tuned_parameters(model): convenience helper.

   - No .prm, .json, or .csv are written anymore.

3) Handling tuneResultCount == 0
   - If Gurobi produces no tuning results (e.g., tuning interrupted, or problem too
     trivial), we still produce a baseline result:
        • Run a single optimize() with the same TimeLimit and MIPGap used per trial,
        • Record status/runtime/objective/gap and a parameter snapshot,
        • Store this as TUNING_RESULTS[0] in the output file.

CLI usage
---------
    python -m src.scuc_sr.tuning \
        --instance matpower/case57/2017-06-24 \
        --tune-time 7200 \
        --tune-trials 8 \
        --time-limit 300 \
        --mip-gap 0.05 \
        --output-dir src/scuc_sr/tuning_results \
        --base-param MIPFocus=1 \
        --base-param Heuristics=0.7

From the SCUC-SR pipeline:

    python -m src.scuc_sr.pipeline \
        --case matpower/case118 \
        --run-split test \
        --tune \
        --tune-time 3600 \
        --tune-trials 5

The pipeline internally calls run_tuning_for_instance() and you will find exactly
one *_parameters.py file in src/scuc_sr/tuning_results for the tuned instance.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from src.data_preparation.read_data import read_benchmark
from src.optimization_model.SCUC_solver.scuc_model_builder import build_model
from src.scuc_sr.utils import case_tag


# Default set of Gurobi parameters that we log and save for every tuning result.
# You can extend this list if you want to monitor additional parameters.
DEFAULT_INSPECT_PARAMS: List[str] = [
    # Core time / stopping
    "TimeLimit",
    "MIPGap",
    "MIPGapAbs",
    # MIP behavior
    "MIPFocus",
    "Method",
    "Presolve",
    "Aggregate",
    "Cuts",
    "MIPSepCuts",
    "FlowCoverCuts",
    "CliqueCuts",
    "ImpliedCuts",
    "GomoryPasses",
    "ZeroHalfCuts",
    "NetworkCuts",
    "RINS",
    "SubMIPNodes",
    "Heuristics",
    "ImproveStartTime",
    "NoRelHeurTime",
    "Symmetry",
    "BranchDir",
    "VarBranch",
    # Numerics / scaling
    "NumericFocus",
    "ScaleFlag",
    # Resource limits
    "Threads",
    "NodefileStart",
    "NodefileDir",
    # Tuning-specific
    "TuneTimeLimit",
    "TuneTrials",
    "TuneMetric",
    "TuneOutput",
    # SCUC-specific (important to monitor)
    "LazyConstraints",  # Important for lazy contingencies
    "PreCrush",  # Can help with presolve
    "AggFill",  # Aggregation fill limit
    "Quad",  # Quad precision (for numerics)
]


def _parse_param_value(text: str) -> Any:
    """
    Best-effort parser for CLI values of the form Name=Value.

    Tries:
      - bool ("true"/"false")
      - int
      - float
      - otherwise: string as-is
    """
    s = text.strip()
    sl = s.lower()
    if sl == "true":
        return True
    if sl == "false":
        return False
    try:
        iv = int(s)
        return iv
    except Exception:
        pass
    try:
        fv = float(s)
        return fv
    except Exception:
        pass
    return s


def _model_metrics(
    model: gp.Model,
) -> Tuple[int, Optional[float], Optional[float], Optional[float]]:
    """
    Extract (status, runtime, objective, gap) from a solved model.
    """
    status = int(getattr(model, "Status", -1))
    try:
        runtime = float(model.Runtime)
    except Exception:
        runtime = None
    try:
        obj = float(model.ObjVal) if getattr(model, "SolCount", 0) > 0 else None
    except Exception:
        obj = None
    try:
        gap = float(model.MIPGap) if getattr(model, "SolCount", 0) > 0 else None
    except Exception:
        gap = None
    return status, runtime, obj, gap


def _py_repr(obj: Any, indent: int = 0) -> str:
    """
    Render a Python literal for obj (dict/list/tuple/float/other), with indentation.
    Floats with inf are rendered as float('inf') / -float('inf').
    """
    sp = " " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines: List[str] = ["{"]
        for k, v in obj.items():
            key_repr = repr(k)
            val_repr = _py_repr(v, indent + 4)
            lines.append(" " * (indent + 4) + f"{key_repr}: {val_repr},")
        lines.append(sp + "}")
        return "\n".join(lines)
    if isinstance(obj, list):
        if not obj:
            return "[]"
        lines = ["["]
        for v in obj:
            val_repr = _py_repr(v, indent + 4)
            lines.append(" " * (indent + 4) + f"{val_repr},")
        lines.append(sp + "]")
        return "\n".join(lines)
    if isinstance(obj, tuple):
        if not obj:
            return "()"
        inner = ", ".join(_py_repr(v, 0) for v in obj)
        if len(obj) == 1:
            inner += ","
        return f"({inner})"
    if isinstance(obj, float):
        if math.isinf(obj):
            return "float('inf')" if obj > 0 else "-float('inf')"
        if math.isnan(obj):
            return "float('nan')"
        return repr(obj)
    return repr(obj)


class SCUCParameterTuner:
    """
    Utility around model.tune() that:

      - configures tuning + trial parameters on a SCUC model,
      - runs tuning,
      - inspects each tuned parameter set (index 0..tuneResultCount-1),
      - saves the best set plus full per-result parameter snapshots into a single
        Python file (<prefix>_parameters.py).

    The list of parameters to inspect is controlled by `inspect_params`.
    """

    def __init__(
        self,
        output_dir: str | Path = "tuning_results",
        inspect_params: Optional[List[str]] = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # ensure uniqueness while preserving order
        params = inspect_params or DEFAULT_INSPECT_PARAMS
        self.inspect_params: List[str] = list(dict.fromkeys(params))
        self.base_params: Dict[str, Any] = {}
        self.reference_params: Dict[str, Any] = {}

    def configure_tuning(
        self, model: gp.Model, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Configure tuning and per-trial parameters on *model*.

        Expected keys in config (with defaults):
          - tune_time_limit (int, seconds)
          - tune_trials     (int)
          - tune_metric     (int: -1 runtime, 0 balanced, 1 gap)
          - time_limit      (int, per trial)
          - mip_gap         (float)
          - tune_output     (int, TuneOutput level)
          - base_params     (dict[str, Any], applied BEFORE tuning)

        Returns the effective configuration actually applied.
        """
        tune_time_limit = int(config.get("tune_time_limit", 3600))
        tune_trials = int(config.get("tune_trials", 5))
        tune_metric = int(config.get("tune_metric", -1))
        time_limit = int(config.get("time_limit", 300))
        mip_gap = float(config.get("mip_gap", 0.05))
        tune_output = int(config.get("tune_output", 3))

        # Apply base parameters (except those controlled by tuner)
        forbidden = {
            "TuneTimeLimit",
            "TuneTrials",
            "TuneMetric",
            "TuneOutput",
            "TimeLimit",
            "MIPGap",
            "MIPGapAbs",
        }
        base_cfg: Dict[str, Any] = dict(config.get("base_params") or {})
        applied: Dict[str, Any] = {}
        if base_cfg:
            print("\nBase Gurobi parameter values (applied before tuning):")
        for name, val in sorted(base_cfg.items()):
            if name in forbidden:
                print(
                    f"  [skip] {name}={val!r} (controlled by tuning config; "
                    f"use time_limit/mip_gap/tune_* instead)"
                )
                continue
            try:
                model.setParam(name, val)
                applied[name] = val
                print(f"  {name} = {val!r}")
            except gp.GurobiError as e:
                print(f"  [warn] could not set {name}={val!r}: {e}")
        self.base_params = applied

        # Tuning controls and per-trial limits
        model.setParam("TuneTimeLimit", tune_time_limit)
        model.setParam("TuneTrials", tune_trials)
        model.setParam("TuneMetric", tune_metric)
        model.setParam("TimeLimit", time_limit)
        model.setParam("MIPGap", mip_gap)
        model.setParam("TuneOutput", tune_output)

        print("\nTuning configuration:")
        print(f"  TuneTimeLimit : {tune_time_limit} s ({tune_time_limit / 60:.1f} min)")
        print(f"  TuneTrials    : {tune_trials}")
        metric_name = {
            -1: "runtime",
            0: "balanced",
            1: "MIP gap",
        }.get(tune_metric, "custom")
        print(f"  TuneMetric    : {tune_metric} ({metric_name})")
        print(f"  Trial TimeLimit : {time_limit} s")
        print(f"  Trial MIPGap    : {mip_gap:.4f}")
        if applied:
            print(f"  Base params applied: {len(applied)} (see logs above)")

        # Snapshot of reference parameters (after base + tuning config)
        self.reference_params = self._snapshot_params(model)

        return {
            "tune_time_limit": tune_time_limit,
            "tune_trials": tune_trials,
            "tune_metric": tune_metric,
            "time_limit": time_limit,
            "mip_gap": mip_gap,
            "tune_output": tune_output,
            "base_params": applied,
        }

    def run_tuning(self, model: gp.Model) -> float:
        """
        Run Gurobi's automatic tuning on the given model.
        Returns elapsed wall-clock time in seconds.
        """
        print("\n=== Running Gurobi tuning (this may take a long time) ===")
        t0 = time.time()
        model.tune()
        elapsed = time.time() - t0
        print(f"✓ Tuning finished in {elapsed:.1f} s ({elapsed / 60:.1f} min)")
        return float(elapsed)

    def _snapshot_params(self, model: gp.Model) -> Dict[str, Any]:
        """
        Capture current values of all inspect_params from model.Params into a dict.
        Missing parameters are silently skipped.
        """
        snap: Dict[str, Any] = {}
        for name in self.inspect_params:
            try:
                val = getattr(model.Params, name)
                snap[name] = val
            except AttributeError:
                # Secondary retrieval via getParamInfo if available
                try:
                    info = model.getParamInfo(name)
                    if info and len(info) >= 5:
                        # getParamInfo returns (PType, PMin, PMax, PDefault, PValue, ...)
                        snap[name] = info[4]
                except Exception:
                    continue
        return snap

    def analyze_results(
        self, model: gp.Model, config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Inspect tuning results on *model* and return a list of result dicts.

        Each dict contains:
          - index        : tuning result index (0-based, 0 is best according to Gurobi)
          - status       : solver status code for this tuned solve
          - runtime      : runtime [s] for this tuned solve (if available)
          - objective    : objective value (or None)
          - gap          : MIPGap (or None)
          - params       : full snapshot of selected parameters at this result
          - delta_params : {name: value} only for parameters that differ from base
        """
        n = int(getattr(model, "tuneResultCount", 0))

        # No tuning results: build a baseline result using a single optimize().
        if n == 0:
            print(
                "\n=== Tuning results: 0 parameter set(s) ===\n"
                "No tuning results available (e.g., model solved too quickly or tuning was interrupted)."
            )

            # Optionally run a baseline solve to record metrics
            status: int
            runtime: Optional[float]
            objective: Optional[float]
            gap: Optional[float]

            baseline_snapped_before = self._snapshot_params(model)

            if config is not None:
                tl = config.get("time_limit")
                mg = config.get("mip_gap")
                if tl is not None:
                    try:
                        model.setParam("TimeLimit", int(tl))
                    except Exception:
                        pass
                if mg is not None:
                    try:
                        model.setParam("MIPGap", float(mg))
                    except Exception:
                        pass
                print("\nRunning a single baseline optimize() to record metrics...")
                model.optimize()
                status, runtime, objective, gap = _model_metrics(model)
            else:
                status, runtime, objective, gap = _model_metrics(model)

            params_snapshot = self._snapshot_params(model)
            ref = self.reference_params or baseline_snapped_before or params_snapshot
            delta = {k: v for k, v in params_snapshot.items() if ref.get(k) != v}

            result = {
                "index": 0,
                "status": status,
                "runtime": runtime,
                "objective": objective,
                "gap": gap,
                "params": params_snapshot,
                "delta_params": delta,
            }

            print("\n--- Baseline parameter set (no tuning performed) ---")
            print(f"Status  : {status}")
            if runtime is not None:
                print(f"Runtime : {runtime:.2f} s")
            if objective is not None:
                print(f"Obj     : {objective:.6g}")
            if gap is not None:
                print(f"MIPGap  : {100.0 * gap:.3f} %")
            if delta:
                print("Changed parameters vs base (if any):")
                for k in sorted(delta.keys()):
                    print(f"  {k} = {delta[k]!r}")
            else:
                print("No parameter changes vs base for this set.")

            return [result]

        # Normal tuning results
        print(f"\n=== Tuning results: {n} parameter set(s) ===")
        results: List[Dict[str, Any]] = []
        ref = self.reference_params or self._snapshot_params(model)

        for i in range(n):
            model.getTuneResult(i)
            status = int(getattr(model, "Status", -1))
            try:
                runtime = float(model.Runtime)
            except Exception:
                runtime = None
            try:
                obj_val = float(model.ObjVal) if model.SolCount > 0 else None
                gap = float(model.MIPGap) if model.SolCount > 0 else None
            except Exception:
                obj_val = None
                gap = None

            params_snapshot = self._snapshot_params(model)
            delta = {k: v for k, v in params_snapshot.items() if ref.get(k) != v}

            results.append(
                {
                    "index": i,
                    "status": status,
                    "runtime": runtime,
                    "objective": obj_val,
                    "gap": gap,
                    "params": params_snapshot,
                    "delta_params": delta,
                }
            )

        # Human-readable summary
        for r in results:
            print(f"\n--- Parameter set #{r['index'] + 1} ---")
            print(f"Status  : {r['status']}")
            if r["runtime"] is not None:
                print(f"Runtime : {r['runtime']:.2f} s")
            if r["objective"] is not None:
                print(f"Obj     : {r['objective']:.6g}")
            if r["gap"] is not None:
                print(f"MIPGap  : {100.0 * r['gap']:.3f} %")
            delta = r.get("delta_params") or {}
            if delta:
                print("Changed parameters vs base:")
                for k in sorted(delta.keys()):
                    print(f"  {k} = {delta[k]!r}")
            else:
                print("No parameter changes vs base for this set.")

        return results

    def save_best_parameters(
        self,
        model: gp.Model,
        results: List[Dict[str, Any]],
        output_prefix: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, str]]:
        """
        Save best parameter set (index 0 in results) and all per-set metrics into
        a single Python module:

            <output_dir>/<output_prefix>_parameters.py

        The file defines:
          - CONFIG, BASE_PARAMS,
          - TUNING_RESULTS, BEST_INDEX, BEST_PARAMS,
          - apply_tuned_parameters(model).

        Returns a dict with {"parameters_py": <path>} or None if no results.
        """
        if not results:
            print("No tuning results to save.")
            return None

        best = results[0]  # by Gurobi ordering (0 = best)

        code_file = self.output_dir / f"{output_prefix}_parameters.py"
        code_file.parent.mkdir(parents=True, exist_ok=True)

        simple_results: List[Dict[str, Any]] = []
        for r in results:
            simple_results.append(
                {
                    "index": int(r.get("index", 0)),
                    "status": int(r.get("status", -1)),
                    "runtime": r.get("runtime"),
                    "objective": r.get("objective"),
                    "gap": r.get("gap"),
                    "params": dict(r.get("params") or {}),
                    "delta_params": dict(r.get("delta_params") or {}),
                }
            )

        cfg = dict(config or {})
        # Avoid duplicating base_params inside CONFIG; exposed separately.
        if "base_params" in cfg:
            cfg_no_base = dict(cfg)
            cfg_no_base.pop("base_params", None)
        else:
            cfg_no_base = cfg

        with code_file.open("w", encoding="utf-8") as fh:
            fh.write("# Auto-generated Gurobi tuning results\n")
            fh.write(f"# Generated at: {datetime.utcnow().isoformat()}Z\n\n")
            fh.write("from typing import Any, Dict, List\n\n")
            fh.write("# Effective tuning configuration (without BASE_PARAMS)\n")
            fh.write("CONFIG: Dict[str, Any] = ")
            fh.write(_py_repr(cfg_no_base, indent=0))
            fh.write("\n\n")
            fh.write("# Base parameters applied before tuning\n")
            fh.write("BASE_PARAMS: Dict[str, Any] = ")
            fh.write(_py_repr(self.base_params, indent=0))
            fh.write("\n\n")
            fh.write("# Per-parameter-set results (baseline and/or tuned)\n")
            fh.write("TUNING_RESULTS: List[Dict[str, Any]] = ")
            fh.write(_py_repr(simple_results, indent=0))
            fh.write("\n\n")
            fh.write("# Best parameter set index according to Gurobi (0-based)\n")
            fh.write("BEST_INDEX: int = 0\n\n")
            fh.write(
                "BEST_PARAMS: Dict[str, Any] = TUNING_RESULTS[BEST_INDEX]['params']\n\n"
            )
            fh.write("def apply_tuned_parameters(model: Any) -> None:\n")
            fh.write(
                '    """Apply tuned Gurobi parameters for the best parameter set."""\n'
            )
            fh.write("    for name, value in BEST_PARAMS.items():\n")
            fh.write("        model.setParam(name, value)\n")

        print(f"\n✓ Wrote tuning parameters + results : {code_file}")

        print("\nHow to use tuned parameters:")
        print("  from {} import apply_tuned_parameters".format(code_file.stem))
        print("  apply_tuned_parameters(model)")

        return {"parameters_py": str(code_file)}


def build_scuc_model_for_instance(
    instance_name: str, use_lazy_contingencies: bool = True
) -> Tuple[gp.Model, object, gp.Env]:
    """
    Build a SCUC model for a given instance using the standard SCUC builder.

    Parameters
    ----------
    instance_name : str
        Dataset name, e.g. 'matpower/case57/2017-06-24'.
    use_lazy_contingencies : bool
        If True, build model with lazy N-1 enforcement; otherwise explicit.

    Returns
    -------
    (model, scenario, env)
    """
    inst = read_benchmark(instance_name, quiet=True)
    sc = inst.deterministic
    env = gp.Env()
    model = build_model(
        scenario=sc,
        use_lazy_contingencies=bool(use_lazy_contingencies),
        env=env,
        name_constraints=False,
        radius_line_whitelist=None,
    )
    # Baseline numerics and heuristics for SCUC
    model.setParam("NumericFocus", 1)
    model.setParam("Method", 1)
    model.setParam("MIPFocus", 1)
    # model.setParam("Aggregate", 2)
    model.setParam("RINS", 0)
    model.setParam("Heuristics", 0.8)
    model.setParam("ScaleFlag", 1)
    model.setParam("LazyConstraints", 1)
    try:
        model._scenario_name = sc.name or instance_name
    except Exception:
        pass
    return model, sc, env


def _sanitize_instance_name(s: str) -> str:
    s = s.strip().strip("/\\").replace("\\", "/")
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    t = "".join(out)
    while "__" in t:
        t = t.replace("__", "_")
    return t.strip("_").lower()


def run_tuning_for_instance(
    instance_name: str,
    case_folder: str,
    *,
    tune_time_limit: int = 3600,
    tune_trials: int = 5,
    tune_metric: int = -1,
    time_limit: int = 300,
    mip_gap: float = 0.05,
    output_dir: str | Path = "tuning_results",
    use_lazy_contingencies: bool = False,
    base_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    High-level helper: build SCUC model for *instance_name*, run Gurobi tuning,
    and save best parameter sets under *output_dir* as a single *_parameters.py file.

    Parameters
    ----------
    instance_name : str
        Dataset name, e.g. 'matpower/case57/2017-06-24'.
    case_folder : str
        Case folder, e.g. 'matpower/case57'.
    tune_time_limit : int
        Overall tuning budget (TuneTimeLimit).
    tune_trials : int
        Number of parameter sets Gurobi will explore (TuneTrials).
    tune_metric : int
        Metric for tuning: -1=runtime, 0=balanced, 1=gap.
    time_limit : int
        Per-trial TimeLimit for MIP solves.
    mip_gap : float
        Per-trial MIPGap.
    output_dir : str | Path
        Directory for tuning outputs.
    use_lazy_contingencies : bool
        Whether to build model with lazy N-1 contingencies.
    base_params : Optional[Dict[str, Any]]
        Base Gurobi parameters to apply *before* tuning.

    Returns
    -------
    dict with:
      - elapsed_time : total tuning time (seconds)
      - results      : list of tuning results (see analyze_results())
      - files        : dict with {'parameters_py': path} or None on failure
      - config       : effective config used (incl. applied base_params)
    """
    tag = case_tag(case_folder)
    inst_tag = _sanitize_instance_name(instance_name)
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== SCUC parameter tuning ===\n"
        f"Case      : {case_folder} (tag={tag})\n"
        f"Instance  : {instance_name}\n"
        f"Output dir: {out_dir}\n"
    )

    model, sc, env = build_scuc_model_for_instance(
        instance_name, use_lazy_contingencies=use_lazy_contingencies
    )

    elapsed: float = 0.0
    results: List[Dict[str, Any]] = []
    files: Optional[Dict[str, str]] = None
    cfg_used: Dict[str, Any] = {}

    try:
        tuner = SCUCParameterTuner(output_dir=out_dir)
        config = {
            "tune_time_limit": int(tune_time_limit),
            "tune_trials": int(tune_trials),
            "tune_metric": int(tune_metric),
            "time_limit": int(time_limit),
            "mip_gap": float(mip_gap),
            "tune_output": 3,
            "base_params": dict(base_params or {}),
        }
        cfg_used = tuner.configure_tuning(model, config)
        elapsed = tuner.run_tuning(model)
        results = tuner.analyze_results(model, config=cfg_used)
        prefix = f"{tag}_{inst_tag}"
        files = tuner.save_best_parameters(
            model, results, output_prefix=prefix, config=cfg_used
        )
    except gp.GurobiError as e:
        print(f"[tuning] GurobiError during tuning: {e}")
    except Exception as e:
        print(f"[tuning] Unexpected error during tuning: {e}")
    finally:
        try:
            model.dispose()
        except Exception:
            pass
        try:
            env.dispose()
        except Exception:
            pass

    return {
        "elapsed_time": elapsed,
        "results": results,
        "files": files,
        "config": cfg_used,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Tune Gurobi parameters for a single SCUC instance."
    )
    ap.add_argument(
        "--instance",
        required=True,
        help="Instance name, e.g. 'matpower/case57/2017-06-24'",
    )
    ap.add_argument(
        "--use-lazy",
        action="store_true",
        default=False,
        help="Build model with lazy N-1 contingencies instead of explicit.",
    )
    ap.add_argument(
        "--tune-time",
        type=int,
        default=3600,
        help="Total tuning time budget in seconds (TuneTimeLimit).",
    )
    ap.add_argument(
        "--tune-trials",
        type=int,
        default=5,
        help="Number of parameter sets to explore (TuneTrials).",
    )
    ap.add_argument(
        "--tune-metric",
        type=int,
        choices=[-1, 0, 1],
        default=-1,
        help="Tuning metric: -1=runtime, 0=balanced, 1=gap.",
    )
    ap.add_argument(
        "--time-limit",
        type=int,
        default=300,
        help="Per-trial TimeLimit parameter (seconds).",
    )
    ap.add_argument(
        "--mip-gap",
        type=float,
        default=0.05,
        help="Per-trial MIPGap parameter.",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="src/scuc_sr/tuning_results",
        help="Directory for tuning outputs.",
    )
    ap.add_argument(
        "--base-param",
        action="append",
        default=[],
        help="Base Gurobi parameter in the form Name=Value. "
        "May be given multiple times.",
    )
    ap.add_argument(
        "--base-param-file",
        type=str,
        default="",
        help="Path to JSON file mapping parameter names to base values.",
    )

    args = ap.parse_args()
    inst = args.instance.strip().strip("/\\").replace("\\", "/")
    case_folder = "/".join(inst.split("/")[:2]) if "/" in inst else inst

    # Build base_params from file + CLI flags
    base_params: Dict[str, Any] = {}
    if args.base_param_file:
        p = Path(args.base_param_file)
        if p.is_file():
            try:
                with p.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    for k, v in data.items():
                        base_params[str(k)] = v
                else:
                    print(
                        f"[tuning] base-param-file {p} does not contain a JSON object."
                    )
            except Exception as e:
                print(f"[tuning] Failed to read base-param-file {p}: {e}")
        else:
            print(f"[tuning] base-param-file {p} not found; ignored.")

    for item in args.base_param or []:
        if "=" not in item:
            print(
                f"[tuning] Ignoring malformed --base-param '{item}' (expected Name=Value)."
            )
            continue
        name, val_str = item.split("=", 1)
        name = name.strip()
        if not name:
            continue
        val = _parse_param_value(val_str)
        base_params[name] = val

    res = run_tuning_for_instance(
        instance_name=inst,
        case_folder=case_folder,
        tune_time_limit=args.tune_time,
        tune_trials=args.tune_trials,
        tune_metric=args.tune_metric,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        output_dir=args.output_dir,
        use_lazy_contingencies=bool(args.use_lazy),
        base_params=base_params or None,
    )

    print("\n=== Tuning complete ===")
    print(f"Elapsed time        : {res['elapsed_time']:.1f} s")
    print(f"Parameter sets tried: {len(res['results'])}")
    files = res.get("files") or {}
    params_path = files.get("parameters_py") if isinstance(files, dict) else None
    if params_path:
        print("\nTuning results + parameters file:")
        print(f"  {params_path}")
    else:
        print("\nNo parameter file was generated (tuning may have failed early).")


if __name__ == "__main__":
    main()
