"""
CommitmentHints: lightweight CatBoost/SKLearn classifier to predict unit commitment
u[g,t] and inject "soft" guidance into the solver (Start, VarHintVal, BranchPriority).
Trains from solved JSON solutions in src/data/output/<case_folder>.

Key features
- Offline training per case folder (e.g., matpower/case118).
- Handles missing CatBoost by falling back to scikit-learn GradientBoostingClassifier.
- Only uses "safe" features available from inputs and simple time descriptors:
  • system load fraction at time t (sum(load)/sum(Pmax))
  • t/T
  • generator min/max as fractions of system Pmax sum
  • ramp_up/down, startup/shutdown as fractions of Pmax
  • must-run flag at time t
  • initial status (binary; ON if >0 at horizon start)
- At inference, applies to a given instance (scenario + model) and:
  • sets Start for commit[g,t] = round(prob) (or keep a soft Start)
  • sets VarHintVal = probRounded, VarHintPri higher for earlier t
  • optionally "fix-safe" bounds LB=UB=0/1 only if prob >= fix_thr (disabled by default)
    Note: we recommend NOT fixing; hints + Start are safer and effective.

CLI examples:
- Train for case118 (CatBoost if available; else SKLearn GBDT):
    python -m src.ml_models.commitment_hints --case matpower/case118 --force

- Inspect trained model path:
    python -m src.ml_models.commitment_hints --case matpower/case118 --report

Usage from code:
    hints = CommitmentHints(case_folder="matpower/case118")
    hints.pretrain(force=False)
    ok = hints.ensure_trained()
    if ok:
        num_hinted = hints.apply_to_model(model, scenario, instance_name, thr=0.98, mode="hint")

"""

from __future__ import annotations

import argparse
import json
import gzip
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional CatBoost
try:
    from catboost import CatBoostClassifier

    HAVE_CATBOOST = True
except Exception:
    HAVE_CATBOOST = False

# Fallback SKLearn
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

from src.data_preparation.params import DataParams
from src.data_preparation.read_data import read_benchmark


def _sanitize_name(s: str) -> str:
    s = (s or "").strip().strip("/\\").replace("\\", "/")
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    t = "".join(out)
    while "__" in t:
        t = t.replace("__", "_")
    return t.strip("_").lower()


def _case_tag(case_folder: str) -> str:
    return _sanitize_name(case_folder)


def _list_outputs(case_folder: str) -> List[Path]:
    case_dir = (DataParams._OUTPUT / case_folder).resolve()
    if not case_dir.exists():
        return []
    return sorted(case_dir.glob("*.json"))


def _load_output_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_input_json(path: Path) -> Optional[dict]:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _instance_name_from_output(p: Path) -> str:
    rel = p.resolve().relative_to(DataParams._OUTPUT.resolve()).as_posix()
    if rel.endswith(".json"):
        rel = rel[: -len(".json")]
    return rel


def _load_system_load_from_input(instance_name: str) -> Optional[List[float]]:
    ip = (DataParams._CACHE / (instance_name + ".json.gz")).resolve()
    j = _read_input_json(ip)
    if not j:
        return None
    buses = j.get("Buses", {})
    T = None
    for b in buses.values():
        ld = b.get("Load (MW)", 0.0)
        if isinstance(ld, list):
            T = len(ld)
            break
    if T is None:
        T = 1
    sys_load = [0.0] * T
    for b in buses.values():
        ld = b.get("Load (MW)", 0.0)
        if isinstance(ld, list):
            for t in range(T):
                sys_load[t] += float(ld[t])
        else:
            sys_load[0] += float(ld)
    return sys_load


class _SimpleModelWrap:
    """Abstraction layer to unify CatBoost and SKLearn calls."""

    def __init__(self, model_type: str = "auto"):
        self.kind = (
            "cb" if HAVE_CATBOOST and model_type in ("auto", "catboost") else "sk"
        )
        if self.kind == "cb":
            self.model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.06,
                depth=6,
                loss_function="Logloss",
                verbose=False,
                random_state=42,
            )
        else:
            if not HAVE_SKLEARN:
                raise RuntimeError("Neither CatBoost nor scikit-learn is available.")
            self.model = GradientBoostingClassifier(
                random_state=42, n_estimators=300, learning_rate=0.06, max_depth=3
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.kind == "cb":
            self.model.fit(X, y)
        else:
            self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "cb":
            p = self.model.predict_proba(X)
            return np.array(p)[:, 1]
        else:
            p = self.model.predict_proba(X)[:, 1]
            return np.array(p)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.kind == "cb":
            self.model.save_model(str(path))
        else:
            # SKLearn: save as JSON-ish with numpy arrays
            payload = {
                "kind": "sk",
                "model_bytes": None,
            }
            # Use joblib if available
            try:
                import joblib

                joblib.dump(self.model, path)
                return
            except Exception:
                pass
            # Fallback: cannot serialize; raise
            raise RuntimeError("Install joblib to save SKLearn model")

    @staticmethod
    def load(path: Path):
        if HAVE_CATBOOST and (path.suffix.lower() in (".cbm", "")):
            # Try CatBoost
            try:
                m = CatBoostClassifier()
                m.load_model(str(path))
                w = _SimpleModelWrap("catboost")
                w.kind = "cb"
                w.model = m
                return w
            except Exception:
                pass
        # Try joblib for SKLearn
        try:
            import joblib

            mdl = joblib.load(path)
            w = _SimpleModelWrap(model_type="auto")
            w.kind = "sk"
            w.model = mdl
            return w
        except Exception as e:
            raise RuntimeError(f"Cannot load model from {path}: {e}")


class CommitmentHints:
    def __init__(self, case_folder: str, model_type: str = "auto"):
        self.case = case_folder.strip().strip("/\\").replace("\\", "/")
        self.tag = _case_tag(self.case)
        self.base_dir = (
            DataParams._INTERMEDIATE / "commitment_hints" / self.tag
        ).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = (
            self.base_dir / "commit_model.cbm"
        ).resolve()  # CatBoost default ext
        self.meta_path = (self.base_dir / "meta.json").resolve()
        self.model_type = model_type
        self.model: Optional[_SimpleModelWrap] = None
        self.meta: Dict = {}

    def _build_training_df(self, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        outs = _list_outputs(self.case)
        if not outs:
            return None
        rows = []
        count = 0
        for op in outs:
            if limit is not None and count >= limit:
                break
            out = _load_output_json(op)
            if not out:
                continue
            name = _instance_name_from_output(op)
            # Load scenario for static features
            try:
                inst = read_benchmark(name, quiet=True)
                sc = inst.deterministic
            except Exception:
                continue
            sys_load = _load_system_load_from_input(name) or [
                sum(b.load[0] for b in sc.buses)
            ]
            T = sc.time
            # Sum Pmax across units (time-average)
            sum_pmax = 0.0
            for g in sc.thermal_units:
                sum_pmax += float(np.mean([float(x) for x in g.max_power]))
            if sum_pmax <= 0:
                sum_pmax = 1.0
            # Labels from solved output
            gens_sol = out.get("generators", {}) or {}
            for g in sc.thermal_units:
                gsol = gens_sol.get(g.name, {})
                u_list = gsol.get("commit", None)
                if u_list is None:
                    # cannot train this gen on this instance
                    continue
                # static features normalized
                ru = float(getattr(g, "ramp_up", 0.0)) / max(1.0, sum_pmax)
                rd = float(getattr(g, "ramp_down", 0.0)) / max(1.0, sum_pmax)
                su = float(getattr(g, "startup_limit", 0.0)) / max(1.0, sum_pmax)
                sd = float(getattr(g, "shutdown_limit", 0.0)) / max(1.0, sum_pmax)
                init_on = (
                    1.0
                    if (g.initial_status is not None and g.initial_status > 0)
                    else 0.0
                )
                for t in range(T):
                    pmin = float(g.min_power[t]) / max(1.0, sum_pmax)
                    pmax = float(g.max_power[t]) / max(1.0, sum_pmax)
                    must = 1.0 if bool(g.must_run[t]) else 0.0
                    load_frac = float(sys_load[min(t, len(sys_load) - 1)]) / max(
                        1.0, sum_pmax
                    )
                    tt = float(t) / max(1, T - 1)
                    y = int(1 if float(u_list[t]) >= 0.5 else 0)
                    rows.append(
                        {
                            "load_frac": load_frac,
                            "t_frac": tt,
                            "pmin_frac": pmin,
                            "pmax_frac": pmax,
                            "ru_frac": ru,
                            "rd_frac": rd,
                            "su_frac": su,
                            "sd_frac": sd,
                            "must": must,
                            "init_on": init_on,
                            "y": y,
                        }
                    )
            count += 1
        if not rows:
            return None
        df = pd.DataFrame(rows)
        return df

    def pretrain(
        self, force: bool = False, limit: Optional[int] = None
    ) -> Optional[Path]:
        if self.model_path.exists() and not force:
            return self.model_path
        df = self._build_training_df(limit=limit)
        if df is None or df.empty:
            print(f"[commit_hints] No training data found for {self.case}")
            return None
        X = df.drop(columns=["y"]).to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=int)
        mdl = _SimpleModelWrap(self.model_type)
        mdl.fit(X, y)
        # Basic AUC on sample (optional)
        try:
            p = mdl.predict_proba(X)
            auc = roc_auc_score(y, p)
        except Exception:
            auc = None
        mdl.save(self.model_path)
        meta = {
            "case_folder": self.case,
            "features": list(df.drop(columns=["y"]).columns),
            "auc_train": None if auc is None else float(auc),
            "saved_model": str(self.model_path),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.model = mdl
        self.meta = meta
        print(f"[commit_hints] Trained model saved to {self.model_path}, AUC={auc}")
        return self.model_path

    def ensure_trained(self) -> bool:
        if self.model is not None:
            return True
        if self.model_path.exists():
            try:
                self.model = _SimpleModelWrap.load(self.model_path)
                self.meta = (
                    json.loads(self.meta_path.read_text(encoding="utf-8"))
                    if self.meta_path.exists()
                    else {}
                )
                return True
            except Exception:
                return False
        return False

    def _features_for_instance(
        self, sc, instance_name: str
    ) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        # Build X for (gen,t) over scenario sc
        sys_load = _load_system_load_from_input(instance_name) or [
            sum(b.load[0] for b in sc.buses)
        ]
        T = sc.time
        sum_pmax = 0.0
        for g in sc.thermal_units:
            sum_pmax += float(np.mean([float(x) for x in g.max_power]))
        if sum_pmax <= 0:
            sum_pmax = 1.0
        rows = []
        keys = []
        for g in sc.thermal_units:
            ru = float(getattr(g, "ramp_up", 0.0)) / max(1.0, sum_pmax)
            rd = float(getattr(g, "ramp_down", 0.0)) / max(1.0, sum_pmax)
            su = float(getattr(g, "startup_limit", 0.0)) / max(1.0, sum_pmax)
            sd = float(getattr(g, "shutdown_limit", 0.0)) / max(1.0, sum_pmax)
            init_on = (
                1.0 if (g.initial_status is not None and g.initial_status > 0) else 0.0
            )
            for t in range(T):
                load_frac = float(sys_load[min(t, len(sys_load) - 1)]) / max(
                    1.0, sum_pmax
                )
                pmin = float(g.min_power[t]) / max(1.0, sum_pmax)
                pmax = float(g.max_power[t]) / max(1.0, sum_pmax)
                must = 1.0 if bool(g.must_run[t]) else 0.0
                tt = float(t) / max(1, T - 1)
                rows.append([load_frac, tt, pmin, pmax, ru, rd, su, sd, must, init_on])
                keys.append((g.name, t))
        X = np.array(rows, dtype=float) if rows else np.zeros((0, 10), dtype=float)
        return X, keys

    def apply_to_model(
        self, model, sc, instance_name: str, thr: float = 0.98, mode: str = "hint"
    ) -> int:
        """
        Apply hints to model.commit[g,t] using predicted p(ON).
        mode: "hint" (recommended), "start", "fix" (not recommended).
        Returns number of variables touched.
        """
        ok = self.ensure_trained()
        if not ok or self.model is None:
            return 0
        commit = getattr(model, "commit", None)
        if commit is None:
            return 0
        X, keys = self._features_for_instance(sc, instance_name)
        if X.shape[0] == 0:
            return 0
        prob = self.model.predict_proba(X)
        applied = 0
        # derive time horizon to set priorities: earlier t higher priority
        times = [t for _, t in keys]
        T = 1 + max(times) if times else 1
        for (gname, t), p in zip(keys, prob):
            try:
                var = commit[gname, t]
            except Exception:
                continue
            # Start
            if mode in ("hint", "start"):
                try:
                    var.Start = 1.0 if p >= 0.5 else 0.0
                except Exception:
                    pass
            # VarHintVal and priorities
            try:
                var.VarHintVal = 1.0 if p >= 0.5 else 0.0
            except Exception:
                pass
            try:
                var.VarHintPri = float(10 + (T - int(t)))
            except Exception:
                pass
            try:
                var.BranchPriority = int(10 + (T - int(t)))
            except Exception:
                pass
            applied += 1
            # Optional fixing (not recommended; only if user insists)
            if mode == "fix" and float(p) >= float(thr):
                try:
                    var.LB = 1.0
                    var.UB = 1.0
                except Exception:
                    pass
            elif mode == "fix" and float(p) <= 1.0 - float(thr):
                try:
                    var.LB = 0.0
                    var.UB = 0.0
                except Exception:
                    pass
        return applied


def main():
    ap = argparse.ArgumentParser(
        description="Train and manage ML-based commitment hints."
    )
    ap.add_argument("--case", required=True, help="Case folder, e.g., matpower/case118")
    ap.add_argument("--force", action="store_true", help="Overwrite existing model")
    ap.add_argument(
        "--limit", type=int, default=0, help="Limit training outputs (debug)"
    )
    ap.add_argument(
        "--report", action="store_true", help="Print model meta if available"
    )
    args = ap.parse_args()

    ch = CommitmentHints(args.case)
    if args.report:
        ok = ch.ensure_trained()
        if not ok:
            print("No trained model found.")
            return
        meta = ch.meta or {}
        print("Model:", ch.model_path)
        print("Meta:", json.dumps(meta, indent=2))
        return

    p = ch.pretrain(
        force=args.force, limit=(args.limit if args.limit and args.limit > 0 else None)
    )
    if p:
        print(f"Model written: {p}")
    else:
        print("Training failed or no data.")


if __name__ == "__main__":
    main()
