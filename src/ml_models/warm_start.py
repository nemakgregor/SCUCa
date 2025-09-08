import json
import gzip
import math
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from src.data_preparation.params import DataParams


def _sanitize_name(s: str) -> str:
    s = (s or "").strip().strip("/\\").replace("\\", "/")
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    t = "".join(out)
    while "__" in t:
        t = t.replace("__", "_")
    return t.strip("_").lower()


def _case_folder_from_instance(instance_name: str) -> str:
    parts = instance_name.strip().strip("/\\").replace("\\", "/").split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] if parts else "case"


def _case_tag(case_folder: str) -> str:
    return _sanitize_name(case_folder)


def _load_input_system_load(input_path: Path) -> Optional[List[float]]:
    try:
        if input_path.suffix == ".gz":
            with gzip.open(input_path, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            with input_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        buses = data.get("Buses", {})
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
    except Exception:
        return None


def _load_output_solution(output_path: Path) -> Optional[dict]:
    try:
        with output_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _zscore(vec: List[float]) -> List[float]:
    if not vec:
        return []
    mean = sum(vec) / len(vec)
    var = sum((x - mean) ** 2 for x in vec) / max(1, len(vec) - 1)
    std = math.sqrt(var) if var > 0 else 1.0
    return [(x - mean) / std for x in vec]


def _l2(v1: List[float], v2: List[float]) -> float:
    if len(v1) != len(v2):
        L = max(len(v1), len(v2))
        v1 = v1 + [v1[-1] if v1 else 0.0] * (L - len(v1))
        v2 = v2 + [v2[-1] if v2 else 0.0] * (L - len(v2))
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def _ensure_list_length(vals: List, T: int, pad_with_last: bool = True, default=0):
    if vals is None:
        return [default] * T
    vals = list(vals)
    if len(vals) < T:
        pad_val = vals[-1] if (pad_with_last and vals) else default
        vals = vals + [pad_val] * (T - len(vals))
    elif len(vals) > T:
        vals = vals[:T]
    return vals


def _hash01(name: str, seed: int) -> float:
    """
    Deterministic pseudo-random in [0,1) from (name, seed).
    """
    h = hashlib.md5(f"{name}::{int(seed)}".encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / float(0xFFFFFFFF)
    return min(max(v, 0.0), math.nextafter(1.0, 0.0))


class WarmStartProvider:
    """
    k-NN warm start over historical solutions (persisted 'index' per case).

    Training (pretrain): see original docstring.

    Inference: generate_and_save_warm_start() writes a warm JSON capturing a neighbor's
    solution, and apply_warm_start_to_model() sets Start values on the current model.

    New in this version:
      - generate_and_save_warm_start(..., auto_fix=True) will automatically build a
        'warm_fixed_<instance>.json' using src.ml_models.fix_warm_start (no Gurobi).
      - apply_warm_start_to_model(...) now prefers a 'warm_fixed_<instance>.json' file
        if present, falling back to 'warm_<instance>.json'.
    """

    def __init__(
        self,
        case_folder: Optional[str] = None,
        coverage_threshold: float = 0.70,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        split_seed: int = 42,
    ):
        self.base_input = DataParams._CACHE
        self.base_output = DataParams._OUTPUT
        self.base_warm = DataParams._WARM_START
        self.case_folder = case_folder  # e.g., "matpower/case14"
        self.coverage_threshold = float(coverage_threshold)

        # Split config
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        if self.train_ratio < 0.0:
            self.train_ratio = 0.0
        if self.val_ratio < 0.0:
            self.val_ratio = 0.0
        if self.train_ratio + self.val_ratio > 1.0:
            self.val_ratio = max(0.0, 1.0 - self.train_ratio)
        self.split_seed = int(split_seed)

        # Runtime state
        self._trained_index: Dict[str, dict] = {}
        self._coverage: float = 0.0
        self._available: bool = False
        self._trained: bool = False

        # Dataset bookkeeping
        self._inputs_list: List[str] = []
        self._outputs_list: List[str] = []
        self._splits: Dict[str, Set[str]] = {
            "train": set(),
            "val": set(),
            "test": set(),
        }

    def _list_inputs(self, case_folder: str) -> List[Path]:
        case_dir = (self.base_input / case_folder).resolve()
        if not case_dir.exists():
            return []
        return sorted(case_dir.glob("*.json.gz"))

    def _list_outputs(self, case_folder: str) -> List[Path]:
        case_dir = (self.base_output / case_folder).resolve()
        if not case_dir.exists():
            return []
        return sorted(case_dir.glob("*.json"))

    def _dataset_name_from_output(self, p: Path) -> str:
        rel = p.resolve().relative_to(self.base_output.resolve()).as_posix()
        if rel.endswith(".json"):
            rel = rel[: -len(".json")]
        return rel

    def _dataset_name_from_input(self, p: Path) -> str:
        rel = p.resolve().relative_to(self.base_input.resolve()).as_posix()
        if rel.endswith(".json.gz"):
            rel = rel[: -len(".json.gz")]
        return rel

    def _index_path(self, case_folder: str) -> Path:
        tag = _case_tag(case_folder)
        return (self.base_warm / f"ws_index_{tag}.json").resolve()

    def _load_index_file(self, case_folder: str) -> bool:
        path = self._index_path(case_folder)
        if not path.exists():
            return False
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if obj.get("case_folder") != case_folder:
                return False

            items = obj.get("items", [])
            idx: Dict[str, dict] = {}
            for it in items:
                name = it.get("instance")
                feats = it.get("features", [])
                commit = it.get("generators", {})
                if not name or not feats or commit is None:
                    continue
                commit_simple = {}
                for g, v in commit.items():
                    try:
                        commit_simple[g] = list(v.get("commit", []))
                    except Exception:
                        commit_simple[g] = []
                idx[name] = {"features": feats, "commit": commit_simple}
            self._trained_index = idx
            self._coverage = float(obj.get("coverage", 0.0))
            self._available = len(self._trained_index) > 0
            self._trained = self._available and (
                self._coverage >= self.coverage_threshold
            )

            self._inputs_list = list(obj.get("inputs", []))
            self._outputs_list = list(obj.get("outputs", []))

            ratios = obj.get("ratios", {})
            self.train_ratio = float(ratios.get("train", self.train_ratio))
            self.val_ratio = float(ratios.get("val", self.val_ratio))
            self.split_seed = int(obj.get("split_seed", self.split_seed))

            split = obj.get("split", None)
            if isinstance(split, dict):
                self._splits = {
                    "train": set(split.get("train", [])),
                    "val": set(split.get("val", [])),
                    "test": set(split.get("test", [])),
                }
            else:
                self._compute_splits_from_index()
            return True
        except Exception:
            return False

    def _save_index_file(self, case_folder: str) -> Path:
        path = self._index_path(case_folder)
        path.parent.mkdir(parents=True, exist_ok=True)
        items = []
        for name, item in self._trained_index.items():
            items.append(
                {
                    "instance": name,
                    "features": list(item["features"]),
                    "generators": {
                        g: {"commit": list(v)} for g, v in item["commit"].items()
                    },
                }
            )
        payload = {
            "case_folder": case_folder,
            "built_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "coverage": float(self._coverage),
            "threshold": float(self.coverage_threshold),
            "ratios": {
                "train": float(self.train_ratio),
                "val": float(self.val_ratio),
                "test": float(max(0.0, 1.0 - self.train_ratio - self.val_ratio)),
            },
            "split_seed": int(self.split_seed),
            "inputs": list(self._inputs_list),
            "outputs": list(self._outputs_list),
            "split": {
                "train": sorted(self._splits["train"]),
                "val": sorted(self._splits["val"]),
                "test": sorted(self._splits["test"]),
            },
            "items": items,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _compute_splits_from_names(self, names: List[str]) -> None:
        tr: Set[str] = set()
        va: Set[str] = set()
        te: Set[str] = set()
        for nm in names:
            r = _hash01(nm, self.split_seed)
            if r < self.train_ratio:
                tr.add(nm)
            elif r < self.train_ratio + self.val_ratio:
                va.add(nm)
            else:
                te.add(nm)
        self._splits = {"train": tr, "val": va, "test": te}

    def _compute_splits_from_index(self) -> None:
        names = sorted(self._trained_index.keys())
        self._compute_splits_from_names(names)

    def _build_index(self, case_folder: str) -> None:
        inputs = self._list_inputs(case_folder)
        outputs = self._list_outputs(case_folder)
        total_inputs = len(inputs)
        total_outputs = len(outputs)
        self._coverage = (total_outputs / total_inputs) if total_inputs > 0 else 0.0

        self._inputs_list = [self._dataset_name_from_input(p) for p in inputs]
        outputs_names = [self._dataset_name_from_output(p) for p in outputs]
        self._outputs_list = outputs_names

        index: Dict[str, dict] = {}
        for out_path in outputs:
            out = _load_output_solution(out_path)
            if not out:
                continue
            meta = out.get("meta", {})
            instance_name = meta.get("instance_name") or self._dataset_name_from_output(
                out_path
            )
            sys_load = None
            try:
                sys_load = out.get("system", {}).get("load", None)
            except Exception:
                sys_load = None
            if not sys_load:
                in_path = (self.base_input / (instance_name + ".json.gz")).resolve()
                sys_load = _load_input_system_load(in_path)
            if not sys_load:
                continue

            gens = out.get("generators", {}) or {}
            commits = {}
            for gname, gsol in gens.items():
                commits[gname] = list(gsol.get("commit", []))
            feats = _zscore([float(x) for x in sys_load])
            index[instance_name] = {"features": feats, "commit": commits}

        self._trained_index = index
        self._available = len(self._trained_index) > 0
        self._trained = self._available and (self._coverage >= self.coverage_threshold)

        self._compute_splits_from_index()

    def pretrain(
        self, case_folder: Optional[str] = None, force: bool = False
    ) -> Optional[Path]:
        cf = case_folder or self.case_folder
        if not cf:
            return None
        idx_path = self._index_path(cf)
        if idx_path.exists() and not force:
            if self._load_index_file(cf):
                return idx_path
        self._build_index(cf)
        if not self._trained_index:
            return None
        return self._save_index_file(cf)

    def ensure_trained(
        self, case_folder: Optional[str] = None, allow_build_if_missing: bool = True
    ) -> Tuple[bool, float]:
        cf = case_folder or self.case_folder
        if not cf:
            return False, 0.0
        if self._load_index_file(cf):
            return (self._available or self._trained), self._coverage
        if allow_build_if_missing:
            self._build_index(cf)
            return (self._available or self._trained), self._coverage
        return False, 0.0

    def _nearest_neighbor(
        self,
        target_feats: List[float],
        restrict_to_names: Optional[Set[str]] = None,
        exclude_names: Optional[Set[str]] = None,
    ) -> Optional[Tuple[str, dict, float]]:
        best_name = None
        best_item = None
        best_dist = float("inf")
        for name, item in self._trained_index.items():
            if restrict_to_names is not None and name not in restrict_to_names:
                continue
            if exclude_names is not None and name in exclude_names:
                continue
            d = _l2(target_feats, item["features"])
            if d < best_dist:
                best_dist = d
                best_item = item
                best_name = name
        if best_item is None:
            return None
        return best_name, best_item, best_dist

    def generate_and_save_warm_start(
        self,
        instance_name: str,
        use_train_index_only: bool = False,
        exclude_self: bool = True,
        auto_fix: bool = True,
    ) -> Optional[Path]:
        case_folder = _case_folder_from_instance(instance_name)

        if not self._trained_index:
            self.ensure_trained(case_folder, allow_build_if_missing=False)
        if not self._trained_index:
            return None

        input_path = (DataParams._CACHE / (instance_name + ".json.gz")).resolve()
        target_load = _load_input_system_load(input_path)
        if not target_load:
            return None
        target_feats = _zscore([float(x) for x in target_load])

        restrict = self._splits["train"] if use_train_index_only else None
        exclude = {instance_name} if exclude_self else None

        nn = self._nearest_neighbor(
            target_feats, restrict_to_names=restrict, exclude_names=exclude
        )
        if nn is None:
            return None
        nn_name, nn_item, nn_dist = nn

        neighbor_json_path = (self.base_output / (nn_name + ".json")).resolve()
        neighbor = _load_output_solution(neighbor_json_path)
        if neighbor is None:
            return None

        gens = neighbor.get("generators", {}) or {}
        reserves = neighbor.get("reserves", {}) or {}
        network = neighbor.get("network", {}) or {}

        payload = {
            "instance_name": instance_name,
            "case_folder": case_folder,
            "neighbor": nn_name,
            "distance": float(nn_dist),
            "coverage": float(self._coverage),
            "generators": gens,
            "reserves": reserves,
            "network": network,
        }

        fname = f"warm_{_sanitize_name(instance_name)}.json"
        out_path = (self.base_warm / fname).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Automatically produce a fixed warm JSON (no Gurobi), if requested
        if auto_fix:
            try:
                from src.ml_models.fix_warm_start import fix_warm_file

                fix_warm_file(instance_name, warm_file=out_path)
            except Exception as e:
                # non-fatal: fall back to raw warm if fixer fails
                print(f"[warm_start] Auto-fix failed for {instance_name}: {e}")

        return out_path

    @staticmethod
    def _startup_shutdown_from_commit(
        commit: List[int], initial_u: int
    ) -> Tuple[List[int], List[int]]:
        T = len(commit)
        v = [0] * T
        w = [0] * T
        u_prev = initial_u
        for t in range(T):
            u_t = int(round(commit[t]))
            delta = u_t - u_prev
            if delta > 0:
                v[t] = 1
            elif delta < 0:
                w[t] = 1
            u_prev = u_t
        return v, w

    def apply_warm_start_to_model(
        self, model, scenario, instance_name: str, mode: str = "repair"
    ) -> int:
        """
        Set Start on model vars using a warm JSON for 'instance_name'.

        Preference order for input file:
          1) warm_fixed_<instance>.json (if exists)
          2) warm_<instance>.json

        mode:
          - "repair"      -> apply and repair to satisfy easy constraints
          - "commit-only" -> set only u, v, w
          - "as-is"       -> raw application
        """
        # Normalize mode
        mode = (mode or "repair").strip().lower()
        if mode not in ("repair", "commit-only", "as-is"):
            mode = "repair"

        # Prefer fixed warm file if present
        tag = _sanitize_name(instance_name)
        fixed_path = (self.base_warm / f"warm_fixed_{tag}.json").resolve()
        if fixed_path.exists():
            fpath = fixed_path
        else:
            fpath = (self.base_warm / f"warm_{tag}.json").resolve()
            if not fpath.exists():
                alt = (
                    self.base_warm / f"warm_{_sanitize_name(scenario.name)}.json"
                ).resolve()
                if not alt.exists():
                    return 0
                fpath = alt

        try:
            warm = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            return 0

        # Pull model var containers (support SCUC and ED)
        commit_td = getattr(model, "commit", None)
        seg_td = getattr(model, "gen_segment_power", None) or getattr(
            model, "seg_power", None
        )
        startup_td = getattr(model, "startup", None)
        shutdown_td = getattr(model, "shutdown", None)
        reserve_td = getattr(model, "reserve", None)
        shortfall_td = getattr(model, "reserve_shortfall", None)
        line_flow_td = getattr(model, "line_flow", None)
        line_ovp_td = getattr(model, "line_overflow_pos", None)
        line_ovn_td = getattr(model, "line_overflow_neg", None)
        cont_ovp_td = getattr(model, "contingency_overflow_pos", None)
        cont_ovn_td = getattr(model, "contingency_overflow_neg", None)

        applied = 0
        T = scenario.time

        # Helper: binary
        def _b(x):
            return int(1 if float(x) >= 0.5 else 0)

        warm_gen = warm.get("generators", {}) or {}

        # 1) Commitment and startup/shutdown
        gen_u: Dict[str, List[int]] = {}
        for gen in scenario.thermal_units:
            gw = warm_gen.get(gen.name, {}) or {}
            if commit_td is not None:
                u_list = _ensure_list_length(gw.get("commit", []), T, True, 0)
                u_list = [_b(x) for x in u_list]
                for t in range(T):
                    try:
                        commit_td[gen.name, t].Start = u_list[t]
                        applied += 1
                    except Exception:
                        pass
            else:
                u_list = _ensure_list_length(gw.get("commit", []), T, True, 0)
                u_list = [_b(x) for x in u_list]
            gen_u[gen.name] = u_list

            # startup/shutdown from commitment and initial status
            init_u = (
                1 if (gen.initial_status is not None and gen.initial_status > 0) else 0
            )
            v_list, w_list = self._startup_shutdown_from_commit(gen_u[gen.name], init_u)
            if startup_td is not None:
                for t in range(T):
                    try:
                        startup_td[gen.name, t].Start = int(v_list[t])
                        applied += 1
                    except Exception:
                        pass
            if shutdown_td is not None:
                for t in range(T):
                    try:
                        shutdown_td[gen.name, t].Start = int(w_list[t])
                        applied += 1
                    except Exception:
                        pass

        if mode == "commit-only":
            return applied

        # 2) Segment power — basic application (repair mode clips caps)
        gen_above_min: Dict[str, List[float]] = {
            g.name: [0.0] * T for g in scenario.thermal_units
        }
        if seg_td is not None:
            for gen in scenario.thermal_units:
                gw = warm_gen.get(gen.name, {}) or {}
                seg2d = gw.get("segment_power", None)
                nS = len(gen.segments) if gen.segments else 0
                if isinstance(seg2d, list):
                    seg2d = _ensure_list_length(seg2d, T, True, [0.0] * nS)
                else:
                    seg2d = [[0.0] * nS for _ in range(T)]

                for t in range(T):
                    row = seg2d[t] if isinstance(seg2d[t], list) else []
                    row = (row + [0.0] * nS)[:nS]
                    u_t = gen_u[gen.name][t]
                    for s in range(nS):
                        cap = max(0.0, float(gen.segments[s].amount[t])) * float(u_t)
                        val = float(row[s])
                        if mode == "repair":
                            val = max(0.0, min(val, cap))
                        try:
                            seg_td[gen.name, t, s].Start = float(val)
                            applied += 1
                        except Exception:
                            pass
                        gen_above_min[gen.name][t] += float(val)

        # Helper: total production for gen,t
        def _p_total(gen, t: int) -> float:
            u = float(gen_u[gen.name][t])
            p = u * float(gen.min_power[t])
            p += float(gen_above_min[gen.name][t])
            return p

        # 3) Reserves — repair to headroom and shortfall (same as previous version)
        warm_res = warm.get("reserves", {}) or {}
        if reserve_td is not None and scenario.reserves:
            # Prepare warm reserve by product and gen
            pb_map: Dict[str, Dict[str, List[float]]] = {}
            for r in scenario.reserves:
                rw = warm_res.get(r.name, {}) or {}
                pb = rw.get("provided_by_gen", {}) or {}
                inner: Dict[str, List[float]] = {}
                for g in r.thermal_units:
                    gl = _ensure_list_length(pb.get(g.name, []), T, True, 0.0)
                    inner[g.name] = [max(0.0, float(x)) for x in gl]
                pb_map[r.name] = inner

            for gen in scenario.thermal_units:
                for t in range(T):
                    u_t = float(gen_u[gen.name][t])
                    headroom = (float(gen.max_power[t]) - float(gen.min_power[t])) * u_t
                    used = float(gen_above_min[gen.name][t])
                    allow_res = max(0.0, headroom - used)
                    r_sum = 0.0
                    for r in scenario.reserves:
                        r_sum += pb_map.get(r.name, {}).get(gen.name, [0.0] * T)[t]
                    scale = (
                        (allow_res / r_sum)
                        if (r_sum > allow_res + 1e-9 and r_sum > 0)
                        else 1.0
                    )
                    for r in scenario.reserves:
                        val = pb_map.get(r.name, {}).get(gen.name, [0.0] * T)[t] * scale
                        try:
                            reserve_td[r.name, gen.name, t].Start = float(val)
                            applied += 1
                        except Exception:
                            pass

            if shortfall_td is not None:
                for r in scenario.reserves:
                    req = [float(r.amount[t]) for t in range(T)]
                    for t in range(T):
                        provided_t = 0.0
                        for g in r.thermal_units:
                            try:
                                provided_t += float(reserve_td[r.name, g.name, t].Start)
                            except Exception:
                                try:
                                    provided_t += float(reserve_td[r.name, g.name, t].X)
                                except Exception:
                                    pass
                        short = max(0.0, float(req[t]) - provided_t)
                        try:
                            shortfall_td[r.name, t].Start = float(short)
                            applied += 1
                        except Exception:
                            pass

        # 4) Network base-case repair (same as previous version)
        if scenario.lines and line_flow_td is not None:
            try:
                from scipy.sparse import csr_matrix, csc_matrix  # noqa: F401

                isf = scenario.isf.tocsr()
                buses = scenario.buses
                lines = scenario.lines
                ref_1b = getattr(scenario, "ptdf_ref_bus_index", buses[0].index)
                non_ref_bus_indices = sorted(
                    [b.index for b in buses if b.index != ref_1b]
                )
                flows: Dict[Tuple[str, int], float] = {}
                for t in range(T):
                    inj_by_bus: Dict[int, float] = {}
                    for b in buses:
                        inj = 0.0
                        for gen in b.thermal_units:
                            inj += _p_total(gen, t)
                        inj -= float(b.load[t])
                        inj_by_bus[b.index] = float(inj)

                    for line in lines:
                        row = isf.getrow(line.index - 1)
                        f_val = 0.0
                        for col, coeff in zip(row.indices.tolist(), row.data.tolist()):
                            bus_1b = non_ref_bus_indices[col]
                            f_val += float(coeff) * float(inj_by_bus[bus_1b])
                        flows[(line.name, t)] = float(f_val)
                        try:
                            line_flow_td[line.name, t].Start = float(f_val)
                            applied += 1
                        except Exception:
                            pass
                        if line_ovp_td is not None and line_ovn_td is not None:
                            F = float(line.normal_limit[t])
                            ovp = max(0.0, f_val - F)
                            ovn = max(0.0, -f_val - F)
                            try:
                                line_ovp_td[line.name, t].Start = float(ovp)
                                applied += 1
                            except Exception:
                                pass
                            try:
                                line_ovn_td[line.name, t].Start = float(ovn)
                                applied += 1
                            except Exception:
                                pass

                # Contingency slacks (shared) remain as in previous version
                if (
                    cont_ovp_td is not None
                    and cont_ovn_td is not None
                    and scenario.contingencies
                ):
                    lodf_csc = scenario.lodf.tocsc()
                    line_by_row = {ln.index - 1: ln for ln in lines}
                    isf_csc = scenario.isf.tocsc()
                    col_by_bus_1b = {
                        b.index: col for col, b in enumerate(buses) if b.index != ref_1b
                    }
                    req_ovp: Dict[Tuple[str, int], float] = {
                        (ln.name, t): 0.0 for ln in lines for t in range(T)
                    }
                    req_ovn: Dict[Tuple[str, int], float] = {
                        (ln.name, t): 0.0 for ln in lines for t in range(T)
                    }
                    _LODF_TOL = 1e-4
                    for cont in scenario.contingencies:
                        if not cont.lines:
                            continue
                        for out_line in cont.lines:
                            mcol = out_line.index - 1
                            col = lodf_csc.getcol(mcol)
                            for l_row, alpha in zip(
                                col.indices.tolist(), col.data.tolist()
                            ):
                                if l_row == mcol or abs(alpha) < _LODF_TOL:
                                    continue
                                line_l = line_by_row.get(l_row)
                                if line_l is None:
                                    continue
                                for t in range(T):
                                    base_l = float(flows[(line_l.name, t)])
                                    base_m = float(flows[(out_line.name, t)])
                                    post = base_l + float(alpha) * base_m
                                    F_em = float(line_l.emergency_limit[t])
                                    req_ovp[(line_l.name, t)] = max(
                                        req_ovp[(line_l.name, t)], max(0.0, post - F_em)
                                    )
                                    req_ovn[(line_l.name, t)] = max(
                                        req_ovn[(line_l.name, t)],
                                        max(0.0, -post - F_em),
                                    )

                    _ISF_TOL = 1e-8
                    for cont in scenario.contingencies:
                        if not getattr(cont, "units", None):
                            continue
                        for gen in cont.units:
                            bidx = gen.bus.index
                            if bidx == ref_1b or bidx not in col_by_bus_1b:
                                continue
                            col = isf_csc.getcol(col_by_bus_1b[bidx])
                            for t in range(T):
                                pg = _p_total(gen, t)
                                for l_row, coeff in zip(
                                    col.indices.tolist(), col.data.tolist()
                                ):
                                    if abs(coeff) < _ISF_TOL:
                                        continue
                                    line_l = line_by_row.get(l_row)
                                    if line_l is None:
                                        continue
                                    base_l = float(flows[(line_l.name, t)])
                                    post = base_l - float(coeff) * float(pg)
                                    F_em = float(line_l.emergency_limit[t])
                                    req_ovp[(line_l.name, t)] = max(
                                        req_ovp[(line_l.name, t)], max(0.0, post - F_em)
                                    )
                                    req_ovn[(line_l.name, t)] = max(
                                        req_ovn[(line_l.name, t)],
                                        max(0.0, -post - F_em),
                                    )

                    for ln in lines:
                        for t in range(T):
                            try:
                                cont_ovp_td[ln.name, t].Start = float(
                                    req_ovp[(ln.name, t)]
                                )
                                applied += 1
                            except Exception:
                                pass
                            try:
                                cont_ovn_td[ln.name, t].Start = float(
                                    req_ovn[(ln.name, t)]
                                )
                                applied += 1
                            except Exception:
                                pass
            except Exception:
                pass

        return applied

    def get_train_instances(self) -> List[str]:
        return sorted(self._splits.get("train", set()))

    def get_val_instances(self) -> List[str]:
        return sorted(self._splits.get("val", set()))

    def get_test_instances(self) -> List[str]:
        return sorted(self._splits.get("test", set()))

    def get_nontrain_instances(self) -> List[str]:
        return sorted(
            (self._splits.get("val", set()) | self._splits.get("test", set()))
        )

    def get_all_inputs(self) -> List[str]:
        return list(self._inputs_list)

    def get_all_outputs(self) -> List[str]:
        return list(self._outputs_list)

    def report_splits(self) -> None:
        tr = self.get_train_instances()
        va = self.get_val_instances()
        te = self.get_test_instances()
        print("Warm-start index split report")
        print(f"- Train ({len(tr)}):")
        for nm in tr:
            print(f"  {nm}")
        print(f"- Verification/Val ({len(va)}):")
        for nm in va:
            print(f"  {nm}")
        print(f"- Test ({len(te)}):")
        for nm in te:
            print(f"  {nm}")

    def generate_warm_for_split(
        self,
        split: str = "test",
        use_train_db: bool = True,
        limit: Optional[int] = None,
    ) -> List[Tuple[str, Optional[Path]]]:
        split = (split or "test").strip().lower()
        if split in ("val", "verification", "valid", "validation"):
            names = self.get_val_instances()
        elif split in ("train",):
            names = self.get_train_instances()
        else:
            names = self.get_test_instances()
        out: List[Tuple[str, Optional[Path]]] = []
        count = 0
        for nm in names:
            if limit is not None and count >= limit:
                break
            count += 1
            p = self.generate_and_save_warm_start(
                nm, use_train_index_only=use_train_db, exclude_self=True, auto_fix=True
            )
            out.append((nm, p))
        return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Warm-start index inspector with train/val/test split."
    )
    ap.add_argument(
        "--case",
        required=True,
        help="Case folder, e.g., 'matpower/case14'",
    )
    ap.add_argument(
        "--pretrain",
        action="store_true",
        help="Build and persist the index (if missing)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild of index (overwrite existing)",
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help="Print the train/val/test split membership",
    )
    ap.add_argument(
        "--generate-for",
        choices=["train", "val", "test"],
        default=None,
        help="Generate warm-start files for the chosen split (auto-fix on)",
    )
    ap.add_argument(
        "--use-train-db",
        action="store_true",
        default=False,
        help="Restrict neighbor DB to training items when generating warm starts",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="Limit number of items for generation"
    )
    ap.add_argument(
        "--train-ratio", type=float, default=0.70, help="Train ratio (default 0.70)"
    )
    ap.add_argument(
        "--val-ratio", type=float, default=0.15, help="Val ratio (default 0.15)"
    )
    ap.add_argument("--seed", type=int, default=42, help="Split seed (default 42)")
    args = ap.parse_args()

    wsp = WarmStartProvider(
        case_folder=args.case,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_seed=args.seed,
    )
    if args.pretrain or args.force:
        p = wsp.pretrain(force=args.force)
        if p:
            print(f"Index written to: {p}")
        else:
            print("No data to build index.")
    else:
        wsp.ensure_trained(args.case, allow_build_if_missing=True)

    if args.report:
        wsp.report_splits()

    if args.generate_for:
        lim = args.limit if args.limit and args.limit > 0 else None
        results = wsp.generate_warm_for_split(
            split=args.generate_for, use_train_db=args.use_train_db, limit=lim
        )
        ok = sum(1 for _, p in results if p is not None)
        print(
            f"Warm-start files generated for split '{args.generate_for}': {ok}/{len(results)}"
        )
