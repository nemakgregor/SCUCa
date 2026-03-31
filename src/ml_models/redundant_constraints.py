"""
RedundancyProvider: k-NN constraint-pruning for contingencies.

This version is memory-safe:
- The persisted index stores ONLY per-instance features (z-scored system load).
- The large per-constraint margins are NOT stored anymore.
- At inference time (make_filter_for_instance), we compute conservative
  pruning metrics ON THE FLY from the nearest-neighbor's saved JSON solution.

How pruning works now
- For each monitored line l and outaged line m (line-outage), we compute:
    s_rel_min(l|m) = min_t (F_em(l,t) - |f_l(t) + α f_m(t)|) / F_em(l,t)
  using the neighbor's base flows and the target scenario's LODF and F_em.
- For each monitored line l and outaged generator g (gen-outage), we compute:
    s_rel_min(l|g) = min_t (F_em(l,t) - |f_l(t) - β p_g(t)|) / F_em(l,t)
  using ISF β and neighbor's p_g(t).
- If s_rel_min >= thr_rel we PRUNE (skip) that pair (for all t); otherwise we keep.
- This is slightly more conservative (we use min across time) but drastically
  reduces memory usage and avoids OOM even on large cases.

Other improvements
- Index file uses json.dump (streaming) without building a huge string.
- Optional restriction to TRAIN split when building the index (smaller file).
- Optional cap on number of outputs used to build index (max_items).

Artifacts
- Index is saved to: src/data/intermediate/redundancy/rc_index_<case_tag>.json
"""

import json
import gzip
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Callable

from scipy.sparse import csc_matrix

from src.data_preparation.params import DataParams
from src.data_preparation.read_data import read_benchmark
from src.ml_models.warm_start import _hash01  # deterministic split helper


def _sanitize_name(s: str) -> str:
    s = (s or "").strip().strip("/\\").replace("\\", "/")
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
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


def _ensure_list_length(vals: List, T: int, pad_with_last: bool = True, default=0.0):
    if vals is None:
        return [default] * T
    vals = list(vals)
    if len(vals) < T:
        pad_val = vals[-1] if (pad_with_last and vals) else default
        vals = vals + [pad_val] * (T - len(vals))
    elif len(vals) > T:
        vals = vals[:T]
    return vals


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


class RedundancyProvider:
    """
    k-NN based predictor to skip (prune) contingency constraints safely.

    Memory-light index:
      - features: z-scored system load vector ONLY
      - no per-constraint arrays are stored

    At inference time:
      - We load the nearest neighbor's JSON and compute per-pair min relative margin
        s_rel_min on the fly, then build a pruning predicate.
    """

    # Numerical tolerances (keep in sync with constraints builder)
    _LODF_TOL = 1e-4
    _ISF_TOL = 1e-8

    def __init__(
        self,
        case_folder: Optional[str] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        split_seed: int = 42,
    ):
        self.base_input = DataParams._CACHE
        self.base_output = DataParams._OUTPUT
        self.base_idx = DataParams._INTERMEDIATE / "redundancy"
        self.base_idx.mkdir(parents=True, exist_ok=True)

        self.case_folder = case_folder
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
        self._index: Dict[str, dict] = {}
        self._coverage: float = 0.0
        self._available: bool = False
        self._inputs_list: List[str] = []
        self._outputs_list: List[str] = []
        self._splits: Dict[str, Set[str]] = {
            "train": set(),
            "val": set(),
            "test": set(),
        }

    def make_masks_for_instance(
        self,
        scenario,
        instance_name: str,
        *,
        thr_rel: float = 0.10,
        use_train_index_only: bool = True,
        exclude_self: bool = True,
    ) -> Optional[Tuple[Tuple[set, set], Dict]]:
        """
        Build pair-level keep masks for contingencies:
          - keep_line_pairs = set{ (line_l.name, out_line.name) } to keep (line outages)
          - keep_gen_pairs  = set{ (line_l.name, gen.name) }     to keep (gen outages)

        A pair is KEPT if min_rel_margin < thr_rel (i.e., likely to bind).
        Returns ((keep_line_pairs, keep_gen_pairs), stats) or None if no neighbor/index.
        """
        case_folder = _case_folder_from_instance(instance_name)
        if not self._index:
            self.ensure_trained(case_folder, allow_build_if_missing=False)
        if not self._index:
            return None

        input_path = (DataParams._CACHE / (instance_name + ".json.gz")).resolve()
        target_load = _load_input_system_load(input_path)
        if not target_load:
            return None
        T = scenario.time
        target_feats = _zscore(_ensure_list_length([float(x) for x in target_load], T, True, 0.0))

        restrict = self._splits["train"] if use_train_index_only else None
        exclude = {instance_name} if exclude_self else None
        nn = self._nearest_neighbor(target_feats, restrict_to_names=restrict, exclude_names=exclude)
        if nn is None:
            return None
        nn_name, _nn_item, nn_dist = nn

        neighbor_json_path = (self.base_output / (nn_name + ".json")).resolve()
        neighbor = _load_output_solution(neighbor_json_path)
        if neighbor is None:
            return None

        line_pairs_minrel, gen_pairs_minrel = self._compute_minrel_from_neighbor(scenario, neighbor)

        thr = float(thr_rel)
        keep_line_pairs = set()
        keep_gen_pairs = set()

        for key, srel in line_pairs_minrel.items():
            # key = (l_name, m_name)
            if float(srel) < thr:
                keep_line_pairs.add((str(key[0]), str(key[1])))

        for key, srel in gen_pairs_minrel.items():
            # key = (l_name, g_name)
            if float(srel) < thr:
                keep_gen_pairs.add((str(key[0]), str(key[1])))

        stats = {
            "neighbor": nn_name,
            "distance": nn_dist,
            "kept_line_pairs": len(keep_line_pairs),
            "kept_gen_pairs": len(keep_gen_pairs),
            "total_line_pairs_scanned": len(line_pairs_minrel),
            "total_gen_pairs_scanned": len(gen_pairs_minrel),
        }
        return (keep_line_pairs, keep_gen_pairs), stats
    
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
        return (self.base_idx / f"rc_index_{tag}.json").resolve()

    def _compute_splits_from_names(self, names: List[str]) -> None:
        tr: Set[str] = set()
        va: Set[str] = set()
        te: Set[str] = set()
        for nm in sorted(names):
            r = _hash01(nm, self.split_seed)
            if r < self.train_ratio:
                tr.add(nm)
            elif r < self.train_ratio + self.val_ratio:
                va.add(nm)
            else:
                te.add(nm)
        self._splits = {"train": tr, "val": va, "test": te}

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
                if not name or feats is None:
                    continue
                idx[name] = {
                    "features": feats,
                }
            self._index = idx
            self._coverage = float(obj.get("coverage", 0.0))
            self._available = len(self._index) > 0

            self._inputs_list = list(obj.get("inputs", []))
            self._outputs_list = list(obj.get("outputs", []))
            split = obj.get("split", None)
            if isinstance(split, dict):
                self._splits = {
                    "train": set(split.get("train", [])),
                    "val": set(split.get("val", [])),
                    "test": set(split.get("test", [])),
                }
            else:
                self._compute_splits_from_names(list(self._index.keys()))
            return True
        except Exception:
            return False

    def _save_index_file(self, case_folder: str) -> Path:
        path = self._index_path(case_folder)
        path.parent.mkdir(parents=True, exist_ok=True)
        items = []
        for name, it in self._index.items():
            items.append(
                {
                    "instance": name,
                    "features": list(it.get("features", [])),
                }
            )
        payload = {
            "case_folder": case_folder,
            "built_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "coverage": float(self._coverage),
            "inputs": list(self._inputs_list),
            "outputs": list(self._outputs_list),
            "split": {
                "train": sorted(self._splits["train"]),
                "val": sorted(self._splits["val"]),
                "test": sorted(self._splits["test"]),
            },
            "items": items,
        }
        # Stream to disk without building a giant string in RAM
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        return path

    def _extract_features_from_output(self, out_json: dict) -> Optional[List[float]]:
        try:
            sys_load = out_json.get("system", {}).get("load", None)
            if not sys_load:
                return None
            feats = _zscore([float(x) for x in sys_load])
            return feats
        except Exception:
            return None

    def _build_index(
        self,
        case_folder: str,
        max_items: Optional[int] = None,
        restrict_to_train: bool = True,
    ) -> None:
        inputs = self._list_inputs(case_folder)
        outputs = self._list_outputs(case_folder)
        total_inputs = len(inputs)
        total_outputs = len(outputs)
        self._coverage = (total_outputs / total_inputs) if total_inputs > 0 else 0.0

        self._inputs_list = [self._dataset_name_from_input(p) for p in inputs]
        outputs_names = [self._dataset_name_from_output(p) for p in outputs]
        self._outputs_list = outputs_names

        # Splits based on output names available
        self._compute_splits_from_names(outputs_names)

        if restrict_to_train:
            outputs = [
                p
                for p in outputs
                if self._dataset_name_from_output(p) in self._splits["train"]
            ]

        if max_items is not None and max_items > 0:
            outputs = outputs[:max_items]

        print(
            f"[redundancy] Building light index for '{case_folder}': using {len(outputs)}/{total_outputs} outputs "
            f"(coverage={self._coverage:.3f})."
        )

        t0 = time.time()
        idx: Dict[str, dict] = {}
        for i, out_path in enumerate(outputs, start=1):
            out = _load_output_solution(out_path)
            if not out:
                continue
            instance_name = self._dataset_name_from_output(out_path)
            feats = self._extract_features_from_output(out)
            # Fallback: derive from input JSON if not present in output
            if not feats:
                in_path = (self.base_input / (instance_name + ".json.gz")).resolve()
                sys_load = _load_input_system_load(in_path)
                if not sys_load:
                    continue
                feats = _zscore([float(x) for x in sys_load])
            idx[instance_name] = {
                "features": feats,
            }
            if i == 1 or (i % 50 == 0) or (i == len(outputs)):
                elapsed = time.time() - t0
                print(
                    f"[redundancy] processed {i}/{len(outputs)} outputs (elapsed {elapsed:.1f}s)"
                )

        self._index = idx
        self._available = len(self._index) > 0
        # Splits already computed on outputs_names
        t1 = time.time()
        print(
            f"[redundancy] Done. Indexed {len(self._index)} item(s) in {t1 - t0:.1f}s."
        )

    def pretrain(
        self,
        case_folder: Optional[str] = None,
        force: bool = False,
        max_items: Optional[int] = None,
        restrict_to_train: bool = True,
    ) -> Optional[Path]:
        cf = case_folder or self.case_folder
        if not cf:
            return None
        path = self._index_path(cf)
        if path.exists() and not force:
            if self._load_index_file(cf):
                return path
        self._build_index(cf, max_items=max_items, restrict_to_train=restrict_to_train)
        if not self._index:
            return None
        return self._save_index_file(cf)

    def ensure_trained(
        self, case_folder: Optional[str] = None, allow_build_if_missing: bool = True
    ) -> Tuple[bool, float]:
        cf = case_folder or self.case_folder
        if not cf:
            return False, 0.0
        if self._load_index_file(cf):
            return (self._available, self._coverage)
        if allow_build_if_missing:
            self._build_index(cf)
            return (self._available, self._coverage)
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
        for name, item in self._index.items():
            if restrict_to_names is not None and name not in restrict_to_names:
                continue
            if exclude_names is not None and name in exclude_names:
                continue
            d = _l2(target_feats, item.get("features", []))
            if d < best_dist:
                best_dist = d
                best_item = item
                best_name = name
        if best_item is None:
            return None
        return best_name, best_item, best_dist

    def _compute_minrel_from_neighbor(
        self,
        scenario,
        neighbor_out: dict,
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
        """
        Compute min relative margin across time for (line|out_line) and (line|gen).
        Returns:
          (line_pairs_min_rel, gen_pairs_min_rel)
          where each dict maps (l_name, m_name) or (l_name, g_name) -> min_rel in [.., 1]
        """
        # Extract flows per line and total power per generator from neighbor's output
        lines = scenario.lines or []
        T = scenario.time
        lodf_csc: csc_matrix = scenario.lodf.tocsc()
        isf_csc: csc_matrix = scenario.isf.tocsc()

        # flows
        net = neighbor_out.get("network", {}) or {}
        lines_out = net.get("lines", {}) or {}
        flows: Dict[str, List[float]] = {}
        for ln in lines:
            obj = lines_out.get(ln.name)
            if not obj:
                # If missing, treat as zeros
                flows[ln.name] = [0.0 for _ in range(T)]
            else:
                flows[ln.name] = _ensure_list_length(
                    [float(x) for x in obj.get("flow", [])], T, True, 0.0
                )

        # pgen
        gens_out = neighbor_out.get("generators", {}) or {}
        pgen: Dict[str, List[float]] = {}
        for g in scenario.thermal_units:
            gobj = gens_out.get(g.name, {}) or {}
            p_list = gobj.get("total_power", None)
            if p_list is not None:
                pgen[g.name] = _ensure_list_length(
                    [float(x) for x in p_list], T, True, 0.0
                )
            else:
                # reconstruct as min + sum(segments)
                commit = _ensure_list_length(gobj.get("commit", []), T, True, 0)
                seg2d = gobj.get("segment_power", None)
                nS = len(g.segments) if g.segments else 0
                if isinstance(seg2d, list):
                    seg2d = _ensure_list_length(seg2d, T, True, [0.0] * nS)
                else:
                    seg2d = [[0.0] * nS for _ in range(T)]
                total = []
                for t in range(T):
                    val = float(commit[t]) * float(g.min_power[t])
                    for s in range(nS):
                        val += float(seg2d[t][s])
                    total.append(val)
                pgen[g.name] = total

        # Mappings
        line_by_row = {ln.index - 1: ln for ln in lines}
        buses = scenario.buses
        ref_1b = getattr(scenario, "ptdf_ref_bus_index", buses[0].index)
        non_ref_bus_indices = sorted([b.index for b in buses if b.index != ref_1b])
        col_by_bus_1b = {bus_1b: col for col, bus_1b in enumerate(non_ref_bus_indices)}

        # Results
        line_pairs_min: Dict[Tuple[str, str], float] = {}
        gen_pairs_min: Dict[Tuple[str, str], float] = {}

        # Line-outage min rel margin
        for cont in scenario.contingencies or []:
            if not cont.lines:
                continue
            for out_line in cont.lines:
                mcol = out_line.index - 1
                col = lodf_csc.getcol(mcol)
                rows = col.indices.tolist()
                vals = col.data.tolist()
                for l_row, alpha in zip(rows, vals):
                    if l_row == mcol or abs(alpha) < self._LODF_TOL:
                        continue
                    line_l = line_by_row.get(l_row)
                    if line_l is None:
                        continue
                    key = (line_l.name, out_line.name)
                    s_rel_min = float("+inf")
                    for t in range(T):
                        f_l = float(flows[line_l.name][t])
                        f_m = float(flows[out_line.name][t])
                        post = f_l + float(alpha) * f_m
                        F_em = float(line_l.emergency_limit[t])
                        if F_em <= 0:
                            continue
                        s_rel = (F_em - abs(post)) / F_em
                        if s_rel < s_rel_min:
                            s_rel_min = s_rel
                    if s_rel_min != float("+inf"):
                        line_pairs_min[key] = float(s_rel_min)

        # Gen-outage min rel margin
        for cont in scenario.contingencies or []:
            if not getattr(cont, "units", None):
                continue
            for gen in cont.units:
                bidx = gen.bus.index
                if bidx == ref_1b or bidx not in col_by_bus_1b:
                    # Effectively coeff=0 => post=f_l; compute min rel per line
                    for line_l in lines:
                        key = (line_l.name, gen.name)
                        s_rel_min = float("+inf")
                        for t in range(T):
                            f_l = float(flows[line_l.name][t])
                            F_em = float(line_l.emergency_limit[t])
                            if F_em <= 0:
                                continue
                            s_rel = (F_em - abs(f_l)) / F_em
                            if s_rel < s_rel_min:
                                s_rel_min = s_rel
                        if s_rel_min != float("+inf"):
                            gen_pairs_min[key] = float(s_rel_min)
                else:
                    col = isf_csc.getcol(col_by_bus_1b[bidx])
                    rows = col.indices.tolist()
                    vals = col.data.tolist()
                    isf_map = {r: v for r, v in zip(rows, vals)}
                    for line_l in lines:
                        beta = float(isf_map.get(line_l.index - 1, 0.0))
                        if abs(beta) < self._ISF_TOL:
                            continue
                        key = (line_l.name, gen.name)
                        s_rel_min = float("+inf")
                        for t in range(T):
                            f_l = float(flows[line_l.name][t])
                            p_g = float(pgen[gen.name][t])
                            post = f_l - beta * p_g
                            F_em = float(line_l.emergency_limit[t])
                            if F_em <= 0:
                                continue
                            s_rel = (F_em - abs(post)) / F_em
                            if s_rel < s_rel_min:
                                s_rel_min = s_rel
                        if s_rel_min != float("+inf"):
                            gen_pairs_min[key] = float(s_rel_min)

        return line_pairs_min, gen_pairs_min

    def make_filter_for_instance(
        self,
        scenario,
        instance_name: str,
        *,
        thr_rel: float = 0.10,
        use_train_index_only: bool = True,
        exclude_self: bool = True,
    ) -> Optional[Tuple[Callable, Dict]]:
        """
        Build a filter predicate closure for contingencies.add_constraints.

        A constraint is pruned (skipped) if neighbor min_rel margin >= thr_rel.
        We use min over time of s_rel(t) per pair to be conservative.

        Returns (predicate, stats) or None if no trained index/neighbor.
        stats = {"neighbor": str, "distance": float, "skipped_line": int, "skipped_gen": int}
        """
        case_folder = _case_folder_from_instance(instance_name)
        if not self._index:
            self.ensure_trained(case_folder, allow_build_if_missing=False)
        if not self._index:
            return None

        # Target features from input JSON system load
        input_path = (DataParams._CACHE / (instance_name + ".json.gz")).resolve()
        target_load = _load_input_system_load(input_path)
        if not target_load:
            return None
        T = scenario.time
        target_feats = _zscore(
            _ensure_list_length([float(x) for x in target_load], T, True, 0.0)
        )

        restrict = self._splits["train"] if use_train_index_only else None
        exclude = {instance_name} if exclude_self else None
        nn = self._nearest_neighbor(
            target_feats, restrict_to_names=restrict, exclude_names=exclude
        )
        if nn is None:
            return None
        nn_name, _nn_item, nn_dist = nn

        # Compute min relative margins on the fly from neighbor JSON
        neighbor_json_path = (self.base_output / (nn_name + ".json")).resolve()
        neighbor = _load_output_solution(neighbor_json_path)
        if neighbor is None:
            return None

        line_pairs_minrel, gen_pairs_minrel = self._compute_minrel_from_neighbor(
            scenario, neighbor
        )

        stats = {
            "neighbor": nn_name,
            "distance": nn_dist,
            "skipped_line": 0,
            "skipped_gen": 0,
        }

        thr = float(thr_rel)

        def _predicate(
            kind: str, line_l, out_obj, t: int, coeff: float, F_em: float
        ) -> bool:
            """
            Return True to include the constraint, False to prune.
            kind: "line" for line-outage; "gen" for gen-outage
            We ignore t here because we used min over time already to decide.
            """
            if kind == "line":
                key = (line_l.name, out_obj.name)
                srel = line_pairs_minrel.get(key, None)
                if srel is None:
                    return True
                if srel >= thr:
                    stats["skipped_line"] += 1
                    return False
                return True
            else:
                key = (line_l.name, out_obj.name)
                srel = gen_pairs_minrel.get(key, None)
                if srel is None:
                    return True
                if srel >= thr:
                    stats["skipped_gen"] += 1
                    return False
                return True

        return _predicate, stats
