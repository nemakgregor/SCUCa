"""
GNN-based line criticality screening (GraphSAGE on a "line graph").

Goal
- Predict line-level "criticality" scores from topology + simple inputs, learned from
  historical solutions (base-case flows vs limits).
- Use these scores to prune contingency pairs:
    • For line-outage (l | m): include only if (critical(l) or critical(m)).
    • For gen-outage (l | g): include only if critical(l).
- This keeps the screening safe (we retain constraints on predicted-critical lines).

Training dataset (per case)
- Nodes = lines
- Edges connect lines that share a bus (source/target) OR have |LODF| > 0.05 (if available)
- Node features from input JSON/scenario:
    • susceptance (S)
    • mean normal limit (MW)
    • mean emergency limit (MW)
    • degree at source bus, degree at target bus
    • mean load at source bus, mean load at target bus
- Labels from outputs:
    • critical[l] = 1 if max_t |flow[l,t]| / F_em[l,t] >= y_thr_label (default 0.70)

Inference
- Build the same graph features from the input JSON.
- Run GNN to get per-line scores in [0,1], mark critical if score >= thr_pred (default 0.6).
- Build a pruning predicate for contingencies.add_constraints(...).

CLI examples:
    python -m src.ml_models.gnn_screening --case matpower/case118 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from torch_geometric.data import Data as GeoData
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import SAGEConv

    HAVE_PYG = True
except Exception:
    HAVE_PYG = False

import networkx as nx

from src.data_preparation.params import DataParams
from src.data_preparation.read_data import read_benchmark


def _sanitize_name(s: str) -> str:
    s = (s or "").strip().strip("/\\").replace("\\", "/")
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_").lower()


def _case_tag(case_folder: str) -> str:
    return _sanitize_name(case_folder)


def _instance_name_from_output(p: Path) -> str:
    rel = p.resolve().relative_to(DataParams._OUTPUT.resolve()).as_posix()
    if rel.endswith(".json"):
        rel = rel[: -len(".json")]
    return rel


def _read_json_auto(path: Path) -> Optional[dict]:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _list_outputs(case_folder: str) -> List[Path]:
    d = (DataParams._OUTPUT / case_folder).resolve()
    if not d.exists():
        return []
    return sorted(d.glob("*.json"))


def _build_line_graph(sc) -> Tuple[nx.Graph, Dict[str, int]]:
    # Nodes are lines; connect if share a bus
    G = nx.Graph()
    idx_by_line = {}
    for i, ln in enumerate(sc.lines):
        idx_by_line[ln.name] = i
        G.add_node(i)
    # share a bus
    for i, ln in enumerate(sc.lines):
        for j, lm in enumerate(sc.lines):
            if j <= i:
                continue
            if (
                (ln.source.index == lm.source.index)
                or (ln.source.index == lm.target.index)
                or (ln.target.index == lm.source.index)
                or (ln.target.index == lm.target.index)
            ):
                G.add_edge(i, j)
    # Optional: add edges via LODF non-negligible
    try:
        lodf = sc.lodf.tocsc()
        for j, lm in enumerate(sc.lines):
            col = lodf.getcol(lm.index - 1)
            rows = col.indices.tolist()
            vals = col.data.tolist()
            for r, v in zip(rows, vals):
                if r == lm.index - 1:
                    continue
                if abs(v) > 0.05:
                    G.add_edge(r, j)
    except Exception:
        pass
    return G, idx_by_line


def _node_features(sc) -> Tuple[np.ndarray, Dict[str, int]]:
    # Build features per line: [susceptance, mean_Fn, mean_Fe, deg_src, deg_tgt, load_src_mean, load_tgt_mean]
    Gbus = nx.Graph()
    for b in sc.buses:
        Gbus.add_node(b.index)
    for ln in sc.lines:
        Gbus.add_edge(ln.source.index, ln.target.index)
    deg = dict(Gbus.degree())

    load_mean = {b.index: float(np.mean([float(x) for x in b.load])) for b in sc.buses}
    feats = []
    idx_map = {}
    for i, ln in enumerate(sc.lines):
        idx_map[ln.name] = i
        susc = float(ln.susceptance)
        F_n = float(np.mean([float(x) for x in ln.normal_limit]))
        F_e = float(np.mean([float(x) for x in ln.emergency_limit]))
        ds = float(deg.get(ln.source.index, 0))
        dt = float(deg.get(ln.target.index, 0))
        ls = float(load_mean.get(ln.source.index, 0.0))
        lt = float(load_mean.get(ln.target.index, 0.0))
        feats.append([susc, F_n, F_e, ds, dt, ls, lt])
    X = np.array(feats, dtype=float)
    # Normalize crude
    if X.size > 0:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-9
        X = (X - mean) / std
    return X, idx_map


def _labels_from_output(sc, out_json: dict, y_thr_label: float = 0.70) -> np.ndarray:
    # critical if max_t |flow| / emergency_limit >= y_thr_label
    T = sc.time
    lab = np.zeros((len(sc.lines),), dtype=int)
    net = out_json.get("network", {}) or {}
    lines_out = net.get("lines", {}) or {}
    for i, ln in enumerate(sc.lines):
        obj = lines_out.get(ln.name)
        if not obj:
            lab[i] = 0
            continue
        flow = [float(x) for x in obj.get("flow", [0.0] * T)]
        ratio = []
        for t in range(T):
            F_em = float(ln.emergency_limit[t])
            if F_em <= 0:
                continue
            ratio.append(abs(flow[min(t, len(flow) - 1)]) / F_em)
        val = max(ratio) if ratio else 0.0
        lab[i] = 1 if val >= y_thr_label else 0
    return lab


class _LineSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = torch.relu(self.c1(x, edge_index))
        h = torch.relu(self.c2(h, edge_index))
        y = torch.sigmoid(self.out(h)).squeeze(-1)
        return y


class GNNLineScreener:
    def __init__(self, case_folder: str):
        self.case = case_folder.strip().strip("/\\").replace("\\", "/")
        self.tag = _case_tag(self.case)
        self.base_dir = (
            DataParams._INTERMEDIATE / "gnn_screening" / self.tag
        ).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = (self.base_dir / "line_sage.pt").resolve()
        self.meta_path = (self.base_dir / "meta.json").resolve()
        self.model: Optional[_LineSAGE] = None

    def _build_dataset(self) -> Optional[List[GeoData]]:
        if not HAVE_PYG:
            print("[gnn_screen] torch-geometric not available; cannot train.")
            return None
        outs = _list_outputs(self.case)
        if not outs:
            return None
        data_list = []
        for op in outs:
            out = _read_json_auto(op)
            if not out:
                continue
            name = _instance_name_from_output(op)
            try:
                inst = read_benchmark(name, quiet=True)
                sc = inst.deterministic
            except Exception:
                continue
            G, _ = _build_line_graph(sc)
            X, _m = _node_features(sc)
            y = _labels_from_output(sc, out_json=out, y_thr_label=0.70)
            if X.shape[0] == 0 or y.shape[0] != X.shape[0]:
                continue
            # edge_index
            edges = (
                np.array(list(G.edges()), dtype=int).T
                if G.number_of_edges() > 0
                else np.zeros((2, 0), dtype=int)
            )
            if edges.size > 0:
                edge_index = torch.tensor(edges, dtype=torch.long)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            gd = GeoData(
                x=torch.tensor(X, dtype=torch.float32),
                edge_index=edge_index,
                y=torch.tensor(y, dtype=torch.float32),
            )
            data_list.append(gd)
        return data_list

    def pretrain(
        self,
        epochs: int = 40,
        lr: float = 1e-3,
        batch_size: int = 1,
        force: bool = False,
    ) -> Optional[Path]:
        if self.model_path.exists() and not force:
            return self.model_path
        data_list = self._build_dataset()
        if not data_list:
            print("[gnn_screen] No data to train.")
            return None
        in_dim = data_list[0].x.shape[1]
        model = _LineSAGE(in_dim=in_dim, hidden=64)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
        model.train()
        for e in range(epochs):
            tot = 0.0
            nb = 0
            for g in loader:
                opt.zero_grad()
                yhat = model(g.x, g.edge_index)
                loss = loss_fn(yhat, g.y)
                loss.backward()
                opt.step()
                tot += float(loss.item())
                nb += 1
            print(f"[gnn_screen] epoch {e:03d} | loss={tot / max(1, nb):.4f}")
        torch.save(model.state_dict(), self.model_path)
        self.model = model
        meta = {"case_folder": self.case, "in_dim": in_dim}
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[gnn_screen] Saved model to {self.model_path}")
        return self.model_path

    def ensure_trained(self) -> bool:
        if self.model is not None:
            return True
        if self.model_path.exists():
            try:
                meta = (
                    json.loads(self.meta_path.read_text(encoding="utf-8"))
                    if self.meta_path.exists()
                    else {}
                )
                in_dim = int(meta.get("in_dim", 7))
                m = _LineSAGE(in_dim=in_dim, hidden=64)
                m.load_state_dict(torch.load(self.model_path, map_location="cpu"))
                self.model = m.eval()
                return True
            except Exception:
                return False
        return False

    def _predict_scores_for_instance(self, sc) -> Dict[str, float]:
        ok = self.ensure_trained()
        if not ok or self.model is None:
            return {}
        X, idx_map = _node_features(sc)
        if X.shape[0] == 0:
            return {}
        Gline, _ = _build_line_graph(sc)
        edges = (
            np.array(list(Gline.edges()), dtype=int).T
            if Gline.number_of_edges() > 0
            else np.zeros((2, 0), dtype=int)
        )
        edge_index = (
            torch.tensor(edges, dtype=torch.long)
            if edges.size > 0
            else torch.zeros((2, 0), dtype=torch.long)
        )
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            y = self.model(x, edge_index).cpu().numpy().tolist()
        scores = {}
        for ln, i in idx_map.items():
            scores[ln] = float(y[i])
        return scores

    def make_pruning_predicate(self, sc, thr_pred: float = 0.60):
        """
        Return a function(kind, line_l, out_obj, t, coeff, F_em) -> bool
        Include constraint if:
          - line_outage: score(line_l) >= thr OR score(out_line) >= thr
          - gen_outage : score(line_l) >= thr
        """
        scores = self._predict_scores_for_instance(sc)
        if not scores:
            return None

        def pred(kind: str, line_l, out_obj, t: int, coeff: float, F_em: float) -> bool:
            s_l = float(scores.get(line_l.name, 0.0))
            if kind == "line":
                s_m = float(scores.get(out_obj.name, 0.0))
                return (s_l >= thr_pred) or (s_m >= thr_pred)
            else:
                return s_l >= thr_pred

        return pred


def main():
    ap = argparse.ArgumentParser(
        description="Train GNN line screening model per case folder."
    )
    ap.add_argument("--case", required=True, help="Case folder, e.g., matpower/case118")
    ap.add_argument("--epochs", type=int, default=40, help="Epochs")
    ap.add_argument("--force", action="store_true", help="Overwrite model")
    args = ap.parse_args()

    gn = GNNLineScreener(args.case)
    p = gn.pretrain(epochs=args.epochs, force=args.force)
    if p:
        print(f"Model written to: {p}")
    else:
        print("Training failed or no data (or torch-geometric missing).")


if __name__ == "__main__":
    main()
