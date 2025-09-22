"""
GRU-based dispatch warm start for segment power.

Idea
- Learn per-generator time series of "above-min" energy fraction relative to headroom:
    r[g,t] = max(0, total_power[g,t] - u[g,t]*Pmin[g,t]) / max(1e-6, u[g,t]*(Pmax-Pmin))
- Inputs per time step: [load_frac_t, t/T] + static features repeated across time:
    [ru_frac, rd_frac, headroom_mean_frac]
- Model: 2-layer GRU -> linear -> sigmoid output per time step in [0,1].

Training data
- From JSON solutions in src/data/output/<case_folder>.
- Uses only per-generator above-min from outputs and loads from input.

Applying to solver
- Predict r_hat[g,t] per gen/time, compute above_min_hat = r_hat * headroom_t.
- Distribute above_min_hat across segments proportional to segment capacities.
- Set Start for gen_segment_power[g,t,s].

CLI:
    python -m src.ml_models.gru_warmstart --case matpower/case118 --epochs 60
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

from src.data_preparation.params import DataParams
from src.data_preparation.read_data import read_benchmark


def _sanitize_name(s: str) -> str:
    return "".join(
        ch if ch.isalnum() else "_" for ch in s.strip().strip("/\\").replace("\\", "/")
    ).lower()


def _case_tag(cf: str) -> str:
    return _sanitize_name(cf)


def _load_json(path: Path) -> Optional[dict]:
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


class _GRU(nn.Module):
    def __init__(self, in_dim: int = 5, hidden: int = 32, layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim, hidden_size=hidden, num_layers=layers, batch_first=True
        )
        self.lin = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: [B, T, in_dim]
        h, _ = self.gru(x)
        y = torch.sigmoid(self.lin(h)).squeeze(-1)  # [B,T]
        return y


class GRUDispatchWarmStart:
    def __init__(self, case_folder: str):
        self.case = case_folder.strip().strip("/\\").replace("\\", "/")
        self.tag = _case_tag(self.case)
        self.base_dir = (
            DataParams._INTERMEDIATE / "gru_warmstart" / self.tag
        ).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = (self.base_dir / "gru.pt").resolve()
        self.meta_path = (self.base_dir / "meta.json").resolve()
        self.model: Optional[_GRU] = None

    def _build_training(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        outs = _list_outputs(self.case)
        X_seq = []
        Y_seq = []
        for op in outs:
            out = _load_json(op)
            if not out:
                continue
            name = op.resolve().relative_to(DataParams._OUTPUT.resolve()).as_posix()
            if name.endswith(".json"):
                name = name[:-5]
            try:
                inst = read_benchmark(name, quiet=True)
                sc = inst.deterministic
            except Exception:
                continue
            sys_load = [sum(b.load[t] for b in sc.buses) for t in range(sc.time)]
            sum_pmax = sum(
                float(np.mean([float(x) for x in g.max_power]))
                for g in sc.thermal_units
            )
            if sum_pmax <= 0:
                sum_pmax = 1.0
            gens = out.get("generators", {}) or {}
            for g in sc.thermal_units:
                gsol = gens.get(g.name, {}) or {}
                total = gsol.get("total_power", None)
                u_list = gsol.get("commit", None)
                if total is None or u_list is None:
                    continue
                T = sc.time
                # Build X: [load_frac, t_frac, ru_frac, rd_frac, headroom_mean_frac]
                load_frac = [float(sys_load[t]) / sum_pmax for t in range(T)]
                tfrac = [float(t) / max(1, T - 1) for t in range(T)]
                hr = [
                    max(0.0, float(g.max_power[t]) - float(g.min_power[t]))
                    for t in range(T)
                ]
                hr_mean_frac = float(np.mean(hr)) / sum_pmax
                ru = float(getattr(g, "ramp_up", 0.0)) / sum_pmax
                rd = float(getattr(g, "ramp_down", 0.0)) / sum_pmax
                X = np.column_stack(
                    [
                        np.array(load_frac, dtype=float),
                        np.array(tfrac, dtype=float),
                        np.array([ru] * T, dtype=float),
                        np.array([rd] * T, dtype=float),
                        np.array([hr_mean_frac] * T, dtype=float),
                    ]
                )
                # Build Y: r = above_min / (u*(Pmax-Pmin))
                y = []
                for t in range(T):
                    u = 1.0 if float(u_list[t]) >= 0.5 else 0.0
                    denom = max(
                        1e-6, u * (float(g.max_power[t]) - float(g.min_power[t]))
                    )
                    above = max(0.0, float(total[t]) - u * float(g.min_power[t]))
                    y.append(min(1.0, max(0.0, above / denom)))
                X_seq.append(X)
                Y_seq.append(np.array(y, dtype=float))
        if not X_seq:
            return None
        # Pack as batches: each sample is [T,5], we batch by padding separately if needed.
        # Here all T are equal within a case; stack.
        Xb = np.stack(X_seq, axis=0)
        Yb = np.stack(Y_seq, axis=0)
        return Xb, Yb

    def pretrain(
        self, epochs: int = 60, lr: float = 2e-3, force: bool = False
    ) -> Optional[Path]:
        if self.model_path.exists() and not force:
            return self.model_path
        data = self._build_training()
        if data is None:
            print("[gru_ws] No training data found.")
            return None
        Xb, Yb = data  # [N, T, 5], [N, T]
        model = _GRU(in_dim=Xb.shape[2], hidden=32, layers=2)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        model.train()
        x = torch.tensor(Xb, dtype=torch.float32)
        y = torch.tensor(Yb, dtype=torch.float32)
        for e in range(epochs):
            opt.zero_grad()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
            print(f"[gru_ws] epoch {e:03d} | loss={float(loss.item()):.6f}")
        torch.save(model.state_dict(), self.model_path)
        self.model = model.eval()
        meta = {"case_folder": self.case, "in_dim": int(Xb.shape[2])}
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[gru_ws] Saved model to {self.model_path}")
        return self.model_path

    def ensure_trained(self) -> bool:
        if self.model is not None:
            return True
        if self.model_path.exists():
            meta = (
                json.loads(self.meta_path.read_text(encoding="utf-8"))
                if self.meta_path.exists()
                else {}
            )
            in_dim = int(meta.get("in_dim", 5))
            model = _GRU(in_dim=in_dim, hidden=32, layers=2)
            model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            self.model = model.eval()
            return True
        return False

    def apply_to_model(self, model, sc) -> int:
        """
        Predict above-min fractions and set Start for segment powers proportional to caps.
        Returns number of segment Start assigned.
        """
        ok = self.ensure_trained()
        if not ok or self.model is None:
            return 0
        seg = getattr(model, "gen_segment_power", None) or getattr(
            model, "seg_power", None
        )
        if seg is None:
            return 0
        # Build X per generator
        sum_pmax = sum(
            float(np.mean([float(x) for x in g.max_power])) for g in sc.thermal_units
        )
        if sum_pmax <= 0:
            sum_pmax = 1.0
        sys_load = [sum(b.load[t] for b in sc.buses) for t in range(sc.time)]
        T = sc.time
        # one-by-one inference
        applied = 0
        for g in sc.thermal_units:
            hr = [
                max(0.0, float(g.max_power[t]) - float(g.min_power[t]))
                for t in range(T)
            ]
            hr_mean_frac = float(np.mean(hr)) / sum_pmax
            ru = float(getattr(g, "ramp_up", 0.0)) / sum_pmax
            rd = float(getattr(g, "ramp_down", 0.0)) / sum_pmax
            X = np.column_stack(
                [
                    np.array(
                        [float(sys_load[t]) / sum_pmax for t in range(T)], dtype=float
                    ),
                    np.array([float(t) / max(1, T - 1) for t in range(T)], dtype=float),
                    np.array([ru] * T, dtype=float),
                    np.array([rd] * T, dtype=float),
                    np.array([hr_mean_frac] * T, dtype=float),
                ]
            )
            with torch.no_grad():
                x = torch.tensor(X[None, :, :], dtype=torch.float32)  # [1,T,5]
                r = self.model(x).cpu().numpy()[0]  # [T]
            # Turn r[t] into above-min MWh and distribute to segments by caps
            nS = len(g.segments) if g.segments else 0
            caps = [
                [(float(g.segments[s].amount[t]) if nS > 0 else 0.0) for s in range(nS)]
                for t in range(T)
            ]
            for t in range(T):
                above = float(r[t]) * float(hr[t])
                if nS == 0 or above <= 1e-9:
                    continue
                cap_row = caps[t]
                cap_sum = sum(float(c) for c in cap_row)
                if cap_sum <= 1e-12:
                    continue
                for s in range(nS):
                    val = above * float(cap_row[s]) / cap_sum
                    try:
                        seg[g.name, t, s].Start = float(val)
                        applied += 1
                    except Exception:
                        pass
        return applied


def main():
    ap = argparse.ArgumentParser(
        description="Train/apply GRU warm start for segment power per case."
    )
    ap.add_argument("--case", required=True, help="Case folder, e.g., matpower/case118")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ws = GRUDispatchWarmStart(args.case)
    p = ws.pretrain(epochs=args.epochs, force=args.force)
    if p:
        print(f"Model saved to: {p}")
    else:
        print("Training failed or no data.")


if __name__ == "__main__":
    main()
