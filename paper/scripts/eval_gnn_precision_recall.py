"""Evaluate the pre-trained GraphSAGE line-criticality classifier.

Uses the model weights already produced by
`python -m src.ml_models.gnn_screening --case matpower/case<N>` and saved in
`results/exp_.../artifacts/gnn_screening/<case>/line_sage.pt`. Produces a small
precision / recall / F1 table per case, plus an aggregated CSV in
`paper/tables/gnn_pr_f1.csv` (and a LaTeX stub).

This script performs NO solver runs and no retraining. It only reloads the
existing weights and evaluates them on the held-out test split of each case.

Usage (no arguments required — discovers the latest artifact folder):

    python paper/scripts/eval_gnn_precision_recall.py

Optional:

    python paper/scripts/eval_gnn_precision_recall.py \
        --artifact-root results/exp_20260417_upto1300_full/artifacts/gnn_screening \
        --y-thr 0.70 --p-thr 0.60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "torch is not installed in this environment. "
        "Run: pip install torch==2.* (CPU build is enough)."
    ) from e

# Re-use the training-time helpers so features and labels are built identically.
from src.data_preparation.read_data import read_benchmark
from src.ml_models.gnn_screening import (
    GNNLineScreener,
    _list_outputs,
    _build_line_graph,
    _node_features,
    _labels_from_output,
    _instance_name_from_output,
    _read_json_auto,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT
    / "results"
    / "exp_20260417_upto1300_full"
    / "artifacts"
    / "gnn_screening"
)


def _test_instance_names(case_folder: str, seed: int = 0) -> list[str]:
    """Recover the same 15% test split the training script held out.

    GNNLineScreener._build_dataset iterates over _list_outputs() deterministically;
    we mimic its 70/15/15 split with the same seed so the P/R/F1 we report comes
    from *unseen* instances only.
    """
    outs = _list_outputs(case_folder)
    names = [_instance_name_from_output(p) for p in outs]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(names))
    rng.shuffle(idx)
    n = len(idx)
    n_test = max(1, int(round(0.15 * n)))
    test_idx = idx[-n_test:]
    return [names[i] for i in test_idx]


def _scores_for_sc(model: torch.nn.Module, sc) -> tuple[np.ndarray, dict]:
    X, idx_map = _node_features(sc)
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
        xt = torch.tensor(X, dtype=torch.float32)
        y = model(xt, edge_index).cpu().numpy().reshape(-1)
    return y, idx_map


def evaluate_case(
    case_folder: str, p_thr: float = 0.60, y_thr: float = 0.70
) -> dict | None:
    screener = GNNLineScreener(case_folder)
    if not screener.ensure_trained():
        print(f"[eval] No trained model for {case_folder}; skipping.")
        return None

    test_names = _test_instance_names(case_folder)
    if not test_names:
        return None

    y_true, y_pred = [], []
    n_test = 0
    for name in test_names:
        try:
            inst = read_benchmark(name, quiet=True)
            sc = inst.deterministic
            out_path = next(iter(_list_outputs(case_folder)), None)
            # match the specific output for this instance
            for p in _list_outputs(case_folder):
                if _instance_name_from_output(p) == name:
                    out_path = p
                    break
            if out_path is None:
                continue
            out_json = _read_json_auto(out_path)
            if out_json is None:
                continue
            labels = _labels_from_output(sc, out_json, y_thr_label=y_thr)
            if labels.size == 0:
                continue
            scores, idx_map = _scores_for_sc(screener.model, sc)
            if scores.shape[0] != labels.shape[0]:
                continue
            preds = (scores >= p_thr).astype(int)
            y_true.append(labels.astype(int))
            y_pred.append(preds)
            n_test += 1
        except Exception as e:  # noqa: BLE001
            print(f"[eval] {name}: {e}")

    if not y_true:
        return None

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * prec * rec / (prec + rec)
        if (prec == prec and rec == rec and (prec + rec) > 0)
        else float("nan")
    )
    pos_rate = float(y_true.mean()) if y_true.size else float("nan")
    return {
        "case_folder": case_folder,
        "n_test_instances": n_test,
        "n_labels": int(y_true.size),
        "positive_rate": round(pos_rate, 4),
        "precision": round(prec, 4) if prec == prec else None,
        "recall": round(rec, 4) if rec == rec else None,
        "f1": round(f1, 4) if f1 == f1 else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    ap.add_argument("--p-thr", type=float, default=0.60)
    ap.add_argument("--y-thr", type=float, default=0.70)
    args = ap.parse_args()

    if not args.artifact_root.exists():
        print(f"No GNN artifacts under {args.artifact_root}")
        return 1

    rows = []
    for case_dir in sorted(args.artifact_root.iterdir()):
        if not (case_dir / "line_sage.pt").exists():
            continue
        # Case folder format: matpower_case<N> -> matpower/case<N>
        case_folder = case_dir.name.replace("matpower_", "matpower/", 1)
        print(f"[eval] {case_folder}")
        r = evaluate_case(case_folder, p_thr=args.p_thr, y_thr=args.y_thr)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No rows to write.")
        return 1

    df = pd.DataFrame(rows)
    out_csv = PAPER_DIR / "tables" / "gnn_pr_f1.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    # LaTeX stub
    lines = [
        "\\begin{table}[t]\\centering",
        "\\caption{GNN line-criticality classifier: precision, recall and F1 on held-out "
        "test instances. ``Positive rate'' is the fraction of lines labeled critical.}",
        "\\label{tab:gnn_pr_f1}",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "\\textbf{Case} & \\textbf{n test} & \\textbf{Pos.\\ rate} & "
        "\\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} \\\\",
        "\\hline",
    ]
    for _, r in df.iterrows():
        case_short = r["case_folder"].replace("matpower/", "")
        lines.append(
            f"{case_short} & {int(r['n_test_instances'])} & "
            f"{r['positive_rate']:.2f} & {r['precision']:.2f} & "
            f"{r['recall']:.2f} & {r['f1']:.2f} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    (PAPER_DIR / "tables" / "gnn_pr_f1.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {PAPER_DIR / 'tables' / 'gnn_pr_f1.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
