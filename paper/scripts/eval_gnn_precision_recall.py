"""Evaluate the GNN line-criticality classifier on held-out paper instances.

Primary use:
1. Reuse archived GNN artifacts when they are compatible with the current
   feature schema.
2. If the archived weights are incompatible (e.g. 11-dim historical features
   vs 7-dim current extractor), optionally re-train a compatible GraphSAGE
   model directly from the archived paper TRAIN solutions and then evaluate it
   on the archived paper TEST solutions.

The script performs no solver runs. It writes a compact precision / recall / F1
summary to `paper/tables/gnn_pr_f1.csv` and a LaTeX stub
`paper/tables/gnn_pr_f1.tex`.

Usage (no arguments required — discovers the latest artifact folder):

    python paper/scripts/eval_gnn_precision_recall.py

Optional:

    python paper/scripts/eval_gnn_precision_recall.py \
        --artifact-root results/paper_upto1354_fullmodes_2026-04-17/artifacts/gnn_screening \
        --train-solution-root results/paper_upto1354_fullmodes_2026-04-17/solutions/train/lazy_all \
        --retrain-if-needed \
        --y-thr 0.70 --p-thr 0.60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

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
    _build_line_graph,
    _node_features,
    _labels_from_output,
    _read_json_auto,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT
    / "results"
    / "paper_upto1354_fullmodes_2026-04-17"
    / "artifacts"
    / "gnn_screening"
)
DEFAULT_SOLUTION_ROOT = (
    REPO_ROOT
    / "results"
    / "paper_upto1354_fullmodes_2026-04-17"
    / "solutions"
    / "test"
    / "raw"
)
DEFAULT_TRAIN_SOLUTION_ROOT = (
    REPO_ROOT
    / "results"
    / "paper_upto1354_fullmodes_2026-04-17"
    / "solutions"
    / "train"
    / "lazy_all"
)
DEFAULT_RETRAIN_ARTIFACT_ROOT = PAPER_DIR / "data" / "gnn_retrained_artifacts"


def _iter_case_outputs(
    solution_root: Path, case_folder: str, max_items: int = 0
) -> list[tuple[str, Path]]:
    """Enumerate archived RAW test solutions for one case.

    Returns pairs `(instance_name, json_path)`, where `instance_name` is in the
    loader format `matpower/case118/2017-01-15`.
    """
    case_dir = (solution_root / case_folder).resolve()
    if not case_dir.exists():
        return []
    rows = []
    for p in sorted(case_dir.glob("*.json")):
        rel = p.resolve().relative_to(solution_root.resolve()).as_posix()
        if rel.endswith(".json"):
            rel = rel[: -len(".json")]
        rows.append((rel, p))
        if max_items and len(rows) >= max_items:
            break
    return rows


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


def _load_model_from_artifact(case_folder: str, case_dir: Optional[Path]):
    """Load a saved GraphSAGE model from the explicitly provided artifact dir.

    The training helper stores weights under
    `<artifact_root>/<matpower_case...>/line_sage.pt`, while the runtime class
    `GNNLineScreener` defaults to `src/data/intermediate/...`. For the paper
    post-processing workflow we want to consume the already archived artifact
    directly, without requiring the user to copy weights into the intermediate
    cache.
    """
    if case_dir is None:
        return None
    model_path = case_dir / "line_sage.pt"
    if not model_path.exists():
        return None

    meta_path = case_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    in_dim = int(meta.get("in_dim", 7))
    model = GNNLineScreener(case_folder).model
    if model is None:
        from src.ml_models.gnn_screening import _LineSAGE

        model = _LineSAGE(in_dim=in_dim, hidden=64)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model.eval()


def _artifact_in_dim(case_dir: Optional[Path]) -> Optional[int]:
    if case_dir is None:
        return None
    meta_path = case_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return int(meta.get("in_dim"))
    except Exception:
        return None


def _feature_dim_for_case(
    case_folder: str, solution_root: Path, max_items: int = 1
) -> Optional[int]:
    for name, _ in _iter_case_outputs(solution_root, case_folder, max_items=max_items):
        try:
            inst = read_benchmark(name, quiet=True)
            sc = inst.deterministic
            X, _ = _node_features(sc)
            return int(X.shape[1])
        except Exception:
            continue
    return None


def _retrain_model(
    case_folder: str,
    train_solution_root: Path,
    retrain_artifact_root: Path,
    epochs: int,
    force: bool,
    max_train_instances: int,
) -> Optional[Path]:
    retrain_artifact_root.mkdir(parents=True, exist_ok=True)
    screener = GNNLineScreener(
        case_folder=case_folder,
        output_root=train_solution_root,
        artifact_root=retrain_artifact_root,
    )
    return screener.pretrain(
        epochs=epochs,
        force=force,
        max_items=(max_train_instances if max_train_instances > 0 else None),
    )


def _discover_case_folders(
    case_folders: Optional[Iterable[str]], solution_root: Path, artifact_root: Path
) -> list[str]:
    if case_folders:
        return [str(x).strip() for x in case_folders if str(x).strip()]

    found = set()
    if artifact_root.exists():
        for case_dir in sorted(artifact_root.iterdir()):
            if case_dir.is_dir():
                found.add(case_dir.name.replace("matpower_", "matpower/", 1))
    if solution_root.exists():
        for base in sorted((solution_root / "matpower").glob("case*")):
            if base.is_dir():
                found.add(f"matpower/{base.name}")
    return sorted(found)


def _fmt_metric(value) -> str:
    if value is None:
        return "--"
    try:
        fv = float(value)
    except Exception:
        return "--"
    if np.isnan(fv):
        return "--"
    return f"{fv:.2f}"


def evaluate_case(
    case_folder: str,
    p_thr: float = 0.60,
    y_thr: float = 0.70,
    case_dir: Optional[Path] = None,
    solution_root: Path = DEFAULT_SOLUTION_ROOT,
    max_test_instances: int = 0,
) -> dict | None:
    screener = GNNLineScreener(case_folder)
    if case_dir is not None:
        try:
            screener.model = _load_model_from_artifact(case_folder, case_dir)
        except Exception:
            screener.model = None
    if screener.model is None and not screener.ensure_trained():
        print(f"[eval] No trained model for {case_folder}; skipping.")
        return None

    archived_outputs = _iter_case_outputs(
        solution_root, case_folder, max_items=max_test_instances
    )
    if not archived_outputs:
        return None

    y_true, y_pred = [], []
    n_test = 0
    for name, out_path in archived_outputs:
        try:
            inst = read_benchmark(name, quiet=True)
            sc = inst.deterministic
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
    ap.add_argument("--solution-root", type=Path, default=DEFAULT_SOLUTION_ROOT)
    ap.add_argument(
        "--train-solution-root",
        type=Path,
        default=DEFAULT_TRAIN_SOLUTION_ROOT,
        help="Archived TRAIN solutions used for compatible re-training if needed.",
    )
    ap.add_argument(
        "--retrain-artifact-root",
        type=Path,
        default=DEFAULT_RETRAIN_ARTIFACT_ROOT,
        help="Where to write re-trained compatible GNN artifacts.",
    )
    ap.add_argument(
        "--case-folders",
        nargs="*",
        default=None,
        help="Optional explicit list such as matpower/case14 matpower/case118.",
    )
    ap.add_argument(
        "--retrain-if-needed",
        action="store_true",
        default=False,
        help="Re-train a compatible GNN from archived TRAIN solutions when artifact dim mismatches current features.",
    )
    ap.add_argument(
        "--retrain-force",
        action="store_true",
        default=False,
        help="Force re-training even if a compatible re-trained artifact already exists.",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Epoch count for compatible re-training.",
    )
    ap.add_argument(
        "--max-train-instances",
        type=int,
        default=0,
        help="Optional cap for TRAIN instances per case (useful for smoke tests).",
    )
    ap.add_argument(
        "--max-test-instances",
        type=int,
        default=0,
        help="Optional cap for TEST instances per case (useful for smoke tests).",
    )
    ap.add_argument("--p-thr", type=float, default=0.60)
    ap.add_argument("--y-thr", type=float, default=0.70)
    args = ap.parse_args()

    if not args.solution_root.exists():
        print(f"No test solutions under {args.solution_root}")
        return 1

    case_folders = _discover_case_folders(
        args.case_folders, args.solution_root, args.artifact_root
    )
    if not case_folders:
        print("No cases discovered for GNN evaluation.")
        return 1

    rows = []
    for case_folder in case_folders:
        print(f"[eval] {case_folder}")
        case_tag = case_folder.replace("/", "_")
        case_dir = args.artifact_root / case_tag
        retrain_case_dir = args.retrain_artifact_root / case_tag
        feature_dim = _feature_dim_for_case(
            case_folder, args.solution_root, max_items=max(1, args.max_test_instances)
        )
        archived_dim = _artifact_in_dim(case_dir)
        use_case_dir = case_dir if (case_dir / "line_sage.pt").exists() else None

        need_retrain = False
        if feature_dim is None:
            print(f"[eval] {case_folder}: could not infer current feature dimension; skipping.")
            continue
        if use_case_dir is None:
            need_retrain = True
            print(f"[eval] {case_folder}: no archived artifact, need retrain.")
        elif archived_dim != feature_dim:
            need_retrain = True
            print(
                f"[eval] {case_folder}: archived in_dim={archived_dim} incompatible with current feature_dim={feature_dim}."
            )

        if need_retrain:
            if not args.retrain_if_needed:
                print(f"[eval] {case_folder}: retraining disabled; skipping.")
                continue
            model_path = _retrain_model(
                case_folder=case_folder,
                train_solution_root=args.train_solution_root,
                retrain_artifact_root=args.retrain_artifact_root,
                epochs=args.epochs,
                force=(args.retrain_force or not (retrain_case_dir / "line_sage.pt").exists()),
                max_train_instances=args.max_train_instances,
            )
            if model_path is None:
                print(f"[eval] {case_folder}: retraining failed.")
                continue
            use_case_dir = retrain_case_dir

        r = evaluate_case(
            case_folder,
            p_thr=args.p_thr,
            y_thr=args.y_thr,
            case_dir=use_case_dir,
            solution_root=args.solution_root,
            max_test_instances=args.max_test_instances,
        )
        if r is not None:
            r["model_source"] = (
                "retrained"
                if use_case_dir is not None
                and use_case_dir.resolve().is_relative_to(args.retrain_artifact_root.resolve())
                else "archived"
            )
            r["feature_dim"] = feature_dim
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
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        "\\textbf{Case} & \\textbf{n test} & \\textbf{Pos.\\ rate} & "
        "\\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{Source} \\\\",
        "\\hline",
    ]
    for _, r in df.iterrows():
        case_short = r["case_folder"].replace("matpower/", "")
        lines.append(
            f"{case_short} & {int(r['n_test_instances'])} & "
            f"{_fmt_metric(r['positive_rate'])} & {_fmt_metric(r['precision'])} & "
            f"{_fmt_metric(r['recall'])} & {_fmt_metric(r['f1'])} & {r['model_source']} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    (PAPER_DIR / "tables" / "gnn_pr_f1.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {PAPER_DIR / 'tables' / 'gnn_pr_f1.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
