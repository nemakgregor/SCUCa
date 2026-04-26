"""One-click runner for the reviewer-facing additional experiments.

Workflow:
1. Preflight: validate environment and required paths.
2. Smoke checks: run tiny end-to-end checks for GNN, fixed-K, and grbtune.
3. Full run: execute the requested steps and aggregate paper-side tables.

Run from repo root, for example:

    PYTHONPATH=. venv/bin/python paper/scripts/run_additional_pipeline.py

Use `--smoke-only` to verify the pipeline before launching the overnight job.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = PAPER_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT
    / "results"
    / "paper_upto1354_fullmodes_2026-04-17"
    / "artifacts"
    / "gnn_screening"
)
DEFAULT_TEST_SOLUTION_ROOT = (
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


class TeeLogger:
    def __init__(self, path: Path):
        self.path = path
        self._fh = path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._fh.close()

    def line(self, text: str = "") -> None:
        print(text, flush=True)
        self._fh.write(text + "\n")
        self._fh.flush()


def _cmd_str(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _run(cmd: list[str], logger: TeeLogger, env: dict[str, str]) -> None:
    logger.line(f"$ {_cmd_str(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        logger.line(line.rstrip("\n"))
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _preflight(args: argparse.Namespace, logger: TeeLogger, env: dict[str, str]) -> None:
    logger.line("[preflight] repo root: " + str(REPO_ROOT))
    logger.line("[preflight] python: " + sys.executable)

    raw_logs_dir = REPO_ROOT / "results" / "raw_logs"
    raw_logs_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        DEFAULT_ARTIFACT_ROOT,
        DEFAULT_TEST_SOLUTION_ROOT,
        DEFAULT_TRAIN_SOLUTION_ROOT,
        raw_logs_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(f"required path is missing: {path}")
        logger.line(f"[preflight] found {path}")

    _run(
        [
            sys.executable,
            "-c",
            "import gurobipy, pandas, numpy; print('ok base libs', gurobipy.gurobi.version())",
        ],
        logger,
        env,
    )

    if not args.skip_gnn:
        _run(
            [
                sys.executable,
                "-c",
                "import torch, torch_geometric; print('ok gnn libs', torch.__version__, torch_geometric.__version__)",
            ],
            logger,
            env,
        )


def _smoke_gnn(logger: TeeLogger, env: dict[str, str]) -> None:
    logger.line("[smoke] GNN retrain-if-needed on case14")
    _run(
        [
            sys.executable,
            "paper/scripts/eval_gnn_precision_recall.py",
            "--case-folders",
            "matpower/case14",
            "--retrain-if-needed",
            "--epochs",
            "1",
            "--max-train-instances",
            "2",
            "--max-test-instances",
            "2",
        ],
        logger,
        env,
    )


def _smoke_fixed_k(logger: TeeLogger, env: dict[str, str]) -> None:
    logger.line("[smoke] fixed-K WARM+LAZY on case14, K=64, one test instance")
    _run(
        [
            sys.executable,
            "-m",
            "src.paper.experiments",
            "--cases",
            "matpower/case14",
            "--modes",
            "WARM+LAZY",
            "--lazy-top-k",
            "64",
            "--lazy-viol-tol",
            "1e-6",
            "--lazy-lodf-tol",
            "1e-4",
            "--lazy-isf-tol",
            "1e-8",
            "--lazy-finalize-exact",
            "--lazy-finalize-rounds",
            "2",
            "--lazy-finalize-time",
            "10",
            "--time-limit",
            "30",
            "--limit-test",
            "1",
            "--train-use-existing-only",
        ],
        logger,
        env,
    )
    _aggregate_lazy_topk(logger)


def _smoke_grbtune(logger: TeeLogger, env: dict[str, str]) -> None:
    logger.line("[smoke] grbtune on case14")
    _run(
        [
            sys.executable,
            "paper/scripts/run_grbtune.py",
            "--case",
            "matpower/case14",
            "--tune-time",
            "5",
            "--solve-time",
            "10",
        ],
        logger,
        env,
    )


def _aggregate_lazy_topk(logger: TeeLogger) -> Path:
    logs = sorted(glob.glob(str(REPO_ROOT / "results" / "raw_logs" / "*.csv")))
    if not logs:
        raise FileNotFoundError("no raw_logs CSV files found")

    df = pd.concat((pd.read_csv(p) for p in logs), ignore_index=True)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
    if {"instance_name", "mode"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["instance_name", "mode"], keep="last")
    sel = df[df["mode"].isin(["WARM+LAZY+K64", "WARM+LAZY+K128", "WARM+LAZY+K256"])]
    if sel.empty:
        raise RuntimeError("no fixed-K rows found in results/raw_logs/*.csv")

    summary = (
        sel.groupby(["case_folder", "mode"])
        .agg(
            n=("runtime_sec", "size"),
            runtime_median_sec=("runtime_sec", "median"),
            runtime_mean_sec=("runtime_sec", "mean"),
            status_ok_rate=(
                "status",
                lambda s: s.astype(str).isin(["OPTIMAL", "SUBOPTIMAL", "TIME_LIMIT"]).mean(),
            ),
            strict_feasible_rate=(
                "violations",
                lambda s: (s.astype(str) == "OK").mean(),
            ),
        )
        .reset_index()
        .sort_values(["case_folder", "mode"])
    )
    out_path = PAPER_DIR / "tables" / "lazy_topk_ablation.csv"
    summary.to_csv(out_path, index=False)
    logger.line(f"[aggregate] wrote {out_path}")
    return out_path


def _run_full_gnn(logger: TeeLogger, env: dict[str, str], epochs: int) -> None:
    logger.line("[full] GNN P/R/F1 with compatible retrain-if-needed")
    _run(
        [
            sys.executable,
            "paper/scripts/eval_gnn_precision_recall.py",
            "--retrain-if-needed",
            "--epochs",
            str(epochs),
        ],
        logger,
        env,
    )


def _run_full_fixed_k(logger: TeeLogger, env: dict[str, str]) -> None:
    for k in ("64", "128", "256"):
        logger.line(f"[full] fixed-K ablation, K={k}")
        _run(
            [
                sys.executable,
                "-m",
                "src.paper.experiments",
                "--cases",
                "matpower/case118",
                "matpower/case300",
                "--modes",
                "WARM+LAZY",
                "--lazy-top-k",
                k,
                "--lazy-viol-tol",
                "1e-6",
                "--lazy-lodf-tol",
                "1e-4",
                "--lazy-isf-tol",
                "1e-8",
                "--lazy-finalize-exact",
                "--lazy-finalize-rounds",
                "3",
                "--lazy-finalize-time",
                "60",
                "--time-limit",
                "600",
                "--skip-solved",
                "--skip-solved-require-ok",
                "--train-use-existing-only",
            ],
            logger,
            env,
        )
    _aggregate_lazy_topk(logger)


def _run_full_grbtune(
    logger: TeeLogger, env: dict[str, str], include_case89pegase: bool
) -> None:
    cases = [
        ("matpower/case118", "1800", "600"),
        ("matpower/case300", "3600", "600"),
    ]
    if include_case89pegase:
        cases.append(("matpower/case89pegase", "600", "600"))
    for case_folder, tune_time, solve_time in cases:
        logger.line(f"[full] grbtune {case_folder}")
        _run(
            [
                sys.executable,
                "paper/scripts/run_grbtune.py",
                "--case",
                case_folder,
                "--tune-time",
                tune_time,
                "--solve-time",
                solve_time,
            ],
            logger,
            env,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true", default=False)
    ap.add_argument("--skip-smoke", action="store_true", default=False)
    ap.add_argument("--skip-gnn", action="store_true", default=False)
    ap.add_argument("--skip-fixed-k", action="store_true", default=False)
    ap.add_argument("--skip-grbtune", action="store_true", default=False)
    ap.add_argument("--include-case89pegase", action="store_true", default=False)
    ap.add_argument(
        "--gnn-epochs",
        type=int,
        default=40,
        help="Epochs for compatible GNN retraining in the full run.",
    )
    args = ap.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"additional_pipeline_{ts}.log"
    logger = TeeLogger(log_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    try:
        logger.line("=== Additional reviewer experiments pipeline ===")
        logger.line(f"log: {log_path}")
        _preflight(args, logger, env)

        if not args.skip_smoke:
            logger.line("=== Smoke checks ===")
            if not args.skip_gnn:
                _smoke_gnn(logger, env)
            if not args.skip_fixed_k:
                _smoke_fixed_k(logger, env)
            if not args.skip_grbtune:
                _smoke_grbtune(logger, env)
            logger.line("=== Smoke checks passed ===")

        if args.smoke_only:
            logger.line("Smoke-only mode: stopping here.")
            return 0

        logger.line("=== Full run ===")
        if not args.skip_gnn:
            _run_full_gnn(logger, env, epochs=args.gnn_epochs)
        if not args.skip_fixed_k:
            _run_full_fixed_k(logger, env)
        if not args.skip_grbtune:
            _run_full_grbtune(logger, env, include_case89pegase=args.include_case89pegase)

        logger.line("[final] running LaTeX sanity check")
        _run([sys.executable, "paper/scripts/sanity_check.py"], logger, env)
        logger.line("=== Pipeline finished successfully ===")
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
