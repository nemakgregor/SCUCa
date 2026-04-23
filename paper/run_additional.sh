#!/usr/bin/env bash
# One-click launcher for the three additional experiments described in
# RUN_ADDITIONAL_EXPERIMENTS.md. Edit RUN_* flags to enable / disable steps.
#
#   - Step 1: GNN precision / recall / F1      (≤ 10 min, no solver)
#   - Step 2: fixed-K ablation for BANDIT      (~2.5 h on case118+case300)
#   - Step 3: Gurobi grbtune baseline          (~2 h on case118+case300)
#
# Run from the repo root:
#     bash paper/run_additional.sh

set -euo pipefail

RUN_GNN_PR=${RUN_GNN_PR:-1}
RUN_FIXED_K=${RUN_FIXED_K:-0}
RUN_GRBTUNE=${RUN_GRBTUNE:-1}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "===== SCUCa revision: additional experiments ====="
echo "repo root: $REPO_ROOT"
echo "steps: GNN_PR=$RUN_GNN_PR  FIXED_K=$RUN_FIXED_K  GRBTUNE=$RUN_GRBTUNE"
echo

if [[ "$RUN_GNN_PR" == "1" ]]; then
    echo "[1/3] GNN precision / recall / F1"
    python paper/scripts/eval_gnn_precision_recall.py \
        --artifact-root results/exp_20260417_upto1300_full/artifacts/gnn_screening \
        --p-thr 0.60 --y-thr 0.70
    echo
fi

if [[ "$RUN_FIXED_K" == "1" ]]; then
    echo "[2/3] Fixed-K ablation (K=64, K=128, K=256) on case118 and case300"
    for K in 64 128 256; do
        python -m src.paper.experiments \
            --cases matpower/case118 matpower/case300 \
            --modes WARM+LAZY \
            --lazy-top-k "$K" \
            --time-limit 600 \
            --skip-solved \
            --train-use-existing-only
    done
    python paper/scripts/postprocess.py
    echo
fi

if [[ "$RUN_GRBTUNE" == "1" ]]; then
    echo "[3/3] Gurobi auto-tuner baseline (case118, case300)"
    python paper/scripts/run_grbtune.py --case matpower/case118 --tune-time 1800 --solve-time 600
    python paper/scripts/run_grbtune.py --case matpower/case300 --tune-time 3600 --solve-time 600
    echo
fi

echo "===== done. Update paper.tex per RUN_ADDITIONAL_EXPERIMENTS.md. ====="
