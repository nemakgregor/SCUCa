#!/usr/bin/env bash
# case1354pegase — comparison runs for paper (Section: Large-Case Results).
#
# Modes compared:
#   1. LAZY-All   : WARM+LAZY, add_top_k=0   (baseline N-1)
#   2. LAZY-K64   : WARM+LAZY, add_top_k=64  (top-K lazy)
#   3. LAZY-K128  : WARM+LAZY, add_top_k=128 (top-K lazy)
#   4. COMMIT     : WARM+LAZY K=0 + 1NN commit hints
#   5. BANDIT     : WARM+LAZY, adaptive K in {64,128,256}
#   6. WARM       : WARM only (no N-1), reference point
#   7. GRU        : WARM+LAZY K=0 + GRU dispatch warm start
#
# Note on violations: for case1354, Gurobi's post-termination LP re-solve
# (fixing integer variables and re-optimising continuous flows) can produce
# post-contingency flows unseen by the lazy callback. Violations=C120 may
# appear even with LAZY-All. We report this and note that the TRAIN runs
# (3600 s, LAZY_ALL) achieve violations=OK; see paper/data/large_case_summary.csv.
#
# finalize disabled (--no-lazy-finalize-exact): case1354 presolve alone ≈105 s,
# making the finalize re-solve infeasible within budget.
#
# Estimated wall time: ≈ 7 × 600 s ≈ 70–105 min (sequential).
# Usage:  bash paper/run_case1354.sh
#         PYTHON_BIN=python3 bash paper/run_case1354.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"

CASE="matpower/case1354pegase"
TL=600    # solver time limit [s] — consistent with other paper cases
MG=0.05   # MIP gap target 5 %

BASE=(
    --cases "$CASE"
    --time-limit "$TL"
    --mip-gap "$MG"
    --limit-test 1
    --train-use-existing-only
    --no-lazy-finalize-exact        # finalize disabled: presolve alone ≈105s for 1354
)

echo "=== [1/7] LAZY-All  (add_top_k=0, baseline) ==="
"$PYTHON_BIN" -m src.paper.experiments "${BASE[@]}" \
    --modes WARM+LAZY --lazy-top-k 0

echo "=== [2/7] LAZY-K64  (top-64 per callback) ==="
"$PYTHON_BIN" -m src.paper.experiments "${BASE[@]}" \
    --modes WARM+LAZY --lazy-top-k 64

echo "=== [3/7] LAZY-K128 (top-128 per callback) ==="
"$PYTHON_BIN" -m src.paper.experiments "${BASE[@]}" \
    --modes WARM+LAZY --lazy-top-k 128

echo "=== [4/7] COMMIT    (1NN commit hints + LAZY-All) ==="
"$PYTHON_BIN" -m src.paper.experiments "${BASE[@]}" \
    --modes WARM+LAZY --lazy-top-k 0 --adv-commit-hints

echo "=== [5/7] BANDIT    (adaptive top-k in {64,128,256}) ==="
"$PYTHON_BIN" -m src.paper.experiments "${BASE[@]}" \
    --modes WARM+LAZY --adv-adaptive-topk \
    --adv-topk-candidates 64 128 256

echo "=== [6/7] WARM      (no N-1, reference point) ==="
"$PYTHON_BIN" -m src.paper.experiments "${BASE[@]}" \
    --modes WARM

echo "=== [7/7] GRU       (GRU dispatch warm start + LAZY-All) ==="
"$PYTHON_BIN" -m src.paper.experiments "${BASE[@]}" \
    --modes WARM+LAZY --lazy-top-k 0 --adv-gru-warm

echo ""
echo "=== Done. Results in results/raw_logs/ ==="
echo "For violations=OK reference: paper/data/large_case_summary.csv (TRAIN, 3600s)"
