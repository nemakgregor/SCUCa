# One-click launcher for the three additional experiments described in
# RUN_ADDITIONAL_EXPERIMENTS.md. Edit $Run* flags to enable / disable steps.
#
#   - Step 1: GNN precision / recall / F1      (<= 10 min, no solver)
#   - Step 2: fixed-K ablation for BANDIT      (~2.5 h on case118+case300)
#   - Step 3: Gurobi grbtune baseline          (~2 h on case118+case300)
#
# Run from the repo root in PowerShell:
#     .\paper\run_additional.ps1

$ErrorActionPreference = "Stop"

$RunGnnPr    = if ($env:RUN_GNN_PR)  { $env:RUN_GNN_PR  } else { "1" }
$RunFixedK   = if ($env:RUN_FIXED_K) { $env:RUN_FIXED_K } else { "0" }
$RunGrbtune  = if ($env:RUN_GRBTUNE) { $env:RUN_GRBTUNE } else { "1" }

$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $RepoRoot

Write-Host "===== SCUCa revision: additional experiments =====" -ForegroundColor Cyan
Write-Host "repo root: $RepoRoot"
Write-Host "steps: GNN_PR=$RunGnnPr  FIXED_K=$RunFixedK  GRBTUNE=$RunGrbtune"
Write-Host ""

if ($RunGnnPr -eq "1") {
    Write-Host "[1/3] GNN precision / recall / F1" -ForegroundColor Yellow
    python paper\scripts\eval_gnn_precision_recall.py `
        --artifact-root results\exp_20260417_upto1300_full\artifacts\gnn_screening `
        --p-thr 0.60 --y-thr 0.70
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host ""
}

if ($RunFixedK -eq "1") {
    Write-Host "[2/3] Fixed-K ablation (K=64, K=128, K=256) on case118 and case300" -ForegroundColor Yellow
    foreach ($K in 64, 128, 256) {
        python -m src.paper.experiments `
            --cases matpower/case118 matpower/case300 `
            --modes WARM+LAZY `
            --lazy-top-k $K `
            --time-limit 600 `
            --skip-solved `
            --train-use-existing-only
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    python paper\scripts\postprocess.py
    Write-Host ""
}

if ($RunGrbtune -eq "1") {
    Write-Host "[3/3] Gurobi auto-tuner baseline (case118, case300)" -ForegroundColor Yellow
    python paper\scripts\run_grbtune.py --case matpower/case118 --tune-time 1800 --solve-time 600
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python paper\scripts\run_grbtune.py --case matpower/case300 --tune-time 3600 --solve-time 600
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host ""
}

Write-Host "===== done. Update paper.tex per RUN_ADDITIONAL_EXPERIMENTS.md. =====" -ForegroundColor Green
