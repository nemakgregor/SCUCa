# IJSE Experiment Audit Runbook

This runbook lists the exact additional experiments and artifacts needed before using the SCUC benchmark results in an International Journal of Sustainable Energy submission. The purpose is to replace fragile claims with auditable evidence on:

- unrestricted `LAZY` and `RAW` checker failures;
- zero-slack security, objective quality, and actual slack magnitudes;
- robust paired runtime summaries instead of an arithmetic-mean headline;
- clustered and repeated-run statistics;
- complete checker-and-fallback wall-clock performance;
- the effect of generator-contingency modeling choices.

Do not synthesize any missing values. If an item below cannot be produced by the current repository state, record that fact in the artifact manifest and either run the listed instrumentation step or narrow the manuscript claim.

If only 2-3 hours of Gurobi time are available, run Section 1A only. It targets missing evidence rather than reproducing already available rows. It cannot support new large-case statistics, repeated-run uncertainty, or generator-contingency claims.

## 1. Checkout and Environment

Use the exact source revision below unless the manuscript is explicitly updated to cite a newer tagged revision.

```bash
git clone https://github.com/nemakgregor/SCUCa.git SCUCa
cd SCUCa
git fetch --all --tags
git checkout d41cf46c89bc323288b46705eb184b00f9807d2c
git switch -c ijse-gurobi-audit-d41cf46
git rev-parse HEAD | tee RUN_COMMIT.txt
```

Create an isolated Python environment and verify that Gurobi can create a model.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt

python - <<'PY'
import gurobipy as gp
print("gurobi_version", gp.gurobi.version())
m = gp.Model()
m.dispose()
PY
```

Record machine and package metadata before running any solves.

```bash
mkdir -p audit_exports
{
  echo "repo_sha=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "hostname=$(hostname)"
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "python=$(python --version 2>&1)"
  python - <<'PY'
import gurobipy as gp
print("gurobi=" + ".".join(map(str, gp.gurobi.version())))
PY
} | tee audit_exports/run_manifest.txt

python -m pip freeze | tee audit_exports/python_freeze.txt
nvidia-smi > audit_exports/nvidia_smi.txt 2>&1 || true
```

Expected fixed paper splits:

```bash
export TEST_DATES="2017-01-15 2017-03-15 2017-05-15 2017-07-15 2017-09-15 2017-11-15"
export TRAIN_DATES="2017-01-05 2017-01-25 2017-02-05 2017-02-25 2017-03-05 2017-03-25 2017-04-05 2017-04-25 2017-05-05 2017-05-25 2017-06-05 2017-06-25 2017-07-05 2017-07-25 2017-08-05 2017-08-25 2017-09-05 2017-09-25 2017-10-05 2017-10-25 2017-11-05 2017-11-25 2017-12-05 2017-12-25"
```

## 1A. Two-to-Three-Hour Missing-Only Audit

This short path reruns only rows that are missing or obligatory for the manuscript's failure and fallback discussion. It does not repeat already available rows and is not a replacement for the confirmatory panels below.

Mandatory failure audit:

- case: `matpower/case57`;
- TEST dates: `2017-09-15` and `2017-11-15`, the known unresolved failure dates in the retained ledger;
- modes: `RAW`, unrestricted `LAZY_ALL`, and `LAZY_COMMIT_HINTS`;
- expected TEST rows: `1 case x 2 dates x 3 modes = 6`;
- worst-case solver time from configured limits: `24 train x 180 s + 6 test x 180 s = 5400 s`, or about 1.5 h, plus setup and data download time.

```bash
export MISSING_RUN_ID=ijse_missing_case57_failure_audit_$(date -u +%Y%m%dT%H%M%SZ)

python -m src.paper.experiments \
  --run-id "$MISSING_RUN_ID" \
  --profile small \
  --only-case matpower/case57 \
  --only-modes RAW LAZY_ALL LAZY_COMMIT_HINTS \
  --train-dates $TRAIN_DATES \
  --only-test-instances 2017-09-15 2017-11-15 \
  --force-rerun
```

Mandatory fallback micro-benchmark if the manuscript keeps any checker-and-fallback workflow claim:

- case: `matpower/case300`;
- TEST date: `2017-01-15`, a retained row where `LAZY_BANDIT` is checker-rejected in the existing ledger;
- modes: `LAZY_BANDIT` and unrestricted `LAZY_ALL`;
- expected TEST rows: `1 case x 1 date x 2 modes = 2`;
- worst-case solver time from configured limits: `2 x 1200 s`, or about 0.7 h.

```bash
export FALLBACK_MICRO_RUN_ID=ijse_missing_fallback_micro_$(date -u +%Y%m%dT%H%M%SZ)

python -m src.paper.experiments \
  --run-id "$FALLBACK_MICRO_RUN_ID" \
  --profile small \
  --only-case matpower/case300 \
  --only-modes LAZY_BANDIT LAZY_ALL \
  --only-test-instances 2017-01-15 \
  --force-rerun
```

Create the missing-only summary:

```bash
python - <<'PY'
import os
from pathlib import Path

import numpy as np
import pandas as pd

run_ids = [os.environ.get("MISSING_RUN_ID", ""), os.environ.get("FALLBACK_MICRO_RUN_ID", "")]
frames = []
for run_id in [x for x in run_ids if x]:
    path = Path("results") / run_id / "results.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["source_run_id"] = run_id
        frames.append(df)
if not frames:
    raise SystemExit("no short-run result CSV found")

df = pd.concat(frames, ignore_index=True)
df = df[df["stage"].astype(str).str.upper().eq("TEST")].copy()
df["runtime"] = pd.to_numeric(df["wall_sec"], errors="coerce")
df["mip_gap_num"] = pd.to_numeric(df["mip_gap"], errors="coerce")
df["pass_num"] = pd.to_numeric(df["pass"], errors="coerce").fillna(0).astype(int)
df["has_incumbent_num"] = pd.to_numeric(df["has_incumbent"], errors="coerce").fillna(0).astype(int)

summary = df.groupby(["case_folder", "mode_id"], dropna=False).agg(
    rows=("runtime", "size"),
    incumbents=("has_incumbent_num", "sum"),
    checker_accepted=("pass_num", "sum"),
    median_wall_sec=("runtime", "median"),
    median_gap=("mip_gap_num", "median"),
    max_residual=("max_constraint_residual", "max"),
).reset_index()

out = Path("audit_exports") / "missing_only_audit_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(out, index=False)
print(out)
print(summary.to_string(index=False))
PY
```

Manuscript use rule for this short path: use the case57 rows to document the RAW/unrestricted-LAZY/LAZY+COMMIT failure dates and use the case300 micro-run only as an illustrative checker-and-fallback timing example. Do not use this path to report new large-case statistics, repeated-run uncertainty, or generator-contingency coverage.

## 2. Minimal Dry Run

Run this before the expensive panel. It should list 24 training instances, 6 test instances, and the requested modes for `case1354pegase`.

```bash
python -m src.paper.experiments \
  --run-id ijse_dryrun \
  --profile large6800 \
  --only-case matpower/case1354pegase \
  --only-modes LAZY_ALL LAZY_COMMIT_HINTS WARM_LAZY_GRU LAZY_BANDIT LAZY_GNN_T080 WARM_PRUNE_LAZY_T030 STREDUCE_LAZY_GRU \
  --train-dates $TRAIN_DATES \
  --test-dates $TEST_DATES \
  --dry-run
```

Do not continue if any mode is reported as unknown.

## 3. Large-Case Confirmation Panel

This is the main replacement for the current large-case claims. It runs four selected large cases, six fixed dates per case, and seven modes. Expected TEST rows: `4 cases x 6 dates x 7 modes = 168`. The run also generates 24 `LAZY_ALL` training solves per case when training artifacts are required.

Modes:

- `LAZY_ALL`: unrestricted lazy generation, main exact workflow.
- `LAZY_COMMIT_HINTS`: lazy generation plus commitment hints.
- `WARM_LAZY_GRU`: lazy generation plus GRU dispatch warm start.
- `LAZY_BANDIT`: lazy generation with bandit top-k policy.
- `LAZY_GNN_T080`: pair-risk stress test with line-level GNN screening.
- `WARM_PRUNE_LAZY_T030`: aggressive reduced model with lazy recovery.
- `STREDUCE_LAZY_GRU`: spatio-temporal reduction with lazy recovery and GRU label where used.

Run one case at a time. This keeps logs easier to audit and avoids wasting a license token after a crash.

```bash
export RUN_ID=ijse_large_audit_$(date -u +%Y%m%dT%H%M%SZ)
export LARGE_MODES="LAZY_ALL LAZY_COMMIT_HINTS WARM_LAZY_GRU LAZY_BANDIT LAZY_GNN_T080 WARM_PRUNE_LAZY_T030 STREDUCE_LAZY_GRU"

python -m src.paper.experiments \
  --run-id "$RUN_ID" \
  --profile large6800 \
  --only-case matpower/case1354pegase \
  --only-modes $LARGE_MODES \
  --train-dates $TRAIN_DATES \
  --test-dates $TEST_DATES \
  --force-rerun

python -m src.paper.experiments \
  --run-id "$RUN_ID" \
  --resume \
  --profile large6800 \
  --only-case matpower/case2383wp \
  --only-modes $LARGE_MODES \
  --train-dates $TRAIN_DATES \
  --test-dates $TEST_DATES \
  --force-rerun

python -m src.paper.experiments \
  --run-id "$RUN_ID" \
  --resume \
  --profile large6800 \
  --only-case matpower/case3375wp \
  --only-modes $LARGE_MODES \
  --train-dates $TRAIN_DATES \
  --test-dates $TEST_DATES \
  --force-rerun

python -m src.paper.experiments \
  --run-id "$RUN_ID" \
  --resume \
  --profile large6800 \
  --only-case matpower/case6515rte \
  --only-modes $LARGE_MODES \
  --train-dates $TRAIN_DATES \
  --test-dates $TEST_DATES \
  --force-rerun
```

After each case, check that a nonempty result file exists:

```bash
test -s "results/$RUN_ID/results.csv"
tail -5 "results/$RUN_ID/results.csv"
find "results/$RUN_ID/logs/gurobi/test" -type f | wc -l
```

## 4. RAW and Unrestricted-LAZY Failure Diagnosis

The manuscript must not claim that a run is simply feasible or reliable. It needs separate labels for:

- solver incumbent exists;
- checker-accepted under the slack-augmented model;
- zero-slack secure or non-zero-slack;
- operationally secure under the stated model limitation.

First rerun the known `case57` RAW panel with `RAW`, unrestricted `LAZY_ALL`, and `LAZY_COMMIT_HINTS`.

```bash
export RAW_RUN_ID=ijse_case57_raw_audit_$(date -u +%Y%m%dT%H%M%SZ)

python -m src.paper.experiments \
  --run-id "$RAW_RUN_ID" \
  --profile small \
  --only-case matpower/case57 \
  --only-modes RAW LAZY_ALL LAZY_COMMIT_HINTS \
  --train-dates $TRAIN_DATES \
  --test-dates $TEST_DATES \
  --force-rerun
```

Then run a small-case cross-check for the exact modes that form the causal comparison:

```bash
export SMALL_EXACT_RUN_ID=ijse_small_exact_audit_$(date -u +%Y%m%dT%H%M%SZ)

python -m src.paper.experiments \
  --run-id "$SMALL_EXACT_RUN_ID" \
  --profile small \
  --only-modes RAW LAZY_ALL LAZY_COMMIT_HINTS WARM_LAZY_GRU \
  --train-dates $TRAIN_DATES \
  --test-dates $TEST_DATES \
  --force-rerun
```

Expected TEST rows for the second command: `6 cases x 6 dates x 4 modes = 144`.

## 5. Fixed Seed and Thread Control

The current runner fixes Python, NumPy, and PyTorch randomness, but it does not expose Gurobi `Seed` and `Threads` as command-line options. Before any repeated deterministic solver runs, apply this local patch. Keep the patch file in the final archive.

```bash
python - <<'PY'
from pathlib import Path
p = Path("src/paper/experiments.py")
s = p.read_text()
needle = "    model.Params.TimeLimit = float(mode.time_limit_sec)\n"
insert = """    model.Params.TimeLimit = float(mode.time_limit_sec)
    if os.environ.get("GUROBI_SEED"):
        model.Params.Seed = int(os.environ["GUROBI_SEED"])
    if os.environ.get("GUROBI_THREADS"):
        model.Params.Threads = int(os.environ["GUROBI_THREADS"])
"""
if insert in s:
    print("seed/thread patch already present")
elif needle in s:
    p.write_text(s.replace(needle, insert), encoding="utf-8")
    print("seed/thread patch applied")
else:
    raise SystemExit("patch anchor not found")
PY

git diff -- src/paper/experiments.py | tee audit_exports/gurobi_seed_threads.patch
```

Determinism smoke test: three identical single-thread runs with the same seed should return identical incumbent objective, status, checker result, and slack summary for each `(case,date,mode)`.

```bash
export GUROBI_THREADS=1
export GUROBI_SEED=42
export DET_MODES="RAW LAZY_ALL LAZY_COMMIT_HINTS WARM_LAZY_GRU"

for rep in 1 2 3; do
  export DET_RUN_ID=ijse_determinism_seed42_rep${rep}_$(date -u +%Y%m%dT%H%M%SZ)
  python -m src.paper.experiments \
    --run-id "$DET_RUN_ID" \
    --profile small \
    --only-case matpower/case57 \
    --only-modes $DET_MODES \
    --train-dates $TRAIN_DATES \
    --test-dates $TEST_DATES \
    --force-rerun
done
```

Repeated solver runs for variance estimates. Use at least five seeds; ten is preferable if license time allows. The full small panel is preferred. If time is constrained, run `case57`, `case118`, and `case300` first.

```bash
export GUROBI_THREADS=1
export REPEAT_MODES="RAW LAZY_ALL LAZY_COMMIT_HINTS WARM_LAZY_GRU"

for seed in 101 102 103 104 105 106 107 108 109 110; do
  export GUROBI_SEED=$seed
  export REPEAT_RUN_ID=ijse_small_repeat_seed${seed}_$(date -u +%Y%m%dT%H%M%SZ)
  python -m src.paper.experiments \
    --run-id "$REPEAT_RUN_ID" \
    --profile small \
    --only-modes $REPEAT_MODES \
    --train-dates $TRAIN_DATES \
    --test-dates $TEST_DATES \
    --force-rerun
done
```

## 6. Checker-and-Fallback Workflow

Permanent screening modes must include checker-triggered fallback time. The relevant metric is:

```text
T_effective = T_screened + I_checker_rejected * T_fallback
```

Run screened modes first, then rerun rejected rows with unrestricted `LAZY_ALL`. Use the same case/date set as the large confirmation panel.

```bash
export SCREEN_RUN_ID=ijse_screen_then_fallback_$(date -u +%Y%m%dT%H%M%SZ)
export SCREEN_MODES="LAZY_GNN_T080 WARM_PRUNE_LAZY_T030 STREDUCE_LAZY_GRU"

for case in matpower/case1354pegase matpower/case2383wp matpower/case3375wp matpower/case6515rte; do
  python -m src.paper.experiments \
    --run-id "$SCREEN_RUN_ID" \
    --resume \
    --profile large6800 \
    --only-case "$case" \
    --only-modes $SCREEN_MODES \
    --train-dates $TRAIN_DATES \
    --test-dates $TEST_DATES \
    --force-rerun
done
```

Generate the rejected instance list:

```bash
python - <<'PY'
import os
import pandas as pd
from pathlib import Path

run_id = os.environ["SCREEN_RUN_ID"]
df = pd.read_csv(Path("results") / run_id / "results.csv")
test = df[df["stage"].astype(str).str.upper().eq("TEST")].copy()
test["pass_num"] = pd.to_numeric(test["pass"], errors="coerce").fillna(0).astype(int)
rej = test[test["pass_num"].ne(1)]
out = Path("audit_exports") / f"{run_id}_fallback_targets.csv"
rej[["case_folder", "instance_name", "mode_id", "status", "has_incumbent", "feasible_ok", "violations", "wall_sec", "runtime_sec", "error_message"]].to_csv(out, index=False)
print(out)
print(rej[["case_folder", "instance_name", "mode_id", "status", "violations"]].to_string(index=False))
PY
```

For every `instance_name` in the fallback target file, run unrestricted `LAZY_ALL`. This shell loop groups targets by case and date.

```bash
export FALLBACK_RUN_ID=${SCREEN_RUN_ID}_fallback_lazy

python - <<'PY' > audit_exports/fallback_commands.sh
import os
import pandas as pd
from pathlib import Path

screen_run_id = os.environ["SCREEN_RUN_ID"]
fallback_run_id = os.environ["FALLBACK_RUN_ID"]
targets = pd.read_csv(Path("audit_exports") / f"{screen_run_id}_fallback_targets.csv")
if targets.empty:
    print("echo no fallback targets")
else:
    for (case, instance), _ in targets.groupby(["case_folder", "instance_name"], sort=True):
        date = str(instance).rsplit("/", 1)[-1]
        print("python -m src.paper.experiments \\")
        print(f"  --run-id {fallback_run_id} \\")
        print("  --resume \\")
        print("  --profile large6800 \\")
        print(f"  --only-case {case} \\")
        print("  --only-modes LAZY_ALL \\")
        print("  --train-dates $TRAIN_DATES \\")
        print(f"  --only-test-instances {date} \\")
        print("  --force-rerun")
PY

bash audit_exports/fallback_commands.sh
```

The final paper table should report screened wall-clock time, fallback wall-clock time, total effective wall-clock time, final checker status, final objective, final gap, and final slack magnitude.

## 7. Slack-Magnitude Extraction

The CSV gives checker pass/fail, residual, objective inconsistency, and candidate JSON paths. Use the saved candidate JSON to report actual slack magnitudes.

Run this for every fresh result file:

```bash
python - <<'PY'
import json
import math
import os
from pathlib import Path

import pandas as pd

run_ids = [
    os.environ.get("RUN_ID", ""),
    os.environ.get("RAW_RUN_ID", ""),
    os.environ.get("SMALL_EXACT_RUN_ID", ""),
    os.environ.get("SCREEN_RUN_ID", ""),
    os.environ.get("FALLBACK_RUN_ID", ""),
]
run_ids = [x for x in run_ids if x]
rows = []

def walk_numbers(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_numbers(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_numbers(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and math.isfinite(float(obj)):
        yield path, float(obj)

slack_tokens = ("shortfall", "overflow", "slack")
for run_id in run_ids:
    csv_path = Path("results") / run_id / "results.csv"
    if not csv_path.exists():
        continue
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        sol_path = str(row.get("candidate_solution_json") or "")
        if not sol_path or not Path(sol_path).exists():
            continue
        data = json.loads(Path(sol_path).read_text())
        vals = [(p, v) for p, v in walk_numbers(data) if any(tok in p.lower() for tok in slack_tokens)]
        rows.append({
            "run_id": run_id,
            "stage": row.get("stage"),
            "case_folder": row.get("case_folder"),
            "instance_name": row.get("instance_name"),
            "mode_id": row.get("mode_id"),
            "status": row.get("status"),
            "pass": row.get("pass"),
            "has_incumbent": row.get("has_incumbent"),
            "feasible_ok": row.get("feasible_ok"),
            "violations": row.get("violations"),
            "obj_val": row.get("obj_val"),
            "mip_gap": row.get("mip_gap"),
            "slack_l1": sum(abs(v) for _, v in vals),
            "slack_max": max([abs(v) for _, v in vals], default=0.0),
            "slack_nonzero_count": sum(1 for _, v in vals if abs(v) > 1e-7),
        })

out = Path("audit_exports") / "slack_magnitude_summary.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(out)
PY
```

Report `checker-accepted` separately from `zero-slack secure`. A result is zero-slack secure only when it is checker-accepted and `slack_max <= 1e-7` under the extraction above or an equivalent directly instrumented slack calculation.

## 8. Robust Runtime and Clustered Statistics

Use this script to produce paired median speedup, geometric-mean speedup, interquartile ranges, censored-run counts, and per-topology summaries. It uses only fresh result folders named in the environment.

```bash
python - <<'PY'
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

run_ids = [
    os.environ.get("SMALL_EXACT_RUN_ID", ""),
    os.environ.get("RUN_ID", ""),
]
run_ids = [x for x in run_ids if x]
frames = []
for run_id in run_ids:
    path = Path("results") / run_id / "results.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["source_run_id"] = run_id
        frames.append(df)
if not frames:
    raise SystemExit("no fresh result CSV found")

df = pd.concat(frames, ignore_index=True)
df = df[df["stage"].astype(str).str.upper().eq("TEST")].copy()
df["runtime"] = pd.to_numeric(df["wall_sec"], errors="coerce")
df["mip_gap_num"] = pd.to_numeric(df["mip_gap"], errors="coerce")
df["pass_num"] = pd.to_numeric(df["pass"], errors="coerce").fillna(0).astype(int)
df["time_limit"] = pd.to_numeric(df["time_limit_sec"], errors="coerce")
df["censored"] = df["runtime"].ge(0.98 * df["time_limit"]) | df["status"].astype(str).str.upper().eq("TIME_LIMIT")

key_cols = ["case_folder", "instance_name"]
baseline = "RAW"
rows = []
for mode in sorted(set(df["mode_id"]) - {baseline}):
    b = df[df["mode_id"].eq(baseline)][key_cols + ["runtime"]].rename(columns={"runtime": "runtime_base"})
    m = df[df["mode_id"].eq(mode)][key_cols + ["runtime", "pass_num", "censored"]]
    paired = b.merge(m, on=key_cols, how="inner")
    paired = paired[(paired["runtime"] > 0) & (paired["runtime_base"] > 0)]
    if paired.empty:
        continue
    ratios = paired["runtime_base"] / paired["runtime"]
    rows.append({
        "mode_id": mode,
        "n_pairs": len(paired),
        "median_speedup": float(np.median(ratios)),
        "geomean_speedup": float(math.exp(np.mean(np.log(ratios)))),
        "mean_speedup_supplementary": float(np.mean(ratios)),
        "speedup_iqr_low": float(np.quantile(ratios, 0.25)),
        "speedup_iqr_high": float(np.quantile(ratios, 0.75)),
        "checker_accept_rate": float(paired["pass_num"].mean()),
        "censored_runs": int(paired["censored"].sum()),
    })

out = Path("audit_exports") / "robust_paired_speedups.csv"
pd.DataFrame(rows).sort_values(["median_speedup"], ascending=False).to_csv(out, index=False)
print(out)

topo = df.groupby(["case_folder", "mode_id"], dropna=False).agg(
    rows=("runtime", "size"),
    accepted=("pass_num", "sum"),
    median_wall_sec=("runtime", "median"),
    median_gap=("mip_gap_num", "median"),
    censored_runs=("censored", "sum"),
).reset_index()
out2 = Path("audit_exports") / "per_topology_summary.csv"
topo.to_csv(out2, index=False)
print(out2)
PY
```

For cluster bootstrap intervals, resample `case_folder`, not rows. With only six small topologies, report leave-one-topology-out sensitivity instead of claiming broad generalization.

## 9. Callback Trace Requirement

The current CSV exports `lazy_added_cont` but not a full callback trace. If any unrestricted `LAZY_ALL` or `LAZY_COMMIT_HINTS` row is checker-rejected, the next rerun must export callback diagnostics before the manuscript explains the failure.

Required fields per row:

- `lazy_incumbents_seen`;
- `lazy_added_cont`;
- `lazy_line_pair_distinct`;
- `lazy_gen_pair_distinct`;
- `lazy_line_cuts_by_pair_json`;
- `lazy_gen_cuts_by_pair_json`;
- `callback_exception_count`;
- `callback_trace_jsonl`;
- `gurobi_log_path`.

Minimum JSONL event schema:

```json
{"event":"mipsol","stage":"TEST","instance_name":"matpower/case57/2017-01-15","mode_id":"LAZY_ALL","mipsol_index":1,"violations_found":4,"cuts_added":4,"max_violation":12.34,"top_kind":"line"}
```

If a callback trace is not available, the manuscript must state only that the row was checker-rejected and that the callback-level cause was not identified. Do not replace this with speculation.

## 10. Generator-Contingency Model Comparison

The current generator outage treatment is reference-bus dependent. Either remove generator contingencies from the claimed N-1 scope or run a model comparison that separates line contingencies from generator contingencies.

Required experiments:

1. Line-outage-only SCUC for `LAZY_ALL` on all six small topologies and the four large cases above.
2. Current line-plus-generator SCUC for the same cases, dates, and modes.
3. If implemented, distributed-slack or reserve-participation generator outage model for the same subset.

The repository does not currently expose a command-line switch for these three event sets. If the code is not changed, the manuscript should state that the submitted benchmark is for the implemented slack-augmented DC model and should not present generator-outage security as an operational certificate.

For any changed event-set code, archive:

- event-set construction patch;
- total line-outage pairs by case;
- total generator-outage pairs by case;
- omitted unsupported line outages by case;
- omitted unsupported generator outages by case;
- reason for omission, especially islanding or unsupported LODF events.

## 11. Predictor Quality Diagnostics

The causal claim must be narrowed unless predictor quality is measured at comparable targets. Export these metrics before making any predictor-vs-entry-point statement:

- GNN: pair-level contingency recall, false-negative rate, precision, calibration curve, and threshold-specific retained-pair fraction.
- BANDIT: runtime reward sensitivity, not only number of cuts.
- GRU: dispatch warm-start error by unit and hour; objective degradation of its incumbent when accepted.
- COMMIT: binary status accuracy, false on/off rates, and number of hinted variables.
- Oracle screening upper bound: perfect screening labels at each entry point for a small subset.

The current runbook does not create those metrics automatically. If they are not added, phrase the manuscript conclusion as: in this implementation, unrestricted lazy generation had a larger measured runtime effect than the particular starts, hints, and screens tested.

## 12. Archive Bundle

After all runs finish, create a reproducible artifact bundle. Include raw ledgers and logs, not only aggregated tables.

```bash
{
  echo "finished_utc=$(date -u +%FT%TZ)"
  echo "repo_sha=$(git rev-parse HEAD)"
  echo "dirty_diff_sha256=$(git diff | sha256sum | awk '{print $1}')"
} | tee -a audit_exports/run_manifest.txt

git diff > audit_exports/source_diff.patch

python -m src.paper.analysis || true
python -m src.paper.plots || true
python -m src.paper.tables || true

tar -czf "audit_exports/ijse_experiment_artifacts_$(date -u +%Y%m%dT%H%M%SZ).tgz" \
  RUN_COMMIT.txt \
  audit_exports \
  results/*/results.csv \
  results/*/state.json \
  results/*/live_status.json \
  results/*/logs \
  results/*/solutions \
  results/*.csv \
  results/tables \
  results/figures

sha256sum audit_exports/*.tgz | tee audit_exports/artifact_sha256.txt
```

Files to send back for manuscript revision:

- `audit_exports/run_manifest.txt`;
- `audit_exports/python_freeze.txt`;
- `audit_exports/source_diff.patch`;
- `results/$RUN_ID/results.csv`;
- `results/$RAW_RUN_ID/results.csv`;
- `results/$SMALL_EXACT_RUN_ID/results.csv`;
- all `ijse_small_repeat_seed*` result folders if repeated runs were completed;
- `audit_exports/slack_magnitude_summary.csv`;
- `audit_exports/robust_paired_speedups.csv`;
- `audit_exports/per_topology_summary.csv`;
- `audit_exports/*fallback*.csv`;
- all Gurobi logs for rejected rows;
- callback JSONL traces if any unrestricted lazy row is rejected.

## 13. Manuscript Use Rules

Use the new results conservatively:

- Replace any headline arithmetic-mean speedup with median paired and geometric-mean speedups.
- Report objective value, MIP gap, checker result, and slack magnitude with runtime.
- Mark modes that permanently screen or reduce constraints as checker-accepted only after post-solve verification.
- Include fallback time for screening modes that are proposed as operational workflows.
- Treat small-case statistics as clustered by topology.
- Do not imply definitive rankings against literature methods when the code uses adaptations rather than original author implementations.
- If renewable/storage experiments are not added, remove sustainability claims from the title, abstract, and main conclusions.
