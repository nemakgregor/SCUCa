# Additional experiments for the revision

**Goal.** Turn the three remaining reviewer comments that are currently only
*asserted* in the text into *measured* evidence. All other comments are
already closed using pre-computed data.

Reviewer targets for the new runs:

| Comment | What it asks for | Current status in paper | New run required |
|---|---|---|---|
| R1-C5 | GNN precision / recall / F1 | Only downstream speedup shown (Tab.~5) | **1. GNN P/R/F1 evaluation** |
| R1-C6 | Justify BANDIT vs fixed-K | Text only; "marginal improvement" | **2. Fixed-K ablation** |
| R2-C4 | Gurobi auto-tuning baseline | Listed as limitation | **3. `grbtune` on case118, case300** |

Nothing else from R1 / R2 needs additional compute — see `response_to_reviewers.tex` for the full closure map.

---

## Prerequisites (once per machine)

```bash
# activate your existing venv (Windows Git-Bash)
source SCUC/Scripts/activate

# confirm Gurobi license is visible
python -c "import gurobipy; print(gurobipy.gurobi.version())"

# confirm torch-geometric is available (only needed for #1)
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

If torch-geometric is missing, skip step 1 — it is nice-to-have, not a
blocker. Everything else uses only `gurobipy` + `pandas` + `numpy` from
`requirements.txt`.

All commands below are run from the repo root
(`c:/Users/egor1/Desktop/Energy/SCUC/SCUCa`).

---

## 1. GNN precision / recall / F1 (closes R1-C5 fully)

**Cost.** Pure postprocessing, no solver runs. ≤ 5 min on CPU.

**Command (bash / zsh):**

```bash
python paper/scripts/eval_gnn_precision_recall.py \
    --artifact-root results/exp_20260417_upto1300_full/artifacts/gnn_screening \
    --p-thr 0.60 --y-thr 0.70
```

**Command (PowerShell):**

```powershell
python .\paper\scripts\eval_gnn_precision_recall.py `
    --artifact-root results\exp_20260417_upto1300_full\artifacts\gnn_screening `
    --p-thr 0.60 --y-thr 0.70
```

**Writes:**

- `paper/tables/gnn_pr_f1.csv` — raw P / R / F1 per case
- `paper/tables/gnn_pr_f1.tex` — LaTeX table stub

**After it runs:** in `paper/paper.tex`, right below the current sentence in
the GNN paragraph ("we treat it as an exploratory ablation"), append

```latex
\input{tables/gnn_pr_f1.tex}
```

and add a single reference: "Tab.~\ref{tab:gnn_pr_f1} reports precision, recall
and F1 of the line-criticality classifier on held-out instances."

> **Caveat.** The GNN artifact in `results/exp_20260417_upto1300_full/artifacts/gnn_screening/` currently contains only `matpower/case118`. If you also want P/R/F1 for the other cases, retrain first via
>
> ```bash
> for C in case14 case30 case57 case89pegase case118 case300; do
>     python -m src.ml_models.gnn_screening --case matpower/$C --epochs 40
> done
> ```
>
> Each call takes 1–10 min. If you skip, the table will contain only case118 (still sufficient to close R1-C5 as an existence proof).

---

## 2. Fixed-K ablation for BANDIT (closes R1-C6 properly)

**Purpose.** The paper currently claims BANDIT is "marginal" over plain LAZY
but does not compare it to fixed K ∈ {64, 128, 256}. This ablation turns the
claim into a proper negative result.

**Cost.**
- case118: 3 runs × 49 test instances ≈ 30 min total (≈ 5 s/instance)
- case300: 3 runs × 59 test instances ≈ 2 h total (≈ 25 s/instance)

Reuse existing warm-start and redundancy artifacts; no retraining.

**Commands (bash / zsh):**

```bash
# Fixed K=64
python -m src.paper.experiments \
    --cases matpower/case118 matpower/case300 \
    --modes WARM+LAZY \
    --lazy-top-k 64 \
    --time-limit 600 \
    --skip-solved \
    --train-use-existing-only

# Fixed K=128
python -m src.paper.experiments \
    --cases matpower/case118 matpower/case300 \
    --modes WARM+LAZY \
    --lazy-top-k 128 \
    --time-limit 600 \
    --skip-solved \
    --train-use-existing-only

# Fixed K=256
python -m src.paper.experiments \
    --cases matpower/case118 matpower/case300 \
    --modes WARM+LAZY \
    --lazy-top-k 256 \
    --time-limit 600 \
    --skip-solved \
    --train-use-existing-only
```

(PowerShell equivalents: replace ` \ ` with `` ` ``.)

**What gets produced.** Rows with `mode_label = "WARM+LAZY+K64"`, `...K128`,
`...K256` will appear in `results/raw_logs/` (the existing logging
infrastructure tags them via `_build_mode_label`, see
[`experiments.py:348`](../src/paper/experiments.py#L348)).

**Postprocessing.** After the runs finish, append the new rows to the CI
table:

```bash
python paper/scripts/postprocess.py     # re-builds tables/speedup_with_ci.tex
```

**Table update in paper.tex.** Add three rows to
`tables/speedup_with_ci.tex` for `WARM+LAZY+K64`, `WARM+LAZY+K128`,
`WARM+LAZY+K256` and reference them in the BANDIT paragraph:

> "Tab.~\ref{tab:speedup_ci} also shows fixed-K ablations (K ∈ {64, 128, 256}):
> the best fixed K reaches $\approx {\color{red}X.XX}\times$ on case300, compared
> to $\approx 13.11\times$ for BANDIT, confirming that the adaptive layer brings
> at most a marginal improvement."

---

## 3. Gurobi `grbtune` baseline (closes R2-C4 fully)

**Purpose.** One representative instance per case, tuned with Gurobi's
built-in parameter tuner, then re-solved with the tuned parameters and
compared against the default configuration. This is the minimal evidence that
our compact-formulation speedup is *not* just a re-implementation of what
`grbtune` would find automatically.

**Cost.**
- case118: ≈ 30 min tune + 5 min verify → ~35 min
- case300: ≈ 60 min tune + 15 min verify → ~75 min
- (optional case89pegase: ≈ 10 min tune + 2 min verify)

Each case is one standalone command. No shared state between them.

**Commands (bash / zsh):**

```bash
python paper/scripts/run_grbtune.py --case matpower/case118 --tune-time 1800 --solve-time 600
python paper/scripts/run_grbtune.py --case matpower/case300 --tune-time 3600 --solve-time 600
# optional:
python paper/scripts/run_grbtune.py --case matpower/case89pegase --tune-time 600 --solve-time 600
```

**Writes:**

- `paper/data/grbtune_<case>.json`   — per-case tuning record
- `paper/data/grbtune_<case>.mps`    — model written before tuning
- `paper/data/grbtune_<case>.prm`    — best parameter file returned by Gurobi
- `paper/tables/grbtune_vs_default.csv` — aggregated comparison

**Paper edit after completion.** In `paper.tex`, in the *Limitations and
threats to validity* subsection, replace the *Solver tuning* bullet with a
short data-backed paragraph:

> **Solver tuning.** On the representative instances of \texttt{case118} and
> \texttt{case300}, Gurobi's built-in parameter tuner (\texttt{Model.tune},
> 30/60 min budget) reduced the default solve time by a factor of
> $\approx {\color{red}X.XX}\times$ and $\approx {\color{red}Y.YY}\times$
> respectively, which is substantially smaller than the speedup delivered by
> \texttt{WARM+PRUNE-0.10} ($8.11\times$, $8.81\times$) or by
> \texttt{WARM+LAZY+BANDIT} ($9.00\times$, $13.11\times$) on the same cases.
> This confirms that the dominant gain comes from the compact formulation /
> lazy enforcement rather than from parameter choices the auto-tuner could
> discover.

Exact numbers come from `paper/tables/grbtune_vs_default.csv` once the runs
finish.

---

## Optional (not required, but cheap and useful)

### Sanity check on LaTeX

```bash
python paper/scripts/sanity_check.py
```

Should print `RESULT: OK`.

### Re-generate all autogenerated tables from current CSVs

```bash
python paper/scripts/postprocess.py
```

### Quick unit check that GNN model is loadable (before running #1)

```bash
python -c "
from src.ml_models.gnn_screening import GNNLineScreener
g = GNNLineScreener('matpower/case118')
print('trained:', g.ensure_trained())
"
```

---

## What happens if any of 1/2/3 fails on your second laptop

- **Step 1 fails** (no torch-geometric / no GNN weights): skip. R1-C5 is
  already *addressed* by the downstream ablation (Tab. 5). P/R/F1 would be
  *stronger* evidence but is not strictly required. Mention in the response
  letter.
- **Step 2 fails** (Gurobi licensing, out-of-memory): skip. The BANDIT
  paragraph already labels itself as a marginal / negative result. Update the
  response letter to leave R1-C6 as partially addressed.
- **Step 3 fails** (Gurobi `tune()` unavailable): document this in the
  response letter as "compute-bound" and leave R2-C4 as a limitation. The
  limitation bullet in the paper already acknowledges this scenario.

None of the three failures blocks submission.

---

## Total additional wall-clock time

On an 8-core laptop with a Gurobi academic license:

| Step | Wall time | Priority |
|---|---|---|
| 1. GNN P/R/F1 | < 10 min | Nice-to-have |
| 2. Fixed-K ablation | ~2.5 h  | Recommended |
| 3. `grbtune` baseline | ~2 h | **Critical** for R2-C4 |

**Minimum recommended:** step 3 only. Everything else is already text-level
addressed; step 3 is the only one a strict reviewer can still demand numbers
for.
