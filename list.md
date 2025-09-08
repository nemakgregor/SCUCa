TASK:
Now             
RULE:
Printout updated files in in copy-past-run style and an explainer.md of what you have modified why and how.
                     
PROJECT FILES:
## OUR CURRENT Files Tree

- README.md
    - drafts/SCUC_v3.tex
        - ml/src/__init__.py
        - ml/src/graph.py
        - ml/src/model.py
        - ml/src/utils.py
- res_compare.md
    - src/__init__.py
            - src/data/logs/compare_logs_matpower_case118_20250907_152327.csv
                    - src/data/output/matpower/case118/compare_matpower_case118_20250904_175820.csv
                    - src/data/output/matpower/case118/compare_matpower_case118_20250905_150049.csv
                    - src/data/output/matpower/case118/compare_matpower_case118_20250907_152327.csv
                    - src/data/output/matpower/case14/perf_matpower_case14_basic_01.csv
                    - src/data/output/matpower/case57/compare_matpower_case57_20250903_153640.csv
                    - src/data/output/test/case14-fixed-status/perf_test_case14_fixed_status_basic_01.csv
                    - src/data/output/test/case14-profiled/perf_test_case14_profiled_basic_01.csv
                    - src/data/output/test/case14-storage/perf_test_case14_storage_basic_01.csv
                    - src/data/output/test/case14-sub-hourly/perf_test_case14_sub_hourly_basic_01.csv
                    - src/data/output/test/case14/perf_test_case14_basic_01.csv
        - src/data_preparation/__init__.py
        - src/data_preparation/data_structure.py
        - src/data_preparation/download_data.py
        - src/data_preparation/params.py
        - src/data_preparation/prepare_data.py
        - src/data_preparation/ptdf_lodf.py
        - src/data_preparation/read_data.py
        - src/data_preparation/utils.py
        - src/ml_models/__init__.py
        - src/ml_models/fix_warm_start.py
        - src/ml_models/pretrain_warm_start.py
        - src/ml_models/redundant_constraints.py
        - src/ml_models/warm_start.py
            - src/optimization_model/SCUC_solver/__init__.py
            - src/optimization_model/SCUC_solver/compare_ml_raw.py
            - src/optimization_model/SCUC_solver/ed_model_builder.py
            - src/optimization_model/SCUC_solver/optimizer.py
            - src/optimization_model/SCUC_solver/scuc_model_builder.py
            - src/optimization_model/SCUC_solver/solve_instances.py
            - src/optimization_model/helpers/__init__.py
            - src/optimization_model/helpers/perf_logger.py
            - src/optimization_model/helpers/restore_solution.py
            - src/optimization_model/helpers/run_utils.py
            - src/optimization_model/helpers/save_json_solution.py
            - src/optimization_model/helpers/save_solution.py
            - src/optimization_model/helpers/verify_solution.py
                    - src/optimization_model/solver/economic_dispatch/constraints/__init__.py
                    - src/optimization_model/solver/economic_dispatch/constraints/commitment_fixing.py
                    - src/optimization_model/solver/economic_dispatch/constraints/linking.py
                    - src/optimization_model/solver/economic_dispatch/constraints/power_balance_segmented.py
                    - src/optimization_model/solver/economic_dispatch/data/load.py
                    - src/optimization_model/solver/economic_dispatch/objectives/__init__.py
                    - src/optimization_model/solver/economic_dispatch/objectives/power_cost_segmented.py
                    - src/optimization_model/solver/economic_dispatch/vars/__init__.py
                    - src/optimization_model/solver/economic_dispatch/vars/commitment.py
                    - src/optimization_model/solver/economic_dispatch/vars/segment_power.py
                    - src/optimization_model/solver/scuc/constraints/__init__.py
                    - src/optimization_model/solver/scuc/constraints/commitment_fixing.py
                    - src/optimization_model/solver/scuc/constraints/contingencies.py
                    - src/optimization_model/solver/scuc/constraints/initial_conditions.py
                    - src/optimization_model/solver/scuc/constraints/line_flow_ptdf.py
                    - src/optimization_model/solver/scuc/constraints/line_limits.py
                    - src/optimization_model/solver/scuc/constraints/linking.py
                    - src/optimization_model/solver/scuc/constraints/min_up_down.py
                    - src/optimization_model/solver/scuc/constraints/power_balance_segmented.py
                    - src/optimization_model/solver/scuc/constraints/reserve.py
                    - src/optimization_model/solver/scuc/constraints/reserve_requirement.py
                    - src/optimization_model/solver/scuc/data/load.py
                    - src/optimization_model/solver/scuc/objectives/__init__.py
                    - src/optimization_model/solver/scuc/objectives/base_overflow_penalty.py
                    - src/optimization_model/solver/scuc/objectives/contingency_overflow_penalty.py
                    - src/optimization_model/solver/scuc/objectives/minimum_output_cost.py
                    - src/optimization_model/solver/scuc/objectives/power_cost_segmented.py
                    - src/optimization_model/solver/scuc/objectives/reserve_shortfall_penalty.py
                    - src/optimization_model/solver/scuc/objectives/segment_power_cost.py
                    - src/optimization_model/solver/scuc/objectives/startup_cost.py
                    - src/optimization_model/solver/scuc/vars/__init__.py
                    - src/optimization_model/solver/scuc/vars/commitment.py
                    - src/optimization_model/solver/scuc/vars/contingency_overflow.py
                    - src/optimization_model/solver/scuc/vars/contingency_redispatch.py
                    - src/optimization_model/solver/scuc/vars/line_flow.py
                    - src/optimization_model/solver/scuc/vars/reserve.py
                    - src/optimization_model/solver/scuc/vars/segment_power.py
                    - src/optimization_model/solver/scuc/vars/startup_shutdown.py

## File Contents

### File: `README.md`

```
# SCUC

This repository provides a minimal, modular implementation to load standard UnitCommitment.jl benchmark instances and solve a simplified Security Constrained Unit Commitment variant: a segmented Economic Dispatch with unit commitment, reserves, and base-case transmission constraints using PTDF (ISF). Network PTDF/LODF matrices are computed from the input data.

Highlights
- Data: automatic download and parsing of UnitCommitment.jl JSON instances
- PTDF/LODF: robust construction that handles parallel lines and explicit reference-bus mapping
- Model: segmented costs, binary commitment, and system-wide power balance
- Reserves with shortfall penalty; transmission constraints with overflow slacks
- N-1 security constraints for contingencies
  - Line-outage constraints using LODF against emergency limits
  - Generator-outage constraints using ISF (PTDF) against emergency limits
- Batch solver to download, solve, and save JSON solutions for multiple instances
- ML warm-starts: k-NN warm start based on historical solutions per case; robust “fixed/repair” mode to make starts solver-friendly

## Quick start

1) Create and activate a virtual environment (optional but recommended)
- Python 3.9+ is required.

2) Install dependencies
- Gurobi requires a valid license. If you do not have one, install and license Gurobi first.
- Then install the Python packages:

```bash
pip install gurobipy numpy scipy requests tqdm scikit-learn torch torch-geometric
```

3) Solve selected instances in batch and save JSON outputs
- Example: solve all instances that contain case57 in their path.
- Solutions are saved under src/data/output mirroring the input path and without .gz extension. For example:
  - input:  src/data/input/matpower/case300/2017-06-24.json.gz
  - output: src/data/output/matpower/case300/2017-06-24.json

```bash
python -m src.optimization_model.SCUC_solver.optimizer --include case57 --time-limit 600 --mip-gap 0.05
```

Useful flags (defaults keep everything OFF):
- --instances ...       explicit dataset names
- --include ...         tokens to match dataset names
- --limit N             solve at most N instances
- --time-limit 600      Gurobi time limit [s]
- --mip-gap 0.05        Gurobi MIP gap
- --skip-existing       do not re-solve if a JSON already exists
- --dry-run             list instances and exit
- --warm                enable warm start (OFF by default)
- --require-pretrained  only use pre-trained warm-start index
- --warm-mode fixed     warm-start mode: fixed (recommended), commit-only, or as-is
- --warm-use-train-db   restrict neighbor DB to the training split
- --save-logs           write human-readable solution and verification logs to src/data/logs (OFF by default)

4) (Optional) Single known instance
- You can also solve a single instance by name:

```bash
python -m src.optimization_model.SCUC_solver.optimizer \
  --instances matpower/case57/2017-06-24 \
  --time-limit 600 --mip-gap 0.05
```

## General pipeline: compare raw Gurobi vs warm-start

We provide a convenience script to run the full pipeline for a case folder (e.g., matpower/case57):
- Stage 1: solve TRAIN split with raw Gurobi (no warm start) and save outputs
- Stage 2: pretrain warm-start model from those outputs
- Stage 3: compare on TEST split raw vs warm-start, recording speed and accuracy, and verifying feasibility of warm-start runs.

Run:

```bash
# Compare on case57 with default splits (70/15/15) and common limits
python -m src.optimization_model.SCUC_solver.compare_warm_start \
  --case matpower/case57 \
  --time-limit 600 --mip-gap 0.05 \
  --train-ratio 0.70 --val-ratio 0.15 \
  --save-logs
```

Notes:
- The script downloads instances as needed, saves raw JSON solutions to src/data/output, builds a warm-start index from TRAIN outputs, and then evaluates warm vs raw on TEST.
- It saves a CSV with per-instance results under:
  src/data/output/<case_folder>/compare_matpower_case57_<timestamp>.csv
- “Feasibility” for warm-start runs is verified using the built-in checker (no strict feasibility if slacks are used; the checker ensures constraints are respected given slacks).

## Warm-start tools

- Pretrain warm-start indexes from your saved outputs:
```bash
# Pretrain a specific case
python -m src.ml_models.pretrain_warm_start --cases matpower/case57

# Auto-discover cases with outputs and pretrain
python -m src.ml_models.pretrain_warm_start --auto-cases
```

- Generate warm-start files per split:
```bash
python -m src.ml_models.warm_start --case matpower/case57 --pretrain --report
python -m src.ml_models.warm_start --case matpower/case57 --generate-for train
```

- Solve with warm start (using a pretrained model), restricting neighbors to the training DB:
```bash
python -m src.optimization_model.SCUC_solver.optimizer \
  --include case57 \
  --warm --require-pretrained --warm-use-train-db \
  --warm-mode fixed \
  --time-limit 600 --mip-gap 0.05
```

## What gets downloaded and where

Instances are fetched on demand from:
- https://axavier.org/UnitCommitment.jl/0.4/instances

They are cached under:
- src/data/input

Solutions are saved as JSON under:
- src/data/output

Warm starts are saved under:
- src/data/intermediate/warm_start

Logs are stored under:
- src/data/logs

Example mapping:
- running with name "matpower/case300/2017-06-24" stores
  - src/data/input/matpower/case300/2017-06-24.json.gz (downloaded)
  - src/data/output/matpower/case300/2017-06-24.json (solution for ML)
  - src/data/intermediate/warm_start/warm_matpower_case300_2017_06_24.json
  - src/data/intermediate/warm_start/warm_fixed_matpower_case300_2017_06_24.json (if fixed)

## Repository layout

- src/data_preparation
  - ... (unchanged)
- src/optimization_model/SCUC_solver
  - scuc_model_builder.py: builds a segmented SCUC by composing modular components
  - optimizer.py: simple solver CLI (all options OFF by default, logs OFF by default)
  - compare_warm_start.py: pipeline to benchmark warm-start vs raw on train/test splits
  - solve_instances.py: batch solver (remote listing, download, solve, JSON save)
  - fix_warm_start.py (in ml_models): repair/generate robust warm-start JSONs
- src/ml_models
  - warm_start.py: k-NN warm-start provider (pretrain, generate warm files, apply to model)
  - pretrain_warm_start.py: convenience tool to prebuild per-case indexes
- src/optimization_model/helpers
  - save_json_solution.py: serialize solutions as JSON under src/data/output
  - verify_solution.py: verify model solution (now with in-memory verification function) and write a report if requested

```

### File: `drafts/SCUC_v3.tex`

```
\documentclass[12pt, a4paper]{article}

% Preamble: Load necessary packages for formatting
\usepackage{amsmath, amsfonts, amssymb} % For advanced math typesetting
\usepackage{geometry} % For setting page margins
\usepackage{url} % For formatting URLs in the bibliography

% Set page geometry
\geometry{a4paper, margin=1in}

\begin{document}

\begin{abstract}
This document provides a comprehensive mathematical formulation for the Security-Constrained Unit Commitment (SCUC) problem. The model is presented as a Mixed-Integer Linear Program (MILP) designed to minimize total system operational costs while adhering to a wide range of physical, operational, and security constraints. It includes detailed models for thermal generators, profiled generators (renewables), and energy storage units. Special emphasis is given to the formulation of transmission network constraints, presenting both the B-Theta (voltage angle) method and the more advanced Shift Factor (PTDF/LODF) method for ensuring N-1 security. The formulation is presented with unique equation numbering, integrated scientific explanations, and descriptions of calculations. An appendix details the sources of input data and the mathematical derivation of key network parameters.
\end{abstract}

\section{Nomenclature}

\subsection{Sets}
\begin{itemize}
\item $T$: Set of time periods in the planning horizon.
\item $G^{\text{Thermal}}$: Set of thermal generators.
\item $G^{\text{Profiled}}$: Set of profiled (renewable) generators.
\item $S$: Set of energy storage units.
\item $B$: Set of buses in the network.
\item $L$: Set of transmission lines.
\item $K_g$: Set of piecewise-linear cost segments for thermal generator $g$.
\item $M_g$: Set of startup cost categories for thermal generator $g$.
\item $R$: Set of reserve requirements (e.g., spinning reserves).
\item $G_r$: Subset of thermal generators eligible to provide reserve $r$.
\end{itemize}

\subsection{Parameters}
\begin{itemize}
\item $P_g^{\min}, P_g^{\max}$: Minimum and maximum power output for thermal generator $g$ (MW).
\item $C_g^{\min}$: No-load or minimum power cost for thermal generator $g$ (\$/period).
\item $C_{g,k}$: Marginal cost of segment $k$ for thermal generator $g$ (\$/MW).
\item $P_{g,k}$: Power breakpoint for segment $k$ of thermal generator $g$ (MW).
\item $S_{g,m}$: Startup cost for category $m$ of thermal generator $g$ (\$).
\item $D_{g,m}$: Downtime delay threshold for startup category $m$ (periods).
\item $R_g^{\text{up}}, R_g^{\text{down}}$: Ramp-up and ramp-down rates for thermal generator $g$ (MW/period).
\item $SU_g, SD_g$: Startup and shutdown power limits for thermal generator $g$ (MW).
\item $UT_g, DT_g$: Minimum up and down times for thermal generator $g$ (periods).
\item $\text{initial status}_g$: Initial commitment status duration for thermal generator $g$ (periods).
\item $\text{initial power}_g$: Initial power output for thermal generator $g$ (MW).
\item $\text{must run}_{g,t}$: Must-run flag for thermal generator $g$ at time $t$ (binary).
\item $P_{g,t}^{\min}, P_{g,t}^{\max}$: Minimum and maximum power for profiled generator $g$ at time $t$ (MW).
\item $C_{g,t}$: Marginal cost for profiled generator $g$ at time $t$ (\$/MW).
\item $L_{s,t}^{\min}, L_{s,t}^{\max}$: Minimum and maximum energy levels for storage $s$ at time $t$ (MWh).
\item $R_{s,t}^{\min\_charge}, R_{s,t}^{\max\_charge}$: Minimum and maximum charging rates for storage $s$ at time $t$ (MW).
\item $R_{s,t}^{\min\_discharge}, R_{s,t}^{\max\_discharge}$: Minimum and maximum discharging rates for storage $s$ at time $t$ (MW).
\item $\eta_{s,t}^{\text{charge}}, \eta_{s,t}^{\text{discharge}}$: Charging and discharging efficiencies for storage $s$ at time $t$.
\item $\lambda_s$: Loss factor for storage $s$ (fraction/period).
\item $C_{s,t}^{\text{charge}}, C_{s,t}^{\text{discharge}}$: Charging cost and discharging revenue for storage $s$ at time $t$ (\$/MW).
\item $D_{b,t}$: Load demand at bus $b$ at time $t$ (MW).
\item $R_t$: Reserve requirement at time $t$ (MW).
\item $B_l$: Susceptance of transmission line $l$ (S).
\item $F_{l,t}^{\text{normal}}, F_{l,t}^{\text{emergency}}$: Normal and emergency flow limits for line $l$ at time $t$ (MW).
\item $\text{PTDF}_{l,b}$: Power Transfer Distribution Factor for line $l$ and bus $b$.
\item $\text{LODF}_{l,c}$: Line Outage Distribution Factor for monitored line $l$ and contingent line $c$.
\item $\text{time\_step}$: Duration of each time period (hours).
\item $\pi^{\text{reserve}}, \pi^{\text{balance}}, \pi^{\text{flow}}$: Penalty costs for reserve shortfall, power imbalance, and flow violations (\$/MW).
\end{itemize}

\subsection{Variables}
\begin{itemize}
\item $x_{g,t}$: Binary commitment status for thermal generator $g$ at time $t$.
\item $p_{g,t}$: Power output for generator $g$ at time $t$ (MW).
\item $y_{g,t,k}$: Power in segment $k$ for thermal generator $g$ at time $t$ (MW).
\item $u_{g,t}, w_{g,t}$: Binary startup and shutdown indicators for thermal generator $g$ at time $t$.
\item $u_{g,t,m}$: Binary startup in category $m$ for thermal generator $g$ at time $t$.
\item $r_{g,t}$: Reserve provided by thermal generator $g$ at time $t$ (MW).
\item $l_{s,t}$: Energy level in storage $s$ at time $t$ (MWh).
\item $c_{s,t}, d_{s,t}$: Charging and discharging power for storage $s$ at time $t$ (MW).
\item $z_{s,t}$: Binary charge/discharge mode for storage $s$ at time $t$.
\item $\theta_{b,t}$: Voltage angle at bus $b$ at time $t$ (radians).
\item $s_{t}^{\text{reserve}}$: Reserve shortfall at time $t$ (MW).
\item $s_{b,t}^{\text{balance}}$: Power balance slack at bus $b$ at time $t$ (MW).
\item $s_{l,t}^{\text{flow}+}, s_{l,t}^{\text{flow}-}$: Positive and negative flow violation slacks for line $l$ at time $t$ (MW).
\end{itemize}

\section{Objective Function}
The objective of the SCUC problem is to minimize the total operational cost of the power system over a given planning horizon. This objective function, formulated as a summation of distinct economic components, drives the optimization towards the most economically efficient schedule that satisfies all operational and security requirements \cite{Wood2013}. The total cost to be minimized is expressed as:
\begin{align}
\min \quad & \sum_{t \in T} \sum_{g \in G^{\text{Thermal}}} \left( C_g^{\min} x_{g,t} + \sum_{k \in K_g} C_{g,k} y_{g,t,k} + \sum_{m \in M_g} S_{g,m} u_{g,t,m} \right) \label{eq:obj_thermal} \\
& + \sum_{t \in T} \sum_{g \in G^{\text{Profiled}}} C_{g,t} p_{g,t} \label{eq:obj_profiled} \\
& + \sum_{t \in T} \sum_{s \in S} \left( C_{s,t}^{\text{charge}} c_{s,t} - C_{s,t}^{\text{discharge}} d_{s,t} \right) \label{eq:obj_storage} \\
& + \sum_{t \in T} \pi^{\text{reserve}} s_{t}^{\text{reserve}} + \sum_{t \in T} \sum_{b \in B} \pi^{\text{balance}} s_{b,t}^{\text{balance}} + \sum_{t \in T} \sum_{l \in L} \pi^{\text{flow}} s_{l,t}^{\text{flow}} \label{eq:obj_penalty}
\end{align}
The first component in line \eqref{eq:obj_thermal} represents the total cost of operating thermal generators. This includes the no-load or minimum power cost ($C_g^{\min} x_{g,t}$), calculated as the fixed cost at the minimum generation level when the unit is committed; the variable production cost, modeled as a piecewise-linear function ($\sum C_{g,k} y_{g,t,k}$), where each segment's marginal cost $C_{g,k}$ is derived as the slope between consecutive breakpoints in the production cost curve (i.e., $C_{g,k} = (C_{g,k} - C_{g,k-1}) / (P_{g,k} - P_{g,k-1})$); and the startup costs ($\sum S_{g,m} u_{g,t,m}$), where multiple categories $m$ allow for downtime-dependent costs, with $S_{g,m}$ directly from input data and $u_{g,t,m}$ selecting the appropriate category based on prior shutdown duration. Line \eqref{eq:obj_profiled} captures the marginal cost of dispatching profiled generators, such as wind and solar, often zero but representing market offers if provided. The economics of energy storage are modeled in line \eqref{eq:obj_storage}, accounting for charging costs and discharging revenues to enable arbitrage. Finally, line \eqref{eq:obj_penalty} applies high penalty costs to slack variables for reserve shortfalls, power imbalances, and flow limit violations, making them economically prohibitive to enforce constraints softly.

\section{Constraints}

\subsection{Thermal Generator Constraints}
The operational characteristics of conventional thermal power plants are governed by a set of constraints modeling their physical and mechanical limitations.
\begin{subequations}
\begin{align}
& P_{g}^{\min} x_{g,t} \leq p_{g,t} \leq P_{g}^{\max} x_{g,t} && \forall g \in G^{\text{Thermal}}, t \in T \label{eq:gen_power_limits}
\end{align}
Equation \eqref{eq:gen_power_limits} is the fundamental power limit constraint, linking the binary commitment decision ($x_{g,t}$) to the continuous power output ($p_{g,t}$). It ensures that if a generator is off ($x_{g,t}=0$), its power output is zero; if it is on ($x_{g,t}=1$), its output is bounded by its minimum stable generation level ($P_g^{\min}$) and maximum capacity ($P_g^{\max}$), both directly from the production cost curve endpoints.

\begin{align}
& p_{g,t} = P_{g}^{\min} x_{g,t} + \sum_{k \in K_g} y_{g,t,k} && \forall g \in G^{\text{Thermal}}, t \in T \label{eq:pwl_sum} \\
& 0 \leq y_{g,t,k} \leq (P_{g,k} - P_{g,k-1}) x_{g,t} && \forall g \in G^{\text{Thermal}}, t \in T, k \in K_g \label{eq:pwl_bounds}
\end{align}
To accurately model the generator's non-linear cost curve, equations \eqref{eq:pwl_sum} and \eqref{eq:pwl_bounds} implement a piecewise-linear approximation. The former decomposes the total power output into the minimum generation level plus the sum of outputs from each cost segment ($y_{g,t,k}$); the latter limits the power in each segment to its width ($P_{g,k} - P_{g,k-1}$), scaled by commitment status.

\begin{align}
& p_{g,t} - p_{g,t-1} \leq \min(R_{g}^{\text{up}}, SU_g) \cdot \text{time\_step} && \forall g \in G^{\text{Thermal}}, t \in T, t>1 \label{eq:ramp_up} \\
& p_{g,t-1} - p_{g,t} \leq \min(R_{g}^{\text{down}}, SD_g) \cdot \text{time\_step} && \forall g \in G^{\text{Thermal}}, t \in T, t>1 \label{eq:ramp_down}
\end{align}
The ramping constraints in \eqref{eq:ramp_up} and \eqref{eq:ramp_down} model the physical inertia of large turbines. They limit the rate of change of power generation between consecutive time periods to the minimum of the unit's ramp rate ($R_g^{\text{up/down}}$) and startup/shutdown limits ($SU_g/SD_g$), scaled by the time step duration.

\begin{align}
& \sum_{\tau = t - UT_{g} + 1}^{t} u_{g,\tau} \leq x_{g,t} && \forall g \in G^{\text{Thermal}}, t \geq UT_g \label{eq:min_up} \\
& \sum_{\tau = t - DT_{g} + 1}^{t} w_{g,\tau} \leq 1 - x_{g,t} && \forall g \in G^{\text{Thermal}}, t \geq DT_g \label{eq:min_down}
\end{align}
Equations \eqref{eq:min_up} and \eqref{eq:min_down} enforce minimum uptime ($UT_g$) and downtime ($DT_g$) requirements by summing startup/shutdown indicators over rolling windows and bounding them by the commitment status.

\begin{align}
& u_{g,t} - w_{g,t} = x_{g,t} - x_{g,t-1} && \forall g \in G^{\text{Thermal}}, t \in T, t>1 \label{eq:su_sd_logic} \\
& u_{g,t} + w_{g,t} \leq 1 && \forall g \in G^{\text{Thermal}}, t \in T \label{eq:su_sd_exclusive}
\end{align}
The logical constraints \eqref{eq:su_sd_logic} and \eqref{eq:su_sd_exclusive} link commitment changes to startup/shutdown events, ensuring $u_{g,t}=1$ only on on-transitions and $w_{g,t}=1$ on off-transitions, with no simultaneous events.

\begin{align}
& u_{g,t} = \sum_{m \in M_g} u_{g,t,m} && \forall g \in G^{\text{Thermal}}, t \in T \label{eq:startup_category_sum} \\
& \sum_{\tau=t - D_{g,m} + 1}^{t-1} (1 - x_{g,\tau}) \geq D_{g,m-1} u_{g,t,m} && \forall g \in G^{\text{Thermal}}, t \in T, m > 1 \label{eq:startup_category_select}
\end{align}
For multi-category startups, \eqref{eq:startup_category_sum} aggregates category-specific startups, and \eqref{eq:startup_category_select} selects the category $m$ based on prior downtime exceeding the lower threshold $D_{g,m-1}$ but not $D_{g,m}$, where $D_{g,m}$ are sorted delay steps. This calculation determines the applicable startup cost based on how long the unit has been offline.

\begin{align}
& x_{g,0} = 1 && \text{if } \text{initial status}_g > 0, && \forall g \in G^{\text{Thermal}} \label{eq:initial_commit} \\
& p_{g,0} = \text{initial power}_g && \forall g \in G^{\text{Thermal}} \label{eq:initial_power} \\
& x_{g,t} = 1 && \text{if } \text{must run}_{g,t} = \text{True}, && \forall g \in G^{\text{Thermal}},\ \forall t \in T \label{eq:must_run}
\end{align}
Initial conditions are enforced by \eqref{eq:initial_commit} and \eqref{eq:initial_power}, fixing pre-horizon status and power. Must-run requirements are added via \eqref{eq:must_run}, ensuring the unit remains committed in specified periods.
\end{subequations}

\subsection{Profiled Generator Constraints}
\begin{align}
& P_{g,t}^{\min} \leq p_{g,t} \leq P_{g,t}^{\max} && \forall g \in G^{\text{Profiled}}, t \in T \label{eq:profiled_limits}
\end{align}
The constraint in \eqref{eq:profiled_limits} governs the output of profiled resources like wind and solar. Their maximum available power ($P_{g,t}^{\max}$) is determined by an external forecast. This inequality ensures their dispatched power ($p_{g,t}$) does not exceed what is available, allowing for curtailment if necessary.

\subsection{Energy Storage Constraints}
\begin{subequations}
\begin{align}
& l_{s,t} = l_{s,t-1} (1 - \lambda_s) + \left( \eta_{s,t}^{\text{charge}} c_{s,t} - \frac{d_{s,t}}{\eta_{s,t}^{\text{discharge}}} \right) \cdot \text{time\_step} && \forall s \in S, t \in T, t>1 \label{eq:storage_balance}
\end{align}
Equation \eqref{eq:storage_balance} calculates the state-of-charge by updating the previous level (adjusted for self-discharge loss $\lambda_s$) with the net energy input/output, scaled by efficiencies and time step. This inter-temporal link enables energy shifting across periods.

\begin{align}
& L_{s,t}^{\min} \leq l_{s,t} \leq L_{s,t}^{\max} && \forall s \in S, t \in T \label{eq:storage_level_limits} \\
& R_{s,t}^{\min\_charge} \leq c_{s,t} \leq R_{s,t}^{\max\_charge} && \forall s \in S, t \in T \label{eq:storage_charge_limits} \\
& R_{s,t}^{\min\_discharge} \leq d_{s,t} \leq R_{s,t}^{\max\_discharge} && \forall s \in S, t \in T \label{eq:storage_discharge_limits}
\end{align}
These inequalities enforce physical bounds on energy storage levels and charge/discharge rates, preventing overcharge or depletion.

\begin{align}
& c_{s,t} \leq R_{s,t}^{\text{max\_charge}} \cdot z_{s,t} && \forall s \in S, t \in T \label{eq:storage_simul_charge} \\
& d_{s,t} \leq R_{s,t}^{\text{max\_discharge}} \cdot (1 - z_{s,t}) && \forall s \in S, t \in T \label{eq:storage_simul_discharge}
\end{align}
These constraints use the binary variable $z_{s,t}$ to enforce mutual exclusivity between charging and discharging modes in each period, reflecting hardware limitations in most storage systems.
\end{subequations}

\subsection{System-Wide and Security Constraints}
\begin{subequations}
\begin{align}
& \sum_{g \in G, g.\text{bus}=b} p_{g,t} + \sum_{s \in S, s.\text{bus}=b} (d_{s,t} - c_{s,t}) - \sum_{l \in L} \text{Flow}_{l,b,t} = D_{b,t} - s_{b,t}^{\text{balance}} && \forall b \in B, t \in T \label{eq:power_balance}
\end{align}
The nodal power balance equation \eqref{eq:power_balance} mandates that injected power (generation + discharge) minus withdrawn power (load + charge + outgoing flows) equals zero at each bus, with a slack variable for imbalances penalized in the objective.

\begin{align}
& \sum_{g \in G_r} r_{g,t} \geq R_{t} - s_{t}^{\text{reserve}} && \forall t \in T, r \in R \label{eq:reserve_req} \\
& r_{g,t} \leq P_{g}^{\max} - p_{g,t} && \forall g \in G_r, t \in T \label{eq:reserve_limit}
\end{align}
Reserves are summed over eligible units $G_r$ (from input data), ensuring the total meets the requirement $R_t$ with shortfall slack, and limited by each unit's headroom (available capacity above scheduled power).
\end{subequations}

\subsubsection{Transmission Security: B-Theta Formulation} \label{sec:btheta}
This is a classic "DC power flow" approximation that uses bus voltage angles as explicit variables.
\begin{subequations}
\begin{align}
& -F_{l,t}^{\text{normal}} \leq B_{l} (\theta_{i,t} - \theta_{j,t}) \leq F_{l,t}^{\text{normal}} && \forall l=(i,j) \in L, t \in T \label{eq:btheta_base} \\
& \theta_{b_{\text{ref}},t} = 0 && \forall t \in T \label{eq:btheta_ref}
\end{align}
In this formulation, equation \eqref{eq:btheta_base} models the power flow on a line as being proportional to the difference in voltage angles ($\theta$) at its terminal buses, scaled by the line's susceptance ($B_l$), and constrains this flow to be within the line's normal thermal limit. Equation \eqref{eq:btheta_ref} sets the angle of a reference bus to zero to provide a fixed reference for the system.
\end{subequations}

\subsubsection{Transmission Security: Shift Factor (PTDF/LODF) Formulation} \label{sec:ptdf}
This advanced formulation uses pre-computed sensitivity factors to model network flows efficiently, which is essential for SCUC. The derivation of these factors is detailed in Appendix \ref{app:data}.
\begin{align}
& P_{\text{net},b,t} = \left(\sum_{g \in G, g.\text{bus}=b} p_{g,t} + \sum_{s \in S, s.\text{bus}=b} (d_{s,t} - c_{s,t})\right) - D_{b,t} && \forall b \in B, t \in T \label{eq:net_injection}
\end{align}
First, equation \eqref{eq:net_injection} defines the net power injection at each bus, consolidating all generation, storage activity, and demand into a single term per bus. This simplifies subsequent flow calculations by aggregating nodal contributions.
\begin{subequations}
\begin{align}
& -F_{l,t}^{\text{normal}} + s_{l,t}^{\text{flow}-} \geq \sum_{b \in B} \text{PTDF}_{l,b} \cdot P_{\text{net},b,t} \geq F_{l,t}^{\text{normal}} - s_{l,t}^{\text{flow}+} && \forall l \in L, t \in T \label{eq:ptdf_base}
\end{align}
The base-case transmission constraint is then expressed in \eqref{eq:ptdf_base} using Power Transfer Distribution Factors (PTDFs). A PTDF is a sensitivity factor that quantifies the change in power flow on line $l$ resulting from a 1 MW injection at bus $b$ (and withdrawal at the reference bus) \cite{Christie2000}. This equation uses the principle of superposition to calculate the total flow on line $l$ as the sum of impacts from all nodal net injections, bounded by normal limits with slacks for violations.

\begin{align}
& \begin{aligned}
-F_{l,t}^{\text{emergency}} \leq & \left( \sum_{b \in B} \text{PTDF}_{l,b} \cdot P_{\text{net},b,t} \right) \\ 
& + \text{LODF}_{l,c} \cdot \left( \sum_{b \in B} \text{PTDF}_{c,b} \cdot P_{\text{net},b,t} \right) \leq F_{l,t}^{\text{emergency}}
\end{aligned} && \forall l,c \in L, l \neq c, t \in T \label{eq:lodf_contingency}
\end{align}
Finally, equation \eqref{eq:lodf_contingency} represents the N-1 security constraint, the cornerstone of the SCUC model. It uses Line Outage Distribution Factors (LODFs) to ensure the system remains secure after the failure of any single line $c$ \cite{Tejada2018}. An LODF quantifies how the pre-outage flow on the failed line $c$ redistributes onto a monitored line $l$. This equation calculates the post-contingency flow on line $l$ by adding its base-case flow (first term) to the redistributed flow from the failed line $c$ (second term), and ensures this new flow does not exceed the line's emergency rating.
\end{subequations}

\appendix
\section{Data Sourcing and Parameter Derivation} \label{app:data}
The parameters used in the SCUC model are sourced from a variety of public and private datasets, or are derived from fundamental network properties.

\subsection{Generator, Load, and System Data}
\begin{itemize}
    \item \textbf{Generator Data:} Physical parameters such as power limits, ramp rates, and minimum up/down times are typically provided by generator owners. Cost data, including startup costs and piecewise-linear production cost curves, are derived from market offers submitted by generation companies to the Independent System Operator (ISO) \cite{CAISO2009}, \cite{PJMData}.
    \item \textbf{Load Forecast Data:} Nodal load forecasts ($D_{b,t}$) are produced by ISOs using sophisticated statistical models that incorporate historical load, weather forecasts, and economic activity \cite{LoadForecast2020}. Publicly available forecast data can often be obtained from ISO websites such as ERCOT and PJM \cite{ERCOTData}, \cite{PJMData}.
    \item \textbf{Network Data:} The transmission network topology, including bus connections and the electrical properties (reactance, resistance) of lines, is maintained by the ISO. Thermal ratings for lines ($F_{l}^{\text{normal}}, F_{l}^{\text{emergency}}$) are also determined by the transmission owners.
\end{itemize}

\subsection{Derivation of Shift Factors}
The PTDF and LODF matrices are not primary data but are pre-calculated sensitivity factors derived from the network's physical topology and impedances, based on the DC power flow approximation \cite{Christie2000}.

\subsubsection{Power Transfer Distribution Factors (PTDFs)}
The PTDF matrix is derived from the bus admittance matrix of the network. The DC power flow model provides a linear relationship between the vector of bus power injections $\mathbf{P}$ and the vector of bus voltage angles $\boldsymbol{\theta}$:
\begin{align}
\mathbf{P} = \mathbf{B'} \boldsymbol{\theta} \label{eq:dc_pf}
\end{align}
where $\mathbf{B'}$ is the DC bus admittance matrix, with the row and column corresponding to a reference (slack) bus removed. By inverting this matrix, we obtain the bus reactance matrix $\mathbf{X} = (\mathbf{B'})^{-1}$.

The power flow on a line $l$ from bus $i$ to bus $j$ with reactance $x_l$ is $F_l = (1/x_l) (\theta_i - \theta_j)$. The PTDF for line $l$ with respect to a power injection at bus $b$ and withdrawal at the reference bus can be calculated directly from the reactance matrix \cite{Guver2006}:
\begin{align}
\text{PTDF}_{l,b} = \frac{1}{x_l} (X_{ib} - X_{jb}) \label{eq:ptdf_calc}
\end{align}
This calculation is performed for each line-bus pair, resulting in a matrix used to compute flows as linear combinations of net injections.

\subsubsection{Line Outage Distribution Factors (LODFs)}
The LODF for a monitored line $l$ with respect to the outage of a contingent line $c$ (connecting buses $m$ and $n$), denoted $\text{LODF}_{l,c}$, is the fraction of the pre-outage flow on line $c$ that is redistributed onto line $l$. LODFs can be calculated directly from the PTDF matrix without re-solving the power flow for each contingency. The formula is given by \cite{Guo2009}, \cite{Ronellenfitsch2017}:
\begin{align}
\text{LODF}_{l,c} = \frac{\text{PTDF}_{l,(m,n)}}{1 - \text{PTDF}_{c,(m,n)}} \label{eq:lodf_calc}
\end{align}
where $\text{PTDF}_{l,(m,n)}$ is the PTDF on line $l$ for a 1 MW transfer between the terminal buses of the outaged line $c$. This pre-computation, as implemented in the accompanying code, enables efficient enforcement of N-1 security constraints.

\begin{thebibliography}{99}

\bibitem{Xavier2024}
A. S. Xavier, A. M. Kazachkov, O. Yurdakul, J. He, and F. Qiu, ``UnitCommitment.jl: A Julia/JuMP Optimization Package for Security-Constrained Unit Commitment (Version 0.4),'' \emph{Zenodo}, 2024. [Online]. Available: \url{https://doi.org/10.5281/zenodo.4269874}

\bibitem{Wood2013}
A. J. Wood, B. F. Wollenberg, and G. B. Sheblé, \emph{Power Generation, Operation, and Control}, 3rd ed. Hoboken, NJ, USA: John Wiley \& Sons, 2013.

\bibitem{Tejada2018}
D. A. Tejada-Arango, P. Sánchez-Martín, and A. Ramos, ``Security constrained unit commitment using line outage distribution factors,'' \emph{IEEE Transactions on Power Systems}, vol. 33, no. 1, pp. 329--337, Jan. 2018.

\bibitem{Christie2000}
R. D. Christie, B. F. Wollenberg, and I. Wangensteen, ``Transmission management in the deregulated environment,'' \emph{Proceedings of the IEEE}, vol. 88, no. 2, pp. 170--195, Feb. 2000.

\bibitem{Merlin1983}
A. Merlin and P. Sandrin, ``A new method for unit commitment at Electricite de France,'' \emph{IEEE Transactions on Power Apparatus and Systems}, vol. PAS-102, no. 5, pp. 1218--1225, May 1983.

\bibitem{Guo2009}
J. Guo and L. Mili, ``Direct calculation of line outage distribution factors,'' \emph{IEEE Transactions on Power Systems}, vol. 24, no. 3, pp. 1633--1634, Aug. 2009.

\bibitem{Ronellenfitsch2017}
H. Ronellenfitsch, D. Manik, J. Horsch, T. Brown, and D. Witthaut, ``A cycle-based approach to calculating power transfer distribution factors,'' \emph{IEEE Transactions on Power Systems}, vol. 32, no. 2, pp. 1255--1264, Mar. 2017.

\bibitem{Guver2006}
T. Guler, G. Gross, and M. Liu, ``Generalized line outage distribution factors,'' \emph{IEEE Power Engineering Society General Meeting}, 2006.

\bibitem{CAISO2009}
California ISO, ``Technical Bulletin: Market Optimization Details,'' Nov. 2009. [Online]. Available: \url{https://www.caiso.com/Documents/TechnicalBulletin-MarketOptimizationDetails.pdf}

\bibitem{PJMData}
PJM Interconnection, ``Data Viewer,'' 2025. [Online]. Available: \url{https://dataviewer.pjm.com/dataviewer/pages/public/load.jsf}

\bibitem{LoadForecast2020}
IBM, ``Load Forecasting,'' 2020. [Online]. Available: \url{https://www.ibm.com/think/topics/load-forecasting}

\bibitem{ERCOTData}
Electric Reliability Council of Texas, ``Load Forecast,'' 2025. [Online]. Available: \url{https://www.ercot.com/gridinfo/load/forecast}

\bibitem{Xie2022}
J. Xie et al., ``Security-Constrained Unit Commitment for Electricity Market: Modeling, Solution Methods, and Future Challenges,'' \emph{IEEE Transactions on Power Systems}, vol. 38, no. 3, pp. 2173--2189, May 2023.

\bibitem{Alqurashi2017}
A. Alqurashi et al., ``Security Constrained Unit Commitment (SCUC) formulation and its solving methodology,'' \emph{Journal of King Saud University - Engineering Sciences}, vol. 29, no. 4, pp. 339--346, Oct. 2017.

\bibitem{Xie2022b}
J. Xie et al., ``Security-Constrained Unit Commitment for Electricity Market: Modeling, Solution Methods, and Future Challenges,'' \emph{NREL Technical Report}, 2022.

\bibitem{Sharma2023}
S. Sharma and S. Chansareewittaya, ``Security-constrained unit commitment: A decomposition approach solving single-period models,'' \emph{European Journal of Operational Research}, vol. 312, no. 2, pp. 639--653, 2023.

\bibitem{Atakan2019}
S. Atakan et al., ``Learning to Solve Large-Scale Security-Constrained Unit Commitment Problems,'' \emph{arXiv preprint arXiv:1902.01697}, 2019.

\bibitem{Castelli}
M. Castelli et al., ``Solving the Security Constrained Unit Commitment problem using Integer Linear Programming,'' \emph{Technical Report}, Carnegie Mellon University.

\bibitem{Xie2022c}
J. Xie et al., ``Security-Constrained Unit Commitment for Electricity Market: Modeling, Solution Methods, and Future Challenges,'' \emph{NREL Technical Report}, 2022.

\bibitem{McCalley}
J. McCalley et al., ``Security Constrained Economic Dispatch Calculation,'' Iowa State University Notes.

\bibitem{Pan2016}
K. Pan and Y. Guan, ``Adaptive Robust Optimization for the Security Constrained Unit Commitment Problem,'' \emph{MIT Sloan Working Paper}, 2016.

\bibitem{Abunima2021}
H. Abunima et al., ``A Comprehensive Review of Security-constrained Unit Commitment,'' \emph{Journal of Modern Power Systems and Clean Energy}, vol. 10, no. 3, pp. 562--579, May 2022.

\end{thebibliography}

\end{document}
```

### File: `ml/src/__init__.py`

```
from .model import EdgeModel
from .graph import build_grid_graph, plot_grid_graph
from .utils import compute_dc_power_flow
```

### File: `ml/src/graph.py`

```
import numpy as np
import matplotlib.pyplot as plt

import json
import networkx as nx

def build_grid_graph(file):

    buses = list(file["Buses"].keys())
    bus_idx = {b: i for i, b in enumerate(buses)}

    G = nx.Graph()

    for lid, line in file["Transmission lines"].items():
        u = bus_idx[line["Source bus"]]
        v = bus_idx[line["Target bus"]]
        G.add_edge(u, v)

    return G, bus_idx


def plot_grid_graph(G, bus_idx, seed = 42):

    pos = nx.spring_layout(G, seed=seed)

    nx.draw_networkx_nodes(G, pos, node_size=300)
    nx.draw_networkx_edges(G, pos, width=1.2)
    inv_idx = {i: b for b, i in bus_idx.items()}
    
    nx.draw_networkx_labels(G, pos, labels=inv_idx, font_size=8)
    plt.title(f"Case {len(bus_idx)}")
    plt
    plt.show()

```

### File: `ml/src/model.py`

```
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class EdgeModel(nn.Module):
    
    def __init__(self, node_in_dim, edge_in_dim):
        super().__init__()
        self.node_conv1 = SAGEConv(node_in_dim, 64)
        self.node_conv2 = SAGEConv(64, 64)
        self.edge_mlp = nn.Sequential(
            nn.Linear(64*2 + edge_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        x = torch.relu(self.node_conv1(x, edge_index))
        x = torch.relu(self.node_conv2(x, edge_index))

        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst], edge_attr], dim=1)  

        out = self.edge_mlp(edge_features)
        return torch.sigmoid(out).squeeze(-1)
    

class GraphTrainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        loader_train,
        loader_val,
        epochs,
        device = "cpu",
        threshold = 0.7,
    ):
        self.model = model.to(device)
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.device = device
        self.epochs = epochs
        self.threshold = threshold

        self.optimizer = optimizer
        self.criterion  = criterion

    def _train_epoch(self):
        self.model.train()
        
        tot_loss = 0.0
        num_batches = 0

        for g in self.loader_train:
            g = g.to(self.device)
            self.optimizer.zero_grad()
            
            y_pred = self.model(g.x, g.edge_index, g.edge_attr)
            y_true = g.y.float()
            loss = self.criterion(y_pred, y_true)

            loss.backward()
            self.optimizer.step()

            tot_loss += loss.item()
            num_batches += 1

        return tot_loss / num_batches


    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        y_true = [] 
        y_prob = []

        for g in self.loader_val:
            g = g.to(self.device)
            p_hat = self.model(g.x, g.edge_index, g.edge_attr)
            y_true.append(g.y.cpu())
            y_prob.append(p_hat.cpu())

        y_true = torch.cat(y_true)
        y_prob  = torch.cat(y_prob)
        y_pred = (y_prob > self.threshold).int()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return acc, prec, recall, f1
       
       
    @torch.no_grad()    
    def predict(self, loader):
        self.model.eval()
        preds, trues = [], []

        for g in loader:
            g = g.to(self.device)
            pred = self.model(g.x, g.edge_index, g.edge_attr)
            preds.append(pred.cpu())
            trues.append(g.y.cpu())

        return torch.cat(preds), torch.cat(trues)


    def fit(self):

        for epoch in range(self.epochs):
            loss = self._train_epoch()
            acc, prec, recall, f1 = self._validate()

            print(
                f"Epoch {epoch:03d} | loss={loss:.4f} | "
                f"acc={acc:.3f} | prec={prec:.3f} | rec={recall:.3f} | f1={f1:.3f}"
            )
```

### File: `ml/src/utils.py`

```
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def compute_dc_power_flow(data):
    buses = data["Buses"]
    lines = data["Transmission lines"]
    gens  = data["Generators"]

    bus_ids = list(buses.keys())
    bus_idx = {bus: i for i, bus in enumerate(bus_ids)}
    n = len(bus_ids)
    m = len(lines)

    rows, cols, vals, x_vals = [], [], [], []

    for iline, line in enumerate(lines.values()):
        i = bus_idx[line["Source bus"]]
        j = bus_idx[line["Target bus"]]
        x = line["Reactance (ohms)"]

        rows += [iline, iline]
        cols += [i, j]
        vals += [1, -1]
        x_vals.append(x)

    A = coo_matrix((vals, (rows, cols)), shape=(m, n)).tocsc()
    B_diag = np.array([1 / x for x in x_vals])
    B = A.T @ coo_matrix((B_diag, (range(m), range(m))), shape=(m, m)) @ A

    P = np.zeros(n)
    for g in gens.values():
        b = bus_idx[g["Bus"]]
        P[b] += g["Initial power (MW)"]

    for b_id, b in buses.items():
        b_idx = bus_idx[b_id]
        load = b.get("Load (MW)", 0.0)
        if isinstance(load, list):
            load = load[0]
        P[b_idx] -= load

    B_red = B[1:, 1:]
    P_red = P[1:]

    theta = np.zeros(n)
    theta[1:] = spsolve(B_red, P_red)

    flows = []
    for line in lines.values():
        i = bus_idx[line["Source bus"]]
        j = bus_idx[line["Target bus"]]
        x = line["Reactance (ohms)"]
        flows.append((theta[i] - theta[j]) / x)

    return np.abs(np.array(flows))

```

### File: `res_compare.md`

```
Below are the updated files. Copy-paste them into your repository in the indicated paths.

File: src/optimization_model/SCUC_solver/compare_ml_raw.py
--------------------------------------------------------------------------------
"""
Benchmark pipeline: compare raw Gurobi vs ML-based (warm-start + redundant-constraint pruning).

What it does:
- Stage 1 (TRAIN): solve with raw Gurobi, save outputs (skip if --skip-existing).
- Stage 2: pretrain warm-start index from TRAIN outputs.
- Stage 2.5 (optional): pretrain redundancy index from TRAIN outputs.
- Stage 3 (TEST): for each instance, solve RAW and WARM(+optional pruning);
  verify feasibility and write:
    • a per-case results CSV under src/data/output/<case>/compare_<tag>_<timestamp>.csv
    • a general logs CSV under src/data/logs/compare_logs_<tag>_<timestamp>.csv
      with (instance, split, method, num vars, num constrs, runtime, max constraint violation, etc.)

Notes:
- Redundancy pruning controls (all OFF unless you pass --rc-enable):
    --rc-enable               Enable pruning
    --rc-thr-abs 20.0         Absolute margin threshold [MW]
    --rc-thr-rel 0.10         Relative margin threshold (fraction of emergency limit)
    --rc-use-train-db         Restrict redundancy k-NN to TRAIN split (default True)
"""

from __future__ import annotations

import argparse
import csv
import math
import time
import requests
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gurobipy import GRB

from src.data_preparation.params import DataParams
from src.data_preparation.read_data import read_benchmark
from src.optimization_model.SCUC_solver.scuc_model_builder import build_model
from src.optimization_model.SCUC_solver.solve_instances import (
    list_remote_instances,
    list_local_cached_instances,
)
from src.optimization_model.helpers.save_json_solution import (
    save_solution_as_json,
    compute_output_path,
)
from src.optimization_model.helpers.verify_solution import (
    verify_solution,
    verify_solution_to_log,
)
from src.optimization_model.helpers.run_utils import allocate_run_id, make_log_filename
from src.ml_models.warm_start import WarmStartProvider, _hash01
from src.ml_models.redundant_constraints import (
    RedundancyProvider as RCProvider,
)  # ML-based constraint pruning provider


def _status_str(code: int) -> str:
    return DataParams.SOLVER_STATUS_STR.get(code, f"STATUS_{code}")


def _case_tag(case_folder: str) -> str:
    t = case_folder.strip().strip("/\\").replace("\\", "/")
    return "".join(ch if ch.isalnum() else "_" for ch in t).strip("_").lower()


def _list_case_instances(case_folder: str) -> List[str]:
    items = list_remote_instances(
        include_filters=[case_folder], roots=["matpower", "test"], max_depth=4
    )
    items = [x for x in items if x.startswith(case_folder)]
    if not items:
        items = list_local_cached_instances(include_filters=[case_folder])
        items = [x for x in items if x.startswith(case_folder)]
    return sorted(set(items))


def _split_instances(
    names: List[str], train_ratio: float, val_ratio: float, seed: int
) -> Tuple[List[str], List[str], List[str]]:
    tr: List[str] = []
    va: List[str] = []
    te: List[str] = []
    for nm in sorted(names):
        r = _hash01(nm, seed)
        if r < train_ratio:
            tr.append(nm)
        elif r < train_ratio + val_ratio:
            va.append(nm)
        else:
            te.append(nm)
    return tr, va, te


@dataclass
class SolveResult:
    instance_name: str
    split: str
    method: str  # raw | warm
    status: str
    status_code: int
    runtime_sec: float
    mip_gap: Optional[float]
    obj_val: Optional[float]
    obj_bound: Optional[float]
    nodes: Optional[float]
    feasible_ok: Optional[bool]
    warm_start_applied_vars: Optional[int] = None
    # New fields for logging/analysis
    num_vars: Optional[int] = None
    num_bin_vars: Optional[int] = None
    num_int_vars: Optional[int] = None
    num_constrs: Optional[int] = None
    max_constraint_violation: Optional[float] = None


def _metrics_from_model(
    model,
) -> Tuple[
    int, float, Optional[float], Optional[float], Optional[float], Optional[float]
]:
    code = int(getattr(model, "Status", -1))
    runtime = float(getattr(model, "Runtime", 0.0) or 0.0)
    try:
        mip_gap = float(model.MIPGap)
    except Exception:
        mip_gap = None
    try:
        obj_val = float(model.ObjVal)
    except Exception:
        obj_val = None
    try:
        obj_bound = float(model.ObjBound)
    except Exception:
        obj_bound = None
    try:
        nodes = float(getattr(model, "NodeCount", 0.0))
    except Exception:
        nodes = None
    return code, runtime, mip_gap, obj_val, obj_bound, nodes


def _model_size(model) -> Tuple[int, int, int, int]:
    """
    Return (num_vars, num_bin_vars, num_int_vars, num_constrs)
    """
    try:
        nvars = int(getattr(model, "NumVars", 0))
    except Exception:
        nvars = 0
    try:
        nbin = int(getattr(model, "NumBinVars", 0))
    except Exception:
        nbin = 0
    try:
        nint = int(getattr(model, "NumIntVars", 0))
    except Exception:
        nint = 0
    try:
        ncons = int(getattr(model, "NumConstrs", 0))
    except Exception:
        ncons = 0
    return nvars, nbin, nint, ncons


def _max_constraint_violation(checks) -> Optional[float]:
    """
    Compute max violation among constraint checks (IDs starting with 'C-').
    Returns None if checks missing; 0.0 if all within tolerance.
    """
    if not checks:
        return None
    vals: List[float] = []
    for c in checks:
        try:
            if isinstance(c.idx, str) and c.idx.startswith("C-"):
                v = float(c.value)
                if math.isfinite(v):
                    vals.append(v)
        except Exception:
            continue
    return max(vals) if vals else 0.0


def _save_logs_if_requested(sc, model, save_logs: bool) -> None:
    if not save_logs:
        return
    run_id = allocate_run_id(sc.name or "scenario")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        sol_fname = make_log_filename("solution", sc.name, run_id, ts)
        _ = Path(DataParams._LOGS / sol_fname)
        from src.optimization_model.helpers.save_solution import (
            save_solution_to_log as _save,
        )

        _save(sc, model, filename=sol_fname)
    except Exception:
        pass
    try:
        ver_fname = make_log_filename("verify", sc.name, run_id, ts)
        verify_solution_to_log(sc, model, filename=ver_fname)
    except Exception:
        pass


def _instance_cache_path_and_url(instance_name: str) -> Tuple[Path, str]:
    gz_name = f"{instance_name}.json.gz"
    local_path = (DataParams._CACHE / gz_name).resolve()
    url = f"{DataParams.INSTANCES_URL.rstrip('/')}/{gz_name}"
    return local_path, url


def _robust_download(instance_name: str, attempts: int, timeout: int) -> bool:
    """
    Try to download instance_name.json.gz to the input cache with retry and timeout.
    Returns True on success, False on final failure.
    """
    local_path, url = _instance_cache_path_and_url(instance_name)
    if local_path.is_file():
        return True

    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")

    backoff_base = 2.0
    for k in range(1, max(1, attempts) + 1):
        try:
            with requests.get(url, stream=True, timeout=max(1, int(timeout))) as r:
                r.raise_for_status()
                with tmp_path.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
            tmp_path.replace(local_path)
            return True
        except Exception:
            if k < attempts:
                sleep_s = min(60.0, backoff_base ** (k - 1))
                time.sleep(sleep_s)
            else:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                return False
    return False


def _prepare_results_csv(case_folder: str) -> Path:
    """
    Create CSV file (under src/data/output/<case>/) and write header. Return path.
    """
    tag = _case_tag(case_folder)
    out_dir = (DataParams._OUTPUT / case_folder).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"compare_{tag}_{ts}.csv"

    header = [
        "timestamp",
        "instance_name",
        "split",
        "method",
        "status",
        "status_code",
        "runtime_sec",
        "mip_gap",
        "obj_val",
        "obj_bound",
        "nodes",
        "feasible_ok",
        "warm_start_applied_vars",
        "num_vars",
        "num_bin_vars",
        "num_int_vars",
        "num_constrs",
        "max_constraint_violation",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
    return path


def _append_result_to_csv(csv_path: Path, r: SolveResult) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                now,
                r.instance_name,
                r.split,
                r.method,
                r.status,
                r.status_code,
                f"{r.runtime_sec:.6f}",
                "" if r.mip_gap is None else f"{r.mip_gap:.8f}",
                "" if r.obj_val is None else f"{r.obj_val:.6f}",
                "" if r.obj_bound is None else f"{r.obj_bound:.6f}",
                "" if r.nodes is None else f"{r.nodes:.0f}",
                "" if r.feasible_ok is None else ("OK" if r.feasible_ok else "FAIL"),
                ""
                if r.warm_start_applied_vars is None
                else int(r.warm_start_applied_vars),
                "" if r.num_vars is None else int(r.num_vars),
                "" if r.num_bin_vars is None else int(r.num_bin_vars),
                "" if r.num_int_vars is None else int(r.num_int_vars),
                "" if r.num_constrs is None else int(r.num_constrs),
                ""
                if r.max_constraint_violation is None
                else f"{float(r.max_constraint_violation):.8f}",
            ]
        )


def _prepare_logs_csv(case_folder: str) -> Path:
    """
    Create a general logs CSV under src/data/logs and write header. Return path.
    Includes general info that the user requested (instance, method, sizes, times, violations).
    """
    tag = _case_tag(case_folder)
    logs_dir = DataParams._LOGS.resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = logs_dir / f"compare_logs_{tag}_{ts}.csv"
    header = [
        "timestamp",
        "instance_name",
        "split",
        "method",
        "status",
        "runtime_sec",
        "obj_val",
        "num_vars",
        "num_constrs",
        "num_bin_vars",
        "num_int_vars",
        "max_constraint_violation",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(header)
    return path


def _append_result_to_logs_csv(csv_path: Path, r: SolveResult) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                now,
                r.instance_name,
                r.split,
                r.method,
                r.status,
                f"{r.runtime_sec:.6f}",
                "" if r.obj_val is None else f"{r.obj_val:.6f}",
                "" if r.num_vars is None else int(r.num_vars),
                "" if r.num_constrs is None else int(r.num_constrs),
                "" if r.num_bin_vars is None else int(r.num_bin_vars),
                "" if r.num_int_vars is None else int(r.num_int_vars),
                ""
                if r.max_constraint_violation is None
                else f"{float(r.max_constraint_violation):.8f}",
            ]
        )


def _build_contingency_filter(
    rc: Optional[RCProvider],
    sc,
    instance_name: str,
    enable: bool,
    thr_abs: float,
    thr_rel: float,
    use_train_db: bool,
) -> Optional[callable]:
    if not enable or rc is None:
        return None
    try:
        res = rc.make_filter_for_instance(
            sc,
            instance_name,
            thr_abs=thr_abs,
            thr_rel=thr_rel,
            use_train_index_only=use_train_db,
            exclude_self=True,
        )
        if res is None:
            return None
        predicate, stats = res
        print(
            f"[redundancy] Using neighbor {stats.get('neighbor')} (dist={stats.get('distance'):.3f}); "
            f"constraints skipped will be counted in builder logs."
        )
        return predicate
    except Exception:
        return None


def _solve_raw(
    instance_name: str,
    time_limit: int,
    mip_gap: float,
    split: str,
    save_logs: bool,
    skip_existing: bool,
    download_attempts: int,
    download_timeout: int,
    *,
    rc_provider: Optional[RCProvider] = None,
    rc_enable: bool = False,
    rc_thr_abs: float = 20.0,
    rc_thr_rel: float = 0.10,
    rc_use_train_db: bool = True,
) -> SolveResult:
    out_json_path = compute_output_path(instance_name)
    if skip_existing and out_json_path.is_file():
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="raw",
            status="SKIPPED_EXISTING",
            status_code=-2,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )

    ok_dl = _robust_download(
        instance_name=instance_name,
        attempts=max(1, int(download_attempts)),
        timeout=max(1, int(download_timeout)),
    )
    if not ok_dl:
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="raw",
            status="DOWNLOAD_FAIL",
            status_code=-3,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )

    try:
        inst = read_benchmark(instance_name, quiet=True)
    except Exception:
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="raw",
            status="READ_FAIL",
            status_code=-4,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )

    sc = inst.deterministic
    cont_filter = _build_contingency_filter(
        rc_provider,
        sc,
        instance_name,
        rc_enable,
        rc_thr_abs,
        rc_thr_rel,
        rc_use_train_db,
    )
    model = build_model(scenario=sc, contingency_filter=cont_filter)
    try:
        model.Params.OutputFlag = 1
        model.Params.MIPGap = mip_gap
        model.Params.TimeLimit = time_limit
        model.Params.NumericFocus = 1
    except Exception:
        pass
    model.optimize()

    try:
        save_solution_as_json(sc, model, instance_name=instance_name)
    except Exception:
        pass

    feasible_ok = None
    max_viol = None
    try:
        feasible_ok, checks, _ = verify_solution(sc, model)
        max_viol = _max_constraint_violation(checks)
    except Exception:
        feasible_ok = None
        max_viol = None

    _save_logs_if_requested(sc, model, save_logs)

    code, runtime, gap, obj, bound, nodes = _metrics_from_model(model)
    nvars, nbin, nint, ncons = _model_size(model)
    return SolveResult(
        instance_name=instance_name,
        split=split,
        method="raw",
        status=_status_str(code),
        status_code=code,
        runtime_sec=runtime,
        mip_gap=gap,
        obj_val=obj,
        obj_bound=bound,
        nodes=nodes,
        feasible_ok=feasible_ok,
        warm_start_applied_vars=None,
        num_vars=nvars,
        num_bin_vars=nbin,
        num_int_vars=nint,
        num_constrs=ncons,
        max_constraint_violation=max_viol,
    )


def _solve_warm(
    instance_name: str,
    wsp: WarmStartProvider,
    time_limit: int,
    mip_gap: float,
    split: str,
    warm_mode: str,
    save_logs: bool,
    skip_existing: bool,
    download_attempts: int,
    download_timeout: int,
    *,
    rc_provider: Optional[RCProvider] = None,
    rc_enable: bool = False,
    rc_thr_abs: float = 20.0,
    rc_thr_rel: float = 0.10,
    rc_use_train_db: bool = True,
) -> SolveResult:
    out_json_path = compute_output_path(instance_name)
    if skip_existing and out_json_path.is_file():
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="warm",
            status="SKIPPED_EXISTING",
            status_code=-2,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )

    ok_dl = _robust_download(
        instance_name=instance_name,
        attempts=max(1, int(download_attempts)),
        timeout=max(1, int(download_timeout)),
    )
    if not ok_dl:
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="warm",
            status="DOWNLOAD_FAIL",
            status_code=-3,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )

    try:
        inst = read_benchmark(instance_name, quiet=True)
    except Exception:
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="warm",
            status="READ_FAIL",
            status_code=-4,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )
    sc = inst.deterministic

    try:
        wsp.generate_and_save_warm_start(
            instance_name, use_train_index_only=True, exclude_self=True, auto_fix=True
        )
    except Exception:
        pass

    cont_filter = _build_contingency_filter(
        rc_provider,
        sc,
        instance_name,
        rc_enable,
        rc_thr_abs,
        rc_thr_rel,
        rc_use_train_db,
    )
    model = build_model(scenario=sc, contingency_filter=cont_filter)

    mode = warm_mode.strip().lower()
    if mode == "fixed":
        mode = "repair"
    applied = 0
    try:
        applied = wsp.apply_warm_start_to_model(model, sc, instance_name, mode=mode)
    except Exception:
        applied = 0

    try:
        model.Params.OutputFlag = 1
        model.Params.MIPGap = mip_gap
        model.Params.TimeLimit = time_limit
        model.Params.NumericFocus = 1
    except Exception:
        pass

    model.optimize()

    try:
        save_solution_as_json(sc, model, instance_name=instance_name)
    except Exception:
        pass

    feasible_ok = None
    max_viol = None
    try:
        feasible_ok, checks, _ = verify_solution(sc, model)
        max_viol = _max_constraint_violation(checks)
    except Exception:
        feasible_ok = None
        max_viol = None

    _save_logs_if_requested(sc, model, save_logs)

    code, runtime, gap, obj, bound, nodes = _metrics_from_model(model)
    nvars, nbin, nint, ncons = _model_size(model)
    return SolveResult(
        instance_name=instance_name,
        split=split,
        method="warm",
        status=_status_str(code),
        status_code=code,
        runtime_sec=runtime,
        mip_gap=gap,
        obj_val=obj,
        obj_bound=bound,
        nodes=nodes,
        feasible_ok=feasible_ok,
        warm_start_applied_vars=applied,
        num_vars=nvars,
        num_bin_vars=nbin,
        num_int_vars=nint,
        num_constrs=ncons,
        max_constraint_violation=max_viol,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Compare raw Gurobi vs warm-start (+ optional redundancy pruning) on TRAIN/TEST for a case."
    )
    ap.add_argument("--case", required=True, help="Case folder, e.g., matpower/case118")
    ap.add_argument(
        "--time-limit",
        type=int,
        default=600,
        help="Time limit [s] for both RAW and WARM",
    )
    ap.add_argument(
        "--mip-gap",
        type=float,
        default=0.05,
        help="Relative MIP gap for both RAW and WARM",
    )
    ap.add_argument(
        "--train-ratio", type=float, default=0.70, help="Train ratio for split"
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation ratio; remainder is test",
    )
    ap.add_argument("--seed", type=int, default=42, help="Split seed")
    ap.add_argument(
        "--limit-train",
        type=int,
        default=0,
        help="Limit number of TRAIN instances (0 => no limit)",
    )
    ap.add_argument(
        "--limit-test",
        type=int,
        default=0,
        help="Limit number of TEST instances (0 => no limit)",
    )
    ap.add_argument(
        "--save-logs",
        action="store_true",
        default=False,
        help="Save human logs to src/data/logs (solution and verification reports)",
    )
    ap.add_argument(
        "--warm-mode",
        choices=["fixed", "commit-only", "as-is"],
        default="fixed",
        help="Warm-start application mode for evaluation",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="TRAIN ONLY: skip instances that already have a JSON solution in src/data/output",
    )
    ap.add_argument(
        "--download-attempts",
        type=int,
        default=3,
        help="Number of attempts to download a missing instance (default: 3)",
    )
    ap.add_argument(
        "--download-timeout",
        type=int,
        default=60,
        help="Per-attempt HTTP timeout for instance download in seconds (default: 60)",
    )
    # Redundancy pruning
    ap.add_argument(
        "--rc-enable",
        action="store_true",
        default=False,
        help="Enable ML-based redundant contingency pruning (OFF by default)",
    )
    ap.add_argument(
        "--rc-thr-abs",
        type=float,
        default=20.0,
        help="Absolute margin threshold [MW] to prune a constraint (default 20 MW)",
    )
    ap.add_argument(
        "--rc-thr-rel",
        type=float,
        default=0.10,
        help="Relative margin threshold (fraction of F_em) to prune (default 0.10)",
    )
    ap.add_argument(
        "--rc-use-train-db",
        action="store_true",
        default=True,
        help="Restrict redundancy k-NN to TRAIN split (default True)",
    )

    args = ap.parse_args()

    case_folder = args.case.strip().strip("/\\").replace("\\", "/")
    all_instances = _list_case_instances(case_folder)
    if not all_instances:
        print(f"No instances found for {case_folder}.")
        return

    train, val, test = _split_instances(
        all_instances, args.train_ratio, args.val_ratio, args.seed
    )
    if not train:
        print("Empty TRAIN split; cannot pretrain warm-start index. Aborting.")
        return
    if not test:
        print("Empty TEST split; nothing to compare. Aborting.")
        return

    if args.limit_train and args.limit_train > 0:
        train = train[: args.limit_train]
    if args.limit_test and args.limit_test > 0:
        test = test[: args.limit_test]

    print(f"Case: {case_folder}")
    print(f"- Train: {len(train)}")
    print(f"- Val  : {len(val)}")
    print(f"- Test : {len(test)}")
    if args.skip_existing:
        print(
            "Note: --skip-existing applies to TRAIN stage only. TEST runs will re-solve even if outputs exist."
        )

    # Prepare CSVs
    csv_path = _prepare_results_csv(case_folder)
    logs_csv_path = _prepare_logs_csv(case_folder)
    print(f"Results will be appended to: {csv_path}")
    print(f"General logs CSV will be appended to: {logs_csv_path}")

    results: List[SolveResult] = []

    # Stage 1: RAW Train (allow skipping existing)
    print("Stage 1: Solving TRAIN with raw Gurobi (no warm start) ...")
    for nm in train:
        r = _solve_raw(
            nm,
            args.time_limit,
            args.mip_gap,
            split="train",
            save_logs=args.save_logs,
            skip_existing=args.skip_existing,
            download_attempts=args.download_attempts,
            download_timeout=args.download_timeout,
            rc_provider=None,
            rc_enable=False,  # Do not prune while building train outputs
        )
        _append_result_to_csv(csv_path, r)
        _append_result_to_logs_csv(logs_csv_path, r)
        results.append(r)

    # Stage 2: Pretrain warm-start index from outputs (existing solutions are used)
    print("Stage 2: Pretraining warm-start index from TRAIN outputs ...")
    wsp = WarmStartProvider(case_folder=case_folder)
    wsp.pretrain(force=True)
    trained, cov = wsp.ensure_trained(case_folder, allow_build_if_missing=False)
    print(f"- Warm-start index: trained={trained}, coverage={cov:.3f}")
    if not trained:
        print(
            "Index not trained (insufficient outputs). Evaluation will still try warm, but coverage may be low."
        )

    # Stage 2.5: Pretrain redundancy index from TRAIN outputs (if enabled)
    rc_provider = None
    if args.rc_enable:
        print("Stage 2.5: Pretraining redundancy index from TRAIN outputs ...")
        rc_provider = RCProvider(case_folder=case_folder)
        rc_provider.pretrain(force=True)
        ok, rc_cov = rc_provider.ensure_trained(
            case_folder, allow_build_if_missing=False
        )
        print(f"- Redundancy index: available={ok}, coverage={rc_cov:.3f}")
        if not ok:
            print("Redundancy index not available. Pruning will be OFF.")
            rc_provider = None

    # Stage 3: Compare on TEST (RAW vs WARM) — always re-solve; ignore skip-existing
    print("Stage 3: Comparing on TEST (RAW vs WARM) ...")
    for nm in test:
        # RAW
        r_raw = _solve_raw(
            nm,
            args.time_limit,
            args.mip_gap,
            split="test",
            save_logs=args.save_logs,
            skip_existing=False,
            download_attempts=args.download_attempts,
            download_timeout=args.download_timeout,
            rc_provider=rc_provider,
            rc_enable=args.rc_enable,
            rc_thr_abs=args.rc_thr_abs,
            rc_thr_rel=args.rc_thr_rel,
            rc_use_train_db=args.rc_use_train_db,
        )
        _append_result_to_csv(csv_path, r_raw)
        _append_result_to_logs_csv(logs_csv_path, r_raw)
        results.append(r_raw)

        # WARM
        r_warm = _solve_warm(
            nm,
            wsp,
            args.time_limit,
            args.mip_gap,
            split="test",
            warm_mode=args.warm_mode,
            save_logs=args.save_logs,
            skip_existing=False,
            download_attempts=args.download_attempts,
            download_timeout=args.download_timeout,
            rc_provider=rc_provider,
            rc_enable=args.rc_enable,
            rc_thr_abs=args.rc_thr_abs,
            rc_thr_rel=args.rc_thr_rel,
            rc_use_train_db=args.rc_use_train_db,
        )
        _append_result_to_csv(csv_path, r_warm)
        _append_result_to_logs_csv(logs_csv_path, r_warm)
        results.append(r_warm)

    print(f"Results appended to: {csv_path}")
    print(f"General logs appended to: {logs_csv_path}")

    # Simple summary
    test_pairs = {}
    for r in results:
        if r.split != "test":
            continue
        test_pairs.setdefault(r.instance_name, {})[r.method] = r
    speed_wins = 0
    obj_wins = 0
    count_pairs = 0
    for nm, pair in test_pairs.items():
        if "raw" in pair and "warm" in pair:
            if pair["raw"].status not in (
                "SKIPPED_EXISTING",
                "DOWNLOAD_FAIL",
                "READ_FAIL",
            ) and pair["warm"].status not in (
                "SKIPPED_EXISTING",
                "DOWNLOAD_FAIL",
                "READ_FAIL",
            ):
                count_pairs += 1
                if pair["warm"].runtime_sec < pair["raw"].runtime_sec:
                    speed_wins += 1
                if (pair["warm"].obj_val is not None) and (
                    pair["raw"].obj_val is not None
                ):
                    if pair["warm"].obj_val <= pair["raw"].obj_val + 1e-6:
                        obj_wins += 1
    if count_pairs > 0:
        print(
            f"Warm faster on {speed_wins}/{count_pairs} test instances; objective <= raw on {obj_wins}/{count_pairs}."
        )


if __name__ == "__main__":
    main()
--------------------------------------------------------------------------------


New file: explainer.md
--------------------------------------------------------------------------------
What was changed and why

Goal
- Provide a complete pipeline to compare raw Gurobi SCUC runs vs ML-enhanced runs (warm start and redundant constraint pruning), then verify solutions and log key metrics, including constraints violations, as requested.

Key updates in src/optimization_model/SCUC_solver/compare_ml_raw.py
1) Fixed an erroneous import
   - Removed a stale alias import from a non-existent module (src.ml_models.redundancy). The actual provider class already exists in src.ml_models.redundant_constraints, and we import it as RCProvider.

2) Added general logging CSV under src/data/logs
   - The script now creates and appends to a CSV in src/data/logs:
     compare_logs_<case_tag>_<timestamp>.csv
   - Columns: timestamp, instance_name, split, method, status, runtime_sec, obj_val, num_vars, num_constrs, num_bin_vars, num_int_vars, max_constraint_violation.
   - This fulfills the requirement “after each instance is solved, save logs to src/data/logs csv table with general info.”

3) Extended the per-case results CSV
   - The existing per-case CSV in src/data/output/<case>/compare_<tag>_<ts>.csv now includes:
     num_vars, num_bin_vars, num_int_vars, num_constrs, max_constraint_violation.
   - This adds deeper analysis to the existing summary.

4) Captured model size and constraints violations
   - After each solve (raw/warm), the code extracts:
     • model size: NumVars, NumBinVars, NumIntVars, NumConstrs
     • max constraint violation using the built-in verification function
   - The verification function returns a list of CheckItem entries; the code computes the maximum violation among those whose IDs start with “C-” (constraints), as requested.

5) Verification integration
   - We already had in-memory verification via verify_solution (step 4*). We now also use it to compute max constraint violation; feasibility “OK/FAIL” is also recorded.

6) Optional ML-based redundancy pruning is integrated
   - If --rc-enable is passed, the script trains src/ml_models/redundant_constraints.py on TRAIN outputs, then applies pruning on TEST via a predicate passed to the SCUC builder. Thresholds are controlled by --rc-thr-abs and --rc-thr-rel.

7) Warm-start pipeline
   - The script pretrains the warm-start index from TRAIN outputs and uses it on TEST instances. Warm-start application prefers “fixed/repair” semantics when the mode is “fixed,” consistent with the robust warm-start repair logic.

8) Robust downloading and logging
   - Instances are downloaded with retry and timeout.
   - Human-readable solution and verification logs can still be written when --save-logs is passed.

How this maps to your requirements

1) Solve certain cases (case 118) by raw Gurobi, or get solved solutions from src/data/output
   - Pass --case matpower/case118. TRAIN stage solves raw; if --skip-existing is set, it uses existing JSONs from src/data/output.

2) Split solutions to train, validate, test
   - Deterministic split with ratios --train-ratio and --val-ratio, seed via --seed.

3) Train two models src/ml_models/redundant_constraints.py and src/ml_models/warm_start.py
   - Stage 2 pretrains warm-start.
   - Stage 2.5 optionally pretrains redundant-constraints (activated by --rc-enable).

4) Run with warm_start and deleted (pruned) constraints
   - TEST stage runs RAW and WARM; if --rc-enable, it also prunes contingency constraints based on the redundancy model.

4*) Check solution for feasibility src/optimization_model/helpers/verify_solution.py
   - After each solve, verify_solution() is called. We record feasibility and the maximum constraints violation.

5) Compare OF values, solution time, constraints violations
   - The per-case CSV and logs CSV include the objective, runtime, and max constraint violation.

6) After each instance is solved, save logs to src/data/logs CSV with general info
   - New logs CSV in src/data/logs captures: instance, split, method, size, runtime, max constraint violation.
   - The existing per-case CSV in src/data/output continues to provide a more detailed comparison.

How to run

Examples:
- Compare on case118 with warm-start and redundancy pruning, saving human-readable logs:
  python -m src.optimization_model.SCUC_solver.compare_ml_raw \
    --case matpower/case118 \
    --time-limit 600 --mip-gap 0.05 \
    --train-ratio 0.70 --val-ratio 0.15 \
    --rc-enable --rc-thr-abs 20 --rc-thr-rel 0.10 \
    --save-logs

- Compare on case118 without pruning:
  python -m src.optimization_model.SCUC_solver.compare_ml_raw \
    --case matpower/case118 \
    --time-limit 600 --mip-gap 0.05

Outputs
- Per-case results CSV:
  src/data/output/<case>/compare_<case_tag>_<timestamp>.csv
- General logs CSV (requested):
  src/data/logs/compare_logs_<case_tag>_<timestamp>.csv
- JSON solutions (for ML and reproducibility): src/data/output/<case>/<instance>.json
- Optional human-readable logs: src/data/logs/SCUC_<scenario>_runNNN_<timestamp>.solution.log and .verify.log

Notes
- Warm-start repair is used when you pass --warm-mode fixed (mapped to “repair” mode internally).
- Redundancy pruning uses nearest-neighbor margins from historical solutions to skip constraints that were safely slack with a comfortable margin.
--------------------------------------------------------------------------------

Summary of changes
- Modified: src/optimization_model/SCUC_solver/compare_ml_raw.py
- Added: explainer.md

These changes are self-contained and require no additional dependencies beyond what’s already specified in your repo.
```

### File: `src/__init__.py`

```
__all__ = []

```

### File: `src/data/logs/compare_logs_matpower_case118_20250907_152327.csv`

```
timestamp,instance_name,split,method,status,runtime_sec,obj_val,num_vars,num_constrs,num_bin_vars,num_int_vars,max_constraint_violation
2025-09-07 15:25:22,matpower/case118/2017-01-02,train,raw,OPTIMAL,55.090000,4844424.994625,49068,1597536,5832,5832,0.00000000
2025-09-07 15:27:08,matpower/case118/2017-01-05,train,raw,OPTIMAL,51.497000,6167079.101966,49068,1597536,5832,5832,0.00000000
2025-09-07 15:28:57,matpower/case118/2017-01-06,train,raw,OPTIMAL,52.439000,6043528.811815,49068,1597536,5832,5832,0.00000000
2025-09-07 15:30:43,matpower/case118/2017-01-07,train,raw,OPTIMAL,49.491000,6299208.822867,49068,1597536,5832,5832,0.00000000
2025-09-07 15:32:29,matpower/case118/2017-01-08,train,raw,OPTIMAL,51.063000,6384833.162047,49068,1597536,5832,5832,0.00000000
2025-09-07 15:34:17,matpower/case118/2017-01-11,train,raw,OPTIMAL,52.731000,5213619.814083,49068,1597536,5832,5832,0.00000000
2025-09-07 15:36:06,matpower/case118/2017-01-12,train,raw,OPTIMAL,51.634000,4984328.982647,49068,1597536,5832,5832,0.00000000
2025-09-07 15:37:55,matpower/case118/2017-01-15,train,raw,OPTIMAL,52.224000,4815570.136330,49068,1597536,5832,5832,0.00000000
2025-09-07 15:39:45,matpower/case118/2017-01-16,train,raw,OPTIMAL,52.766000,5480572.300569,49068,1597536,5832,5832,0.00000000
2025-09-07 15:41:35,matpower/case118/2017-01-17,train,raw,OPTIMAL,51.716000,5497060.046869,49068,1597536,5832,5832,0.00000000
2025-09-07 15:43:20,matpower/case118/2017-01-18,train,raw,OPTIMAL,46.575000,5393659.533144,49068,1597536,5832,5832,0.00000000
2025-09-07 15:45:00,matpower/case118/2017-01-19,train,raw,OPTIMAL,47.299000,5469032.706635,49068,1597536,5832,5832,0.00000000
2025-09-07 15:46:43,matpower/case118/2017-01-21,train,raw,OPTIMAL,46.435000,5011221.001296,49068,1597536,5832,5832,0.00000000
2025-09-07 15:48:34,matpower/case118/2017-01-22,train,raw,OPTIMAL,51.166000,5191372.832008,49068,1597536,5832,5832,0.00000000
2025-09-07 15:50:24,matpower/case118/2017-01-23,train,raw,OPTIMAL,53.137000,5868275.365160,49068,1597536,5832,5832,0.00000000
2025-09-07 15:52:13,matpower/case118/2017-01-26,train,raw,OPTIMAL,50.482000,5269491.012978,49068,1597536,5832,5832,0.00000000
2025-09-07 15:54:04,matpower/case118/2017-01-27,train,raw,OPTIMAL,52.845000,4828403.541371,49068,1597536,5832,5832,0.00000000
2025-09-07 15:55:53,matpower/case118/2017-01-31,train,raw,OPTIMAL,50.383000,5631243.555296,49068,1597536,5832,5832,0.00000000
2025-09-07 15:57:35,matpower/case118/2017-02-01,train,raw,OPTIMAL,48.977000,5728872.209972,49068,1597536,5832,5832,0.00000000
2025-09-07 15:59:26,matpower/case118/2017-02-02,train,raw,OPTIMAL,54.495000,5964267.232530,49068,1597536,5832,5832,0.00000000
2025-09-07 16:01:17,matpower/case118/2017-02-03,train,raw,OPTIMAL,53.007000,5857914.303498,49068,1597536,5832,5832,0.00000000
2025-09-07 16:03:08,matpower/case118/2017-02-05,train,raw,OPTIMAL,52.201000,5226625.044439,49068,1597536,5832,5832,0.00000000
2025-09-07 16:04:49,matpower/case118/2017-02-06,train,raw,OPTIMAL,46.947000,5337348.583769,49068,1597536,5832,5832,0.00000000
2025-09-07 16:06:28,matpower/case118/2017-02-08,train,raw,OPTIMAL,46.954000,5032278.961333,49068,1597536,5832,5832,0.00000000
2025-09-07 16:08:09,matpower/case118/2017-02-09,train,raw,OPTIMAL,45.544000,6543758.221165,49068,1597536,5832,5832,0.00000000
2025-09-07 16:09:49,matpower/case118/2017-02-11,train,raw,OPTIMAL,47.102000,5629484.377440,49068,1597536,5832,5832,0.00000000
2025-09-07 16:11:32,matpower/case118/2017-02-13,train,raw,OPTIMAL,50.509000,6487317.536677,49068,1597536,5832,5832,0.00000000
2025-09-07 16:13:16,matpower/case118/2017-02-14,train,raw,OPTIMAL,50.835000,6516423.306406,49068,1597536,5832,5832,0.00000000
2025-09-07 16:14:55,matpower/case118/2017-02-15,train,raw,OPTIMAL,47.938000,6058511.991679,49068,1597536,5832,5832,0.00000000
2025-09-07 16:16:40,matpower/case118/2017-02-16,train,raw,OPTIMAL,50.733000,6435605.444325,49068,1597536,5832,5832,0.00000000
2025-09-07 16:18:27,matpower/case118/2017-02-17,train,raw,OPTIMAL,50.999000,5172179.310743,49068,1597536,5832,5832,0.00000000
2025-09-07 16:20:19,matpower/case118/2017-02-18,train,raw,OPTIMAL,54.471000,4306920.128827,49068,1597536,5832,5832,0.00000000
2025-09-07 16:22:07,matpower/case118/2017-02-19,train,raw,OPTIMAL,50.094000,4181299.025532,49068,1597536,5832,5832,0.00000000
2025-09-07 16:23:54,matpower/case118/2017-02-20,train,raw,OPTIMAL,50.395000,4841724.610617,49068,1597536,5832,5832,0.00000000
2025-09-07 16:25:51,matpower/case118/2017-02-21,train,raw,OPTIMAL,55.093000,4949937.818419,49068,1597536,5832,5832,0.00000000
2025-09-07 16:27:42,matpower/case118/2017-02-23,train,raw,OPTIMAL,55.086000,4396036.178819,49068,1597536,5832,5832,0.00000000
2025-09-07 16:29:26,matpower/case118/2017-02-24,train,raw,OPTIMAL,50.703000,4062168.967666,49068,1597536,5832,5832,0.00000000
2025-09-07 16:31:17,matpower/case118/2017-02-26,train,raw,OPTIMAL,50.700000,3919573.303562,49068,1597536,5832,5832,0.00000000
2025-09-07 16:33:04,matpower/case118/2017-02-27,train,raw,OPTIMAL,51.598000,4576743.106220,49068,1597536,5832,5832,0.00000000
2025-09-07 16:34:49,matpower/case118/2017-03-01,train,raw,OPTIMAL,49.696000,4271665.729725,49068,1597536,5832,5832,0.00000000
2025-09-07 17:35:52,matpower/case118/2017-03-02,train,raw,TIME_LIMIT,3611.311000,37469411.810272,49068,1597536,5832,5832,0.00000000
2025-09-07 17:36:56,matpower/case118/2017-03-03,train,raw,OPTIMAL,30.794000,4575087.766783,49068,1597536,5832,5832,0.00000000
2025-09-07 17:38:03,matpower/case118/2017-03-05,train,raw,OPTIMAL,35.391000,4529769.924424,49068,1597536,5832,5832,0.00000000
2025-09-07 17:39:16,matpower/case118/2017-03-06,train,raw,OPTIMAL,34.486000,4199555.092940,49068,1597536,5832,5832,0.00000000
2025-09-07 17:40:57,matpower/case118/2017-03-08,train,raw,OPTIMAL,48.579000,4011638.378017,49068,1597536,5832,5832,0.00000000
2025-09-07 17:42:38,matpower/case118/2017-03-12,train,raw,OPTIMAL,48.008000,4973945.243072,49068,1597536,5832,5832,0.00000000
2025-09-07 17:44:21,matpower/case118/2017-03-13,train,raw,OPTIMAL,49.651000,5205316.056590,49068,1597536,5832,5832,0.00000000
2025-09-07 17:46:00,matpower/case118/2017-03-15,train,raw,OPTIMAL,46.031000,5716822.119742,49068,1597536,5832,5832,0.00000000
2025-09-07 17:47:48,matpower/case118/2017-03-16,train,raw,OPTIMAL,53.990000,5566720.195358,49068,1597536,5832,5832,0.00000000
2025-09-07 17:49:26,matpower/case118/2017-03-17,train,raw,OPTIMAL,45.756000,4566031.760944,49068,1597536,5832,5832,0.00000000
2025-09-07 17:51:06,matpower/case118/2017-03-18,train,raw,OPTIMAL,47.892000,3952126.170027,49068,1597536,5832,5832,0.00000000
2025-09-07 17:52:46,matpower/case118/2017-03-19,train,raw,OPTIMAL,46.852000,4254333.768088,49068,1597536,5832,5832,0.00000000
2025-09-07 17:54:25,matpower/case118/2017-03-21,train,raw,OPTIMAL,46.823000,4432492.031231,49068,1597536,5832,5832,0.00000000
2025-09-07 17:56:04,matpower/case118/2017-03-22,train,raw,OPTIMAL,47.094000,5235438.723326,49068,1597536,5832,5832,0.00000000
2025-09-07 17:57:49,matpower/case118/2017-03-24,train,raw,OPTIMAL,52.311000,4128644.461442,49068,1597536,5832,5832,0.00000000
2025-09-07 17:59:22,matpower/case118/2017-03-27,train,raw,OPTIMAL,45.238000,3912062.165023,49068,1597536,5832,5832,0.00000000
2025-09-07 18:00:34,matpower/case118/2017-03-28,train,raw,OPTIMAL,38.330000,3893639.718969,49068,1597536,5832,5832,0.00000000
2025-09-07 18:01:43,matpower/case118/2017-03-31,train,raw,OPTIMAL,35.702000,4343389.856328,49068,1597536,5832,5832,0.00000000
2025-09-07 18:02:54,matpower/case118/2017-04-01,train,raw,OPTIMAL,37.516000,3797531.537966,49068,1597536,5832,5832,0.00000000
2025-09-07 18:04:02,matpower/case118/2017-04-03,train,raw,OPTIMAL,34.919000,4207331.420769,49068,1597536,5832,5832,0.00000000
2025-09-07 18:05:11,matpower/case118/2017-04-04,train,raw,OPTIMAL,35.043000,4160707.421025,49068,1597536,5832,5832,0.00000000
2025-09-07 18:06:18,matpower/case118/2017-04-05,train,raw,OPTIMAL,33.562000,4171037.070334,49068,1597536,5832,5832,0.00000000
2025-09-07 18:07:26,matpower/case118/2017-04-06,train,raw,OPTIMAL,34.224000,4377494.529453,49068,1597536,5832,5832,0.00000000
2025-09-07 18:08:32,matpower/case118/2017-04-08,train,raw,OPTIMAL,32.845000,3839159.623285,49068,1597536,5832,5832,0.00000000
2025-09-07 18:09:38,matpower/case118/2017-04-09,train,raw,OPTIMAL,32.474000,3783646.272427,49068,1597536,5832,5832,0.00000000
2025-09-07 18:10:47,matpower/case118/2017-04-10,train,raw,OPTIMAL,35.507000,4148696.243938,49068,1597536,5832,5832,0.00000000
2025-09-07 18:11:54,matpower/case118/2017-04-11,train,raw,OPTIMAL,33.866000,4243802.950110,49068,1597536,5832,5832,0.00000000
2025-09-07 18:13:05,matpower/case118/2017-04-12,train,raw,OPTIMAL,36.950000,4104851.344508,49068,1597536,5832,5832,0.00000000
2025-09-07 18:14:11,matpower/case118/2017-04-13,train,raw,OPTIMAL,33.572000,3954991.706927,49068,1597536,5832,5832,0.00000000
2025-09-07 18:15:21,matpower/case118/2017-04-15,train,raw,OPTIMAL,35.853000,3415974.662532,49068,1597536,5832,5832,0.00000000
2025-09-07 18:16:30,matpower/case118/2017-04-16,train,raw,OPTIMAL,35.656000,3867060.292944,49068,1597536,5832,5832,0.00000000
2025-09-07 18:17:39,matpower/case118/2017-04-18,train,raw,OPTIMAL,35.009000,3712398.412625,49068,1597536,5832,5832,0.00000000
2025-09-07 18:18:48,matpower/case118/2017-04-19,train,raw,OPTIMAL,35.521000,3622366.935837,49068,1597536,5832,5832,0.00000000
2025-09-07 18:19:59,matpower/case118/2017-04-21,train,raw,OPTIMAL,37.145000,3716777.948732,49068,1597536,5832,5832,0.00000000
2025-09-07 18:21:10,matpower/case118/2017-04-23,train,raw,OPTIMAL,36.935000,3471886.716562,49068,1597536,5832,5832,0.00000000
2025-09-07 18:22:20,matpower/case118/2017-04-24,train,raw,OPTIMAL,36.313000,3946647.354799,49068,1597536,5832,5832,0.00000000
2025-09-07 18:23:29,matpower/case118/2017-04-26,train,raw,OPTIMAL,35.215000,3862320.875425,49068,1597536,5832,5832,0.00000000
2025-09-07 18:24:36,matpower/case118/2017-04-27,train,raw,OPTIMAL,33.127000,4197050.723042,49068,1597536,5832,5832,0.00000000
2025-09-07 18:25:51,matpower/case118/2017-04-28,train,raw,OPTIMAL,41.455000,4098212.276202,49068,1597536,5832,5832,0.00000000
2025-09-07 18:27:02,matpower/case118/2017-04-29,train,raw,OPTIMAL,37.671000,3664533.362046,49068,1597536,5832,5832,0.00000000
2025-09-07 18:28:16,matpower/case118/2017-04-30,train,raw,OPTIMAL,40.990000,3334215.590560,49068,1597536,5832,5832,0.00000000
2025-09-07 18:29:33,matpower/case118/2017-05-01,train,raw,OPTIMAL,42.983000,3641876.035388,49068,1597536,5832,5832,0.00000000
2025-09-07 18:30:45,matpower/case118/2017-05-02,train,raw,OPTIMAL,38.651000,3425601.368909,49068,1597536,5832,5832,0.00000000
2025-09-07 18:31:59,matpower/case118/2017-05-03,train,raw,OPTIMAL,39.827000,3148494.536617,49068,1597536,5832,5832,0.00000000
2025-09-07 18:33:11,matpower/case118/2017-05-05,train,raw,OPTIMAL,39.102000,3457470.714886,49068,1597536,5832,5832,0.00000000
2025-09-07 18:34:34,matpower/case118/2017-05-06,train,raw,OPTIMAL,49.437000,3146014.863390,49068,1597536,5832,5832,0.00000000
2025-09-07 18:35:47,matpower/case118/2017-05-07,train,raw,OPTIMAL,38.135000,3328166.488035,49068,1597536,5832,5832,0.00000000
2025-09-07 18:37:01,matpower/case118/2017-05-08,train,raw,OPTIMAL,39.332000,3839142.063299,49068,1597536,5832,5832,0.00000000
2025-09-07 18:38:11,matpower/case118/2017-05-10,train,raw,OPTIMAL,36.447000,4097579.202842,49068,1597536,5832,5832,0.00000000
2025-09-07 18:39:24,matpower/case118/2017-05-11,train,raw,OPTIMAL,38.933000,4286083.017911,49068,1597536,5832,5832,0.00000000
2025-09-07 18:40:32,matpower/case118/2017-05-12,train,raw,OPTIMAL,34.749000,4202750.880985,49068,1597536,5832,5832,0.00000000
2025-09-07 18:41:40,matpower/case118/2017-05-14,train,raw,OPTIMAL,34.363000,3859630.151647,49068,1597536,5832,5832,0.00000000
2025-09-07 18:42:48,matpower/case118/2017-05-15,train,raw,OPTIMAL,33.966000,4237803.206300,49068,1597536,5832,5832,0.00000000
2025-09-07 18:43:55,matpower/case118/2017-05-17,train,raw,OPTIMAL,33.366000,5198766.939996,49068,1597536,5832,5832,0.00000000
2025-09-07 18:45:20,matpower/case118/2017-05-19,train,raw,OPTIMAL,51.779000,4869262.594291,49068,1597536,5832,5832,0.00000000
2025-09-07 18:46:44,matpower/case118/2017-05-20,train,raw,OPTIMAL,50.788000,3421401.080674,49068,1597536,5832,5832,0.00000000
2025-09-07 18:47:52,matpower/case118/2017-05-21,train,raw,OPTIMAL,33.866000,3343782.037784,49068,1597536,5832,5832,0.00000000
2025-09-07 18:49:01,matpower/case118/2017-05-23,train,raw,OPTIMAL,35.188000,3869732.579024,49068,1597536,5832,5832,0.00000000
2025-09-07 18:50:08,matpower/case118/2017-05-24,train,raw,OPTIMAL,34.304000,4005402.262152,49068,1597536,5832,5832,0.00000000
2025-09-07 18:51:19,matpower/case118/2017-05-26,train,raw,OPTIMAL,36.440000,3682976.603357,49068,1597536,5832,5832,0.00000000
2025-09-07 18:52:28,matpower/case118/2017-05-29,train,raw,OPTIMAL,36.457000,3347867.449056,49068,1597536,5832,5832,0.00000000
2025-09-07 18:53:37,matpower/case118/2017-05-30,train,raw,OPTIMAL,34.654000,3792689.726057,49068,1597536,5832,5832,0.00000000
2025-09-07 18:54:44,matpower/case118/2017-05-31,train,raw,OPTIMAL,33.312000,4029163.925489,49068,1597536,5832,5832,0.00000000
2025-09-07 18:55:50,matpower/case118/2017-06-01,train,raw,OPTIMAL,32.486000,4022639.377375,49068,1597536,5832,5832,0.00000000
2025-09-07 18:56:57,matpower/case118/2017-06-02,train,raw,OPTIMAL,33.593000,4069430.341564,49068,1597536,5832,5832,0.00000000
2025-09-07 18:58:05,matpower/case118/2017-06-03,train,raw,OPTIMAL,34.876000,3830197.760638,49068,1597536,5832,5832,0.00000000
2025-09-07 18:59:14,matpower/case118/2017-06-04,train,raw,OPTIMAL,34.281000,4135732.697740,49068,1597536,5832,5832,0.00000000
2025-09-07 19:00:21,matpower/case118/2017-06-05,train,raw,OPTIMAL,34.147000,4602207.658313,49068,1597536,5832,5832,0.00000000
2025-09-07 19:01:27,matpower/case118/2017-06-06,train,raw,OPTIMAL,32.508000,4279056.578223,49068,1597536,5832,5832,0.00000000
2025-09-07 19:02:38,matpower/case118/2017-06-07,train,raw,OPTIMAL,36.936000,3993973.402546,49068,1597536,5832,5832,0.00000000
2025-09-07 19:03:58,matpower/case118/2017-06-08,train,raw,OPTIMAL,46.189000,4096615.493010,49068,1597536,5832,5832,0.00000000
2025-09-07 19:05:04,matpower/case118/2017-06-09,train,raw,OPTIMAL,32.890000,4384073.933709,49068,1597536,5832,5832,0.00000000
2025-09-07 19:06:14,matpower/case118/2017-06-11,train,raw,OPTIMAL,36.197000,5561841.926280,49068,1597536,5832,5832,0.00000000
2025-09-07 19:07:26,matpower/case118/2017-06-12,train,raw,OPTIMAL,38.419000,6831236.907647,49068,1597536,5832,5832,0.00000000
2025-09-07 19:08:35,matpower/case118/2017-06-13,train,raw,OPTIMAL,35.624000,7028557.702271,49068,1597536,5832,5832,0.00000000
2025-09-07 19:09:43,matpower/case118/2017-06-16,train,raw,OPTIMAL,33.579000,5157460.944670,49068,1597536,5832,5832,0.00000000
2025-09-07 19:10:51,matpower/case118/2017-06-17,train,raw,OPTIMAL,34.958000,5352297.256413,49068,1597536,5832,5832,0.00000000
2025-09-07 19:12:19,matpower/case118/2017-06-19,train,raw,OPTIMAL,50.931000,6499380.324378,49068,1597536,5832,5832,0.00000000
2025-09-07 19:13:50,matpower/case118/2017-06-20,train,raw,OPTIMAL,42.686000,6254019.129143,49068,1597536,5832,5832,0.00000000
2025-09-07 19:15:27,matpower/case118/2017-06-21,train,raw,OPTIMAL,47.914000,6115423.044228,49068,1597536,5832,5832,0.00000000
2025-09-07 19:16:50,matpower/case118/2017-06-22,train,raw,OPTIMAL,41.564000,6943090.599400,49068,1597536,5832,5832,0.00000000
2025-09-07 19:18:07,matpower/case118/2017-06-23,train,raw,OPTIMAL,38.693000,6768858.612653,49068,1597536,5832,5832,0.00000000
2025-09-07 19:19:30,matpower/case118/2017-06-24,train,raw,OPTIMAL,39.191000,5943870.983599,49068,1597536,5832,5832,0.00000000
2025-09-07 19:20:53,matpower/case118/2017-06-26,train,raw,OPTIMAL,40.365000,5399307.877929,49068,1597536,5832,5832,0.00000000
2025-09-07 19:22:13,matpower/case118/2017-06-29,train,raw,OPTIMAL,40.955000,6912090.455089,49068,1597536,5832,5832,0.00000000
2025-09-07 19:23:38,matpower/case118/2017-06-30,train,raw,OPTIMAL,44.173000,7285093.940495,49068,1597536,5832,5832,0.00000000
2025-09-07 19:24:57,matpower/case118/2017-07-01,train,raw,OPTIMAL,39.156000,6841148.159642,49068,1597536,5832,5832,0.00000000
2025-09-07 19:26:26,matpower/case118/2017-07-02,train,raw,OPTIMAL,44.897000,6750133.232013,49068,1597536,5832,5832,0.00000000
2025-09-07 19:27:48,matpower/case118/2017-07-04,train,raw,OPTIMAL,40.266000,6170160.162100,49068,1597536,5832,5832,0.00000000
2025-09-07 19:29:07,matpower/case118/2017-07-05,train,raw,OPTIMAL,37.872000,6290768.857808,49068,1597536,5832,5832,0.00000000
2025-09-07 19:30:34,matpower/case118/2017-07-06,train,raw,OPTIMAL,42.421000,6144488.226836,49068,1597536,5832,5832,0.00000000
2025-09-07 19:31:55,matpower/case118/2017-07-07,train,raw,OPTIMAL,40.728000,6606131.315061,49068,1597536,5832,5832,0.00000000
2025-09-07 19:33:12,matpower/case118/2017-07-08,train,raw,OPTIMAL,38.370000,6163530.041813,49068,1597536,5832,5832,0.00000000
2025-09-07 19:34:30,matpower/case118/2017-07-10,train,raw,OPTIMAL,40.601000,6723077.062295,49068,1597536,5832,5832,0.00000000
2025-09-07 19:35:59,matpower/case118/2017-07-11,train,raw,OPTIMAL,47.174000,7574380.225344,49068,1597536,5832,5832,0.00000000
2025-09-07 19:37:25,matpower/case118/2017-07-12,train,raw,OPTIMAL,44.265000,8043371.313534,49068,1597536,5832,5832,0.00000000
2025-09-07 19:39:04,matpower/case118/2017-07-13,train,raw,OPTIMAL,52.204000,8244642.286241,49068,1597536,5832,5832,0.00000000
2025-09-07 19:40:51,matpower/case118/2017-07-15,train,raw,OPTIMAL,56.484000,6031414.621420,49068,1597536,5832,5832,0.00000000
2025-09-07 19:42:29,matpower/case118/2017-07-17,train,raw,OPTIMAL,49.776000,6803831.274196,49068,1597536,5832,5832,0.00000000
2025-09-07 19:44:03,matpower/case118/2017-07-20,train,raw,OPTIMAL,47.025000,8294303.589356,49068,1597536,5832,5832,0.00000000
2025-09-07 19:45:33,matpower/case118/2017-07-22,train,raw,OPTIMAL,43.207000,6952032.417987,49068,1597536,5832,5832,0.00000000
2025-09-07 19:46:48,matpower/case118/2017-07-23,train,raw,OPTIMAL,37.050000,6306408.303979,49068,1597536,5832,5832,0.00000000
2025-09-07 19:48:06,matpower/case118/2017-07-24,train,raw,OPTIMAL,40.182000,6542528.028154,49068,1597536,5832,5832,0.00000000
2025-09-07 19:49:25,matpower/case118/2017-07-25,train,raw,OPTIMAL,40.880000,5640035.352644,49068,1597536,5832,5832,0.00000000
2025-09-07 19:50:40,matpower/case118/2017-07-27,train,raw,OPTIMAL,34.618000,6325908.445528,49068,1597536,5832,5832,0.00000000
2025-09-07 19:51:58,matpower/case118/2017-07-28,train,raw,OPTIMAL,40.782000,5905925.848222,49068,1597536,5832,5832,0.00000000
2025-09-07 19:53:21,matpower/case118/2017-07-29,train,raw,OPTIMAL,44.685000,4784025.638924,49068,1597536,5832,5832,0.00000000
2025-09-07 19:54:40,matpower/case118/2017-07-30,train,raw,OPTIMAL,41.779000,4977556.442801,49068,1597536,5832,5832,0.00000000
2025-09-07 19:55:58,matpower/case118/2017-08-01,train,raw,OPTIMAL,40.114000,7032106.233454,49068,1597536,5832,5832,0.00000000
2025-09-07 19:57:15,matpower/case118/2017-08-02,train,raw,OPTIMAL,38.662000,7191421.118098,49068,1597536,5832,5832,0.00000000
2025-09-07 19:58:33,matpower/case118/2017-08-03,train,raw,OPTIMAL,40.518000,6928621.044160,49068,1597536,5832,5832,0.00000000
2025-09-07 19:59:42,matpower/case118/2017-08-04,train,raw,OPTIMAL,31.740000,5716370.772767,49068,1597536,5832,5832,0.00000000
2025-09-07 20:00:51,matpower/case118/2017-08-05,train,raw,OPTIMAL,30.528000,4473785.203581,49068,1597536,5832,5832,0.00000000
2025-09-07 20:02:02,matpower/case118/2017-08-06,train,raw,OPTIMAL,33.735000,4462039.173785,49068,1597536,5832,5832,0.00000000
2025-09-07 20:03:15,matpower/case118/2017-08-07,train,raw,OPTIMAL,34.512000,4894355.582060,49068,1597536,5832,5832,0.00000000
2025-09-07 20:04:23,matpower/case118/2017-08-08,train,raw,OPTIMAL,30.152000,4678418.722312,49068,1597536,5832,5832,0.00000000
2025-09-07 20:05:34,matpower/case118/2017-08-10,train,raw,OPTIMAL,32.986000,5255912.625233,49068,1597536,5832,5832,0.00000000
2025-09-07 20:06:45,matpower/case118/2017-08-11,train,raw,OPTIMAL,33.113000,5088299.336459,49068,1597536,5832,5832,0.00000000
2025-09-07 20:07:53,matpower/case118/2017-08-12,train,raw,OPTIMAL,31.003000,4682626.480520,49068,1597536,5832,5832,0.00000000
2025-09-07 20:09:07,matpower/case118/2017-08-13,train,raw,OPTIMAL,35.757000,5067979.391381,49068,1597536,5832,5832,0.00000000
2025-09-07 20:10:20,matpower/case118/2017-08-15,train,raw,OPTIMAL,35.184000,6166905.223574,49068,1597536,5832,5832,0.00000000
2025-09-07 20:11:33,matpower/case118/2017-08-16,train,raw,OPTIMAL,35.942000,6270556.525602,49068,1597536,5832,5832,0.00000000
2025-09-07 20:12:52,matpower/case118/2017-08-19,train,raw,OPTIMAL,36.861000,5963298.153728,49068,1597536,5832,5832,0.00000000
2025-09-07 20:14:18,matpower/case118/2017-08-20,train,raw,OPTIMAL,38.808000,5618304.246837,49068,1597536,5832,5832,0.00000000
2025-09-07 20:15:36,matpower/case118/2017-08-23,train,raw,OPTIMAL,34.178000,5527847.505455,49068,1597536,5832,5832,0.00000000
2025-09-07 20:16:46,matpower/case118/2017-08-24,train,raw,OPTIMAL,32.022000,5041526.885678,49068,1597536,5832,5832,0.00000000
2025-09-07 20:17:56,matpower/case118/2017-08-25,train,raw,OPTIMAL,31.557000,4649000.289135,49068,1597536,5832,5832,0.00000000
2025-09-07 20:19:09,matpower/case118/2017-08-27,train,raw,OPTIMAL,33.987000,4726128.564341,49068,1597536,5832,5832,0.00000000
2025-09-07 20:20:22,matpower/case118/2017-08-28,train,raw,OPTIMAL,35.501000,5073149.711993,49068,1597536,5832,5832,0.00000000
2025-09-07 20:21:37,matpower/case118/2017-08-29,train,raw,OPTIMAL,36.468000,4972191.972984,49068,1597536,5832,5832,0.00000000
2025-09-07 20:22:53,matpower/case118/2017-08-30,train,raw,OPTIMAL,37.031000,5139353.357207,49068,1597536,5832,5832,0.00000000
2025-09-07 20:24:04,matpower/case118/2017-08-31,train,raw,OPTIMAL,33.092000,5205242.626132,49068,1597536,5832,5832,0.00000000
2025-09-07 20:25:19,matpower/case118/2017-09-01,train,raw,OPTIMAL,36.753000,4445141.324649,49068,1597536,5832,5832,0.00000000
2025-09-07 20:26:29,matpower/case118/2017-09-03,train,raw,OPTIMAL,31.560000,3958686.928733,49068,1597536,5832,5832,0.00000000
2025-09-07 20:27:38,matpower/case118/2017-09-04,train,raw,OPTIMAL,31.663000,4596881.544678,49068,1597536,5832,5832,0.00000000
2025-09-07 20:28:54,matpower/case118/2017-09-05,train,raw,OPTIMAL,36.959000,5377747.169863,49068,1597536,5832,5832,0.00000000
2025-09-07 20:30:07,matpower/case118/2017-09-06,train,raw,OPTIMAL,35.454000,4507919.277645,49068,1597536,5832,5832,0.00000000
2025-09-07 20:31:21,matpower/case118/2017-09-07,train,raw,OPTIMAL,35.406000,4250606.715186,49068,1597536,5832,5832,0.00000000
2025-09-07 20:32:34,matpower/case118/2017-09-08,train,raw,OPTIMAL,34.934000,4234277.016958,49068,1597536,5832,5832,0.00000000
2025-09-07 20:33:44,matpower/case118/2017-09-09,train,raw,OPTIMAL,31.958000,3890682.501230,49068,1597536,5832,5832,0.00000000
2025-09-07 20:34:55,matpower/case118/2017-09-10,train,raw,OPTIMAL,32.027000,3855912.578037,49068,1597536,5832,5832,0.00000000
2025-09-07 20:36:05,matpower/case118/2017-09-11,train,raw,OPTIMAL,32.459000,4344944.704115,49068,1597536,5832,5832,0.00000000
2025-09-07 20:37:25,matpower/case118/2017-09-13,train,raw,OPTIMAL,42.088000,5108285.700511,49068,1597536,5832,5832,0.00000000
2025-09-07 20:38:39,matpower/case118/2017-09-15,train,raw,OPTIMAL,36.910000,5347011.299289,49068,1597536,5832,5832,0.00000000
2025-09-07 20:40:02,matpower/case118/2017-09-16,train,raw,OPTIMAL,45.275000,4853432.130091,49068,1597536,5832,5832,0.00000000
2025-09-07 20:41:18,matpower/case118/2017-09-19,train,raw,OPTIMAL,38.505000,5522254.393697,49068,1597536,5832,5832,0.00000000
2025-09-07 20:42:24,matpower/case118/2017-09-20,train,raw,OPTIMAL,29.104000,6047014.629529,49068,1597536,5832,5832,0.00000000
2025-09-07 20:43:31,matpower/case118/2017-09-21,train,raw,OPTIMAL,29.854000,6782521.398857,49068,1597536,5832,5832,0.00000000
2025-09-07 20:44:46,matpower/case118/2017-09-23,train,raw,OPTIMAL,36.374000,5914977.160948,49068,1597536,5832,5832,0.00000000
2025-09-07 20:45:55,matpower/case118/2017-09-24,train,raw,OPTIMAL,30.294000,5965216.148182,49068,1597536,5832,5832,0.00000000
2025-09-07 20:47:07,matpower/case118/2017-09-25,train,raw,OPTIMAL,33.440000,5921083.880496,49068,1597536,5832,5832,0.00000000
2025-09-07 20:48:19,matpower/case118/2017-09-26,train,raw,OPTIMAL,33.230000,5701961.203263,49068,1597536,5832,5832,0.00000000
2025-09-07 20:49:31,matpower/case118/2017-09-27,train,raw,OPTIMAL,32.351000,5892580.996132,49068,1597536,5832,5832,0.00000000
2025-09-07 20:50:52,matpower/case118/2017-09-28,train,raw,OPTIMAL,42.250000,4918034.734452,49068,1597536,5832,5832,0.00000000
2025-09-07 20:52:10,matpower/case118/2017-09-29,train,raw,OPTIMAL,38.005000,4216099.256562,49068,1597536,5832,5832,0.00000000
2025-09-07 20:53:23,matpower/case118/2017-10-03,train,raw,OPTIMAL,32.782000,4034790.446439,49068,1597536,5832,5832,0.00000000
2025-09-07 20:54:31,matpower/case118/2017-10-04,train,raw,OPTIMAL,31.075000,4211363.866819,49068,1597536,5832,5832,0.00000000
2025-09-07 20:55:46,matpower/case118/2017-10-08,train,raw,OPTIMAL,36.968000,4936522.777999,49068,1597536,5832,5832,0.00000000
2025-09-07 20:57:05,matpower/case118/2017-10-09,train,raw,OPTIMAL,39.777000,5136305.153035,49068,1597536,5832,5832,0.00000000
2025-09-07 20:58:25,matpower/case118/2017-10-10,train,raw,OPTIMAL,38.858000,5193624.828247,49068,1597536,5832,5832,0.00000000
2025-09-07 20:59:49,matpower/case118/2017-10-11,train,raw,OPTIMAL,39.542000,4846560.943793,49068,1597536,5832,5832,0.00000000
2025-09-07 21:01:01,matpower/case118/2017-10-12,train,raw,OPTIMAL,31.846000,4919827.308497,49068,1597536,5832,5832,0.00000000
2025-09-07 21:02:31,matpower/case118/2017-10-13,train,raw,OPTIMAL,43.143000,4815672.855264,49068,1597536,5832,5832,0.00000000
2025-09-07 21:03:46,matpower/case118/2017-10-14,train,raw,OPTIMAL,33.134000,4329656.218314,49068,1597536,5832,5832,0.00000000
2025-09-07 21:04:57,matpower/case118/2017-10-15,train,raw,OPTIMAL,33.864000,4758333.790294,49068,1597536,5832,5832,0.00000000
2025-09-07 21:06:05,matpower/case118/2017-10-16,train,raw,OPTIMAL,30.625000,4525207.399472,49068,1597536,5832,5832,0.00000000
2025-09-07 21:07:14,matpower/case118/2017-10-17,train,raw,OPTIMAL,31.833000,4501749.530244,49068,1597536,5832,5832,0.00000000
2025-09-07 21:08:23,matpower/case118/2017-10-18,train,raw,OPTIMAL,29.960000,4275350.373972,49068,1597536,5832,5832,0.00000000
2025-09-07 21:09:32,matpower/case118/2017-10-19,train,raw,OPTIMAL,31.040000,4402520.319732,49068,1597536,5832,5832,0.00000000
2025-09-07 21:10:35,matpower/case118/2017-10-21,train,raw,OPTIMAL,25.739000,3724422.274467,49068,1597536,5832,5832,0.00000000
2025-09-07 21:11:43,matpower/case118/2017-10-22,train,raw,OPTIMAL,28.669000,3937706.231190,49068,1597536,5832,5832,0.00000000
2025-09-07 21:12:52,matpower/case118/2017-10-23,train,raw,OPTIMAL,31.315000,4671109.504259,49068,1597536,5832,5832,0.00000000
2025-09-07 21:14:02,matpower/case118/2017-10-24,train,raw,OPTIMAL,30.951000,4788186.196762,49068,1597536,5832,5832,0.00000000
2025-09-07 21:15:10,matpower/case118/2017-10-25,train,raw,OPTIMAL,29.791000,4615796.170281,49068,1597536,5832,5832,0.00000000
2025-09-07 21:16:12,matpower/case118/2017-10-26,train,raw,OPTIMAL,24.307000,4521758.454111,49068,1597536,5832,5832,0.00000000
2025-09-07 21:17:17,matpower/case118/2017-10-27,train,raw,OPTIMAL,26.241000,4324752.648898,49068,1597536,5832,5832,0.00000000
2025-09-07 21:18:24,matpower/case118/2017-10-30,train,raw,OPTIMAL,29.573000,4435014.326699,49068,1597536,5832,5832,0.00000000
2025-09-07 21:19:34,matpower/case118/2017-10-31,train,raw,OPTIMAL,32.016000,4397912.966829,49068,1597536,5832,5832,0.00000000
2025-09-07 21:20:45,matpower/case118/2017-11-02,train,raw,OPTIMAL,32.292000,4383480.832697,49068,1597536,5832,5832,0.00000000
2025-09-07 21:21:56,matpower/case118/2017-11-03,train,raw,OPTIMAL,31.544000,4073100.570532,49068,1597536,5832,5832,0.00000000
2025-09-07 21:23:03,matpower/case118/2017-11-04,train,raw,OPTIMAL,29.361000,3708800.252684,49068,1597536,5832,5832,0.00000000
2025-09-07 21:24:13,matpower/case118/2017-11-05,train,raw,OPTIMAL,31.139000,3853896.665174,49068,1597536,5832,5832,0.00000000
2025-09-07 21:25:24,matpower/case118/2017-11-07,train,raw,OPTIMAL,32.882000,4412630.268749,49068,1597536,5832,5832,0.00000000
2025-09-07 21:26:38,matpower/case118/2017-11-09,train,raw,OPTIMAL,36.122000,4988764.882854,49068,1597536,5832,5832,0.00000000
2025-09-07 21:27:52,matpower/case118/2017-11-10,train,raw,OPTIMAL,36.142000,5343158.469200,49068,1597536,5832,5832,0.00000000
2025-09-07 21:29:16,matpower/case118/2017-11-11,train,raw,OPTIMAL,45.503000,4911581.917400,49068,1597536,5832,5832,0.00000000
2025-09-07 21:30:27,matpower/case118/2017-11-12,train,raw,OPTIMAL,34.589000,4588471.293036,49068,1597536,5832,5832,0.00000000
2025-09-07 21:31:44,matpower/case118/2017-11-15,train,raw,OPTIMAL,39.581000,4711000.468680,49068,1597536,5832,5832,0.00000000
2025-09-07 21:32:57,matpower/case118/2017-11-16,train,raw,OPTIMAL,36.247000,4830258.443147,49068,1597536,5832,5832,0.00000000
2025-09-07 21:34:11,matpower/case118/2017-11-17,train,raw,OPTIMAL,37.296000,4865411.332893,49068,1597536,5832,5832,0.00000000
2025-09-07 21:35:23,matpower/case118/2017-11-18,train,raw,OPTIMAL,35.032000,4189228.854765,49068,1597536,5832,5832,0.00000000
2025-09-07 21:36:34,matpower/case118/2017-11-19,train,raw,OPTIMAL,33.397000,4533965.018477,49068,1597536,5832,5832,0.00000000
2025-09-07 21:38:01,matpower/case118/2017-11-20,train,raw,OPTIMAL,49.616000,5394645.833010,49068,1597536,5832,5832,0.00000000
2025-09-07 21:39:18,matpower/case118/2017-11-21,train,raw,OPTIMAL,40.574000,4915607.260212,49068,1597536,5832,5832,0.00000000
2025-09-07 21:40:28,matpower/case118/2017-11-22,train,raw,OPTIMAL,32.761000,5039652.747192,49068,1597536,5832,5832,0.00000000
2025-09-07 21:41:39,matpower/case118/2017-11-23,train,raw,OPTIMAL,33.648000,5071048.862307,49068,1597536,5832,5832,0.00000000
2025-09-07 21:42:50,matpower/case118/2017-11-25,train,raw,OPTIMAL,33.746000,4389501.766386,49068,1597536,5832,5832,0.00000000
2025-09-07 21:44:01,matpower/case118/2017-11-26,train,raw,OPTIMAL,33.303000,4804721.659951,49068,1597536,5832,5832,0.00000000
2025-09-07 21:45:12,matpower/case118/2017-11-27,train,raw,OPTIMAL,35.092000,5425652.974413,49068,1597536,5832,5832,0.00000000
2025-09-07 21:46:26,matpower/case118/2017-11-28,train,raw,OPTIMAL,36.470000,5055762.295693,49068,1597536,5832,5832,0.00000000
2025-09-07 21:47:34,matpower/case118/2017-11-29,train,raw,OPTIMAL,30.681000,5093437.414405,49068,1597536,5832,5832,0.00000000
2025-09-07 21:48:41,matpower/case118/2017-12-01,train,raw,OPTIMAL,30.682000,5038969.844867,49068,1597536,5832,5832,0.00000000
2025-09-07 21:49:54,matpower/case118/2017-12-03,train,raw,OPTIMAL,35.411000,5230572.745015,49068,1597536,5832,5832,0.00000000
2025-09-07 21:51:05,matpower/case118/2017-12-04,train,raw,OPTIMAL,33.119000,5541161.130095,49068,1597536,5832,5832,0.00000000
2025-09-07 21:52:17,matpower/case118/2017-12-07,train,raw,OPTIMAL,34.894000,6248550.667583,49068,1597536,5832,5832,0.00000000
2025-09-07 21:53:31,matpower/case118/2017-12-08,train,raw,OPTIMAL,36.899000,6742566.350919,49068,1597536,5832,5832,0.00000000
2025-09-07 21:54:48,matpower/case118/2017-12-09,train,raw,OPTIMAL,39.574000,7122495.967886,49068,1597536,5832,5832,0.00000000
2025-09-07 21:56:07,matpower/case118/2017-12-10,train,raw,OPTIMAL,42.491000,7008845.524673,49068,1597536,5832,5832,0.00000000
2025-09-07 21:57:29,matpower/case118/2017-12-11,train,raw,OPTIMAL,44.786000,7530100.866536,49068,1597536,5832,5832,0.00000000
2025-09-07 21:58:39,matpower/case118/2017-12-12,train,raw,OPTIMAL,32.786000,7856643.244367,49068,1597536,5832,5832,0.00000000
2025-09-07 21:59:51,matpower/case118/2017-12-13,train,raw,OPTIMAL,35.365000,8290571.738578,49068,1597536,5832,5832,0.00000000
2025-09-07 22:01:14,matpower/case118/2017-12-15,train,raw,OPTIMAL,45.566000,8527591.871877,49068,1597536,5832,5832,0.00000000
2025-09-07 22:02:33,matpower/case118/2017-12-16,train,raw,OPTIMAL,42.494000,7412840.649605,49068,1597536,5832,5832,0.00000000
2025-09-07 22:03:52,matpower/case118/2017-12-17,train,raw,OPTIMAL,41.073000,7250400.380731,49068,1597536,5832,5832,0.00000000
2025-09-07 22:05:15,matpower/case118/2017-12-19,train,raw,OPTIMAL,45.562000,6842979.051138,49068,1597536,5832,5832,0.00000000
2025-09-07 22:06:40,matpower/case118/2017-12-21,train,raw,OPTIMAL,48.641000,8016928.793169,49068,1597536,5832,5832,0.00000000
2025-09-07 22:07:58,matpower/case118/2017-12-23,train,raw,OPTIMAL,40.595000,5669620.096569,49068,1597536,5832,5832,0.00000000
2025-09-07 22:09:14,matpower/case118/2017-12-24,train,raw,OPTIMAL,38.680000,5917742.516003,49068,1597536,5832,5832,0.00000000
2025-09-07 22:10:26,matpower/case118/2017-12-25,train,raw,OPTIMAL,34.179000,6668344.960969,49068,1597536,5832,5832,0.00000000
2025-09-07 22:11:34,matpower/case118/2017-12-26,train,raw,OPTIMAL,30.980000,7790026.485482,49068,1597536,5832,5832,0.00000000
2025-09-07 22:12:51,matpower/case118/2017-12-27,train,raw,OPTIMAL,39.780000,8499646.127207,49068,1597536,5832,5832,0.00000000
2025-09-07 22:13:58,matpower/case118/2017-12-28,train,raw,OPTIMAL,30.367000,8858302.923714,49068,1597536,5832,5832,0.00000000
2025-09-07 22:15:36,matpower/case118/2017-12-29,train,raw,TIME_LIMIT,60.325000,20127913.524390,49068,1597536,5832,5832,0.00000000
2025-09-07 22:17:00,matpower/case118/2017-12-30,train,raw,OPTIMAL,47.161000,7855732.650394,49068,1597536,5832,5832,0.00000000
2025-09-07 22:18:14,matpower/case118/2017-12-31,train,raw,OPTIMAL,37.167000,8256728.593570,49068,1597536,5832,5832,0.00000000
2025-09-07 22:34:08,matpower/case118/2017-01-09,test,raw,OPTIMAL,14.115000,7078282.549776,49068,83644,5832,5832,0.00000000
2025-09-07 22:34:38,matpower/case118/2017-01-09,test,warm,OPTIMAL,14.850000,7077589.987506,49068,83644,5832,5832,0.00000000
2025-09-07 22:35:00,matpower/case118/2017-01-10,test,raw,OPTIMAL,8.652000,6163863.107085,49068,62022,5832,5832,0.00000000
2025-09-07 22:35:22,matpower/case118/2017-01-10,test,warm,OPTIMAL,8.721000,6440214.280056,49068,62022,5832,5832,8.61214296
2025-09-07 22:35:48,matpower/case118/2017-01-24,test,raw,OPTIMAL,11.785000,5687639.859782,49068,89436,5832,5832,0.00000000
2025-09-07 22:36:16,matpower/case118/2017-01-24,test,warm,OPTIMAL,12.707000,5679662.326917,49068,89436,5832,5832,0.00000000
2025-09-07 22:36:42,matpower/case118/2017-01-25,test,raw,OPTIMAL,13.074000,5423005.512082,49068,97416,5832,5832,0.00000000
2025-09-07 22:37:10,matpower/case118/2017-01-25,test,warm,OPTIMAL,13.206000,5422726.468085,49068,97416,5832,5832,0.00000000
2025-09-07 22:37:30,matpower/case118/2017-01-28,test,raw,OPTIMAL,6.680000,5574898.725893,49068,45276,5832,5832,0.00000000
2025-09-07 22:37:50,matpower/case118/2017-01-28,test,warm,OPTIMAL,6.575000,5574898.725893,49068,45276,5832,5832,0.00000000
2025-09-07 22:38:11,matpower/case118/2017-01-29,test,raw,OPTIMAL,8.055000,5548562.422801,49068,57526,5832,5832,0.00000000
2025-09-07 22:38:33,matpower/case118/2017-01-29,test,warm,OPTIMAL,8.064000,5757720.644063,49068,57526,5832,5832,0.00000000
2025-09-07 22:38:51,matpower/case118/2017-02-07,test,raw,OPTIMAL,5.305000,4913930.102406,49068,42358,5832,5832,0.00000000
2025-09-07 22:39:14,matpower/case118/2017-02-07,test,warm,OPTIMAL,4.820000,5129339.685880,49068,42358,5832,5832,0.00000000
2025-09-07 22:39:37,matpower/case118/2017-02-12,test,raw,OPTIMAL,9.175000,5592444.453819,49068,57526,5832,5832,0.00000000
2025-09-07 22:40:00,matpower/case118/2017-02-12,test,warm,OPTIMAL,8.863000,5788334.661243,49068,57526,5832,5832,0.00000000
2025-09-07 22:40:25,matpower/case118/2017-02-28,test,raw,OPTIMAL,12.005000,4372284.034888,49068,82254,5832,5832,0.00000000
2025-09-07 22:40:51,matpower/case118/2017-02-28,test,warm,OPTIMAL,11.925000,4520283.601179,49068,82254,5832,5832,0.00000000
2025-09-07 22:41:20,matpower/case118/2017-03-04,test,raw,OPTIMAL,14.228000,4226453.400210,49068,132058,5832,5832,0.00000000
2025-09-07 22:41:49,matpower/case118/2017-03-04,test,warm,OPTIMAL,14.372000,4420338.842747,49068,132058,5832,5832,0.00000000
2025-09-07 22:42:10,matpower/case118/2017-03-07,test,raw,OPTIMAL,8.069000,4038847.598909,49068,51804,5832,5832,0.00000000
2025-09-07 22:42:33,matpower/case118/2017-03-07,test,warm,OPTIMAL,8.924000,4205155.078381,49068,51804,5832,5832,0.00000000
2025-09-07 22:43:01,matpower/case118/2017-03-09,test,raw,OPTIMAL,14.075000,4110255.769613,49068,121116,5832,5832,0.00000000
2025-09-07 22:43:32,matpower/case118/2017-03-09,test,warm,OPTIMAL,15.276000,4109484.929990,49068,121116,5832,5832,0.00000000
2025-09-07 22:43:59,matpower/case118/2017-03-11,test,raw,OPTIMAL,13.778000,5314355.445488,49068,132058,5832,5832,0.00000000
2025-09-07 22:44:33,matpower/case118/2017-03-11,test,warm,OPTIMAL,13.759000,5502584.427333,49068,132058,5832,5832,0.00000000
2025-09-07 22:45:01,matpower/case118/2017-03-14,test,raw,OPTIMAL,14.432000,5411312.272747,49068,93644,5832,5832,0.00000000
2025-09-07 22:45:31,matpower/case118/2017-03-14,test,warm,OPTIMAL,15.179000,5411192.349260,49068,93644,5832,5832,0.00000000
2025-09-07 22:45:55,matpower/case118/2017-03-20,test,raw,OPTIMAL,11.545000,4582671.400448,49068,69108,5832,5832,0.00000000
2025-09-07 22:46:22,matpower/case118/2017-03-20,test,warm,OPTIMAL,11.824000,4581161.418841,49068,69108,5832,5832,0.00000000
2025-09-07 22:46:45,matpower/case118/2017-03-25,test,raw,OPTIMAL,10.506000,3556036.940042,49068,69384,5832,5832,0.00000000
2025-09-07 22:47:10,matpower/case118/2017-03-25,test,warm,OPTIMAL,10.273000,3554699.699659,49068,69384,5832,5832,0.00000000
2025-09-07 22:47:29,matpower/case118/2017-04-07,test,raw,OPTIMAL,6.594000,4321691.114373,49068,42794,5832,5832,0.00000000
2025-09-07 22:47:50,matpower/case118/2017-04-07,test,warm,OPTIMAL,6.964000,4321691.114373,49068,42794,5832,5832,0.00000000
2025-09-07 22:48:17,matpower/case118/2017-04-17,test,raw,OPTIMAL,13.677000,3845241.572704,49068,52836,5832,5832,0.00000000
2025-09-07 22:48:45,matpower/case118/2017-04-17,test,warm,OPTIMAL,14.460000,4000033.226012,49068,52836,5832,5832,0.00000000
2025-09-07 22:49:12,matpower/case118/2017-04-22,test,raw,OPTIMAL,13.883000,3365017.901353,49068,50336,5832,5832,0.00000000
2025-09-07 22:49:39,matpower/case118/2017-04-22,test,warm,OPTIMAL,13.468000,3520504.209117,49068,50336,5832,5832,0.00000000
2025-09-07 22:50:05,matpower/case118/2017-04-25,test,raw,OPTIMAL,13.094000,4110042.807877,49068,64474,5832,5832,0.00000000
2025-09-07 22:50:37,matpower/case118/2017-04-25,test,warm,OPTIMAL,12.982000,4112476.888227,49068,64474,5832,5832,0.00000000
2025-09-07 22:51:06,matpower/case118/2017-05-04,test,raw,OPTIMAL,16.476000,3529328.254132,49068,51102,5832,5832,0.00000000
2025-09-07 22:51:34,matpower/case118/2017-05-04,test,warm,OPTIMAL,13.899000,3574197.817749,49068,51102,5832,5832,0.00000000
2025-09-07 22:51:59,matpower/case118/2017-05-18,test,raw,OPTIMAL,12.424000,5655723.809015,49068,59890,5832,5832,0.00000000
2025-09-07 22:52:25,matpower/case118/2017-05-18,test,warm,OPTIMAL,11.633000,5448976.933598,49068,59890,5832,5832,0.00000000
2025-09-07 22:52:54,matpower/case118/2017-05-22,test,raw,OPTIMAL,16.445000,3841570.655895,49068,62766,5832,5832,0.00000000
2025-09-07 22:53:22,matpower/case118/2017-05-22,test,warm,OPTIMAL,13.608000,3841570.655895,49068,62766,5832,5832,0.00000000
2025-09-07 22:53:52,matpower/case118/2017-05-27,test,raw,OPTIMAL,17.217000,3269129.631247,49068,61492,5832,5832,0.00000000
2025-09-07 22:54:20,matpower/case118/2017-05-27,test,warm,OPTIMAL,14.058000,3269047.507229,49068,61492,5832,5832,0.00000000
2025-09-07 22:54:53,matpower/case118/2017-05-28,test,raw,OPTIMAL,19.378000,3167345.199860,49068,61492,5832,5832,0.00000000
2025-09-07 22:55:24,matpower/case118/2017-05-28,test,warm,OPTIMAL,16.690000,3167265.289054,49068,61492,5832,5832,0.00000000
2025-09-07 22:55:52,matpower/case118/2017-06-10,test,raw,OPTIMAL,15.467000,4724457.798041,49068,60324,5832,5832,0.00000000
2025-09-07 22:56:18,matpower/case118/2017-06-10,test,warm,OPTIMAL,12.291000,4724518.455499,49068,60324,5832,5832,0.00000000
2025-09-07 22:56:43,matpower/case118/2017-06-15,test,raw,OPTIMAL,10.950000,5055652.450509,49068,79642,5832,5832,0.00000000
2025-09-07 22:57:13,matpower/case118/2017-06-15,test,warm,OPTIMAL,11.656000,5052222.268122,49068,79642,5832,5832,0.00000000
2025-09-07 22:57:41,matpower/case118/2017-06-25,test,raw,OPTIMAL,14.652000,5378323.189359,49068,59566,5832,5832,0.00000000
2025-09-07 22:58:08,matpower/case118/2017-06-25,test,warm,OPTIMAL,13.364000,5389890.703334,49068,59566,5832,5832,0.00000000
2025-09-07 22:58:35,matpower/case118/2017-06-28,test,raw,OPTIMAL,14.380000,5601949.039070,49068,59350,5832,5832,0.00000000
2025-09-07 22:59:02,matpower/case118/2017-06-28,test,warm,OPTIMAL,12.636000,5640929.531583,49068,59350,5832,5832,0.00000000
2025-09-07 22:59:30,matpower/case118/2017-07-03,test,raw,OPTIMAL,14.894000,6405253.484085,49068,58980,5832,5832,0.00000000
2025-09-07 22:59:58,matpower/case118/2017-07-03,test,warm,OPTIMAL,13.804000,6430836.422458,49068,58980,5832,5832,0.00000000
2025-09-07 23:00:24,matpower/case118/2017-07-09,test,raw,OPTIMAL,11.111000,5918012.021684,49068,68980,5832,5832,0.00000000
2025-09-07 23:00:50,matpower/case118/2017-07-09,test,warm,OPTIMAL,11.316000,6131203.699964,49068,68980,5832,5832,0.00000000
2025-09-07 23:01:17,matpower/case118/2017-07-14,test,raw,OPTIMAL,14.430000,6883565.115446,49068,60762,5832,5832,0.00000000
2025-09-07 23:01:46,matpower/case118/2017-07-14,test,warm,OPTIMAL,14.639000,6883565.115446,49068,60762,5832,5832,0.00000000
2025-09-07 23:02:11,matpower/case118/2017-07-21,test,raw,OPTIMAL,11.777000,8006369.083503,49068,63420,5832,5832,0.00000000
2025-09-07 23:02:37,matpower/case118/2017-07-21,test,warm,OPTIMAL,12.151000,8006369.083503,49068,63420,5832,5832,0.00000000
2025-09-07 23:03:10,matpower/case118/2017-07-26,test,raw,OPTIMAL,19.910000,5573157.058580,49068,59350,5832,5832,0.00000000
2025-09-07 23:03:42,matpower/case118/2017-07-26,test,warm,OPTIMAL,14.234000,5573751.309895,49068,59350,5832,5832,0.00000000
2025-09-07 23:04:12,matpower/case118/2017-08-09,test,raw,OPTIMAL,16.736000,4804874.167417,49068,69466,5832,5832,0.00000000
2025-09-07 23:04:42,matpower/case118/2017-08-09,test,warm,OPTIMAL,15.174000,4935348.821968,49068,69466,5832,5832,0.00000000
2025-09-07 23:05:14,matpower/case118/2017-08-17,test,raw,OPTIMAL,17.616000,6204389.575916,49068,69466,5832,5832,0.00000000
2025-09-07 23:05:43,matpower/case118/2017-08-17,test,warm,OPTIMAL,14.557000,6264348.289453,49068,69466,5832,5832,0.00000000
2025-09-07 23:06:09,matpower/case118/2017-08-18,test,raw,OPTIMAL,12.824000,6314055.736189,49068,77376,5832,5832,0.00000000
2025-09-07 23:06:35,matpower/case118/2017-08-18,test,warm,OPTIMAL,12.150000,6609327.008976,49068,77376,5832,5832,0.00000000
2025-09-07 23:07:02,matpower/case118/2017-09-02,test,raw,OPTIMAL,13.877000,3677044.847255,49068,61492,5832,5832,0.00000000
2025-09-07 23:07:31,matpower/case118/2017-09-02,test,warm,OPTIMAL,14.343000,3677044.847255,49068,61492,5832,5832,0.00000000
2025-09-07 23:07:55,matpower/case118/2017-09-17,test,raw,OPTIMAL,11.573000,4947218.321797,49068,58366,5832,5832,0.00000000
2025-09-07 23:08:22,matpower/case118/2017-09-17,test,warm,OPTIMAL,12.178000,4988603.841760,49068,58366,5832,5832,0.00000000
2025-09-07 23:08:45,matpower/case118/2017-09-18,test,raw,OPTIMAL,10.821000,5460232.276392,49068,64444,5832,5832,0.00000000
2025-09-07 23:09:11,matpower/case118/2017-09-18,test,warm,OPTIMAL,11.562000,5522056.788278,49068,64444,5832,5832,0.00000000
2025-09-07 23:09:33,matpower/case118/2017-10-01,test,raw,OPTIMAL,9.556000,3415562.132509,49068,49954,5832,5832,0.00000000
2025-09-07 23:09:59,matpower/case118/2017-10-01,test,warm,OPTIMAL,7.261000,3415562.132509,49068,49954,5832,5832,0.00000000
2025-09-07 23:10:22,matpower/case118/2017-10-28,test,raw,OPTIMAL,10.360000,3956805.386788,49068,65272,5832,5832,0.00000000
2025-09-07 23:10:46,matpower/case118/2017-10-28,test,warm,OPTIMAL,9.122000,3842296.222064,49068,65272,5832,5832,0.00000000
2025-09-07 23:11:09,matpower/case118/2017-10-29,test,raw,OPTIMAL,10.651000,4129792.460667,49068,49954,5832,5832,0.00000000
2025-09-07 23:11:31,matpower/case118/2017-10-29,test,warm,OPTIMAL,8.080000,4255410.835487,49068,49954,5832,5832,0.00000000
2025-09-07 23:11:58,matpower/case118/2017-11-13,test,raw,OPTIMAL,13.112000,5006780.584562,49068,104516,5832,5832,0.00000000
2025-09-07 23:12:26,matpower/case118/2017-11-13,test,warm,OPTIMAL,12.681000,5014144.647585,49068,104516,5832,5832,0.00000000
2025-09-07 23:12:51,matpower/case118/2017-11-30,test,raw,OPTIMAL,11.646000,5327057.835042,49068,97416,5832,5832,0.00000000
2025-09-07 23:13:18,matpower/case118/2017-11-30,test,warm,OPTIMAL,12.471000,5273903.440782,49068,97416,5832,5832,0.00000000
2025-09-07 23:13:36,matpower/case118/2017-12-06,test,raw,OPTIMAL,5.588000,5591136.135557,49068,42192,5832,5832,0.00000000
2025-09-07 23:13:56,matpower/case118/2017-12-06,test,warm,OPTIMAL,5.930000,5585426.320581,49068,42192,5832,5832,0.00000000
2025-09-07 23:14:28,matpower/case118/2017-12-18,test,raw,OPTIMAL,18.830000,7224018.338950,49068,83644,5832,5832,3.41305107
2025-09-07 23:15:02,matpower/case118/2017-12-18,test,warm,OPTIMAL,19.114000,7239584.899674,49068,83644,5832,5832,0.00000000
2025-09-07 23:15:44,matpower/case118/2017-12-20,test,raw,OPTIMAL,23.329000,6853264.635246,49068,120008,5832,5832,0.00000000
2025-09-07 23:16:22,matpower/case118/2017-12-20,test,warm,OPTIMAL,23.314000,7024966.634181,49068,120008,5832,5832,0.00000000
2025-09-07 23:16:55,matpower/case118/2017-12-22,test,raw,OPTIMAL,19.772000,6710759.707876,49068,62022,5832,5832,0.00000000
2025-09-07 23:17:23,matpower/case118/2017-12-22,test,warm,OPTIMAL,14.105000,6992303.159182,49068,62022,5832,5832,27.91104808

```

### File: `src/data/output/matpower/case118/compare_matpower_case118_20250904_175820.csv`

```
timestamp,instance_name,split,method,status,status_code,runtime_sec,mip_gap,obj_val,obj_bound,nodes,feasible_ok,warm_start_applied_vars
2025-09-04 18:00:01,matpower/case118/2017-01-02,train,raw,OPTIMAL,2,49.088000,0.00013321,4844424.994625,4843779.661003,1,OK,
2025-09-04 18:01:39,matpower/case118/2017-01-05,train,raw,OPTIMAL,2,49.747000,0.00017842,6167079.101966,6165978.780926,1,OK,
2025-09-04 18:03:17,matpower/case118/2017-01-06,train,raw,OPTIMAL,2,49.696000,0.00000486,6043528.811815,6043499.429915,1,OK,
2025-09-04 18:04:54,matpower/case118/2017-01-07,train,raw,OPTIMAL,2,49.777000,0.00023482,6299208.822867,6297729.629954,1,OK,
2025-09-04 18:06:33,matpower/case118/2017-01-08,train,raw,OPTIMAL,2,50.421000,0.00000000,6384833.162047,6384833.162047,1,OK,
2025-09-04 18:09:02,matpower/case118/2017-01-11,train,raw,OPTIMAL,2,83.409000,0.00002041,5213619.814083,5213513.406415,1,OK,
2025-09-04 21:22:25,matpower/case118/2017-01-12,train,raw,OPTIMAL,2,57.432000,0.00031706,4984328.982647,4982748.659045,1,OK,
2025-09-04 21:24:52,matpower/case118/2017-01-15,train,raw,OPTIMAL,2,71.525000,0.00002470,4815570.136330,4815451.177755,1,OK,
2025-09-04 21:27:30,matpower/case118/2017-01-16,train,raw,OPTIMAL,2,82.454000,0.00000602,5480572.300569,5480539.284721,1,OK,
2025-09-04 21:29:56,matpower/case118/2017-01-17,train,raw,OPTIMAL,2,69.302000,0.00001376,5497060.046869,5496984.431462,1,OK,
2025-09-04 21:32:25,matpower/case118/2017-01-18,train,raw,OPTIMAL,2,73.647000,0.00013971,5393659.533144,5392905.972862,1,OK,
2025-09-04 21:34:52,matpower/case118/2017-01-19,train,raw,OPTIMAL,2,70.350000,0.00001726,5469032.706635,5468938.286915,1,OK,
2025-09-04 21:37:21,matpower/case118/2017-01-21,train,raw,OPTIMAL,2,73.334000,0.00011080,5011221.001296,5010665.777526,1,OK,
2025-09-04 21:39:48,matpower/case118/2017-01-22,train,raw,OPTIMAL,2,69.856000,0.00002969,5191372.832008,5191218.712735,1,OK,
2025-09-04 21:42:28,matpower/case118/2017-01-23,train,raw,OPTIMAL,2,84.522000,0.00005933,5868275.365160,5867927.225284,1,OK,
2025-09-04 21:44:16,matpower/case118/2017-01-26,train,raw,OPTIMAL,2,52.210000,0.00006735,5269491.012978,5269136.109674,1,OK,
2025-09-04 21:45:58,matpower/case118/2017-01-27,train,raw,OPTIMAL,2,53.220000,0.00102171,4828403.541371,4823470.321043,1,OK,
2025-09-04 21:47:37,matpower/case118/2017-01-31,train,raw,OPTIMAL,2,50.360000,0.00024672,5631243.555296,5629854.233487,1,OK,
2025-09-04 21:49:19,matpower/case118/2017-02-01,train,raw,OPTIMAL,2,52.430000,0.00004152,5728872.209972,5728634.350212,1,OK,
2025-09-04 21:51:01,matpower/case118/2017-02-02,train,raw,OPTIMAL,2,53.035000,0.00027208,5964267.232530,5962644.484264,1,OK,
2025-09-04 21:52:49,matpower/case118/2017-02-03,train,raw,OPTIMAL,2,59.148000,0.00010822,5857914.303498,5857280.377313,1,OK,
2025-09-04 21:54:32,matpower/case118/2017-02-05,train,raw,OPTIMAL,2,54.180000,0.00010915,5226625.044439,5226054.562683,1,OK,
2025-09-04 21:56:14,matpower/case118/2017-02-06,train,raw,OPTIMAL,2,52.032000,0.00009528,5337348.583769,5336840.037145,1,OK,
2025-09-04 21:57:54,matpower/case118/2017-02-08,train,raw,OPTIMAL,2,51.607000,0.00007520,5032278.961333,5031900.513826,1,OK,
2025-09-04 21:59:33,matpower/case118/2017-02-09,train,raw,OPTIMAL,2,50.296000,0.00001388,6543758.221165,6543667.379458,1,OK,
2025-09-04 22:01:16,matpower/case118/2017-02-11,train,raw,OPTIMAL,2,52.678000,0.00011450,5629484.377440,5628839.812782,1,OK,
2025-09-04 22:03:09,matpower/case118/2017-02-13,train,raw,OPTIMAL,2,62.390000,0.00025829,6487317.536677,6485641.956185,1,OK,
2025-09-04 22:05:20,matpower/case118/2017-02-14,train,raw,OPTIMAL,2,76.541000,0.00020207,6516423.306406,6515106.506763,1,OK,
2025-09-04 22:07:50,matpower/case118/2017-02-15,train,raw,OPTIMAL,2,74.106000,0.00015354,6058511.991679,6057581.778726,1,OK,
2025-09-04 22:10:22,matpower/case118/2017-02-16,train,raw,OPTIMAL,2,76.683000,0.00001678,6435605.444325,6435497.481817,1,OK,
2025-09-04 22:12:32,matpower/case118/2017-02-17,train,raw,OPTIMAL,2,62.700000,0.00026493,5172179.310743,5170809.036971,1,OK,
2025-09-04 22:14:29,matpower/case118/2017-02-18,train,raw,OPTIMAL,2,62.096000,0.00121541,4306920.128827,4301685.454944,1,OK,
2025-09-04 22:16:27,matpower/case118/2017-02-19,train,raw,OPTIMAL,2,62.803000,0.00001128,4181299.025532,4181251.850114,1,OK,
2025-09-04 22:18:18,matpower/case118/2017-02-20,train,raw,OPTIMAL,2,55.896000,0.00010706,4841724.610617,4841206.232551,1,OK,
2025-09-04 22:20:16,matpower/case118/2017-02-21,train,raw,OPTIMAL,2,63.458000,0.00122625,4949937.818419,4943867.967207,1,OK,
2025-09-04 22:22:16,matpower/case118/2017-02-23,train,raw,OPTIMAL,2,62.352000,0.00008628,4396036.178819,4395656.908955,1,OK,
2025-09-04 22:24:10,matpower/case118/2017-02-24,train,raw,OPTIMAL,2,59.531000,0.00027180,4062168.967666,4061064.861371,1,OK,
2025-09-04 22:26:20,matpower/case118/2017-02-26,train,raw,OPTIMAL,2,73.363000,0.00012060,3919573.303562,3919100.602888,1,OK,
2025-09-04 22:28:15,matpower/case118/2017-02-27,train,raw,OPTIMAL,2,59.323000,0.00008324,4576743.106220,4576362.123883,1,OK,
2025-09-04 22:30:05,matpower/case118/2017-03-01,train,raw,OPTIMAL,2,54.642000,0.00002489,4271665.729725,4271559.404029,1,OK,
2025-09-04 22:32:01,matpower/case118/2017-03-02,train,raw,OPTIMAL,2,59.859000,0.00006748,4526296.514793,4525991.057720,1,OK,
2025-09-04 22:33:51,matpower/case118/2017-03-03,train,raw,OPTIMAL,2,55.311000,0.00007830,4575087.766783,4574729.544363,1,OK,
2025-09-04 22:35:54,matpower/case118/2017-03-05,train,raw,OPTIMAL,2,67.923000,0.00026129,4529769.924424,4528586.338568,1,OK,
2025-09-04 22:37:45,matpower/case118/2017-03-06,train,raw,OPTIMAL,2,55.406000,0.00006951,4199555.092940,4199263.201445,1,OK,
2025-09-04 22:39:48,matpower/case118/2017-03-08,train,raw,OPTIMAL,2,67.327000,0.00075721,4011638.378017,4008600.718498,1,OK,
2025-09-04 22:41:45,matpower/case118/2017-03-12,train,raw,OPTIMAL,2,61.095000,0.00009135,4973945.243072,4973490.851736,1,OK,
2025-09-04 22:43:42,matpower/case118/2017-03-13,train,raw,OPTIMAL,2,62.492000,0.00000671,5205316.056590,5205281.147765,1,OK,
2025-09-04 22:45:32,matpower/case118/2017-03-15,train,raw,OPTIMAL,2,54.378000,0.00000656,5716822.119742,5716784.590326,1,OK,
2025-09-04 22:47:32,matpower/case118/2017-03-16,train,raw,OPTIMAL,2,63.936000,0.00019209,5566720.195358,5565650.865681,1,OK,
2025-09-04 22:49:22,matpower/case118/2017-03-17,train,raw,OPTIMAL,2,55.105000,0.00012173,4566031.760944,4565475.918943,1,OK,
2025-09-04 22:51:19,matpower/case118/2017-03-18,train,raw,OPTIMAL,2,60.586000,0.00041452,3952126.170027,3950487.945040,1,OK,
2025-09-04 22:53:10,matpower/case118/2017-03-19,train,raw,OPTIMAL,2,56.201000,0.00019533,4254333.768088,4253502.761597,1,OK,
2025-09-04 22:55:03,matpower/case118/2017-03-21,train,raw,OPTIMAL,2,57.455000,0.00013985,4432492.031231,4431872.162791,1,OK,
2025-09-04 22:56:57,matpower/case118/2017-03-22,train,raw,OPTIMAL,2,57.857000,0.00010062,5235438.723326,5234911.957915,1,OK,
2025-09-04 22:59:10,matpower/case118/2017-03-24,train,raw,OPTIMAL,2,77.966000,0.00220204,4128644.461442,4119553.002757,1,OK,
2025-09-04 23:01:06,matpower/case118/2017-03-27,train,raw,OPTIMAL,2,59.453000,0.00069278,3912062.165023,3909351.953485,1,OK,
2025-09-04 23:03:14,matpower/case118/2017-03-28,train,raw,OPTIMAL,2,72.610000,0.00097807,3893639.718969,3889831.470274,1,OK,
2025-09-04 23:05:09,matpower/case118/2017-03-31,train,raw,OPTIMAL,2,58.574000,0.00083525,4343389.856328,4339762.043418,1,OK,
2025-09-04 23:07:16,matpower/case118/2017-04-01,train,raw,OPTIMAL,2,71.521000,0.00123847,3797531.537966,3792828.418058,1,OK,
2025-09-04 23:09:08,matpower/case118/2017-04-03,train,raw,OPTIMAL,2,56.806000,0.00045189,4207331.420769,4205430.187890,1,OK,
2025-09-04 23:11:05,matpower/case118/2017-04-04,train,raw,OPTIMAL,2,62.259000,0.00049959,4160707.421025,4158628.774128,1,OK,
2025-09-04 23:13:02,matpower/case118/2017-04-05,train,raw,OPTIMAL,2,61.907000,0.00038201,4171037.070334,4169443.676456,1,OK,
2025-09-04 23:14:55,matpower/case118/2017-04-06,train,raw,OPTIMAL,2,57.873000,0.00036959,4377494.529453,4375876.646490,1,OK,
2025-09-04 23:16:45,matpower/case118/2017-04-08,train,raw,OPTIMAL,2,55.303000,0.00069701,3839159.623285,3836483.681331,1,OK,
2025-09-04 23:18:39,matpower/case118/2017-04-09,train,raw,OPTIMAL,2,58.496000,0.00012598,3783646.272427,3783169.621677,1,OK,
2025-09-04 23:20:33,matpower/case118/2017-04-10,train,raw,OPTIMAL,2,58.420000,0.00032470,4148696.243938,4147349.169809,1,OK,
2025-09-04 23:22:27,matpower/case118/2017-04-11,train,raw,OPTIMAL,2,58.375000,0.00016938,4243802.950110,4243084.153859,1,OK,
2025-09-04 23:24:24,matpower/case118/2017-04-12,train,raw,OPTIMAL,2,61.474000,0.00031951,4104851.344508,4103539.822577,1,OK,
2025-09-04 23:26:16,matpower/case118/2017-04-13,train,raw,OPTIMAL,2,56.752000,0.00049238,3954991.706927,3953044.350984,1,OK,
2025-09-04 23:28:21,matpower/case118/2017-04-15,train,raw,OPTIMAL,2,69.565000,0.00076377,3415974.662532,3413365.630568,1,OK,
2025-09-04 23:30:17,matpower/case118/2017-04-16,train,raw,OPTIMAL,2,60.404000,0.00005378,3867060.292944,3866852.316098,1,OK,
2025-09-04 23:32:10,matpower/case118/2017-04-18,train,raw,OPTIMAL,2,57.929000,0.00014489,3712398.412625,3711860.514176,1,OK,
2025-09-04 23:34:08,matpower/case118/2017-04-19,train,raw,OPTIMAL,2,62.155000,0.00002227,3622366.935837,3622286.278298,1,OK,
2025-09-04 23:36:13,matpower/case118/2017-04-21,train,raw,OPTIMAL,2,69.487000,0.00111022,3716777.948732,3712651.502026,1,OK,
2025-09-04 23:38:26,matpower/case118/2017-04-23,train,raw,OPTIMAL,2,77.870000,0.00018814,3471886.716562,3471233.531714,1,OK,
2025-09-04 23:40:21,matpower/case118/2017-04-24,train,raw,OPTIMAL,2,60.035000,0.00028920,3946647.354799,3945506.002194,1,OK,
2025-09-04 23:42:15,matpower/case118/2017-04-26,train,raw,OPTIMAL,2,58.160000,0.00019985,3862320.875425,3861548.980824,1,OK,
2025-09-04 23:44:09,matpower/case118/2017-04-27,train,raw,OPTIMAL,2,58.773000,0.00094325,4197050.723042,4193091.854769,1,OK,
2025-09-04 23:46:20,matpower/case118/2017-04-28,train,raw,OPTIMAL,2,74.955000,0.00058895,4058195.873775,4055805.802042,1,OK,
2025-09-04 23:48:22,matpower/case118/2017-04-29,train,raw,OPTIMAL,2,65.974000,0.00817939,3664533.362046,3634559.730077,1,OK,
2025-09-04 23:50:35,matpower/case118/2017-04-30,train,raw,OPTIMAL,2,77.410000,0.00305894,3334215.590560,3324016.420115,1,OK,
2025-09-04 23:52:45,matpower/case118/2017-05-01,train,raw,OPTIMAL,2,74.075000,0.00890603,3641876.035388,3609441.389461,1,OK,
2025-09-04 23:55:06,matpower/case118/2017-05-02,train,raw,OPTIMAL,2,85.364000,0.00524392,3425601.368909,3407637.800477,1,OK,
2025-09-04 23:57:14,matpower/case118/2017-05-03,train,raw,OPTIMAL,2,71.526000,0.00154855,3148494.536617,3143618.933748,1,OK,
2025-09-04 23:59:15,matpower/case118/2017-05-05,train,raw,OPTIMAL,2,65.655000,0.00954300,3457470.714886,3424476.060190,1,OK,
2025-09-05 00:01:31,matpower/case118/2017-05-06,train,raw,OPTIMAL,2,79.012000,0.00380641,3146014.863390,3134039.845981,1,OK,
2025-09-05 00:03:37,matpower/case118/2017-05-07,train,raw,OPTIMAL,2,70.643000,0.00328891,3328166.488035,3317220.463329,1,OK,
2025-09-05 00:05:39,matpower/case118/2017-05-08,train,raw,OPTIMAL,2,65.715000,0.00665494,3839142.063299,3813592.821740,1,OK,
2025-09-05 00:07:47,matpower/case118/2017-05-10,train,raw,OPTIMAL,2,71.529000,0.00006355,4097579.202842,4097318.801294,1,OK,
2025-09-05 00:09:48,matpower/case118/2017-05-11,train,raw,OPTIMAL,2,65.800000,0.00036067,4286083.017911,4284537.166960,1,OK,
2025-09-05 00:11:48,matpower/case118/2017-05-12,train,raw,OPTIMAL,2,64.178000,0.00015107,4202750.880985,4202115.951028,1,OK,
2025-09-05 00:13:40,matpower/case118/2017-05-14,train,raw,OPTIMAL,2,56.018000,0.00024556,3859630.151647,3858682.394414,1,OK,
2025-09-05 00:15:31,matpower/case118/2017-05-15,train,raw,OPTIMAL,2,55.349000,0.00017517,4237803.206300,4237060.883140,1,OK,
2025-09-05 00:17:26,matpower/case118/2017-05-17,train,raw,OPTIMAL,2,58.552000,0.00028353,5198766.939996,5197292.912878,1,OK,
2025-09-05 00:19:47,matpower/case118/2017-05-19,train,raw,OPTIMAL,2,85.123000,0.00057032,4869262.594291,4866485.548236,1,OK,
2025-09-05 00:22:00,matpower/case118/2017-05-20,train,raw,OPTIMAL,2,77.425000,0.00091184,3421401.080674,3418281.308771,1,OK,
2025-09-05 00:24:05,matpower/case118/2017-05-21,train,raw,OPTIMAL,2,69.718000,0.00057589,3343782.037784,3341856.393096,1,OK,
2025-09-05 00:25:57,matpower/case118/2017-05-23,train,raw,OPTIMAL,2,56.882000,0.00015243,3869732.579024,3869142.730609,1,OK,
2025-09-05 00:27:53,matpower/case118/2017-05-24,train,raw,OPTIMAL,2,60.390000,0.00038523,4005402.262152,4003859.259188,1,OK,
2025-09-05 00:29:46,matpower/case118/2017-05-26,train,raw,OPTIMAL,2,57.466000,0.00116224,3682976.603357,3678696.092710,1,OK,
2025-09-05 00:31:40,matpower/case118/2017-05-29,train,raw,OPTIMAL,2,57.908000,0.00049275,3347867.449056,3346217.781049,1,OK,
2025-09-05 00:33:39,matpower/case118/2017-05-30,train,raw,OPTIMAL,2,64.049000,0.00019024,3792689.726057,3791968.222661,1,OK,
2025-09-05 00:35:31,matpower/case118/2017-05-31,train,raw,OPTIMAL,2,55.663000,0.00008555,4029163.925489,4028819.240005,1,OK,
2025-09-05 00:37:23,matpower/case118/2017-06-01,train,raw,OPTIMAL,2,55.241000,0.00007735,4022639.377375,4022328.208051,1,OK,
2025-09-05 00:39:14,matpower/case118/2017-06-02,train,raw,OPTIMAL,2,55.577000,0.00087651,4069430.341564,4065863.447773,1,OK,
2025-09-05 00:41:07,matpower/case118/2017-06-03,train,raw,OPTIMAL,2,56.724000,0.00060897,3830197.760638,3827865.296418,1,OK,
2025-09-05 00:42:59,matpower/case118/2017-06-04,train,raw,OPTIMAL,2,56.449000,0.00024814,4135732.697740,4134706.471033,1,OK,
2025-09-05 00:44:49,matpower/case118/2017-06-05,train,raw,OPTIMAL,2,54.491000,0.00004190,4602207.658313,4602014.821766,1,OK,
2025-09-05 00:46:38,matpower/case118/2017-06-06,train,raw,OPTIMAL,2,52.922000,0.00039852,4279056.578223,4277351.291420,1,OK,
2025-09-05 00:48:58,matpower/case118/2017-06-07,train,raw,OPTIMAL,2,84.473000,0.00058886,3993973.402546,3991621.491664,1,OK,
2025-09-05 00:51:21,matpower/case118/2017-06-08,train,raw,OPTIMAL,2,87.201000,0.00026052,4096615.493010,4095548.256975,1,OK,
2025-09-05 00:53:09,matpower/case118/2017-06-09,train,raw,OPTIMAL,2,52.491000,0.00013745,4384073.933709,4383471.344904,1,OK,
2025-09-05 00:55:06,matpower/case118/2017-06-11,train,raw,OPTIMAL,2,61.385000,0.00004834,5561841.926280,5561573.092459,1,OK,
2025-09-05 00:57:04,matpower/case118/2017-06-12,train,raw,OPTIMAL,2,62.039000,0.00001985,6831236.907647,6831101.322005,1,OK,
2025-09-05 00:59:08,matpower/case118/2017-06-13,train,raw,OPTIMAL,2,69.781000,0.00000685,7028557.702271,7028509.536819,1,OK,
2025-09-05 01:00:59,matpower/case118/2017-06-16,train,raw,OPTIMAL,2,55.484000,0.00010594,5157460.944670,5156914.557476,1,OK,
2025-09-05 01:02:55,matpower/case118/2017-06-17,train,raw,OPTIMAL,2,60.205000,0.00003364,5352297.256413,5352117.188762,1,OK,
2025-09-05 01:05:13,matpower/case118/2017-06-19,train,raw,OPTIMAL,2,83.445000,0.00013679,6499380.324378,6498491.306408,1,OK,
2025-09-05 01:07:07,matpower/case118/2017-06-20,train,raw,OPTIMAL,2,57.586000,0.00005363,6254019.129143,6253683.754984,1,OK,
2025-09-05 01:09:02,matpower/case118/2017-06-21,train,raw,OPTIMAL,2,59.633000,0.00002396,6115423.044228,6115276.508357,1,OK,
2025-09-05 01:11:11,matpower/case118/2017-06-22,train,raw,OPTIMAL,2,72.200000,0.00003845,6943090.599400,6942823.659349,1,OK,
2025-09-05 01:13:07,matpower/case118/2017-06-23,train,raw,OPTIMAL,2,59.472000,0.00003503,6768858.612653,6768621.487034,1,OK,
2025-09-05 01:14:58,matpower/case118/2017-06-24,train,raw,OPTIMAL,2,55.382000,0.00008181,5943870.983599,5943384.725491,1,OK,
2025-09-05 01:16:53,matpower/case118/2017-06-26,train,raw,OPTIMAL,2,59.986000,0.00017983,5399307.877929,5398336.906687,1,OK,
2025-09-05 01:18:50,matpower/case118/2017-06-29,train,raw,OPTIMAL,2,60.270000,0.00011915,6912090.455089,6911266.913504,1,OK,
2025-09-05 01:21:07,matpower/case118/2017-06-30,train,raw,OPTIMAL,2,81.226000,0.00007095,7285093.940495,7284577.098428,1,OK,
2025-09-05 01:23:01,matpower/case118/2017-07-01,train,raw,OPTIMAL,2,58.449000,0.00005560,6841148.159642,6840767.768128,1,OK,
2025-09-05 01:24:57,matpower/case118/2017-07-02,train,raw,OPTIMAL,2,60.064000,0.00005425,6750133.232013,6749767.009055,1,OK,
2025-09-05 01:26:48,matpower/case118/2017-07-04,train,raw,OPTIMAL,2,55.568000,0.00018195,6170160.162100,6169037.473785,1,OK,
2025-09-05 01:28:39,matpower/case118/2017-07-05,train,raw,OPTIMAL,2,54.718000,0.00010846,6290768.857808,6290086.532938,1,OK,
2025-09-05 01:30:48,matpower/case118/2017-07-06,train,raw,OPTIMAL,2,73.356000,0.00002101,6144488.226836,6144359.122413,1,OK,
2025-09-05 01:32:43,matpower/case118/2017-07-07,train,raw,OPTIMAL,2,59.277000,0.00007238,6606131.315061,6605653.164195,1,OK,
2025-09-05 01:34:35,matpower/case118/2017-07-08,train,raw,OPTIMAL,2,56.951000,0.00046291,6163530.041813,6160676.892809,1,OK,
2025-09-05 01:36:40,matpower/case118/2017-07-10,train,raw,OPTIMAL,2,68.419000,0.00001629,6723077.062295,6722967.553910,1,OK,
2025-09-05 01:38:33,matpower/case118/2017-07-11,train,raw,OPTIMAL,2,57.412000,0.00005229,7574380.225344,7573984.128393,1,OK,
2025-09-05 01:40:24,matpower/case118/2017-07-12,train,raw,OPTIMAL,2,55.832000,0.00001188,8043371.313534,8043275.734671,1,OK,
2025-09-05 01:42:20,matpower/case118/2017-07-13,train,raw,OPTIMAL,2,63.472000,0.00002076,8244642.286241,8244471.129915,1,OK,
2025-09-05 01:44:08,matpower/case118/2017-07-15,train,raw,OPTIMAL,2,55.280000,0.00014274,6031414.621420,6030553.684128,1,OK,
2025-09-05 01:46:08,matpower/case118/2017-07-17,train,raw,OPTIMAL,2,66.842000,0.00001579,6803831.274196,6803723.842323,1,OK,
2025-09-05 01:47:53,matpower/case118/2017-07-20,train,raw,OPTIMAL,2,52.929000,0.00010573,8294303.589356,8293426.637875,1,OK,
2025-09-05 01:49:51,matpower/case118/2017-07-22,train,raw,OPTIMAL,2,64.902000,0.00002377,6952032.417987,6951867.201355,1,OK,
2025-09-05 01:51:40,matpower/case118/2017-07-23,train,raw,OPTIMAL,2,56.270000,0.00010763,6306408.303979,6305729.543678,1,OK,
2025-09-05 01:53:38,matpower/case118/2017-07-24,train,raw,OPTIMAL,2,64.355000,0.00072802,6542528.028154,6537764.957128,1,OK,
2025-09-05 01:55:32,matpower/case118/2017-07-25,train,raw,OPTIMAL,2,61.111000,0.00206662,5640035.352644,5628379.554110,1,OK,
2025-09-05 01:57:15,matpower/case118/2017-07-27,train,raw,OPTIMAL,2,49.998000,0.00062803,6325908.445528,6321935.596728,1,OK,
2025-09-05 01:59:12,matpower/case118/2017-07-28,train,raw,OPTIMAL,2,64.227000,0.00066950,5905925.848222,5901971.855841,1,OK,
2025-09-05 02:01:21,matpower/case118/2017-07-29,train,raw,OPTIMAL,2,76.160000,0.00276640,4784025.638924,4770791.129892,1,OK,
2025-09-05 02:03:18,matpower/case118/2017-07-30,train,raw,OPTIMAL,2,63.298000,0.00027684,4977556.442801,4976178.454363,1,OK,
2025-09-05 02:05:06,matpower/case118/2017-08-01,train,raw,OPTIMAL,2,55.632000,0.00022294,7032106.233454,7030538.524885,1,OK,
2025-09-05 02:06:58,matpower/case118/2017-08-02,train,raw,OPTIMAL,2,58.600000,0.00026981,7191421.118098,7189480.833978,1,OK,
2025-09-05 02:08:52,matpower/case118/2017-08-03,train,raw,OPTIMAL,2,61.806000,0.00017339,6928621.044160,6927419.700564,1,OK,
2025-09-05 02:10:30,matpower/case118/2017-08-04,train,raw,OPTIMAL,2,44.520000,0.00180183,5716370.772767,5706070.844306,1,OK,
2025-09-05 02:12:06,matpower/case118/2017-08-05,train,raw,OPTIMAL,2,43.630000,0.00362973,4473785.203581,4457546.553193,1,OK,
2025-09-05 02:13:46,matpower/case118/2017-08-06,train,raw,OPTIMAL,2,46.339000,0.00340541,4462039.173785,4446844.087670,1,OK,
2025-09-05 02:15:27,matpower/case118/2017-08-07,train,raw,OPTIMAL,2,48.282000,0.00193645,4894355.582060,4884877.922209,1,OK,
2025-09-05 02:17:02,matpower/case118/2017-08-08,train,raw,OPTIMAL,2,42.044000,0.00278416,4678418.722312,4665393.268065,1,OK,
2025-09-05 02:18:42,matpower/case118/2017-08-10,train,raw,OPTIMAL,2,46.925000,0.00314521,5255912.625233,5239381.672621,1,OK,
2025-09-05 02:20:23,matpower/case118/2017-08-11,train,raw,OPTIMAL,2,47.400000,0.00303890,5088299.336459,5072836.509972,1,OK,
2025-09-05 02:22:00,matpower/case118/2017-08-12,train,raw,OPTIMAL,2,43.895000,0.00255336,4682626.480520,4670670.061041,1,OK,
2025-09-05 02:23:22,matpower/case118/2017-08-13,train,raw,OPTIMAL,2,39.828000,0.00167357,5067979.391381,5059497.758431,1,OK,
2025-09-05 02:24:32,matpower/case118/2017-08-15,train,raw,OPTIMAL,2,35.728000,0.00194650,6166905.223574,6154901.340473,1,OK,
2025-09-05 02:25:42,matpower/case118/2017-08-16,train,raw,OPTIMAL,2,36.259000,0.00550494,6270556.525602,6236037.476899,1,OK,
2025-09-05 02:26:50,matpower/case118/2017-08-19,train,raw,OPTIMAL,2,35.654000,0.00283568,5963298.153728,5946388.151580,1,OK,
2025-09-05 02:27:56,matpower/case118/2017-08-20,train,raw,OPTIMAL,2,32.864000,0.00244420,5618304.246837,5604571.966875,1,OK,
2025-09-05 02:29:03,matpower/case118/2017-08-23,train,raw,OPTIMAL,2,33.251000,0.00128486,5527847.505455,5520745.017293,1,OK,
2025-09-05 02:30:08,matpower/case118/2017-08-24,train,raw,OPTIMAL,2,32.082000,0.00303763,5041526.885678,5026212.586751,1,OK,
2025-09-05 02:31:12,matpower/case118/2017-08-25,train,raw,OPTIMAL,2,31.070000,0.00309978,4649000.289135,4634589.398220,1,OK,
2025-09-05 02:32:20,matpower/case118/2017-08-27,train,raw,OPTIMAL,2,34.380000,0.00448554,4726128.564341,4704929.324100,1,OK,
2025-09-05 02:33:29,matpower/case118/2017-08-28,train,raw,OPTIMAL,2,35.783000,0.00507211,5073149.711993,5047418.145142,1,OK,
2025-09-05 02:34:39,matpower/case118/2017-08-29,train,raw,OPTIMAL,2,36.933000,0.00659824,4972191.972984,4939384.259491,1,OK,
2025-09-05 02:35:58,matpower/case118/2017-08-30,train,raw,OPTIMAL,2,45.267000,0.00379993,5104348.315411,5084952.124960,1,OK,
2025-09-05 02:37:04,matpower/case118/2017-08-31,train,raw,OPTIMAL,2,33.176000,0.00551577,5205242.626132,5176531.682459,1,OK,
2025-09-05 02:38:15,matpower/case118/2017-09-01,train,raw,OPTIMAL,2,37.566000,0.00614726,4445141.324649,4417815.905042,1,OK,
2025-09-05 02:39:27,matpower/case118/2017-09-03,train,raw,OPTIMAL,2,38.528000,0.00084052,3914094.625660,3910804.736784,1,OK,
2025-09-05 02:40:32,matpower/case118/2017-09-04,train,raw,OPTIMAL,2,32.405000,0.00541114,4596881.544678,4572007.193799,1,OK,
2025-09-05 02:41:59,matpower/case118/2017-09-05,train,raw,OPTIMAL,2,53.010000,0.00875048,5301570.769755,5255179.485068,1,OK,
2025-09-05 02:43:09,matpower/case118/2017-09-06,train,raw,OPTIMAL,2,36.417000,0.00366164,4507919.277645,4491412.880954,1,OK,
2025-09-05 02:44:18,matpower/case118/2017-09-07,train,raw,OPTIMAL,2,36.351000,0.00530556,4250606.715186,4228054.886350,1,OK,
2025-09-05 02:45:27,matpower/case118/2017-09-08,train,raw,OPTIMAL,2,35.521000,0.00739557,4234277.016958,4202962.103778,1,OK,
2025-09-05 02:46:34,matpower/case118/2017-09-09,train,raw,OPTIMAL,2,33.202000,0.00627426,3890682.501230,3866271.360346,1,OK,
2025-09-05 02:47:39,matpower/case118/2017-09-10,train,raw,OPTIMAL,2,32.241000,0.00391544,3855912.578037,3840814.975114,1,OK,
2025-09-05 02:48:46,matpower/case118/2017-09-11,train,raw,OPTIMAL,2,33.446000,0.00634008,4344944.704115,4317397.402762,1,OK,
2025-09-05 02:50:03,matpower/case118/2017-09-13,train,raw,OPTIMAL,2,43.546000,0.00426000,5108285.700511,5086524.413786,1,OK,
2025-09-05 02:51:21,matpower/case118/2017-09-15,train,raw,OPTIMAL,2,45.018000,0.00064853,5287935.239764,5284505.838363,1,OK,
2025-09-05 02:52:41,matpower/case118/2017-09-16,train,raw,OPTIMAL,2,46.397000,0.00489860,4853432.130091,4829657.088994,1,OK,
2025-09-05 02:53:54,matpower/case118/2017-09-19,train,raw,OPTIMAL,2,39.869000,0.00255760,5522254.393697,5508130.685984,1,OK,
2025-09-05 02:54:57,matpower/case118/2017-09-20,train,raw,OPTIMAL,2,29.993000,0.00228206,6047014.629529,6033214.967394,1,OK,
2025-09-05 02:56:02,matpower/case118/2017-09-21,train,raw,OPTIMAL,2,31.518000,0.00224078,6782521.398857,6767323.237977,1,OK,
2025-09-05 02:57:21,matpower/case118/2017-09-23,train,raw,OPTIMAL,2,46.180000,0.00065473,5797074.896608,5793279.400570,1,OK,
2025-09-05 02:58:33,matpower/case118/2017-09-24,train,raw,OPTIMAL,2,38.940000,0.00907768,5899366.650939,5845814.060464,1,OK,
2025-09-05 02:59:49,matpower/case118/2017-09-25,train,raw,OPTIMAL,2,42.130000,0.00044895,5829177.667720,5826560.662388,1,OK,
2025-09-05 03:01:04,matpower/case118/2017-09-26,train,raw,OPTIMAL,2,41.822000,0.00041004,5635382.423357,5633071.700070,1,OK,
2025-09-05 03:02:18,matpower/case118/2017-09-27,train,raw,OPTIMAL,2,41.065000,0.00049092,5807844.815137,5804993.599277,1,OK,
2025-09-05 03:03:37,matpower/case118/2017-09-28,train,raw,OPTIMAL,2,45.297000,0.00090052,4854729.684401,4850357.922095,1,OK,
2025-09-05 03:04:57,matpower/case118/2017-09-29,train,raw,OPTIMAL,2,46.382000,0.00898968,4172766.404152,4135254.565549,1,OK,
2025-09-05 03:06:02,matpower/case118/2017-10-03,train,raw,OPTIMAL,2,31.775000,0.00146082,4034790.446439,4028896.332090,1,OK,
2025-09-05 03:07:08,matpower/case118/2017-10-04,train,raw,OPTIMAL,2,32.575000,0.00398676,4211363.866819,4194574.155944,1,OK,
2025-09-05 03:08:19,matpower/case118/2017-10-08,train,raw,OPTIMAL,2,38.723000,0.00606631,4936522.777999,4906576.318591,1,OK,
2025-09-05 03:09:34,matpower/case118/2017-10-09,train,raw,OPTIMAL,2,41.814000,0.00299920,5136305.153035,5120900.341627,1,OK,
2025-09-05 03:10:46,matpower/case118/2017-10-10,train,raw,OPTIMAL,2,38.058000,0.00480958,5193624.828247,5168645.676545,1,OK,
2025-09-05 03:11:55,matpower/case118/2017-10-11,train,raw,OPTIMAL,2,35.928000,0.00242745,4846560.943793,4834796.160647,1,OK,
2025-09-05 03:13:01,matpower/case118/2017-10-12,train,raw,OPTIMAL,2,32.674000,0.00419984,4919827.308497,4899164.801468,1,OK,
2025-09-05 03:14:09,matpower/case118/2017-10-13,train,raw,OPTIMAL,2,34.718000,0.00581008,4815672.855264,4787693.430939,1,OK,
2025-09-05 03:15:14,matpower/case118/2017-10-14,train,raw,OPTIMAL,2,31.028000,0.00765368,4329656.218314,4296518.422930,1,OK,
2025-09-05 03:16:22,matpower/case118/2017-10-15,train,raw,OPTIMAL,2,35.056000,0.00494888,4758333.790294,4734785.376369,1,OK,
2025-09-05 03:17:27,matpower/case118/2017-10-16,train,raw,OPTIMAL,2,31.582000,0.00367372,4525207.399472,4508583.069409,1,OK,
2025-09-05 03:18:33,matpower/case118/2017-10-17,train,raw,OPTIMAL,2,32.786000,0.00707002,4501749.530244,4469922.052100,1,OK,
2025-09-05 03:19:36,matpower/case118/2017-10-18,train,raw,OPTIMAL,2,30.081000,0.00467743,4275350.373972,4255352.711756,1,OK,
2025-09-05 03:20:42,matpower/case118/2017-10-19,train,raw,OPTIMAL,2,32.398000,0.00435176,4402520.319732,4383361.629522,1,OK,
2025-09-05 03:21:43,matpower/case118/2017-10-21,train,raw,OPTIMAL,2,27.535000,0.00717463,3724422.274467,3697700.930448,1,OK,
2025-09-05 03:22:46,matpower/case118/2017-10-22,train,raw,OPTIMAL,2,29.637000,0.00629618,3937706.231190,3912913.730976,1,OK,
2025-09-05 03:23:52,matpower/case118/2017-10-23,train,raw,OPTIMAL,2,32.439000,0.00607410,4671109.504259,4642736.714382,1,OK,
2025-09-05 03:24:57,matpower/case118/2017-10-24,train,raw,OPTIMAL,2,31.709000,0.00408508,4788186.196762,4768626.090885,1,OK,
2025-09-05 03:26:02,matpower/case118/2017-10-25,train,raw,OPTIMAL,2,30.776000,0.00371394,4615796.170281,4598653.393433,1,OK,
2025-09-05 03:27:17,matpower/case118/2017-10-26,train,raw,OPTIMAL,2,42.564000,0.00482624,4495028.989556,4473334.904342,1,OK,
2025-09-05 03:28:25,matpower/case118/2017-10-27,train,raw,OPTIMAL,2,34.459000,0.00267831,4278341.577176,4266882.872776,1,OK,
2025-09-05 03:30:16,matpower/case118/2017-10-30,train,raw,OPTIMAL,2,77.840000,0.00765138,4413266.951069,4379499.355176,458,OK,
2025-09-05 03:31:28,matpower/case118/2017-10-31,train,raw,OPTIMAL,2,38.487000,0.00987495,4397912.966829,4354483.807316,1,OK,
2025-09-05 03:32:44,matpower/case118/2017-11-02,train,raw,OPTIMAL,2,42.182000,0.00539630,4344565.610110,4321121.035168,1,OK,
2025-09-05 03:33:59,matpower/case118/2017-11-03,train,raw,OPTIMAL,2,42.626000,0.00574055,4024725.965623,4001621.841486,1,OK,
2025-09-05 03:35:15,matpower/case118/2017-11-04,train,raw,OPTIMAL,2,42.043000,0.00669428,3654550.816790,3630086.213556,1,OK,
2025-09-05 03:36:30,matpower/case118/2017-11-05,train,raw,OPTIMAL,2,42.174000,0.00277189,3777658.848989,3767187.612785,1,OK,
2025-09-05 03:37:38,matpower/case118/2017-11-07,train,raw,OPTIMAL,2,33.761000,0.00340941,4412630.268749,4397585.824775,1,OK,
2025-09-05 03:38:48,matpower/case118/2017-11-09,train,raw,OPTIMAL,2,36.634000,0.00190435,4988764.882854,4979264.515599,1,OK,
2025-09-05 03:39:58,matpower/case118/2017-11-10,train,raw,OPTIMAL,2,37.203000,0.00138402,5343158.469200,5335763.415674,1,OK,
2025-09-05 03:41:17,matpower/case118/2017-11-11,train,raw,OPTIMAL,2,45.434000,0.00073984,4911581.917400,4907948.110634,1,OK,
2025-09-05 03:42:26,matpower/case118/2017-11-12,train,raw,OPTIMAL,2,35.623000,0.00167103,4588471.293036,4580803.805497,1,OK,
2025-09-05 03:43:41,matpower/case118/2017-11-15,train,raw,OPTIMAL,2,41.045000,0.00154771,4711000.468680,4703709.229439,1,OK,
2025-09-05 03:44:52,matpower/case118/2017-11-16,train,raw,OPTIMAL,2,37.616000,0.00252680,4830258.443147,4818053.363726,1,OK,
2025-09-05 03:46:04,matpower/case118/2017-11-17,train,raw,OPTIMAL,2,39.061000,0.00308600,4865411.332893,4850396.658329,1,OK,
2025-09-05 03:47:14,matpower/case118/2017-11-18,train,raw,OPTIMAL,2,36.456000,0.00702622,4189228.854765,4159794.391421,1,OK,
2025-09-05 03:48:22,matpower/case118/2017-11-19,train,raw,OPTIMAL,2,34.573000,0.00368542,4533965.018477,4517255.455387,1,OK,
2025-09-05 03:49:46,matpower/case118/2017-11-20,train,raw,OPTIMAL,2,50.811000,0.00158665,5394645.833010,5386086.400461,1,OK,
2025-09-05 03:51:01,matpower/case118/2017-11-21,train,raw,OPTIMAL,2,42.047000,0.00185355,4915607.260212,4906495.943066,1,OK,
2025-09-05 03:52:08,matpower/case118/2017-11-22,train,raw,OPTIMAL,2,33.919000,0.00147601,5039652.747192,5032214.168887,1,OK,
2025-09-05 03:53:17,matpower/case118/2017-11-23,train,raw,OPTIMAL,2,35.157000,0.00342918,5071048.862307,5053659.332442,1,OK,
2025-09-05 03:54:25,matpower/case118/2017-11-25,train,raw,OPTIMAL,2,35.307000,0.00791125,4389501.766386,4354775.339806,1,OK,
2025-09-05 03:55:33,matpower/case118/2017-11-26,train,raw,OPTIMAL,2,34.476000,0.00679836,4804721.659951,4772057.409588,1,OK,
2025-09-05 03:56:43,matpower/case118/2017-11-27,train,raw,OPTIMAL,2,36.515000,0.00211410,5425652.974413,5414182.577126,1,OK,
2025-09-05 03:57:55,matpower/case118/2017-11-28,train,raw,OPTIMAL,2,38.023000,0.00359099,5055762.295693,5037607.089971,1,OK,
2025-09-05 03:59:00,matpower/case118/2017-11-29,train,raw,OPTIMAL,2,32.232000,0.00341952,5093437.414405,5076020.297999,1,OK,
2025-09-05 04:00:26,matpower/case118/2017-12-01,train,raw,OPTIMAL,2,51.958000,0.00777122,4967270.144699,4928668.409369,1,OK,
2025-09-05 04:01:41,matpower/case118/2017-12-03,train,raw,OPTIMAL,2,42.211000,0.00479390,5202923.481033,5177981.192386,1,OK,
2025-09-05 04:02:49,matpower/case118/2017-12-04,train,raw,OPTIMAL,2,34.428000,0.00474536,5541161.130095,5514866.299925,1,OK,
2025-09-05 04:03:59,matpower/case118/2017-12-07,train,raw,OPTIMAL,2,36.340000,0.00314928,6248550.667583,6228872.207686,1,OK,
2025-09-05 04:05:11,matpower/case118/2017-12-08,train,raw,OPTIMAL,2,38.554000,0.00213864,6742566.350919,6728146.411198,1,OK,
2025-09-05 04:06:25,matpower/case118/2017-12-09,train,raw,OPTIMAL,2,41.290000,0.00502672,7122495.967886,7086693.145565,1,OK,
2025-09-05 04:07:42,matpower/case118/2017-12-10,train,raw,OPTIMAL,2,43.874000,0.00352191,7008845.524673,6984161.022042,1,OK,
2025-09-05 04:09:02,matpower/case118/2017-12-11,train,raw,OPTIMAL,2,46.509000,0.00516161,7530100.866536,7491233.418681,1,OK,
2025-09-05 04:10:10,matpower/case118/2017-12-12,train,raw,OPTIMAL,2,34.215000,0.00216574,7856643.244367,7839627.773662,1,OK,
2025-09-05 04:11:19,matpower/case118/2017-12-13,train,raw,OPTIMAL,2,36.011000,0.00166546,8290571.738578,8276764.083371,1,OK,
2025-09-05 04:12:40,matpower/case118/2017-12-15,train,raw,OPTIMAL,2,47.205000,0.00092982,8527591.871877,8519662.787130,1,OK,
2025-09-05 04:13:57,matpower/case118/2017-12-16,train,raw,OPTIMAL,2,44.143000,0.00038107,7412840.649605,7410015.858057,1,OK,
2025-09-05 04:15:13,matpower/case118/2017-12-17,train,raw,OPTIMAL,2,42.581000,0.00071502,7250400.380731,7245216.219131,1,OK,
2025-09-05 04:16:33,matpower/case118/2017-12-19,train,raw,OPTIMAL,2,46.715000,0.00345386,6842979.051138,6819344.363651,1,OK,
2025-09-05 04:17:56,matpower/case118/2017-12-21,train,raw,OPTIMAL,2,49.442000,0.00102934,8016928.793169,8008676.641747,1,OK,
2025-09-05 04:19:12,matpower/case118/2017-12-23,train,raw,OPTIMAL,2,42.440000,0.00266114,5669620.096569,5654532.458357,1,OK,
2025-09-05 04:20:25,matpower/case118/2017-12-24,train,raw,OPTIMAL,2,40.307000,0.00367562,5917742.516003,5895991.127329,1,OK,
2025-09-05 04:21:35,matpower/case118/2017-12-25,train,raw,OPTIMAL,2,36.115000,0.00087254,6668344.960969,6662526.545740,1,OK,
2025-09-05 04:22:41,matpower/case118/2017-12-26,train,raw,OPTIMAL,2,32.235000,0.00152286,7790026.485482,7778163.361906,1,OK,
2025-09-05 04:23:55,matpower/case118/2017-12-27,train,raw,OPTIMAL,2,40.725000,0.00071552,8499646.127207,8493564.430381,1,OK,
2025-09-05 04:25:00,matpower/case118/2017-12-28,train,raw,OPTIMAL,2,31.196000,0.00037209,8858302.923714,8855006.817097,1,OK,
2025-09-05 04:27:34,matpower/case118/2017-12-29,train,raw,TIME_LIMIT,9,120.477000,0.06913624,19126397.884002,17804070.608028,488,OK,
2025-09-05 04:28:56,matpower/case118/2017-12-30,train,raw,OPTIMAL,2,48.418000,0.00270676,7855732.650394,7834469.055556,1,OK,
2025-09-05 04:30:08,matpower/case118/2017-12-31,train,raw,OPTIMAL,2,38.338000,0.00854952,8256728.593570,8186137.523675,1,OK,
2025-09-05 04:31:27,matpower/case118/2017-01-09,test,raw,OPTIMAL,2,39.750000,0.00010158,7078282.549776,7077563.503192,1,OK,
2025-09-05 04:32:41,matpower/case118/2017-01-09,test,warm,OPTIMAL,2,39.693000,0.00010158,7078282.549776,7077563.503192,1,OK,49068
2025-09-05 04:33:57,matpower/case118/2017-01-10,test,raw,OPTIMAL,2,42.217000,0.00009282,6163863.107085,6163290.950552,1,OK,
2025-09-05 04:35:14,matpower/case118/2017-01-10,test,warm,OPTIMAL,2,42.565000,0.00009282,6163863.107085,6163290.950552,1,OK,49068
2025-09-05 04:36:25,matpower/case118/2017-01-24,test,raw,OPTIMAL,2,36.995000,0.00013530,5680339.258746,5679570.712498,1,OK,
2025-09-05 04:37:37,matpower/case118/2017-01-24,test,warm,OPTIMAL,2,37.248000,0.00013530,5680339.258746,5679570.712498,1,OK,49068
2025-09-05 04:38:46,matpower/case118/2017-01-25,test,raw,OPTIMAL,2,35.946000,0.00006686,5423005.512082,5422642.909132,1,OK,
2025-09-05 04:39:58,matpower/case118/2017-01-25,test,warm,OPTIMAL,2,36.332000,0.00006686,5423005.512082,5422642.909132,1,OK,49068
2025-09-05 04:41:10,matpower/case118/2017-01-28,test,raw,OPTIMAL,2,39.231000,0.00026017,5377699.100291,5376299.991124,1,OK,
2025-09-05 04:42:24,matpower/case118/2017-01-28,test,warm,OPTIMAL,2,39.407000,0.00026017,5377699.100291,5376299.991124,1,OK,49068
2025-09-05 04:43:35,matpower/case118/2017-01-29,test,raw,OPTIMAL,2,37.244000,0.00001236,5548629.395114,5548560.808519,1,OK,
2025-09-05 04:44:47,matpower/case118/2017-01-29,test,warm,OPTIMAL,2,37.443000,0.00001236,5548629.395114,5548560.808519,1,OK,49068
2025-09-05 04:45:59,matpower/case118/2017-02-07,test,raw,OPTIMAL,2,38.010000,0.00053507,4915818.099098,4913187.810217,1,OK,
2025-09-05 04:47:12,matpower/case118/2017-02-07,test,warm,OPTIMAL,2,38.002000,0.00053507,4915818.099098,4913187.810217,1,OK,49068
2025-09-05 04:48:24,matpower/case118/2017-02-12,test,raw,OPTIMAL,2,39.019000,0.00015424,5592940.435255,5592077.788462,1,OK,
2025-09-05 04:49:38,matpower/case118/2017-02-12,test,warm,OPTIMAL,2,39.481000,0.00015424,5592940.435255,5592077.788462,1,OK,49068
2025-09-05 04:50:50,matpower/case118/2017-02-28,test,raw,OPTIMAL,2,37.831000,0.00008621,4372284.034888,4371907.119566,1,OK,
2025-09-05 04:52:03,matpower/case118/2017-02-28,test,warm,OPTIMAL,2,38.171000,0.00008621,4372284.034888,4371907.119566,1,OK,49068
2025-09-05 04:53:17,matpower/case118/2017-03-04,test,raw,OPTIMAL,2,40.515000,0.00025248,4226453.400210,4225386.326142,1,OK,
2025-09-05 04:54:32,matpower/case118/2017-03-04,test,warm,OPTIMAL,2,40.132000,0.00025248,4226453.400210,4225386.326142,1,OK,49068
2025-09-05 04:55:45,matpower/case118/2017-03-07,test,raw,OPTIMAL,2,39.592000,0.00009310,4039072.644948,4038696.620060,1,OK,
2025-09-05 04:57:00,matpower/case118/2017-03-07,test,warm,OPTIMAL,2,40.245000,0.00009310,4039072.644948,4038696.620060,1,OK,49068
2025-09-05 04:58:10,matpower/case118/2017-03-09,test,raw,OPTIMAL,2,36.769000,0.00040454,4110195.405278,4108532.653944,1,OK,
2025-09-05 04:59:22,matpower/case118/2017-03-09,test,warm,OPTIMAL,2,36.849000,0.00040454,4110195.405278,4108532.653944,1,OK,49068
2025-09-05 05:00:35,matpower/case118/2017-03-11,test,raw,OPTIMAL,2,38.992000,0.00002665,5314355.445488,5314213.804288,1,OK,
2025-09-05 05:01:49,matpower/case118/2017-03-11,test,warm,OPTIMAL,2,39.395000,0.00002665,5314355.445488,5314213.804288,1,OK,49068
2025-09-05 05:03:00,matpower/case118/2017-03-14,test,raw,OPTIMAL,2,36.887000,0.00007630,5411292.421686,5410879.551717,1,OK,
2025-09-05 05:04:12,matpower/case118/2017-03-14,test,warm,OPTIMAL,2,37.026000,0.00007630,5411292.421686,5410879.551717,1,OK,49068
2025-09-05 05:05:23,matpower/case118/2017-03-20,test,raw,OPTIMAL,2,37.260000,0.00047321,4582718.994585,4580550.404924,1,OK,
2025-09-05 05:06:33,matpower/case118/2017-03-20,test,warm,OPTIMAL,2,35.635000,0.00013338,4581161.418841,4580550.404924,1,OK,49068
2025-09-05 05:07:47,matpower/case118/2017-03-25,test,raw,OPTIMAL,2,40.394000,0.00112670,3556036.940042,3552030.354564,1,OK,
2025-09-05 05:09:04,matpower/case118/2017-03-25,test,warm,OPTIMAL,2,41.460000,0.00075093,3554699.699659,3552030.354564,1,OK,49068
2025-09-05 05:10:14,matpower/case118/2017-04-07,test,raw,OPTIMAL,2,36.206000,0.00103491,4325645.638341,4321168.987042,1,OK,
2025-09-05 05:11:25,matpower/case118/2017-04-07,test,warm,OPTIMAL,2,36.124000,0.00103491,4325645.638341,4321168.987042,1,OK,49068
2025-09-05 05:12:37,matpower/case118/2017-04-17,test,raw,OPTIMAL,2,38.829000,0.00030116,3840804.544250,3839647.853950,1,OK,
2025-09-05 05:13:51,matpower/case118/2017-04-17,test,warm,OPTIMAL,2,39.163000,0.00030116,3840804.544250,3839647.853950,1,OK,49068
2025-09-05 05:15:03,matpower/case118/2017-04-22,test,raw,OPTIMAL,2,38.366000,0.00035321,3366014.130211,3364825.220954,1,OK,
2025-09-05 05:16:17,matpower/case118/2017-04-22,test,warm,OPTIMAL,2,38.770000,0.00035321,3366014.130211,3364825.220954,1,OK,49068
2025-09-05 05:17:30,matpower/case118/2017-04-25,test,raw,OPTIMAL,2,40.069000,0.00032387,4110042.807877,4108711.686998,1,OK,
2025-09-05 05:18:45,matpower/case118/2017-04-25,test,warm,OPTIMAL,2,39.817000,0.00032387,4110042.807877,4108711.686998,1,OK,49068
2025-09-05 05:20:04,matpower/case118/2017-05-04,test,raw,OPTIMAL,2,45.414000,0.00439822,3542435.970127,3526855.556878,1,OK,
2025-09-05 05:21:25,matpower/case118/2017-05-04,test,warm,OPTIMAL,2,45.938000,0.00439822,3542435.970127,3526855.556878,1,OK,49068
2025-09-05 05:22:38,matpower/case118/2017-05-18,test,raw,OPTIMAL,2,38.714000,0.00001574,5449009.793365,5448924.049622,1,OK,
2025-09-05 05:23:51,matpower/case118/2017-05-18,test,warm,OPTIMAL,2,38.380000,0.00000971,5448976.933598,5448924.049622,1,OK,49068
2025-09-05 05:25:32,matpower/case118/2017-05-22,test,raw,OPTIMAL,2,66.819000,0.00013082,3841992.113988,3841489.511058,1,OK,
2025-09-05 05:27:24,matpower/case118/2017-05-22,test,warm,OPTIMAL,2,76.886000,0.00002112,3841570.655895,3841489.511058,1,OK,49068
2025-09-05 05:28:50,matpower/case118/2017-05-27,test,raw,OPTIMAL,2,52.451000,0.00160532,3274053.811309,3268797.906080,1,OK,
2025-09-05 05:30:17,matpower/case118/2017-05-27,test,warm,OPTIMAL,2,52.388000,0.00007635,3269047.507229,3268797.906080,1,OK,49068
2025-09-05 05:31:39,matpower/case118/2017-05-28,test,raw,OPTIMAL,2,48.142000,0.00250902,3175011.631087,3167045.455365,1,OK,
2025-09-05 05:33:02,matpower/case118/2017-05-28,test,warm,OPTIMAL,2,47.903000,0.00006941,3167265.289054,3167045.455365,1,OK,49068
2025-09-05 05:34:11,matpower/case118/2017-06-10,test,raw,OPTIMAL,2,35.655000,0.00010081,4724614.214362,4724137.904835,1,OK,
2025-09-05 05:35:22,matpower/case118/2017-06-10,test,warm,OPTIMAL,2,35.055000,0.00008055,4724518.455499,4724137.904835,1,OK,49068
2025-09-05 05:36:32,matpower/case118/2017-06-15,test,raw,OPTIMAL,2,37.322000,0.00014779,5052711.324524,5051964.605425,1,OK,
2025-09-05 05:37:45,matpower/case118/2017-06-15,test,warm,OPTIMAL,2,37.460000,0.00014779,5052711.324524,5051964.605425,1,OK,49068
2025-09-05 05:38:55,matpower/case118/2017-06-25,test,raw,OPTIMAL,2,36.389000,0.00013557,5378888.954384,5378159.714348,1,OK,
2025-09-05 05:40:06,matpower/case118/2017-06-25,test,warm,OPTIMAL,2,36.515000,0.00013557,5378888.954384,5378159.714348,1,OK,49068
2025-09-05 05:41:16,matpower/case118/2017-06-28,test,raw,OPTIMAL,2,36.764000,0.00007813,5598857.170751,5598419.746970,1,OK,
2025-09-05 05:42:28,matpower/case118/2017-06-28,test,warm,OPTIMAL,2,37.145000,0.00007813,5598857.170751,5598419.746970,1,OK,49068
2025-09-05 05:43:41,matpower/case118/2017-07-03,test,raw,OPTIMAL,2,39.414000,0.00002372,6405260.252280,6405108.346207,1,OK,
2025-09-05 05:44:56,matpower/case118/2017-07-03,test,warm,OPTIMAL,2,39.993000,0.00002372,6405260.252280,6405108.346207,1,OK,49068
2025-09-05 05:46:07,matpower/case118/2017-07-09,test,raw,OPTIMAL,2,37.597000,0.00048273,5893338.954531,5890494.067289,1,OK,
2025-09-05 05:47:19,matpower/case118/2017-07-09,test,warm,OPTIMAL,2,37.623000,0.00048273,5893338.954531,5890494.067289,1,OK,49068
2025-09-05 05:48:33,matpower/case118/2017-07-14,test,raw,OPTIMAL,2,39.823000,0.00003712,6883738.230410,6883482.716362,1,OK,
2025-09-05 05:49:48,matpower/case118/2017-07-14,test,warm,OPTIMAL,2,39.989000,0.00003712,6883738.230410,6883482.716362,1,OK,49068
2025-09-05 05:51:02,matpower/case118/2017-07-21,test,raw,OPTIMAL,2,40.880000,0.00003206,8006379.735924,8006123.079069,1,OK,
2025-09-05 05:52:18,matpower/case118/2017-07-21,test,warm,OPTIMAL,2,40.978000,0.00003206,8006379.735924,8006123.079069,1,OK,49068
2025-09-05 05:53:31,matpower/case118/2017-07-26,test,raw,OPTIMAL,2,39.199000,0.00169584,5582224.927740,5572758.360956,1,OK,
2025-09-05 05:54:44,matpower/case118/2017-07-26,test,warm,OPTIMAL,2,37.598000,0.00017815,5573751.309895,5572758.360956,1,OK,49068
2025-09-05 05:55:51,matpower/case118/2017-08-09,test,raw,OPTIMAL,2,33.752000,0.00228506,4812609.962136,4801612.873402,1,OK,
2025-09-05 05:57:00,matpower/case118/2017-08-09,test,warm,OPTIMAL,2,33.875000,0.00228506,4812609.962136,4801612.873402,1,OK,49068
2025-09-05 05:58:12,matpower/case118/2017-08-17,test,raw,OPTIMAL,2,38.944000,0.00287065,6209853.965343,6192027.665850,1,OK,
2025-09-05 05:59:31,matpower/case118/2017-08-17,test,warm,OPTIMAL,2,43.310000,0.00045140,6195461.687028,6192665.036512,1,OK,49068
2025-09-05 06:00:39,matpower/case118/2017-08-18,test,raw,OPTIMAL,2,35.061000,0.00125632,6321063.610758,6313122.319320,1,OK,
2025-09-05 06:01:50,matpower/case118/2017-08-18,test,warm,OPTIMAL,2,35.052000,0.00125632,6321063.610758,6313122.319320,1,OK,49068
2025-09-05 06:02:49,matpower/case118/2017-09-02,test,raw,OPTIMAL,2,25.201000,0.00792493,3698225.471472,3668917.311889,1,OK,
2025-09-05 06:03:50,matpower/case118/2017-09-02,test,warm,OPTIMAL,2,25.639000,0.00792493,3698225.471472,3668917.311889,1,OK,49068
2025-09-05 06:05:09,matpower/case118/2017-09-17,test,raw,OPTIMAL,2,45.643000,0.00427842,4966854.939666,4945604.635252,1,OK,
2025-09-05 06:06:30,matpower/case118/2017-09-17,test,warm,OPTIMAL,2,46.270000,0.00427842,4966854.939666,4945604.635252,1,OK,49068
2025-09-05 06:07:45,matpower/case118/2017-09-18,test,raw,OPTIMAL,2,41.450000,0.00260356,5460355.671228,5446139.299981,1,OK,
2025-09-05 06:09:02,matpower/case118/2017-09-18,test,warm,OPTIMAL,2,41.463000,0.00260356,5460355.671228,5446139.299981,1,OK,49068
2025-09-05 06:10:03,matpower/case118/2017-10-01,test,raw,OPTIMAL,2,28.176000,0.00333508,3425743.947933,3414318.818970,1,OK,
2025-09-05 06:11:06,matpower/case118/2017-10-01,test,warm,OPTIMAL,2,27.976000,0.00036401,3415562.132509,3414318.818965,1,OK,49068
2025-09-05 06:12:13,matpower/case118/2017-10-28,test,raw,OPTIMAL,2,33.673000,0.00279330,3839467.350402,3828742.563708,1,OK,
2025-09-05 06:13:23,matpower/case118/2017-10-28,test,warm,OPTIMAL,2,33.930000,0.00279330,3839467.350402,3828742.563708,1,OK,49068
2025-09-05 06:14:37,matpower/case118/2017-10-29,test,raw,OPTIMAL,2,41.741000,0.00457810,4114951.236035,4096112.587346,1,OK,
2025-09-05 06:15:55,matpower/case118/2017-10-29,test,warm,OPTIMAL,2,41.769000,0.00457810,4114951.236035,4096112.587346,1,OK,49068
2025-09-05 06:17:10,matpower/case118/2017-11-13,test,raw,OPTIMAL,2,41.727000,0.00133344,5012223.116866,5005539.629156,1,OK,
2025-09-05 06:18:27,matpower/case118/2017-11-13,test,warm,OPTIMAL,2,42.051000,0.00133344,5012223.116866,5005539.629156,1,OK,49068
2025-09-05 06:19:49,matpower/case118/2017-11-30,test,raw,OPTIMAL,2,48.457000,0.00564327,5282985.673529,5253172.347661,1,OK,
2025-09-05 06:21:13,matpower/case118/2017-11-30,test,warm,OPTIMAL,2,48.666000,0.00564327,5282985.673529,5253172.347661,1,OK,49068
2025-09-05 06:22:22,matpower/case118/2017-12-06,test,raw,OPTIMAL,2,34.881000,0.00465069,5607516.697857,5581437.862958,1,OK,
2025-09-05 06:23:32,matpower/case118/2017-12-06,test,warm,OPTIMAL,2,34.891000,0.00465069,5607516.697857,5581437.862958,1,OK,49068
2025-09-05 06:24:58,matpower/case118/2017-12-18,test,raw,OPTIMAL,2,52.355000,0.00339724,7219878.219159,7195350.534469,1,OK,
2025-09-05 06:26:26,matpower/case118/2017-12-18,test,warm,OPTIMAL,2,52.382000,0.00339724,7219878.219159,7195350.534469,1,OK,49068
2025-09-05 06:27:47,matpower/case118/2017-12-20,test,raw,OPTIMAL,2,47.664000,0.00122386,6853264.635246,6844877.229826,1,OK,
2025-09-05 06:29:10,matpower/case118/2017-12-20,test,warm,OPTIMAL,2,47.919000,0.00122386,6853264.635246,6844877.229826,1,OK,49068
2025-09-05 06:30:26,matpower/case118/2017-12-22,test,raw,OPTIMAL,2,43.028000,0.00405483,6725556.517372,6698285.523088,1,OK,
2025-09-05 06:31:45,matpower/case118/2017-12-22,test,warm,OPTIMAL,2,43.301000,0.00405483,6725556.517372,6698285.523088,1,OK,49068

```

### File: `src/data/output/matpower/case118/compare_matpower_case118_20250905_150049.csv`

```
timestamp,instance_name,split,method,status,status_code,runtime_sec,mip_gap,obj_val,obj_bound,nodes,feasible_ok,warm_start_applied_vars,num_vars,num_bin_vars,num_int_vars,num_constrs,max_constraint_violation
2025-09-05 15:02:34,matpower/case118/2017-01-02,train,raw,OPTIMAL,2,48.123000,0.00013321,4844424.994625,4843779.661003,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:04:16,matpower/case118/2017-01-05,train,raw,OPTIMAL,2,48.792000,0.00017842,6167079.101966,6165978.780926,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:06:00,matpower/case118/2017-01-06,train,raw,OPTIMAL,2,49.567000,0.00000486,6043528.811815,6043499.429915,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:07:44,matpower/case118/2017-01-07,train,raw,OPTIMAL,2,47.880000,0.00023482,6299208.822867,6297729.629954,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:09:31,matpower/case118/2017-01-08,train,raw,OPTIMAL,2,50.741000,0.00000000,6384833.162047,6384833.162047,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:11:25,matpower/case118/2017-01-11,train,raw,OPTIMAL,2,53.730000,0.00002041,5213619.814083,5213513.406415,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:13:13,matpower/case118/2017-01-12,train,raw,OPTIMAL,2,50.565000,0.00031706,4984328.982647,4982748.659045,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:15:01,matpower/case118/2017-01-15,train,raw,OPTIMAL,2,53.387000,0.00002470,4815570.136330,4815451.177755,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:16:49,matpower/case118/2017-01-16,train,raw,OPTIMAL,2,50.656000,0.00000602,5480572.300569,5480539.284721,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:18:32,matpower/case118/2017-01-17,train,raw,OPTIMAL,2,47.082000,0.00001376,5497060.046869,5496984.431462,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:20:16,matpower/case118/2017-01-18,train,raw,OPTIMAL,2,47.697000,0.00013971,5393659.533144,5392905.972862,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:22:01,matpower/case118/2017-01-19,train,raw,OPTIMAL,2,48.844000,0.00001726,5469032.706635,5468938.286915,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:23:45,matpower/case118/2017-01-21,train,raw,OPTIMAL,2,48.196000,0.00011080,5011221.001296,5010665.777526,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:25:27,matpower/case118/2017-01-22,train,raw,OPTIMAL,2,45.837000,0.00002969,5191372.832008,5191218.712735,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:27:10,matpower/case118/2017-01-23,train,raw,OPTIMAL,2,50.644000,0.00005933,5868275.365160,5867927.225284,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:28:49,matpower/case118/2017-01-26,train,raw,OPTIMAL,2,46.744000,0.00006735,5269491.012978,5269136.109674,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:30:28,matpower/case118/2017-01-27,train,raw,OPTIMAL,2,46.474000,0.00102171,4828403.541371,4823470.321043,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:32:04,matpower/case118/2017-01-31,train,raw,OPTIMAL,2,44.492000,0.00024672,5631243.555296,5629854.233487,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:33:44,matpower/case118/2017-02-01,train,raw,OPTIMAL,2,46.628000,0.00004152,5728872.209972,5728634.350212,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:35:26,matpower/case118/2017-02-02,train,raw,OPTIMAL,2,48.529000,0.00027208,5964267.232530,5962644.484264,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 15:37:13,matpower/case118/2017-02-03,train,raw,OPTIMAL,2,53.213000,0.00010822,5857914.303498,5857280.377313,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 17:01:02,matpower/case118/2017-02-05,train,raw,TIME_LIMIT,9,4988.721000,inf,inf,-inf,0,,,49068,5832,5832,1597536,
2025-09-05 17:02:15,matpower/case118/2017-02-06,train,raw,OPTIMAL,2,34.237000,0.00009528,5337348.583769,5336840.037145,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 17:03:53,matpower/case118/2017-02-08,train,raw,OPTIMAL,2,50.879000,0.00007520,5032278.961333,5031900.513826,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 17:05:38,matpower/case118/2017-02-09,train,raw,OPTIMAL,2,46.880000,0.00001388,6543758.221165,6543667.379458,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 17:07:20,matpower/case118/2017-02-11,train,raw,OPTIMAL,2,47.277000,0.00011450,5629484.377440,5628839.812782,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 17:09:15,matpower/case118/2017-02-13,train,raw,OPTIMAL,2,57.449000,0.00025829,6487317.536677,6485641.956185,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-05 17:11:03,matpower/case118/2017-02-14,train,raw,OPTIMAL,2,53.191000,0.00020207,6516423.306406,6515106.506763,1,OK,,49068,5832,5832,1597536,0.00000000

```

### File: `src/data/output/matpower/case118/compare_matpower_case118_20250907_152327.csv`

```
timestamp,instance_name,split,method,status,status_code,runtime_sec,mip_gap,obj_val,obj_bound,nodes,feasible_ok,warm_start_applied_vars,num_vars,num_bin_vars,num_int_vars,num_constrs,max_constraint_violation
2025-09-07 15:25:22,matpower/case118/2017-01-02,train,raw,OPTIMAL,2,55.090000,0.00013321,4844424.994625,4843779.661003,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:27:08,matpower/case118/2017-01-05,train,raw,OPTIMAL,2,51.497000,0.00017842,6167079.101966,6165978.780926,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:28:57,matpower/case118/2017-01-06,train,raw,OPTIMAL,2,52.439000,0.00000486,6043528.811815,6043499.429915,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:30:43,matpower/case118/2017-01-07,train,raw,OPTIMAL,2,49.491000,0.00023482,6299208.822867,6297729.629954,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:32:29,matpower/case118/2017-01-08,train,raw,OPTIMAL,2,51.063000,0.00000000,6384833.162047,6384833.162047,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:34:17,matpower/case118/2017-01-11,train,raw,OPTIMAL,2,52.731000,0.00002041,5213619.814083,5213513.406415,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:36:06,matpower/case118/2017-01-12,train,raw,OPTIMAL,2,51.634000,0.00031706,4984328.982647,4982748.659045,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:37:55,matpower/case118/2017-01-15,train,raw,OPTIMAL,2,52.224000,0.00002470,4815570.136330,4815451.177755,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:39:45,matpower/case118/2017-01-16,train,raw,OPTIMAL,2,52.766000,0.00000602,5480572.300569,5480539.284721,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:41:35,matpower/case118/2017-01-17,train,raw,OPTIMAL,2,51.716000,0.00001376,5497060.046869,5496984.431462,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:43:20,matpower/case118/2017-01-18,train,raw,OPTIMAL,2,46.575000,0.00013971,5393659.533144,5392905.972862,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:45:00,matpower/case118/2017-01-19,train,raw,OPTIMAL,2,47.299000,0.00001726,5469032.706635,5468938.286915,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:46:43,matpower/case118/2017-01-21,train,raw,OPTIMAL,2,46.435000,0.00011080,5011221.001296,5010665.777526,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:48:34,matpower/case118/2017-01-22,train,raw,OPTIMAL,2,51.166000,0.00002969,5191372.832008,5191218.712735,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:50:24,matpower/case118/2017-01-23,train,raw,OPTIMAL,2,53.137000,0.00005933,5868275.365160,5867927.225284,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:52:13,matpower/case118/2017-01-26,train,raw,OPTIMAL,2,50.482000,0.00006735,5269491.012978,5269136.109674,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:54:04,matpower/case118/2017-01-27,train,raw,OPTIMAL,2,52.845000,0.00102171,4828403.541371,4823470.321043,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:55:53,matpower/case118/2017-01-31,train,raw,OPTIMAL,2,50.383000,0.00024672,5631243.555296,5629854.233487,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:57:35,matpower/case118/2017-02-01,train,raw,OPTIMAL,2,48.977000,0.00004152,5728872.209972,5728634.350212,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 15:59:26,matpower/case118/2017-02-02,train,raw,OPTIMAL,2,54.495000,0.00027208,5964267.232530,5962644.484264,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:01:17,matpower/case118/2017-02-03,train,raw,OPTIMAL,2,53.007000,0.00010822,5857914.303498,5857280.377313,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:03:08,matpower/case118/2017-02-05,train,raw,OPTIMAL,2,52.201000,0.00010915,5226625.044439,5226054.562683,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:04:49,matpower/case118/2017-02-06,train,raw,OPTIMAL,2,46.947000,0.00009528,5337348.583769,5336840.037145,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:06:28,matpower/case118/2017-02-08,train,raw,OPTIMAL,2,46.954000,0.00007520,5032278.961333,5031900.513826,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:08:09,matpower/case118/2017-02-09,train,raw,OPTIMAL,2,45.544000,0.00001388,6543758.221165,6543667.379458,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:09:49,matpower/case118/2017-02-11,train,raw,OPTIMAL,2,47.102000,0.00011450,5629484.377440,5628839.812782,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:11:32,matpower/case118/2017-02-13,train,raw,OPTIMAL,2,50.509000,0.00025829,6487317.536677,6485641.956185,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:13:16,matpower/case118/2017-02-14,train,raw,OPTIMAL,2,50.835000,0.00020207,6516423.306406,6515106.506763,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:14:55,matpower/case118/2017-02-15,train,raw,OPTIMAL,2,47.938000,0.00015354,6058511.991679,6057581.778726,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:16:40,matpower/case118/2017-02-16,train,raw,OPTIMAL,2,50.733000,0.00001678,6435605.444325,6435497.481817,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:18:27,matpower/case118/2017-02-17,train,raw,OPTIMAL,2,50.999000,0.00026493,5172179.310743,5170809.036971,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:20:19,matpower/case118/2017-02-18,train,raw,OPTIMAL,2,54.471000,0.00121541,4306920.128827,4301685.454944,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:22:07,matpower/case118/2017-02-19,train,raw,OPTIMAL,2,50.094000,0.00001128,4181299.025532,4181251.850114,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:23:54,matpower/case118/2017-02-20,train,raw,OPTIMAL,2,50.395000,0.00010706,4841724.610617,4841206.232551,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:25:51,matpower/case118/2017-02-21,train,raw,OPTIMAL,2,55.093000,0.00122625,4949937.818419,4943867.967207,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:27:42,matpower/case118/2017-02-23,train,raw,OPTIMAL,2,55.086000,0.00008628,4396036.178819,4395656.908955,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:29:26,matpower/case118/2017-02-24,train,raw,OPTIMAL,2,50.703000,0.00027180,4062168.967666,4061064.861371,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:31:17,matpower/case118/2017-02-26,train,raw,OPTIMAL,2,50.700000,0.00012060,3919573.303562,3919100.602888,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:33:04,matpower/case118/2017-02-27,train,raw,OPTIMAL,2,51.598000,0.00008324,4576743.106220,4576362.123883,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 16:34:49,matpower/case118/2017-03-01,train,raw,OPTIMAL,2,49.696000,0.00002489,4271665.729725,4271559.404029,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 17:35:52,matpower/case118/2017-03-02,train,raw,TIME_LIMIT,9,3611.311000,0.87920838,37469411.810272,4525991.057720,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 17:36:56,matpower/case118/2017-03-03,train,raw,OPTIMAL,2,30.794000,0.00007830,4575087.766783,4574729.544363,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 17:38:03,matpower/case118/2017-03-05,train,raw,OPTIMAL,2,35.391000,0.00026129,4529769.924424,4528586.338568,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 17:39:16,matpower/case118/2017-03-06,train,raw,OPTIMAL,2,34.486000,0.00006951,4199555.092940,4199263.201445,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 17:40:57,matpower/case118/2017-03-08,train,raw,OPTIMAL,2,48.579000,0.00075721,4011638.378017,4008600.718498,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 17:42:38,matpower/case118/2017-03-12,train,raw,OPTIMAL,2,48.008000,0.00009135,4973945.243072,4973490.851736,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 17:44:21,matpower/case118/2017-03-13,train,raw,OPTIMAL,2,49.651000,0.00000671,5205316.056590,5205281.147765,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 17:46:00,matpower/case118/2017-03-15,train,raw,OPTIMAL,2,46.031000,0.00000656,5716822.119742,5716784.590326,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 17:47:48,matpower/case118/2017-03-16,train,raw,OPTIMAL,2,53.990000,0.00019209,5566720.195358,5565650.865681,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 17:49:26,matpower/case118/2017-03-17,train,raw,OPTIMAL,2,45.756000,0.00012173,4566031.760944,4565475.918943,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 17:51:06,matpower/case118/2017-03-18,train,raw,OPTIMAL,2,47.892000,0.00041452,3952126.170027,3950487.945040,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 17:52:46,matpower/case118/2017-03-19,train,raw,OPTIMAL,2,46.852000,0.00019533,4254333.768088,4253502.761597,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 17:54:25,matpower/case118/2017-03-21,train,raw,OPTIMAL,2,46.823000,0.00013985,4432492.031231,4431872.162791,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 17:56:04,matpower/case118/2017-03-22,train,raw,OPTIMAL,2,47.094000,0.00010062,5235438.723326,5234911.957915,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 17:57:49,matpower/case118/2017-03-24,train,raw,OPTIMAL,2,52.311000,0.00220204,4128644.461442,4119553.002757,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 17:59:22,matpower/case118/2017-03-27,train,raw,OPTIMAL,2,45.238000,0.00069278,3912062.165023,3909351.953485,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:00:34,matpower/case118/2017-03-28,train,raw,OPTIMAL,2,38.330000,0.00097807,3893639.718969,3889831.470274,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:01:43,matpower/case118/2017-03-31,train,raw,OPTIMAL,2,35.702000,0.00083525,4343389.856328,4339762.043418,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:02:54,matpower/case118/2017-04-01,train,raw,OPTIMAL,2,37.516000,0.00123847,3797531.537966,3792828.418058,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:04:02,matpower/case118/2017-04-03,train,raw,OPTIMAL,2,34.919000,0.00045189,4207331.420769,4205430.187890,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:05:11,matpower/case118/2017-04-04,train,raw,OPTIMAL,2,35.043000,0.00049959,4160707.421025,4158628.774128,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:06:18,matpower/case118/2017-04-05,train,raw,OPTIMAL,2,33.562000,0.00038201,4171037.070334,4169443.676456,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:07:26,matpower/case118/2017-04-06,train,raw,OPTIMAL,2,34.224000,0.00036959,4377494.529453,4375876.646490,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:08:32,matpower/case118/2017-04-08,train,raw,OPTIMAL,2,32.845000,0.00069701,3839159.623285,3836483.681331,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:09:38,matpower/case118/2017-04-09,train,raw,OPTIMAL,2,32.474000,0.00012598,3783646.272427,3783169.621677,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:10:47,matpower/case118/2017-04-10,train,raw,OPTIMAL,2,35.507000,0.00032470,4148696.243938,4147349.169809,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:11:54,matpower/case118/2017-04-11,train,raw,OPTIMAL,2,33.866000,0.00016938,4243802.950110,4243084.153859,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:13:05,matpower/case118/2017-04-12,train,raw,OPTIMAL,2,36.950000,0.00031951,4104851.344508,4103539.822577,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:14:11,matpower/case118/2017-04-13,train,raw,OPTIMAL,2,33.572000,0.00049238,3954991.706927,3953044.350984,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:15:21,matpower/case118/2017-04-15,train,raw,OPTIMAL,2,35.853000,0.00076377,3415974.662532,3413365.630568,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:16:30,matpower/case118/2017-04-16,train,raw,OPTIMAL,2,35.656000,0.00005378,3867060.292944,3866852.316098,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:17:39,matpower/case118/2017-04-18,train,raw,OPTIMAL,2,35.009000,0.00014489,3712398.412625,3711860.514176,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:18:48,matpower/case118/2017-04-19,train,raw,OPTIMAL,2,35.521000,0.00002227,3622366.935837,3622286.278298,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:19:59,matpower/case118/2017-04-21,train,raw,OPTIMAL,2,37.145000,0.00111022,3716777.948732,3712651.502026,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:21:10,matpower/case118/2017-04-23,train,raw,OPTIMAL,2,36.935000,0.00018814,3471886.716562,3471233.531714,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:22:20,matpower/case118/2017-04-24,train,raw,OPTIMAL,2,36.313000,0.00028920,3946647.354799,3945506.002194,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:23:29,matpower/case118/2017-04-26,train,raw,OPTIMAL,2,35.215000,0.00019985,3862320.875425,3861548.980824,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:24:36,matpower/case118/2017-04-27,train,raw,OPTIMAL,2,33.127000,0.00094325,4197050.723042,4193091.854769,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:25:51,matpower/case118/2017-04-28,train,raw,OPTIMAL,2,41.455000,0.01034755,4098212.276202,4055805.802042,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:27:02,matpower/case118/2017-04-29,train,raw,OPTIMAL,2,37.671000,0.00817939,3664533.362046,3634559.730077,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:28:16,matpower/case118/2017-04-30,train,raw,OPTIMAL,2,40.990000,0.00305894,3334215.590560,3324016.420115,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:29:33,matpower/case118/2017-05-01,train,raw,OPTIMAL,2,42.983000,0.00890603,3641876.035388,3609441.389461,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:30:45,matpower/case118/2017-05-02,train,raw,OPTIMAL,2,38.651000,0.00524392,3425601.368909,3407637.800477,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:31:59,matpower/case118/2017-05-03,train,raw,OPTIMAL,2,39.827000,0.00154855,3148494.536617,3143618.933748,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:33:11,matpower/case118/2017-05-05,train,raw,OPTIMAL,2,39.102000,0.00954300,3457470.714886,3424476.060190,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:34:34,matpower/case118/2017-05-06,train,raw,OPTIMAL,2,49.437000,0.00380641,3146014.863390,3134039.845981,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:35:47,matpower/case118/2017-05-07,train,raw,OPTIMAL,2,38.135000,0.00328891,3328166.488035,3317220.463329,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:37:01,matpower/case118/2017-05-08,train,raw,OPTIMAL,2,39.332000,0.00665494,3839142.063299,3813592.821740,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:38:11,matpower/case118/2017-05-10,train,raw,OPTIMAL,2,36.447000,0.00006355,4097579.202842,4097318.801294,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:39:24,matpower/case118/2017-05-11,train,raw,OPTIMAL,2,38.933000,0.00036067,4286083.017911,4284537.166960,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:40:32,matpower/case118/2017-05-12,train,raw,OPTIMAL,2,34.749000,0.00015107,4202750.880985,4202115.951028,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:41:40,matpower/case118/2017-05-14,train,raw,OPTIMAL,2,34.363000,0.00024556,3859630.151647,3858682.394414,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:42:48,matpower/case118/2017-05-15,train,raw,OPTIMAL,2,33.966000,0.00017517,4237803.206300,4237060.883140,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:43:55,matpower/case118/2017-05-17,train,raw,OPTIMAL,2,33.366000,0.00028353,5198766.939996,5197292.912878,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:45:20,matpower/case118/2017-05-19,train,raw,OPTIMAL,2,51.779000,0.00057032,4869262.594291,4866485.548236,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:46:44,matpower/case118/2017-05-20,train,raw,OPTIMAL,2,50.788000,0.00091184,3421401.080674,3418281.308771,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:47:52,matpower/case118/2017-05-21,train,raw,OPTIMAL,2,33.866000,0.00057589,3343782.037784,3341856.393096,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:49:01,matpower/case118/2017-05-23,train,raw,OPTIMAL,2,35.188000,0.00015243,3869732.579024,3869142.730609,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:50:08,matpower/case118/2017-05-24,train,raw,OPTIMAL,2,34.304000,0.00038523,4005402.262152,4003859.259188,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:51:19,matpower/case118/2017-05-26,train,raw,OPTIMAL,2,36.440000,0.00116224,3682976.603357,3678696.092710,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:52:28,matpower/case118/2017-05-29,train,raw,OPTIMAL,2,36.457000,0.00049275,3347867.449056,3346217.781049,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:53:37,matpower/case118/2017-05-30,train,raw,OPTIMAL,2,34.654000,0.00019024,3792689.726057,3791968.222661,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:54:44,matpower/case118/2017-05-31,train,raw,OPTIMAL,2,33.312000,0.00008555,4029163.925489,4028819.240005,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:55:50,matpower/case118/2017-06-01,train,raw,OPTIMAL,2,32.486000,0.00007735,4022639.377375,4022328.208051,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:56:57,matpower/case118/2017-06-02,train,raw,OPTIMAL,2,33.593000,0.00087651,4069430.341564,4065863.447773,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:58:05,matpower/case118/2017-06-03,train,raw,OPTIMAL,2,34.876000,0.00060897,3830197.760638,3827865.296418,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 18:59:14,matpower/case118/2017-06-04,train,raw,OPTIMAL,2,34.281000,0.00024814,4135732.697740,4134706.471033,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:00:21,matpower/case118/2017-06-05,train,raw,OPTIMAL,2,34.147000,0.00004190,4602207.658313,4602014.821766,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:01:27,matpower/case118/2017-06-06,train,raw,OPTIMAL,2,32.508000,0.00039852,4279056.578223,4277351.291420,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:02:38,matpower/case118/2017-06-07,train,raw,OPTIMAL,2,36.936000,0.00058886,3993973.402546,3991621.491664,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:03:58,matpower/case118/2017-06-08,train,raw,OPTIMAL,2,46.189000,0.00026052,4096615.493010,4095548.256975,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:05:04,matpower/case118/2017-06-09,train,raw,OPTIMAL,2,32.890000,0.00013745,4384073.933709,4383471.344904,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:06:14,matpower/case118/2017-06-11,train,raw,OPTIMAL,2,36.197000,0.00004834,5561841.926280,5561573.092459,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:07:26,matpower/case118/2017-06-12,train,raw,OPTIMAL,2,38.419000,0.00001985,6831236.907647,6831101.322005,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:08:35,matpower/case118/2017-06-13,train,raw,OPTIMAL,2,35.624000,0.00000685,7028557.702271,7028509.536819,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:09:43,matpower/case118/2017-06-16,train,raw,OPTIMAL,2,33.579000,0.00010594,5157460.944670,5156914.557476,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:10:51,matpower/case118/2017-06-17,train,raw,OPTIMAL,2,34.958000,0.00003364,5352297.256413,5352117.188762,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:12:19,matpower/case118/2017-06-19,train,raw,OPTIMAL,2,50.931000,0.00013679,6499380.324378,6498491.306408,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:13:50,matpower/case118/2017-06-20,train,raw,OPTIMAL,2,42.686000,0.00005363,6254019.129143,6253683.754984,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:15:27,matpower/case118/2017-06-21,train,raw,OPTIMAL,2,47.914000,0.00002396,6115423.044228,6115276.508357,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:16:50,matpower/case118/2017-06-22,train,raw,OPTIMAL,2,41.564000,0.00003845,6943090.599400,6942823.659349,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:18:07,matpower/case118/2017-06-23,train,raw,OPTIMAL,2,38.693000,0.00003503,6768858.612653,6768621.487034,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:19:30,matpower/case118/2017-06-24,train,raw,OPTIMAL,2,39.191000,0.00008181,5943870.983599,5943384.725491,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:20:53,matpower/case118/2017-06-26,train,raw,OPTIMAL,2,40.365000,0.00017983,5399307.877929,5398336.906687,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:22:13,matpower/case118/2017-06-29,train,raw,OPTIMAL,2,40.955000,0.00011915,6912090.455089,6911266.913504,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:23:38,matpower/case118/2017-06-30,train,raw,OPTIMAL,2,44.173000,0.00007095,7285093.940495,7284577.098428,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:24:57,matpower/case118/2017-07-01,train,raw,OPTIMAL,2,39.156000,0.00005560,6841148.159642,6840767.768128,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:26:26,matpower/case118/2017-07-02,train,raw,OPTIMAL,2,44.897000,0.00005425,6750133.232013,6749767.009055,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:27:48,matpower/case118/2017-07-04,train,raw,OPTIMAL,2,40.266000,0.00018195,6170160.162100,6169037.473785,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:29:07,matpower/case118/2017-07-05,train,raw,OPTIMAL,2,37.872000,0.00010846,6290768.857808,6290086.532938,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:30:34,matpower/case118/2017-07-06,train,raw,OPTIMAL,2,42.421000,0.00002101,6144488.226836,6144359.122413,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:31:55,matpower/case118/2017-07-07,train,raw,OPTIMAL,2,40.728000,0.00007238,6606131.315061,6605653.164195,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:33:12,matpower/case118/2017-07-08,train,raw,OPTIMAL,2,38.370000,0.00046291,6163530.041813,6160676.892809,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:34:30,matpower/case118/2017-07-10,train,raw,OPTIMAL,2,40.601000,0.00001629,6723077.062295,6722967.553910,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:35:59,matpower/case118/2017-07-11,train,raw,OPTIMAL,2,47.174000,0.00005229,7574380.225344,7573984.128393,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:37:25,matpower/case118/2017-07-12,train,raw,OPTIMAL,2,44.265000,0.00001188,8043371.313534,8043275.734671,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:39:04,matpower/case118/2017-07-13,train,raw,OPTIMAL,2,52.204000,0.00002076,8244642.286241,8244471.129915,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:40:51,matpower/case118/2017-07-15,train,raw,OPTIMAL,2,56.484000,0.00014274,6031414.621420,6030553.684128,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:42:29,matpower/case118/2017-07-17,train,raw,OPTIMAL,2,49.776000,0.00001579,6803831.274196,6803723.842323,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:44:03,matpower/case118/2017-07-20,train,raw,OPTIMAL,2,47.025000,0.00010573,8294303.589356,8293426.637875,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:45:33,matpower/case118/2017-07-22,train,raw,OPTIMAL,2,43.207000,0.00002377,6952032.417987,6951867.201355,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:46:48,matpower/case118/2017-07-23,train,raw,OPTIMAL,2,37.050000,0.00010763,6306408.303979,6305729.543678,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:48:06,matpower/case118/2017-07-24,train,raw,OPTIMAL,2,40.182000,0.00072802,6542528.028154,6537764.957128,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:49:25,matpower/case118/2017-07-25,train,raw,OPTIMAL,2,40.880000,0.00206662,5640035.352644,5628379.554110,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:50:40,matpower/case118/2017-07-27,train,raw,OPTIMAL,2,34.618000,0.00062803,6325908.445528,6321935.596728,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:51:58,matpower/case118/2017-07-28,train,raw,OPTIMAL,2,40.782000,0.00066950,5905925.848222,5901971.855841,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:53:21,matpower/case118/2017-07-29,train,raw,OPTIMAL,2,44.685000,0.00276640,4784025.638924,4770791.129892,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:54:40,matpower/case118/2017-07-30,train,raw,OPTIMAL,2,41.779000,0.00027684,4977556.442801,4976178.454363,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:55:58,matpower/case118/2017-08-01,train,raw,OPTIMAL,2,40.114000,0.00022294,7032106.233454,7030538.524885,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:57:15,matpower/case118/2017-08-02,train,raw,OPTIMAL,2,38.662000,0.00026981,7191421.118098,7189480.833978,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:58:33,matpower/case118/2017-08-03,train,raw,OPTIMAL,2,40.518000,0.00017339,6928621.044160,6927419.700564,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 19:59:42,matpower/case118/2017-08-04,train,raw,OPTIMAL,2,31.740000,0.00180183,5716370.772767,5706070.844306,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:00:51,matpower/case118/2017-08-05,train,raw,OPTIMAL,2,30.528000,0.00362973,4473785.203581,4457546.553193,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:02:02,matpower/case118/2017-08-06,train,raw,OPTIMAL,2,33.735000,0.00340541,4462039.173785,4446844.087670,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:03:15,matpower/case118/2017-08-07,train,raw,OPTIMAL,2,34.512000,0.00193645,4894355.582060,4884877.922209,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:04:23,matpower/case118/2017-08-08,train,raw,OPTIMAL,2,30.152000,0.00278416,4678418.722312,4665393.268065,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:05:34,matpower/case118/2017-08-10,train,raw,OPTIMAL,2,32.986000,0.00314521,5255912.625233,5239381.672621,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:06:45,matpower/case118/2017-08-11,train,raw,OPTIMAL,2,33.113000,0.00303890,5088299.336459,5072836.509972,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:07:53,matpower/case118/2017-08-12,train,raw,OPTIMAL,2,31.003000,0.00255336,4682626.480520,4670670.061041,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:09:07,matpower/case118/2017-08-13,train,raw,OPTIMAL,2,35.757000,0.00167357,5067979.391381,5059497.758431,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:10:20,matpower/case118/2017-08-15,train,raw,OPTIMAL,2,35.184000,0.00194650,6166905.223574,6154901.340473,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:11:33,matpower/case118/2017-08-16,train,raw,OPTIMAL,2,35.942000,0.00550494,6270556.525602,6236037.476899,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:12:52,matpower/case118/2017-08-19,train,raw,OPTIMAL,2,36.861000,0.00283568,5963298.153728,5946388.151580,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:14:18,matpower/case118/2017-08-20,train,raw,OPTIMAL,2,38.808000,0.00244420,5618304.246837,5604571.966875,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:15:36,matpower/case118/2017-08-23,train,raw,OPTIMAL,2,34.178000,0.00128486,5527847.505455,5520745.017293,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:16:46,matpower/case118/2017-08-24,train,raw,OPTIMAL,2,32.022000,0.00303763,5041526.885678,5026212.586751,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:17:56,matpower/case118/2017-08-25,train,raw,OPTIMAL,2,31.557000,0.00309978,4649000.289135,4634589.398220,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:19:09,matpower/case118/2017-08-27,train,raw,OPTIMAL,2,33.987000,0.00448554,4726128.564341,4704929.324100,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:20:22,matpower/case118/2017-08-28,train,raw,OPTIMAL,2,35.501000,0.00507211,5073149.711993,5047418.145142,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:21:37,matpower/case118/2017-08-29,train,raw,OPTIMAL,2,36.468000,0.00659824,4972191.972984,4939384.259491,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:22:53,matpower/case118/2017-08-30,train,raw,OPTIMAL,2,37.031000,0.01075697,5139353.357207,5084069.471908,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:24:04,matpower/case118/2017-08-31,train,raw,OPTIMAL,2,33.092000,0.00551577,5205242.626132,5176531.682459,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:25:19,matpower/case118/2017-09-01,train,raw,OPTIMAL,2,36.753000,0.00614726,4445141.324649,4417815.905042,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:26:29,matpower/case118/2017-09-03,train,raw,OPTIMAL,2,31.560000,0.01236207,3958686.928733,3909749.356983,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:27:38,matpower/case118/2017-09-04,train,raw,OPTIMAL,2,31.663000,0.00541114,4596881.544678,4572007.193799,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:28:54,matpower/case118/2017-09-05,train,raw,OPTIMAL,2,36.959000,0.02325981,5377747.169863,5252661.802852,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:30:07,matpower/case118/2017-09-06,train,raw,OPTIMAL,2,35.454000,0.00366164,4507919.277645,4491412.880954,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:31:21,matpower/case118/2017-09-07,train,raw,OPTIMAL,2,35.406000,0.00530556,4250606.715186,4228054.886350,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:32:34,matpower/case118/2017-09-08,train,raw,OPTIMAL,2,34.934000,0.00739557,4234277.016958,4202962.103778,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:33:44,matpower/case118/2017-09-09,train,raw,OPTIMAL,2,31.958000,0.00627426,3890682.501230,3866271.360346,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:34:55,matpower/case118/2017-09-10,train,raw,OPTIMAL,2,32.027000,0.00391544,3855912.578037,3840814.975114,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:36:05,matpower/case118/2017-09-11,train,raw,OPTIMAL,2,32.459000,0.00634008,4344944.704115,4317397.402762,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:37:25,matpower/case118/2017-09-13,train,raw,OPTIMAL,2,42.088000,0.00426000,5108285.700511,5086524.413786,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:38:39,matpower/case118/2017-09-15,train,raw,OPTIMAL,2,36.910000,0.01183609,5347011.299289,5283723.593437,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:40:02,matpower/case118/2017-09-16,train,raw,OPTIMAL,2,45.275000,0.00489860,4853432.130091,4829657.088994,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:41:18,matpower/case118/2017-09-19,train,raw,OPTIMAL,2,38.505000,0.00255760,5522254.393697,5508130.685984,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:42:24,matpower/case118/2017-09-20,train,raw,OPTIMAL,2,29.104000,0.00228206,6047014.629529,6033214.967394,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:43:31,matpower/case118/2017-09-21,train,raw,OPTIMAL,2,29.854000,0.00224078,6782521.398857,6767323.237977,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:44:46,matpower/case118/2017-09-23,train,raw,OPTIMAL,2,36.374000,0.02103012,5914977.160948,5790584.494989,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:45:55,matpower/case118/2017-09-24,train,raw,OPTIMAL,2,30.294000,0.02036468,5965216.148182,5843736.410224,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:47:07,matpower/case118/2017-09-25,train,raw,OPTIMAL,2,33.440000,0.01618338,5921083.880496,5825260.737342,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:48:19,matpower/case118/2017-09-26,train,raw,OPTIMAL,2,33.230000,0.01228320,5701961.203263,5631922.877686,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:49:31,matpower/case118/2017-09-27,train,raw,OPTIMAL,2,32.351000,0.01507281,5892580.996132,5803763.252180,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:50:52,matpower/case118/2017-09-28,train,raw,OPTIMAL,2,42.250000,0.01376095,4918034.734452,4850357.922095,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:52:10,matpower/case118/2017-09-29,train,raw,OPTIMAL,2,38.005000,0.01950029,4216099.256562,4133884.109243,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:53:23,matpower/case118/2017-10-03,train,raw,OPTIMAL,2,32.782000,0.00146082,4034790.446439,4028896.332090,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:54:31,matpower/case118/2017-10-04,train,raw,OPTIMAL,2,31.075000,0.00398676,4211363.866819,4194574.155944,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:55:46,matpower/case118/2017-10-08,train,raw,OPTIMAL,2,36.968000,0.00606631,4936522.777999,4906576.318591,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:57:05,matpower/case118/2017-10-09,train,raw,OPTIMAL,2,39.777000,0.00299920,5136305.153035,5120900.341627,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:58:25,matpower/case118/2017-10-10,train,raw,OPTIMAL,2,38.858000,0.00480958,5193624.828247,5168645.676545,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 20:59:49,matpower/case118/2017-10-11,train,raw,OPTIMAL,2,39.542000,0.00242745,4846560.943793,4834796.160647,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:01:01,matpower/case118/2017-10-12,train,raw,OPTIMAL,2,31.846000,0.00419984,4919827.308497,4899164.801468,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:02:31,matpower/case118/2017-10-13,train,raw,OPTIMAL,2,43.143000,0.00581008,4815672.855264,4787693.430939,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:03:46,matpower/case118/2017-10-14,train,raw,OPTIMAL,2,33.134000,0.00765368,4329656.218314,4296518.422930,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:04:57,matpower/case118/2017-10-15,train,raw,OPTIMAL,2,33.864000,0.00494888,4758333.790294,4734785.376369,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:06:05,matpower/case118/2017-10-16,train,raw,OPTIMAL,2,30.625000,0.00367372,4525207.399472,4508583.069409,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:07:14,matpower/case118/2017-10-17,train,raw,OPTIMAL,2,31.833000,0.00707002,4501749.530244,4469922.052100,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:08:23,matpower/case118/2017-10-18,train,raw,OPTIMAL,2,29.960000,0.00467743,4275350.373972,4255352.711756,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:09:32,matpower/case118/2017-10-19,train,raw,OPTIMAL,2,31.040000,0.00435176,4402520.319732,4383361.629522,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:10:35,matpower/case118/2017-10-21,train,raw,OPTIMAL,2,25.739000,0.00717463,3724422.274467,3697700.930448,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:11:43,matpower/case118/2017-10-22,train,raw,OPTIMAL,2,28.669000,0.00629618,3937706.231190,3912913.730976,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:12:52,matpower/case118/2017-10-23,train,raw,OPTIMAL,2,31.315000,0.00607410,4671109.504259,4642736.714382,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:14:02,matpower/case118/2017-10-24,train,raw,OPTIMAL,2,30.951000,0.00408508,4788186.196762,4768626.090885,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:15:10,matpower/case118/2017-10-25,train,raw,OPTIMAL,2,29.791000,0.00371394,4615796.170281,4598653.393433,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:16:12,matpower/case118/2017-10-26,train,raw,OPTIMAL,2,24.307000,0.01192908,4521758.454111,4467818.024416,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:17:17,matpower/case118/2017-10-27,train,raw,OPTIMAL,2,26.241000,0.01444141,4324752.648898,4262297.138188,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:18:24,matpower/case118/2017-10-30,train,raw,OPTIMAL,2,29.573000,0.01389526,4435014.326699,4373388.653523,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:19:34,matpower/case118/2017-10-31,train,raw,OPTIMAL,2,32.016000,0.01073866,4397912.966829,4350685.267356,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:20:45,matpower/case118/2017-11-02,train,raw,OPTIMAL,2,32.292000,0.01540634,4383480.832697,4315947.425322,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:21:56,matpower/case118/2017-11-03,train,raw,OPTIMAL,2,31.544000,0.01891210,4073100.570532,3996069.691465,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:23:03,matpower/case118/2017-11-04,train,raw,OPTIMAL,2,29.361000,0.02231974,3708800.252684,3626020.805770,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:24:13,matpower/case118/2017-11-05,train,raw,OPTIMAL,2,31.139000,0.02413618,3853896.665174,3760878.333319,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:25:24,matpower/case118/2017-11-07,train,raw,OPTIMAL,2,32.882000,0.00340941,4412630.268749,4397585.824775,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:26:38,matpower/case118/2017-11-09,train,raw,OPTIMAL,2,36.122000,0.00190435,4988764.882854,4979264.515599,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:27:52,matpower/case118/2017-11-10,train,raw,OPTIMAL,2,36.142000,0.00138402,5343158.469200,5335763.415674,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:29:16,matpower/case118/2017-11-11,train,raw,OPTIMAL,2,45.503000,0.00073984,4911581.917400,4907948.110634,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:30:27,matpower/case118/2017-11-12,train,raw,OPTIMAL,2,34.589000,0.00167103,4588471.293036,4580803.805497,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:31:44,matpower/case118/2017-11-15,train,raw,OPTIMAL,2,39.581000,0.00154771,4711000.468680,4703709.229439,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:32:57,matpower/case118/2017-11-16,train,raw,OPTIMAL,2,36.247000,0.00252680,4830258.443147,4818053.363726,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:34:11,matpower/case118/2017-11-17,train,raw,OPTIMAL,2,37.296000,0.00308600,4865411.332893,4850396.658329,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:35:23,matpower/case118/2017-11-18,train,raw,OPTIMAL,2,35.032000,0.00702622,4189228.854765,4159794.391421,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:36:34,matpower/case118/2017-11-19,train,raw,OPTIMAL,2,33.397000,0.00368542,4533965.018477,4517255.455387,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:38:01,matpower/case118/2017-11-20,train,raw,OPTIMAL,2,49.616000,0.00158665,5394645.833010,5386086.400461,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:39:18,matpower/case118/2017-11-21,train,raw,OPTIMAL,2,40.574000,0.00185355,4915607.260212,4906495.943066,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:40:28,matpower/case118/2017-11-22,train,raw,OPTIMAL,2,32.761000,0.00147601,5039652.747192,5032214.168887,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:41:39,matpower/case118/2017-11-23,train,raw,OPTIMAL,2,33.648000,0.00342918,5071048.862307,5053659.332442,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:42:50,matpower/case118/2017-11-25,train,raw,OPTIMAL,2,33.746000,0.00791125,4389501.766386,4354775.339806,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:44:01,matpower/case118/2017-11-26,train,raw,OPTIMAL,2,33.303000,0.00679836,4804721.659951,4772057.409588,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:45:12,matpower/case118/2017-11-27,train,raw,OPTIMAL,2,35.092000,0.00211410,5425652.974413,5414182.577126,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:46:26,matpower/case118/2017-11-28,train,raw,OPTIMAL,2,36.470000,0.00359099,5055762.295693,5037607.089971,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:47:34,matpower/case118/2017-11-29,train,raw,OPTIMAL,2,30.681000,0.00341952,5093437.414405,5076020.297999,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:48:41,matpower/case118/2017-12-01,train,raw,OPTIMAL,2,30.682000,0.02269650,5038969.844867,4924602.850239,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:49:54,matpower/case118/2017-12-03,train,raw,OPTIMAL,2,35.411000,0.01035915,5230572.745015,5176388.476905,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:51:05,matpower/case118/2017-12-04,train,raw,OPTIMAL,2,33.119000,0.00474536,5541161.130095,5514866.299925,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:52:17,matpower/case118/2017-12-07,train,raw,OPTIMAL,2,34.894000,0.00314928,6248550.667583,6228872.207686,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:53:31,matpower/case118/2017-12-08,train,raw,OPTIMAL,2,36.899000,0.00213864,6742566.350919,6728146.411198,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:54:48,matpower/case118/2017-12-09,train,raw,OPTIMAL,2,39.574000,0.00502672,7122495.967886,7086693.145565,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:56:07,matpower/case118/2017-12-10,train,raw,OPTIMAL,2,42.491000,0.00352191,7008845.524673,6984161.022042,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:57:29,matpower/case118/2017-12-11,train,raw,OPTIMAL,2,44.786000,0.00516161,7530100.866536,7491233.418681,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:58:39,matpower/case118/2017-12-12,train,raw,OPTIMAL,2,32.786000,0.00216574,7856643.244367,7839627.773662,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 21:59:51,matpower/case118/2017-12-13,train,raw,OPTIMAL,2,35.365000,0.00166546,8290571.738578,8276764.083371,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 22:01:14,matpower/case118/2017-12-15,train,raw,OPTIMAL,2,45.566000,0.00092982,8527591.871877,8519662.787130,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 22:02:33,matpower/case118/2017-12-16,train,raw,OPTIMAL,2,42.494000,0.00038107,7412840.649605,7410015.858057,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 22:03:52,matpower/case118/2017-12-17,train,raw,OPTIMAL,2,41.073000,0.00071502,7250400.380731,7245216.219131,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 22:05:15,matpower/case118/2017-12-19,train,raw,OPTIMAL,2,45.562000,0.00345386,6842979.051138,6819344.363651,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 22:06:40,matpower/case118/2017-12-21,train,raw,OPTIMAL,2,48.641000,0.00102934,8016928.793169,8008676.641747,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 22:07:58,matpower/case118/2017-12-23,train,raw,OPTIMAL,2,40.595000,0.00266114,5669620.096569,5654532.458357,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 22:09:14,matpower/case118/2017-12-24,train,raw,OPTIMAL,2,38.680000,0.00367562,5917742.516003,5895991.127329,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 22:10:26,matpower/case118/2017-12-25,train,raw,OPTIMAL,2,34.179000,0.00087254,6668344.960969,6662526.545740,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 22:11:34,matpower/case118/2017-12-26,train,raw,OPTIMAL,2,30.980000,0.00152286,7790026.485482,7778163.361906,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 22:12:51,matpower/case118/2017-12-27,train,raw,OPTIMAL,2,39.780000,0.00071552,8499646.127207,8493564.430381,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 22:13:58,matpower/case118/2017-12-28,train,raw,OPTIMAL,2,30.367000,0.00037209,8858302.923714,8855006.817097,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 22:15:36,matpower/case118/2017-12-29,train,raw,TIME_LIMIT,9,60.325000,0.11552223,20127913.524390,17802692.155517,2,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 22:17:00,matpower/case118/2017-12-30,train,raw,OPTIMAL,2,47.161000,0.00270676,7855732.650394,7834469.055556,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 22:18:14,matpower/case118/2017-12-31,train,raw,OPTIMAL,2,37.167000,0.00854952,8256728.593570,8186137.523675,1,OK,,49068,5832,5832,1597536,0.00000000
2025-09-07 22:34:08,matpower/case118/2017-01-09,test,raw,OPTIMAL,2,14.115000,0.00010158,7078282.549776,7077563.503192,1,OK,,49068,5832,5832,83644,0.00000000
2025-09-07 22:34:38,matpower/case118/2017-01-09,test,warm,OPTIMAL,2,14.850000,0.00000374,7077589.987506,7077563.503192,1,OK,49068,49068,5832,5832,83644,0.00000000
2025-09-07 22:35:00,matpower/case118/2017-01-10,test,raw,OPTIMAL,2,8.652000,0.00009282,6163863.107085,6163290.950552,1,OK,,49068,5832,5832,62022,0.00000000
2025-09-07 22:35:22,matpower/case118/2017-01-10,test,warm,OPTIMAL,2,8.721000,0.04299909,6440214.280056,6163290.950552,1,FAIL,49068,49068,5832,5832,62022,8.61214296
2025-09-07 22:35:48,matpower/case118/2017-01-24,test,raw,OPTIMAL,2,11.785000,0.00141872,5687639.859782,5679570.712498,1,OK,,49068,5832,5832,89436,0.00000000
2025-09-07 22:36:16,matpower/case118/2017-01-24,test,warm,OPTIMAL,2,12.707000,0.00001613,5679662.326917,5679570.712498,1,OK,49068,49068,5832,5832,89436,0.00000000
2025-09-07 22:36:42,matpower/case118/2017-01-25,test,raw,OPTIMAL,2,13.074000,0.00006686,5423005.512082,5422642.909132,1,OK,,49068,5832,5832,97416,0.00000000
2025-09-07 22:37:10,matpower/case118/2017-01-25,test,warm,OPTIMAL,2,13.206000,0.00001541,5422726.468085,5422642.909132,1,OK,49068,49068,5832,5832,97416,0.00000000
2025-09-07 22:37:30,matpower/case118/2017-01-28,test,raw,OPTIMAL,2,6.680000,0.03562374,5574898.725893,5376299.991124,1,OK,,49068,5832,5832,45276,0.00000000
2025-09-07 22:37:50,matpower/case118/2017-01-28,test,warm,OPTIMAL,2,6.575000,0.03562374,5574898.725893,5376299.991124,1,OK,49068,49068,5832,5832,45276,0.00000000
2025-09-07 22:38:11,matpower/case118/2017-01-29,test,raw,OPTIMAL,2,8.055000,0.00000029,5548562.422801,5548560.808519,1,OK,,49068,5832,5832,57526,0.00000000
2025-09-07 22:38:33,matpower/case118/2017-01-29,test,warm,OPTIMAL,2,8.064000,0.03632685,5757720.644063,5548560.808519,1,OK,49068,49068,5832,5832,57526,0.00000000
2025-09-07 22:38:51,matpower/case118/2017-02-07,test,raw,OPTIMAL,2,5.305000,0.00013799,4913930.102406,4913252.012326,1,OK,,49068,5832,5832,42358,0.00000000
2025-09-07 22:39:14,matpower/case118/2017-02-07,test,warm,OPTIMAL,2,4.820000,0.04214029,5129339.685880,4913187.810217,1,OK,49068,49068,5832,5832,42358,0.00000000
2025-09-07 22:39:37,matpower/case118/2017-02-12,test,raw,OPTIMAL,2,9.175000,0.00006556,5592444.453819,5592077.788462,1,OK,,49068,5832,5832,57526,0.00000000
2025-09-07 22:40:00,matpower/case118/2017-02-12,test,warm,OPTIMAL,2,8.863000,0.03390558,5788334.661243,5592077.788462,1,OK,49068,49068,5832,5832,57526,0.00000000
2025-09-07 22:40:25,matpower/case118/2017-02-28,test,raw,OPTIMAL,2,12.005000,0.00008621,4372284.034888,4371907.119566,1,OK,,49068,5832,5832,82254,0.00000000
2025-09-07 22:40:51,matpower/case118/2017-02-28,test,warm,OPTIMAL,2,11.925000,0.03282460,4520283.601179,4371907.119566,1,OK,49068,49068,5832,5832,82254,0.00000000
2025-09-07 22:41:20,matpower/case118/2017-03-04,test,raw,OPTIMAL,2,14.228000,0.00025248,4226453.400210,4225386.326136,1,OK,,49068,5832,5832,132058,0.00000000
2025-09-07 22:41:49,matpower/case118/2017-03-04,test,warm,OPTIMAL,2,14.372000,0.04410352,4420338.842747,4225386.326136,1,OK,49068,49068,5832,5832,132058,0.00000000
2025-09-07 22:42:10,matpower/case118/2017-03-07,test,raw,OPTIMAL,2,8.069000,0.00003738,4038847.598909,4038696.620060,1,OK,,49068,5832,5832,51804,0.00000000
2025-09-07 22:42:33,matpower/case118/2017-03-07,test,warm,OPTIMAL,2,8.924000,0.03958438,4205155.078381,4038696.620060,1,OK,49068,49068,5832,5832,51804,0.00000000
2025-09-07 22:43:01,matpower/case118/2017-03-09,test,raw,OPTIMAL,2,14.075000,0.00041922,4110255.769613,4108532.653944,1,OK,,49068,5832,5832,121116,0.00000000
2025-09-07 22:43:32,matpower/case118/2017-03-09,test,warm,OPTIMAL,2,15.276000,0.00023173,4109484.929990,4108532.653944,1,OK,49068,49068,5832,5832,121116,0.00000000
2025-09-07 22:43:59,matpower/case118/2017-03-11,test,raw,OPTIMAL,2,13.778000,0.00002665,5314355.445488,5314213.804288,1,OK,,49068,5832,5832,132058,0.00000000
2025-09-07 22:44:33,matpower/case118/2017-03-11,test,warm,OPTIMAL,2,13.759000,0.03423312,5502584.427333,5314213.804288,1,OK,49068,49068,5832,5832,132058,0.00000000
2025-09-07 22:45:01,matpower/case118/2017-03-14,test,raw,OPTIMAL,2,14.432000,0.00007997,5411312.272747,5410879.551717,1,OK,,49068,5832,5832,93644,0.00000000
2025-09-07 22:45:31,matpower/case118/2017-03-14,test,warm,OPTIMAL,2,15.179000,0.00005781,5411192.349260,5410879.551717,1,OK,49068,49068,5832,5832,93644,0.00000000
2025-09-07 22:45:55,matpower/case118/2017-03-20,test,raw,OPTIMAL,2,11.545000,0.00046283,4582671.400448,4580550.404924,1,OK,,49068,5832,5832,69108,0.00000000
2025-09-07 22:46:22,matpower/case118/2017-03-20,test,warm,OPTIMAL,2,11.824000,0.00013338,4581161.418841,4580550.404924,1,OK,49068,49068,5832,5832,69108,0.00000000
2025-09-07 22:46:45,matpower/case118/2017-03-25,test,raw,OPTIMAL,2,10.506000,0.00112670,3556036.940042,3552030.354564,1,OK,,49068,5832,5832,69384,0.00000000
2025-09-07 22:47:10,matpower/case118/2017-03-25,test,warm,OPTIMAL,2,10.273000,0.00075093,3554699.699659,3552030.354564,1,OK,49068,49068,5832,5832,69384,0.00000000
2025-09-07 22:47:29,matpower/case118/2017-04-07,test,raw,OPTIMAL,2,6.594000,0.00011712,4321691.114373,4321184.963976,1,OK,,49068,5832,5832,42794,0.00000000
2025-09-07 22:47:50,matpower/case118/2017-04-07,test,warm,OPTIMAL,2,6.964000,0.00010134,4321691.114373,4321253.147043,1,OK,49068,49068,5832,5832,42794,0.00000000
2025-09-07 22:48:17,matpower/case118/2017-04-17,test,raw,OPTIMAL,2,13.677000,0.00145471,3845241.572704,3839647.853950,1,OK,,49068,5832,5832,52836,0.00000000
2025-09-07 22:48:45,matpower/case118/2017-04-17,test,warm,OPTIMAL,2,14.460000,0.04009601,4000033.226012,3839647.853950,1,OK,49068,49068,5832,5832,52836,0.00000000
2025-09-07 22:49:12,matpower/case118/2017-04-22,test,raw,OPTIMAL,2,13.883000,0.00005726,3365017.901353,3364825.220954,1,OK,,49068,5832,5832,50336,0.00000000
2025-09-07 22:49:39,matpower/case118/2017-04-22,test,warm,OPTIMAL,2,13.468000,0.04422065,3520504.209117,3364825.220954,1,OK,49068,49068,5832,5832,50336,0.00000000
2025-09-07 22:50:05,matpower/case118/2017-04-25,test,raw,OPTIMAL,2,13.094000,0.00032387,4110042.807877,4108711.686998,1,OK,,49068,5832,5832,64474,0.00000000
2025-09-07 22:50:37,matpower/case118/2017-04-25,test,warm,OPTIMAL,2,12.982000,0.00091556,4112476.888227,4108711.686998,1,OK,49068,49068,5832,5832,64474,0.00000000
2025-09-07 22:51:06,matpower/case118/2017-05-04,test,raw,OPTIMAL,2,16.476000,0.00053810,3529328.254132,3527429.133431,1,OK,,49068,5832,5832,51102,0.00000000
2025-09-07 22:51:34,matpower/case118/2017-05-04,test,warm,OPTIMAL,2,13.899000,0.01324556,3574197.817749,3526855.556842,1,OK,49068,49068,5832,5832,51102,0.00000000
2025-09-07 22:51:59,matpower/case118/2017-05-18,test,raw,OPTIMAL,2,12.424000,0.03656468,5655723.809015,5448924.049622,1,OK,,49068,5832,5832,59890,0.00000000
2025-09-07 22:52:25,matpower/case118/2017-05-18,test,warm,OPTIMAL,2,11.633000,0.00000971,5448976.933598,5448924.049622,1,OK,49068,49068,5832,5832,59890,0.00000000
2025-09-07 22:52:54,matpower/case118/2017-05-22,test,raw,OPTIMAL,2,16.445000,0.00000216,3841570.655895,3841562.356197,1,OK,,49068,5832,5832,62766,0.00000000
2025-09-07 22:53:22,matpower/case118/2017-05-22,test,warm,OPTIMAL,2,13.608000,0.00002112,3841570.655895,3841489.511058,1,OK,49068,49068,5832,5832,62766,0.00000000
2025-09-07 22:53:52,matpower/case118/2017-05-27,test,raw,OPTIMAL,2,17.217000,0.00003665,3269129.631247,3269009.818058,1,OK,,49068,5832,5832,61492,0.00000000
2025-09-07 22:54:20,matpower/case118/2017-05-27,test,warm,OPTIMAL,2,14.058000,0.00007635,3269047.507229,3268797.906080,1,OK,49068,49068,5832,5832,61492,0.00000000
2025-09-07 22:54:53,matpower/case118/2017-05-28,test,raw,OPTIMAL,2,19.378000,0.00005484,3167345.199860,3167171.498272,1,OK,,49068,5832,5832,61492,0.00000000
2025-09-07 22:55:24,matpower/case118/2017-05-28,test,warm,OPTIMAL,2,16.690000,0.00006941,3167265.289054,3167045.455365,1,OK,49068,49068,5832,5832,61492,0.00000000
2025-09-07 22:55:52,matpower/case118/2017-06-10,test,raw,OPTIMAL,2,15.467000,0.00003544,4724457.798041,4724290.365510,1,OK,,49068,5832,5832,60324,0.00000000
2025-09-07 22:56:18,matpower/case118/2017-06-10,test,warm,OPTIMAL,2,12.291000,0.00008055,4724518.455499,4724137.904835,1,OK,49068,49068,5832,5832,60324,0.00000000
2025-09-07 22:56:43,matpower/case118/2017-06-15,test,raw,OPTIMAL,2,10.950000,0.00072945,5055652.450509,5051964.605425,1,OK,,49068,5832,5832,79642,0.00000000
2025-09-07 22:57:13,matpower/case118/2017-06-15,test,warm,OPTIMAL,2,11.656000,0.00005100,5052222.268122,5051964.605425,1,OK,49068,49068,5832,5832,79642,0.00000000
2025-09-07 22:57:41,matpower/case118/2017-06-25,test,raw,OPTIMAL,2,14.652000,0.00001276,5378323.189359,5378254.558653,1,OK,,49068,5832,5832,59566,0.00000000
2025-09-07 22:58:08,matpower/case118/2017-06-25,test,warm,OPTIMAL,2,13.364000,0.00217648,5389890.703334,5378159.714348,1,OK,49068,49068,5832,5832,59566,0.00000000
2025-09-07 22:58:35,matpower/case118/2017-06-28,test,raw,OPTIMAL,2,14.380000,0.00061248,5601949.039070,5598517.937670,1,OK,,49068,5832,5832,59350,0.00000000
2025-09-07 22:59:02,matpower/case118/2017-06-28,test,warm,OPTIMAL,2,12.636000,0.00753595,5640929.531583,5598419.746970,1,OK,49068,49068,5832,5832,59350,0.00000000
2025-09-07 22:59:30,matpower/case118/2017-07-03,test,raw,OPTIMAL,2,14.894000,0.00001258,6405253.484085,6405172.930127,1,OK,,49068,5832,5832,58980,0.00000000
2025-09-07 22:59:58,matpower/case118/2017-07-03,test,warm,OPTIMAL,2,13.804000,0.00400074,6430836.422458,6405108.346207,1,OK,49068,49068,5832,5832,58980,0.00000000
2025-09-07 23:00:24,matpower/case118/2017-07-09,test,raw,OPTIMAL,2,11.111000,0.00464986,5918012.021684,5890494.067289,1,OK,,49068,5832,5832,68980,0.00000000
2025-09-07 23:00:50,matpower/case118/2017-07-09,test,warm,OPTIMAL,2,11.316000,0.03925977,6131203.699964,5890494.067289,1,OK,49068,49068,5832,5832,68980,0.00000000
2025-09-07 23:01:17,matpower/case118/2017-07-14,test,raw,OPTIMAL,2,14.430000,0.00000342,6883565.115446,6883541.560361,1,OK,,49068,5832,5832,60762,0.00000000
2025-09-07 23:01:46,matpower/case118/2017-07-14,test,warm,OPTIMAL,2,14.639000,0.00000342,6883565.115446,6883541.560361,1,OK,49068,49068,5832,5832,60762,0.00000000
2025-09-07 23:02:11,matpower/case118/2017-07-21,test,raw,OPTIMAL,2,11.777000,0.00002016,8006369.083503,8006207.670216,1,OK,,49068,5832,5832,63420,0.00000000
2025-09-07 23:02:37,matpower/case118/2017-07-21,test,warm,OPTIMAL,2,12.151000,0.00002016,8006369.083503,8006207.670216,1,OK,49068,49068,5832,5832,63420,0.00000000
2025-09-07 23:03:10,matpower/case118/2017-07-26,test,raw,OPTIMAL,2,19.910000,0.00001597,5573157.058580,5573068.077846,1,OK,,49068,5832,5832,59350,0.00000000
2025-09-07 23:03:42,matpower/case118/2017-07-26,test,warm,OPTIMAL,2,14.234000,0.00017815,5573751.309895,5572758.360956,1,OK,49068,49068,5832,5832,59350,0.00000000
2025-09-07 23:04:12,matpower/case118/2017-08-09,test,raw,OPTIMAL,2,16.736000,0.00053045,4804874.167417,4802325.418750,1,OK,,49068,5832,5832,69466,0.00000000
2025-09-07 23:04:42,matpower/case118/2017-08-09,test,warm,OPTIMAL,2,15.174000,0.02709757,4935348.821968,4801612.873402,1,OK,49068,49068,5832,5832,69466,0.00000000
2025-09-07 23:05:14,matpower/case118/2017-08-17,test,raw,OPTIMAL,2,17.616000,0.00189132,6204389.575916,6192655.076915,1,OK,,49068,5832,5832,69466,0.00000000
2025-09-07 23:05:43,matpower/case118/2017-08-17,test,warm,OPTIMAL,2,14.557000,0.01154480,6264348.289453,6192027.665841,1,OK,49068,49068,5832,5832,69466,0.00000000
2025-09-07 23:06:09,matpower/case118/2017-08-18,test,raw,OPTIMAL,2,12.824000,0.00014783,6314055.736189,6313122.319320,1,OK,,49068,5832,5832,77376,0.00000000
2025-09-07 23:06:35,matpower/case118/2017-08-18,test,warm,OPTIMAL,2,12.150000,0.04481616,6609327.008976,6313122.319320,1,OK,49068,49068,5832,5832,77376,0.00000000
2025-09-07 23:07:02,matpower/case118/2017-09-02,test,raw,OPTIMAL,2,13.877000,0.00203056,3677044.847255,3669578.399028,1,OK,,49068,5832,5832,61492,0.00000000
2025-09-07 23:07:31,matpower/case118/2017-09-02,test,warm,OPTIMAL,2,14.343000,0.00203056,3677044.847255,3669578.399028,1,OK,49068,49068,5832,5832,61492,0.00000000
2025-09-07 23:07:55,matpower/case118/2017-09-17,test,raw,OPTIMAL,2,11.573000,0.00032618,4947218.321797,4945604.635221,1,OK,,49068,5832,5832,58366,0.00000000
2025-09-07 23:08:22,matpower/case118/2017-09-17,test,warm,OPTIMAL,2,12.178000,0.00861949,4988603.841760,4945604.635221,1,OK,49068,49068,5832,5832,58366,0.00000000
2025-09-07 23:08:45,matpower/case118/2017-09-18,test,raw,OPTIMAL,2,10.821000,0.00258102,5460232.276392,5446139.299980,1,OK,,49068,5832,5832,64444,0.00000000
2025-09-07 23:09:11,matpower/case118/2017-09-18,test,warm,OPTIMAL,2,11.562000,0.01374805,5522056.788278,5446139.299980,1,OK,49068,49068,5832,5832,64444,0.00000000
2025-09-07 23:09:33,matpower/case118/2017-10-01,test,raw,OPTIMAL,2,9.556000,0.00017402,3415562.132509,3414967.744682,1,OK,,49068,5832,5832,49954,0.00000000
2025-09-07 23:09:59,matpower/case118/2017-10-01,test,warm,OPTIMAL,2,7.261000,0.00036401,3415562.132509,3414318.818927,1,OK,49068,49068,5832,5832,49954,0.00000000
2025-09-07 23:10:22,matpower/case118/2017-10-28,test,raw,OPTIMAL,2,10.360000,0.03342773,3956805.386788,3824538.368458,1,OK,,49068,5832,5832,65272,0.00000000
2025-09-07 23:10:46,matpower/case118/2017-10-28,test,warm,OPTIMAL,2,9.122000,0.00469390,3842296.222064,3824260.867304,1,OK,49068,49068,5832,5832,65272,0.00000000
2025-09-07 23:11:09,matpower/case118/2017-10-29,test,raw,OPTIMAL,2,10.651000,0.00822962,4129792.460667,4095805.831364,1,OK,,49068,5832,5832,49954,0.00000000
2025-09-07 23:11:31,matpower/case118/2017-10-29,test,warm,OPTIMAL,2,8.080000,0.03923926,4255410.835487,4088431.654585,1,OK,49068,49068,5832,5832,49954,0.00000000
2025-09-07 23:11:58,matpower/case118/2017-11-13,test,raw,OPTIMAL,2,13.112000,0.00024785,5006780.584562,5005539.629156,1,OK,,49068,5832,5832,104516,0.00000000
2025-09-07 23:12:26,matpower/case118/2017-11-13,test,warm,OPTIMAL,2,12.681000,0.00171615,5014144.647585,5005539.629156,1,OK,49068,49068,5832,5832,104516,0.00000000
2025-09-07 23:12:51,matpower/case118/2017-11-30,test,raw,OPTIMAL,2,11.646000,0.01460136,5327057.835042,5249275.533842,1,OK,,49068,5832,5832,97416,0.00000000
2025-09-07 23:13:18,matpower/case118/2017-11-30,test,warm,OPTIMAL,2,12.471000,0.00466977,5273903.440782,5249275.533842,1,OK,49068,49068,5832,5832,97416,0.00000000
2025-09-07 23:13:36,matpower/case118/2017-12-06,test,raw,OPTIMAL,2,5.588000,0.00171213,5591136.135557,5581563.381035,1,OK,,49068,5832,5832,42192,0.00000000
2025-09-07 23:13:56,matpower/case118/2017-12-06,test,warm,OPTIMAL,2,5.930000,0.00068988,5585426.320581,5581573.066048,1,OK,49068,49068,5832,5832,42192,0.00000000
2025-09-07 23:14:28,matpower/case118/2017-12-18,test,raw,OPTIMAL,2,18.830000,0.00396952,7224018.338950,7195342.440439,1,FAIL,,49068,5832,5832,83644,3.41305107
2025-09-07 23:15:02,matpower/case118/2017-12-18,test,warm,OPTIMAL,2,19.114000,0.00611119,7239584.899674,7195342.440439,1,OK,49068,49068,5832,5832,83644,0.00000000
2025-09-07 23:15:44,matpower/case118/2017-12-20,test,raw,OPTIMAL,2,23.329000,0.00122386,6853264.635246,6844877.229826,1,OK,,49068,5832,5832,120008,0.00000000
2025-09-07 23:16:22,matpower/case118/2017-12-20,test,warm,OPTIMAL,2,23.314000,0.02563562,7024966.634181,6844877.229826,1,OK,49068,49068,5832,5832,120008,0.00000000
2025-09-07 23:16:55,matpower/case118/2017-12-22,test,raw,OPTIMAL,2,19.772000,0.00164992,6710759.707876,6699687.513449,1,OK,,49068,5832,5832,62022,0.00000000
2025-09-07 23:17:23,matpower/case118/2017-12-22,test,warm,OPTIMAL,2,14.105000,0.04205850,6992303.159182,6698217.410673,1,FAIL,49068,49068,5832,5832,62022,27.91104808

```

### File: `src/data/output/matpower/case14/perf_matpower_case14_basic_01.csv`

```
timestamp,instance_name,case_folder,technique,run_id,status,status_code,runtime_sec,mip_gap,obj_val,obj_bound,nodes,num_vars,num_bin_vars,num_int_vars,num_cont_vars,num_constrs,num_nzs,buses,units,lines,reserves,time_steps,time_step_min,ptdf_nnz,lodf_nnz,startup_count,startup_cost,reserve_shortfall_mw,reserve_shortfall_penalty,base_overflow_mw,base_overflow_penalty,cont_overflow_mw,cont_overflow_penalty
2025-08-18 19:16:03,matpower/case14/2017-01-01,matpower/case14,basic,1,OPTIMAL,2,0.119000,0.00000000,359042.038425,359042.038425,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,10985.830000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:04,matpower/case14/2017-01-02,matpower/case14,basic,1,OPTIMAL,2,0.116000,0.00000000,371163.126513,371163.126513,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,10985.830000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:05,matpower/case14/2017-01-03,matpower/case14,basic,1,OPTIMAL,2,0.118000,0.00000000,369818.552627,369818.552627,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,10985.830000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:06,matpower/case14/2017-01-04,matpower/case14,basic,1,OPTIMAL,2,0.106000,0.00920179,394367.995088,390739.105534,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,15374.530000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:07,matpower/case14/2017-01-05,matpower/case14,basic,1,OPTIMAL,2,0.106000,0.00033360,451224.865749,451074.338050,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,15353.910000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:08,matpower/case14/2017-01-06,matpower/case14,basic,1,OPTIMAL,2,0.129000,0.01506701,470510.532774,463421.344329,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,15345.110000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:09,matpower/case14/2017-01-07,matpower/case14,basic,1,OPTIMAL,2,0.147000,0.02882829,499832.987726,485423.658097,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,15555.310000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:10,matpower/case14/2017-01-08,matpower/case14,basic,1,OPTIMAL,2,0.112000,0.01824887,494924.019816,485892.216757,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,4.0000,26172.030000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:11,matpower/case14/2017-01-09,matpower/case14,basic,1,OPTIMAL,2,0.115000,0.01480151,524451.803986,516689.126227,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,4.0000,26120.900000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:12,matpower/case14/2017-01-10,matpower/case14,basic,1,OPTIMAL,2,0.118000,0.02993883,472806.292160,458651.023665,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,4.0000,26120.900000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:13,matpower/case14/2017-01-11,matpower/case14,basic,1,OPTIMAL,2,0.113000,0.04738889,410585.120248,391127.945629,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,4.0000,26120.900000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:14,matpower/case14/2017-01-12,matpower/case14,basic,1,OPTIMAL,2,0.111000,0.03502958,382857.052139,369445.730749,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,15490.100000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:15,matpower/case14/2017-01-13,matpower/case14,basic,1,OPTIMAL,2,0.114000,0.00050083,422686.419622,422474.727318,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,9006.340000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:16,matpower/case14/2017-01-14,matpower/case14,basic,1,OPTIMAL,2,0.113000,0.03104764,432114.298341,418698.167195,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,15398.870000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:17,matpower/case14/2017-01-15,matpower/case14,basic,1,OPTIMAL,2,0.110000,0.02257043,378051.117818,369518.339655,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,15370.900000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:19,matpower/case14/2017-01-16,matpower/case14,basic,1,OPTIMAL,2,0.120000,0.00156652,399451.317778,398825.570425,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,8967.700000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:19,matpower/case14/2017-01-17,matpower/case14,basic,1,OPTIMAL,2,0.115000,0.00000000,407843.945560,407843.945560,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,4577.990000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:21,matpower/case14/2017-01-18,matpower/case14,basic,1,OPTIMAL,2,0.155000,0.00169908,396616.144732,395942.261440,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,8951.600000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:22,matpower/case14/2017-01-19,matpower/case14,basic,1,OPTIMAL,2,0.115000,0.00000000,393008.779669,393008.779669,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,4568.740000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:23,matpower/case14/2017-01-20,matpower/case14,basic,1,OPTIMAL,2,0.119000,0.00061849,448264.771708,447987.525818,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,10942.520000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:24,matpower/case14/2017-01-21,matpower/case14,basic,1,OPTIMAL,2,0.158000,0.03736787,409122.003512,393833.984045,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,15313.640000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:25,matpower/case14/2017-01-22,matpower/case14,basic,1,OPTIMAL,2,0.139000,0.00394875,402799.356424,401208.801163,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,10923.240000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:26,matpower/case14/2017-01-23,matpower/case14,basic,1,OPTIMAL,2,0.113000,0.01574481,461521.743920,454255.173820,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,15251.710000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:27,matpower/case14/2017-01-24,matpower/case14,basic,1,OPTIMAL,2,0.137000,0.00038886,443366.600240,443194.192780,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,10892.910000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:28,matpower/case14/2017-01-25,matpower/case14,basic,1,OPTIMAL,2,0.113000,0.02215717,436330.125648,426662.285703,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,15251.710000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:29,matpower/case14/2017-01-26,matpower/case14,basic,1,OPTIMAL,2,0.108000,0.03826229,426971.621958,410634.709979,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,15287.420000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:30,matpower/case14/2017-01-27,matpower/case14,basic,1,OPTIMAL,2,0.127000,0.00665056,376682.242497,374177.094221,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,15403.360000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:31,matpower/case14/2017-01-28,matpower/case14,basic,1,OPTIMAL,2,0.145000,0.00511175,420605.422445,418455.391133,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,10961.010000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:32,matpower/case14/2017-01-29,matpower/case14,basic,1,OPTIMAL,2,0.138000,0.00453472,435185.369986,433211.926813,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,11000.050000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:33,matpower/case14/2017-01-30,matpower/case14,basic,1,OPTIMAL,2,0.131000,0.04016193,417406.303456,400642.462597,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,17035.180000,1.854908,1854.908347,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:34,matpower/case14/2017-01-31,matpower/case14,basic,1,OPTIMAL,2,0.150000,0.01475622,382834.204293,377185.017351,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6454.990000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:35,matpower/case14/2017-02-01,matpower/case14,basic,1,OPTIMAL,2,0.104000,0.00000000,367223.595557,367223.595557,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:36,matpower/case14/2017-02-02,matpower/case14,basic,1,OPTIMAL,2,0.193000,0.01531618,389310.725168,383347.973895,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6456.320000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:37,matpower/case14/2017-02-03,matpower/case14,basic,1,OPTIMAL,2,0.128000,0.01696658,386318.731163,379764.224612,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6487.140000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:38,matpower/case14/2017-02-04,matpower/case14,basic,1,OPTIMAL,2,0.113000,0.01994875,370575.567236,363183.048064,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6462.000000,1.479797,1479.797350,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:39,matpower/case14/2017-02-05,matpower/case14,basic,1,OPTIMAL,2,0.112000,0.00000000,348022.484785,348022.484785,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:40,matpower/case14/2017-02-06,matpower/case14,basic,1,OPTIMAL,2,0.120000,0.00000000,353223.285707,353223.285707,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:41,matpower/case14/2017-02-07,matpower/case14,basic,1,OPTIMAL,2,0.094000,0.00000000,328019.427553,328019.427553,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:42,matpower/case14/2017-02-08,matpower/case14,basic,1,OPTIMAL,2,0.124000,0.00148562,348395.484024,347877.900857,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6488.090000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:43,matpower/case14/2017-02-09,matpower/case14,basic,1,OPTIMAL,2,0.137000,0.00031847,449000.436944,448857.444648,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,11125.980000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:44,matpower/case14/2017-02-10,matpower/case14,basic,1,OPTIMAL,2,0.108000,0.03773110,522515.851385,502800.752632,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,15518.210000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:45,matpower/case14/2017-02-11,matpower/case14,basic,1,OPTIMAL,2,0.137000,0.00167624,423562.258116,422852.266129,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,4611.730000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:46,matpower/case14/2017-02-12,matpower/case14,basic,1,OPTIMAL,2,0.134000,0.00093426,431402.131832,430999.090698,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,8982.040000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:47,matpower/case14/2017-02-13,matpower/case14,basic,1,OPTIMAL,2,0.119000,0.00736761,469095.977428,465639.863050,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,39315.450000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:48,matpower/case14/2017-02-14,matpower/case14,basic,1,OPTIMAL,2,0.122000,0.00238421,464028.015100,462921.672797,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,35235.970000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:49,matpower/case14/2017-02-15,matpower/case14,basic,1,OPTIMAL,2,0.138000,0.00020130,449771.808954,449681.267838,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6430.720000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:50,matpower/case14/2017-02-16,matpower/case14,basic,1,OPTIMAL,2,0.141000,0.00003014,471518.679278,471504.466274,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6430.720000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:51,matpower/case14/2017-02-17,matpower/case14,basic,1,OPTIMAL,2,0.132000,0.00000000,383339.388731,383339.388731,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,11087.110000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:52,matpower/case14/2017-02-18,matpower/case14,basic,1,OPTIMAL,2,0.137000,0.00000000,317536.321101,317536.321101,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:53,matpower/case14/2017-02-19,matpower/case14,basic,1,OPTIMAL,2,0.113000,0.00000000,308077.982485,308077.982485,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:54,matpower/case14/2017-02-20,matpower/case14,basic,1,OPTIMAL,2,0.110000,0.00000000,343297.174765,343297.174765,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:55,matpower/case14/2017-02-21,matpower/case14,basic,1,OPTIMAL,2,0.132000,0.00000000,349473.950840,349473.950840,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:56,matpower/case14/2017-02-22,matpower/case14,basic,1,OPTIMAL,2,0.120000,0.00000000,333599.165161,333599.165161,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:57,matpower/case14/2017-02-23,matpower/case14,basic,1,OPTIMAL,2,0.108000,0.00000000,315252.385082,315252.385082,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:58,matpower/case14/2017-02-24,matpower/case14,basic,1,OPTIMAL,2,0.123000,0.00000000,292411.187726,292411.187726,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:16:59,matpower/case14/2017-02-25,matpower/case14,basic,1,OPTIMAL,2,0.161000,0.00000000,290063.004293,290063.004293,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:00,matpower/case14/2017-02-26,matpower/case14,basic,1,OPTIMAL,2,0.113000,0.00000000,294355.142550,294355.142550,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,4736.780000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:01,matpower/case14/2017-02-27,matpower/case14,basic,1,OPTIMAL,2,0.100000,0.00000000,333662.618098,333662.618098,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:02,matpower/case14/2017-02-28,matpower/case14,basic,1,OPTIMAL,2,0.108000,0.00000000,316978.336505,316978.336505,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:03,matpower/case14/2017-03-01,matpower/case14,basic,1,OPTIMAL,2,0.103000,0.00000000,304284.425935,304284.425935,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:04,matpower/case14/2017-03-02,matpower/case14,basic,1,OPTIMAL,2,0.122000,0.00000000,331626.327677,331626.327677,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:05,matpower/case14/2017-03-03,matpower/case14,basic,1,OPTIMAL,2,0.119000,0.01774258,341563.492377,335503.274140,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,6.368175,6368.175354,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:06,matpower/case14/2017-03-04,matpower/case14,basic,1,OPTIMAL,2,0.116000,0.00344371,332640.109574,331494.592795,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,11438.170000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:07,matpower/case14/2017-03-05,matpower/case14,basic,1,OPTIMAL,2,0.107000,0.00000000,325762.230988,325762.230988,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:08,matpower/case14/2017-03-06,matpower/case14,basic,1,OPTIMAL,2,0.112000,0.00000000,326937.149884,326937.149884,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,5725.530000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:09,matpower/case14/2017-03-07,matpower/case14,basic,1,OPTIMAL,2,0.114000,0.00000000,303337.919535,303337.919535,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:10,matpower/case14/2017-03-08,matpower/case14,basic,1,OPTIMAL,2,0.113000,0.00000000,298112.377066,298112.377066,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:11,matpower/case14/2017-03-09,matpower/case14,basic,1,OPTIMAL,2,0.107000,0.00000000,301179.741157,301179.741157,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:12,matpower/case14/2017-03-10,matpower/case14,basic,1,OPTIMAL,2,0.097000,0.00000000,343283.962470,343283.962470,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:13,matpower/case14/2017-03-11,matpower/case14,basic,1,OPTIMAL,2,0.095000,0.00000000,355498.421201,355498.421201,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:14,matpower/case14/2017-03-12,matpower/case14,basic,1,OPTIMAL,2,0.151000,0.02521283,374429.845292,364989.409414,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,4554.100000,4.624129,4624.129367,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:15,matpower/case14/2017-03-13,matpower/case14,basic,1,OPTIMAL,2,0.137000,0.03673546,390934.801792,376573.632667,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,4565.260000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:16,matpower/case14/2017-03-14,matpower/case14,basic,1,OPTIMAL,2,0.155000,0.04073818,405017.439865,388517.768098,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,4565.260000,8.388159,8388.158735,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:17,matpower/case14/2017-03-15,matpower/case14,basic,1,OPTIMAL,2,0.144000,0.04721535,424233.707672,404203.364799,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,10868.760000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:18,matpower/case14/2017-03-16,matpower/case14,basic,1,OPTIMAL,2,0.144000,0.04969399,421249.393420,400315.831326,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,10870.820000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:19,matpower/case14/2017-03-17,matpower/case14,basic,1,OPTIMAL,2,0.150000,0.00453231,354511.254227,352904.500201,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,4829.390000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:20,matpower/case14/2017-03-18,matpower/case14,basic,1,OPTIMAL,2,0.114000,0.00000000,282431.561032,282431.561032,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:21,matpower/case14/2017-03-19,matpower/case14,basic,1,OPTIMAL,2,0.130000,0.00000000,313263.036846,313263.036846,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:22,matpower/case14/2017-03-20,matpower/case14,basic,1,OPTIMAL,2,0.115000,0.00000000,335221.061021,335221.061021,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:24,matpower/case14/2017-03-21,matpower/case14,basic,1,OPTIMAL,2,0.136000,0.00000000,318115.692808,318115.692808,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:25,matpower/case14/2017-03-22,matpower/case14,basic,1,OPTIMAL,2,0.153000,0.03131744,365154.210688,353718.515814,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,12.074659,12074.659399,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:26,matpower/case14/2017-03-23,matpower/case14,basic,1,OPTIMAL,2,0.180000,0.01005778,334555.318879,331190.435417,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,6.824237,6824.236695,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:27,matpower/case14/2017-03-24,matpower/case14,basic,1,OPTIMAL,2,0.107000,0.00000000,282667.766232,282667.766232,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:28,matpower/case14/2017-03-25,matpower/case14,basic,1,OPTIMAL,2,0.112000,0.00000000,254011.243759,254011.243759,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:29,matpower/case14/2017-03-26,matpower/case14,basic,1,OPTIMAL,2,0.099000,0.00000000,264499.016027,264499.016027,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:30,matpower/case14/2017-03-27,matpower/case14,basic,1,OPTIMAL,2,0.126000,0.00000000,279109.723880,279109.723880,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:31,matpower/case14/2017-03-28,matpower/case14,basic,1,OPTIMAL,2,0.108000,0.00000000,278567.851990,278567.851990,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:32,matpower/case14/2017-03-29,matpower/case14,basic,1,OPTIMAL,2,0.096000,0.00000000,289965.659541,289965.659541,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:33,matpower/case14/2017-03-30,matpower/case14,basic,1,OPTIMAL,2,0.104000,0.00000000,316566.282134,316566.282134,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:34,matpower/case14/2017-03-31,matpower/case14,basic,1,OPTIMAL,2,0.106000,0.00000000,310862.747487,310862.747487,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:35,matpower/case14/2017-04-01,matpower/case14,basic,1,OPTIMAL,2,0.099000,0.00000000,270347.563975,270347.563975,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:36,matpower/case14/2017-04-02,matpower/case14,basic,1,OPTIMAL,2,0.100000,0.00000000,270170.127909,270170.127909,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:37,matpower/case14/2017-04-03,matpower/case14,basic,1,OPTIMAL,2,0.118000,0.00000000,292563.865259,292563.865259,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:38,matpower/case14/2017-04-04,matpower/case14,basic,1,OPTIMAL,2,0.096000,0.00000000,289072.165282,289072.165282,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:39,matpower/case14/2017-04-05,matpower/case14,basic,1,OPTIMAL,2,0.099000,0.00000000,279426.298553,279426.298553,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:40,matpower/case14/2017-04-06,matpower/case14,basic,1,OPTIMAL,2,0.104000,0.00000000,302714.211737,302714.211737,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:41,matpower/case14/2017-04-07,matpower/case14,basic,1,OPTIMAL,2,0.097000,0.00000000,304138.378541,304138.378541,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:42,matpower/case14/2017-04-08,matpower/case14,basic,1,OPTIMAL,2,0.104000,0.00000000,277744.893438,277744.893438,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:43,matpower/case14/2017-04-09,matpower/case14,basic,1,OPTIMAL,2,0.093000,0.00000000,273172.643767,273172.643767,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:44,matpower/case14/2017-04-10,matpower/case14,basic,1,OPTIMAL,2,0.105000,0.00000000,285914.653508,285914.653508,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:45,matpower/case14/2017-04-11,matpower/case14,basic,1,OPTIMAL,2,0.095000,0.00000000,291712.034215,291712.034215,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:46,matpower/case14/2017-04-12,matpower/case14,basic,1,OPTIMAL,2,0.099000,0.00000000,285368.430550,285368.430550,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:47,matpower/case14/2017-04-13,matpower/case14,basic,1,OPTIMAL,2,0.107000,0.00000000,277857.786384,277857.786384,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:48,matpower/case14/2017-04-14,matpower/case14,basic,1,OPTIMAL,2,0.096000,0.00000000,268228.788862,268228.788862,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:49,matpower/case14/2017-04-15,matpower/case14,basic,1,OPTIMAL,2,0.098000,0.00000000,242422.243530,242422.243530,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:50,matpower/case14/2017-04-16,matpower/case14,basic,1,OPTIMAL,2,0.095000,0.00000000,263519.951108,263519.951108,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:51,matpower/case14/2017-04-17,matpower/case14,basic,1,OPTIMAL,2,0.100000,0.00000000,262040.055975,262040.055975,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:52,matpower/case14/2017-04-18,matpower/case14,basic,1,OPTIMAL,2,0.101000,0.00000000,254657.214293,254657.214293,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:53,matpower/case14/2017-04-19,matpower/case14,basic,1,OPTIMAL,2,0.098000,0.00000000,258747.471343,258747.471343,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:54,matpower/case14/2017-04-20,matpower/case14,basic,1,OPTIMAL,2,0.096000,0.00000000,273782.730141,273782.730141,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:55,matpower/case14/2017-04-21,matpower/case14,basic,1,OPTIMAL,2,0.120000,0.00000000,266111.576796,266111.576796,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:56,matpower/case14/2017-04-22,matpower/case14,basic,1,OPTIMAL,2,0.095000,0.00000000,243162.081491,243162.081491,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:57,matpower/case14/2017-04-23,matpower/case14,basic,1,OPTIMAL,2,0.103000,0.00000000,259845.006040,259845.006040,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:58,matpower/case14/2017-04-24,matpower/case14,basic,1,OPTIMAL,2,0.097000,0.00000000,278496.301199,278496.301199,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:17:59,matpower/case14/2017-04-25,matpower/case14,basic,1,OPTIMAL,2,0.101000,0.00000000,287403.068522,287403.068522,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:00,matpower/case14/2017-04-26,matpower/case14,basic,1,OPTIMAL,2,0.097000,0.00000000,269648.558626,269648.558626,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:01,matpower/case14/2017-04-27,matpower/case14,basic,1,OPTIMAL,2,0.159000,0.00000000,287503.815677,287503.815677,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:02,matpower/case14/2017-04-28,matpower/case14,basic,1,OPTIMAL,2,0.158000,0.00000000,284673.166617,284673.166617,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:03,matpower/case14/2017-04-29,matpower/case14,basic,1,OPTIMAL,2,0.105000,0.00000000,276469.672186,276469.672186,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:04,matpower/case14/2017-04-30,matpower/case14,basic,1,OPTIMAL,2,0.101000,0.00000000,256686.239765,256686.239765,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:05,matpower/case14/2017-05-01,matpower/case14,basic,1,OPTIMAL,2,0.108000,0.00000000,281062.874988,281062.874988,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:06,matpower/case14/2017-05-02,matpower/case14,basic,1,OPTIMAL,2,0.102000,0.00000000,270484.559296,270484.559296,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:07,matpower/case14/2017-05-03,matpower/case14,basic,1,OPTIMAL,2,0.136000,0.00085118,255416.443273,255199.037564,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,1.0000,4851.260000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:08,matpower/case14/2017-05-04,matpower/case14,basic,1,OPTIMAL,2,0.107000,0.00000000,278064.735859,278064.735859,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:09,matpower/case14/2017-05-05,matpower/case14,basic,1,OPTIMAL,2,0.105000,0.00000000,269069.695548,269069.695548,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:09,matpower/case14/2017-05-06,matpower/case14,basic,1,OPTIMAL,2,0.120000,0.00000000,245786.504783,245786.504783,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:10,matpower/case14/2017-05-07,matpower/case14,basic,1,OPTIMAL,2,0.107000,0.00000000,261118.555015,261118.555015,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:11,matpower/case14/2017-05-08,matpower/case14,basic,1,OPTIMAL,2,0.094000,0.00000000,269179.565069,269179.565069,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:12,matpower/case14/2017-05-09,matpower/case14,basic,1,OPTIMAL,2,0.094000,0.00000000,261776.428866,261776.428866,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:13,matpower/case14/2017-05-10,matpower/case14,basic,1,OPTIMAL,2,0.204000,0.00000000,321950.875487,321950.875487,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:14,matpower/case14/2017-05-11,matpower/case14,basic,1,OPTIMAL,2,0.145000,0.00000000,335439.920413,335439.920413,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:15,matpower/case14/2017-05-12,matpower/case14,basic,1,OPTIMAL,2,0.155000,0.00000000,326496.450830,326496.450830,1,5076,540,540,4536,28836,97677,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:16,matpower/case14/2017-05-13,matpower/case14,basic,1,OPTIMAL,2,0.143000,0.00000000,303937.945831,303937.945831,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:18,matpower/case14/2017-05-14,matpower/case14,basic,1,OPTIMAL,2,0.160000,0.00000000,310737.596730,310737.596730,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:19,matpower/case14/2017-05-15,matpower/case14,basic,1,OPTIMAL,2,0.149000,0.00643117,328111.892486,326001.750660,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:20,matpower/case14/2017-05-16,matpower/case14,basic,1,OPTIMAL,2,0.131000,0.02948844,362358.633216,351673.241503,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,2.0000,8622.720000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:21,matpower/case14/2017-05-17,matpower/case14,basic,1,OPTIMAL,2,0.165000,0.03487207,342337.397109,330399.383754,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,2.0000,8766.820000,2.363714,2363.714356,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:22,matpower/case14/2017-05-18,matpower/case14,basic,1,OPTIMAL,2,0.138000,0.04777623,384243.650888,365885.938653,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,2.0000,8687.590000,2.423631,2423.631363,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:23,matpower/case14/2017-05-19,matpower/case14,basic,1,OPTIMAL,2,0.179000,0.02420762,334469.033738,326372.334387,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:24,matpower/case14/2017-05-20,matpower/case14,basic,1,OPTIMAL,2,0.100000,0.00000000,239637.083679,239637.083679,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:25,matpower/case14/2017-05-21,matpower/case14,basic,1,OPTIMAL,2,0.093000,0.00000000,234782.674697,234782.674697,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:26,matpower/case14/2017-05-22,matpower/case14,basic,1,OPTIMAL,2,0.090000,0.00000000,262321.390132,262321.390132,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:27,matpower/case14/2017-05-23,matpower/case14,basic,1,OPTIMAL,2,0.105000,0.00000000,265030.912828,265030.912828,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:28,matpower/case14/2017-05-24,matpower/case14,basic,1,OPTIMAL,2,0.111000,0.00000000,277057.787038,277057.787038,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:29,matpower/case14/2017-05-25,matpower/case14,basic,1,OPTIMAL,2,0.108000,0.00000000,272730.859803,272730.859803,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:30,matpower/case14/2017-05-26,matpower/case14,basic,1,OPTIMAL,2,0.106000,0.00000000,263459.443897,263459.443897,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:31,matpower/case14/2017-05-27,matpower/case14,basic,1,OPTIMAL,2,0.102000,0.00000000,243050.278174,243050.278174,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:32,matpower/case14/2017-05-28,matpower/case14,basic,1,OPTIMAL,2,0.107000,0.00000000,235093.882205,235093.882205,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:33,matpower/case14/2017-05-29,matpower/case14,basic,1,OPTIMAL,2,0.091000,0.00000000,243989.845942,243989.845942,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:34,matpower/case14/2017-05-30,matpower/case14,basic,1,OPTIMAL,2,0.095000,0.00000000,272578.579048,272578.579048,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:35,matpower/case14/2017-05-31,matpower/case14,basic,1,OPTIMAL,2,0.091000,0.00000000,287924.833155,287924.833155,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:36,matpower/case14/2017-06-01,matpower/case14,basic,1,OPTIMAL,2,0.104000,0.00000000,288433.482419,288433.482419,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:37,matpower/case14/2017-06-02,matpower/case14,basic,1,OPTIMAL,2,0.100000,0.00000000,285914.942503,285914.942503,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:38,matpower/case14/2017-06-03,matpower/case14,basic,1,OPTIMAL,2,0.121000,0.00000000,262612.996848,262612.996848,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:39,matpower/case14/2017-06-04,matpower/case14,basic,1,OPTIMAL,2,0.092000,0.00000000,287104.480310,287104.480310,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:40,matpower/case14/2017-06-05,matpower/case14,basic,1,OPTIMAL,2,0.094000,0.00000000,304385.872458,304385.872458,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:41,matpower/case14/2017-06-06,matpower/case14,basic,1,OPTIMAL,2,0.101000,0.00000000,289878.493278,289878.493278,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:42,matpower/case14/2017-06-07,matpower/case14,basic,1,OPTIMAL,2,0.094000,0.00000000,274081.044481,274081.044481,1,5076,540,540,4536,28836,100086,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:43,matpower/case14/2017-06-08,matpower/case14,basic,1,OPTIMAL,2,0.100000,0.00000000,282698.984883,282698.984883,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:44,matpower/case14/2017-06-09,matpower/case14,basic,1,OPTIMAL,2,0.099000,0.00000000,296047.512649,296047.512649,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:45,matpower/case14/2017-06-10,matpower/case14,basic,1,OPTIMAL,2,0.101000,0.00000000,309291.597369,309291.597369,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:46,matpower/case14/2017-06-11,matpower/case14,basic,1,OPTIMAL,2,0.132000,0.03785943,378908.876261,364563.601884,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,12949.730000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:47,matpower/case14/2017-06-12,matpower/case14,basic,1,OPTIMAL,2,0.156000,0.04113587,458763.216089,439891.590667,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,4.0000,18990.690000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:48,matpower/case14/2017-06-13,matpower/case14,basic,1,OPTIMAL,2,0.161000,0.01780363,443682.928320,435783.762061,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,10423.440000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:49,matpower/case14/2017-06-14,matpower/case14,basic,1,OPTIMAL,2,0.162000,0.03257439,370996.953828,358911.955623,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,8676.190000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:50,matpower/case14/2017-06-15,matpower/case14,basic,1,OPTIMAL,2,0.117000,0.00016380,334825.270915,334770.425207,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:51,matpower/case14/2017-06-16,matpower/case14,basic,1,OPTIMAL,2,0.113000,0.02635251,352834.619542,343536.542982,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,9.504437,9504.436786,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:52,matpower/case14/2017-06-17,matpower/case14,basic,1,OPTIMAL,2,0.123000,0.04002707,358449.101971,344101.436420,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,14.900218,14900.218205,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:53,matpower/case14/2017-06-18,matpower/case14,basic,1,OPTIMAL,2,0.144000,0.02050510,402056.198062,393811.995875,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6080.690000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:54,matpower/case14/2017-06-19,matpower/case14,basic,1,OPTIMAL,2,0.132000,0.04324150,426257.930136,407825.899143,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,10512.560000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:55,matpower/case14/2017-06-20,matpower/case14,basic,1,OPTIMAL,2,0.131000,0.03539773,408970.812352,394494.173895,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6082.380000,3.098737,3098.737384,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:56,matpower/case14/2017-06-21,matpower/case14,basic,1,OPTIMAL,2,0.148000,0.02690558,398451.156190,387730.596450,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6082.380000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:57,matpower/case14/2017-06-22,matpower/case14,basic,1,OPTIMAL,2,0.133000,0.02797144,449477.514172,436904.982707,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,10504.250000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:58,matpower/case14/2017-06-23,matpower/case14,basic,1,OPTIMAL,2,0.127000,0.03166814,467045.100931,452254.651577,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,13095.730000,3.277902,3277.901582,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:18:59,matpower/case14/2017-06-24,matpower/case14,basic,1,OPTIMAL,2,0.128000,0.02645476,404817.412083,394108.064903,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,8692.650000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:00,matpower/case14/2017-06-25,matpower/case14,basic,1,OPTIMAL,2,0.121000,0.01435871,369146.684812,363846.213159,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,4400.940000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:01,matpower/case14/2017-06-26,matpower/case14,basic,1,OPTIMAL,2,0.139000,0.02585386,361221.453013,351882.485622,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,4403.270000,4.138042,4138.041582,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:02,matpower/case14/2017-06-27,matpower/case14,basic,1,OPTIMAL,2,0.127000,0.00000000,413900.298043,413900.298043,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:03,matpower/case14/2017-06-28,matpower/case14,basic,1,OPTIMAL,2,0.143000,0.02405101,439900.272178,429320.225382,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,8690.100000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:04,matpower/case14/2017-06-29,matpower/case14,basic,1,OPTIMAL,2,0.146000,0.00739293,512052.092339,508266.527243,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6126.950000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:05,matpower/case14/2017-06-30,matpower/case14,basic,1,OPTIMAL,2,0.130000,0.02222481,486551.190030,475737.683475,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,10585.170000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:06,matpower/case14/2017-07-01,matpower/case14,basic,1,OPTIMAL,2,0.139000,0.02139431,450836.848989,441191.507125,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6118.610000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:07,matpower/case14/2017-07-02,matpower/case14,basic,1,OPTIMAL,2,0.121000,0.04003554,446578.630262,428699.611847,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,10544.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:08,matpower/case14/2017-07-03,matpower/case14,basic,1,OPTIMAL,2,0.139000,0.02052990,413428.517404,404940.872450,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6087.530000,5.631482,5631.481582,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:09,matpower/case14/2017-07-04,matpower/case14,basic,1,OPTIMAL,2,0.122000,0.03160848,404423.273756,391640.069921,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6087.530000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:10,matpower/case14/2017-07-05,matpower/case14,basic,1,OPTIMAL,2,0.128000,0.01352454,403747.021722,398286.530541,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6087.530000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:11,matpower/case14/2017-07-06,matpower/case14,basic,1,OPTIMAL,2,0.117000,0.04968628,416167.379182,395489.570768,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,13066.590000,2.168639,2168.639174,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:12,matpower/case14/2017-07-07,matpower/case14,basic,1,OPTIMAL,2,0.133000,0.03213399,452596.049373,438052.330987,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6084.830000,4.782318,4782.317591,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:13,matpower/case14/2017-07-08,matpower/case14,basic,1,OPTIMAL,2,0.142000,0.03629865,438037.722214,422137.542232,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,10352.170000,5.262068,5262.067591,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:14,matpower/case14/2017-07-09,matpower/case14,basic,1,OPTIMAL,2,0.121000,0.03251682,424304.259681,410507.233916,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,12943.900000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:15,matpower/case14/2017-07-10,matpower/case14,basic,1,OPTIMAL,2,0.154000,0.01140505,452126.506086,446969.979639,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,8552.440000,1.773527,1773.526593,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:16,matpower/case14/2017-07-11,matpower/case14,basic,1,OPTIMAL,2,0.154000,0.00492603,498602.710857,496146.580895,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,10394.490000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:17,matpower/case14/2017-07-12,matpower/case14,basic,1,OPTIMAL,2,0.188000,0.01519322,534373.902075,526255.040716,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,14670.710000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:18,matpower/case14/2017-07-13,matpower/case14,basic,1,OPTIMAL,2,0.170000,0.03968920,573829.566895,551054.728781,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,4.0000,29411.050000,2.716842,2716.841582,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:19,matpower/case14/2017-07-14,matpower/case14,basic,1,OPTIMAL,2,0.133000,0.03246052,492103.593185,476129.652897,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,4.0000,17337.900000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:20,matpower/case14/2017-07-15,matpower/case14,basic,1,OPTIMAL,2,0.134000,0.04835422,452205.280057,430339.248147,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,19007.450000,1.644197,1644.197178,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:21,matpower/case14/2017-07-16,matpower/case14,basic,1,OPTIMAL,2,0.145000,0.01055738,449414.138687,444669.504684,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6005.030000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:22,matpower/case14/2017-07-17,matpower/case14,basic,1,OPTIMAL,2,0.140000,0.02841736,472996.560231,459555.248844,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,13030.780000,1.609469,1609.468589,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:23,matpower/case14/2017-07-18,matpower/case14,basic,1,OPTIMAL,2,0.134000,0.01607056,486724.324225,478902.392571,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,13030.780000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:24,matpower/case14/2017-07-19,matpower/case14,basic,1,OPTIMAL,2,0.154000,0.04069921,559034.847041,536282.569098,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,4.0000,25104.060000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:25,matpower/case14/2017-07-20,matpower/case14,basic,1,OPTIMAL,2,0.146000,0.03670683,559246.455628,538718.289532,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,18972.630000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:26,matpower/case14/2017-07-21,matpower/case14,basic,1,OPTIMAL,2,0.169000,0.01165564,526072.188696,519940.480564,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,10380.230000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:27,matpower/case14/2017-07-22,matpower/case14,basic,1,OPTIMAL,2,0.145000,0.01683795,474276.555065,466290.707908,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,10377.130000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:28,matpower/case14/2017-07-23,matpower/case14,basic,1,OPTIMAL,2,0.162000,0.00793457,467635.641891,463925.155894,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,5994.240000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:30,matpower/case14/2017-07-24,matpower/case14,basic,1,OPTIMAL,2,0.149000,0.00986305,479766.471300,475034.511112,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,4276.570000,1.428789,1428.788589,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:31,matpower/case14/2017-07-25,matpower/case14,basic,1,OPTIMAL,2,0.113000,0.03939464,437899.961233,420649.051216,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,8660.890000,5.916797,5916.797178,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:32,matpower/case14/2017-07-26,matpower/case14,basic,1,OPTIMAL,2,0.161000,0.04838884,403200.016609,383689.636970,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,14736.050000,2.288463,2288.463187,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:33,matpower/case14/2017-07-27,matpower/case14,basic,1,OPTIMAL,2,0.147000,0.02227872,435505.856684,425803.344359,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6007.470000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:34,matpower/case14/2017-07-28,matpower/case14,basic,1,OPTIMAL,2,0.147000,0.01158088,423876.778752,418967.910732,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6003.430000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:35,matpower/case14/2017-07-29,matpower/case14,basic,1,OPTIMAL,2,0.116000,0.00047782,344320.376383,344155.852987,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:36,matpower/case14/2017-07-30,matpower/case14,basic,1,OPTIMAL,2,0.106000,0.01316840,340155.597479,335676.294142,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,4.751427,4751.426593,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:37,matpower/case14/2017-07-31,matpower/case14,basic,1,OPTIMAL,2,0.154000,0.03131751,412313.706445,399401.066550,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,10260.580000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:38,matpower/case14/2017-08-01,matpower/case14,basic,1,OPTIMAL,2,0.181000,0.04247071,475853.510599,455643.672787,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,20617.820000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:39,matpower/case14/2017-08-02,matpower/case14,basic,1,OPTIMAL,2,0.196000,0.03071535,490046.031356,474994.098410,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,16366.540000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:40,matpower/case14/2017-08-03,matpower/case14,basic,1,OPTIMAL,2,0.163000,0.03402222,482339.193055,465928.940972,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,16642.070000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:41,matpower/case14/2017-08-04,matpower/case14,basic,1,OPTIMAL,2,0.147000,0.04400253,452288.851264,432386.998988,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,19430.380000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:42,matpower/case14/2017-08-05,matpower/case14,basic,1,OPTIMAL,2,0.105000,0.02349572,357737.851367,349332.542709,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,4494.790000,3.849863,3849.863165,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:43,matpower/case14/2017-08-06,matpower/case14,basic,1,OPTIMAL,2,0.122000,0.01116707,349062.550943,345164.546177,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,4.158491,4158.491169,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:44,matpower/case14/2017-08-07,matpower/case14,basic,1,OPTIMAL,2,0.163000,0.04175668,384737.864270,368672.490297,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,10526.060000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:45,matpower/case14/2017-08-08,matpower/case14,basic,1,OPTIMAL,2,0.131000,0.01902827,366148.459503,359181.288505,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,4415.670000,0.581480,581.479587,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:46,matpower/case14/2017-08-09,matpower/case14,basic,1,OPTIMAL,2,0.130000,0.02466540,374031.726927,364806.083632,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,8972.360000,0.142143,142.142580,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:47,matpower/case14/2017-08-10,matpower/case14,basic,1,OPTIMAL,2,0.242000,0.02579611,410367.529878,399781.642811,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,10592.060000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:48,matpower/case14/2017-08-11,matpower/case14,basic,1,OPTIMAL,2,0.149000,0.03298642,398715.333103,385563.143005,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,4484.070000,8.258966,8258.965767,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:49,matpower/case14/2017-08-12,matpower/case14,basic,1,OPTIMAL,2,0.090000,0.00000000,356296.475617,356296.475617,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:50,matpower/case14/2017-08-13,matpower/case14,basic,1,OPTIMAL,2,0.134000,0.02277120,408475.156153,399173.687772,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,8883.950000,0.370387,370.386593,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:51,matpower/case14/2017-08-14,matpower/case14,basic,1,OPTIMAL,2,0.149000,0.01701434,478747.144350,470601.577792,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6074.710000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:52,matpower/case14/2017-08-15,matpower/case14,basic,1,OPTIMAL,2,0.149000,0.02705656,516551.086807,502574.991550,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,14947.550000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:53,matpower/case14/2017-08-16,matpower/case14,basic,1,OPTIMAL,2,0.147000,0.02900628,511167.994797,496340.913992,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,14947.550000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:55,matpower/case14/2017-08-17,matpower/case14,basic,1,OPTIMAL,2,0.152000,0.02708260,524524.448208,510318.964388,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,14950.310000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:56,matpower/case14/2017-08-18,matpower/case14,basic,1,OPTIMAL,2,0.159000,0.04106098,543857.782166,521526.450959,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,25483.120000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:57,matpower/case14/2017-08-19,matpower/case14,basic,1,OPTIMAL,2,0.143000,0.02247224,518272.746584,506625.996469,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,14944.830000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:58,matpower/case14/2017-08-20,matpower/case14,basic,1,OPTIMAL,2,0.153000,0.01388385,452271.595831,445992.326275,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,6079.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:19:59,matpower/case14/2017-08-21,matpower/case14,basic,1,OPTIMAL,2,0.137000,0.00735526,483265.045202,479710.506099,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,10528.460000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:00,matpower/case14/2017-08-22,matpower/case14,basic,1,OPTIMAL,2,0.174000,0.02948780,529009.141047,513409.822874,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,21016.460000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:01,matpower/case14/2017-08-23,matpower/case14,basic,1,OPTIMAL,2,0.138000,0.02972249,448316.982546,434991.885918,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,10528.460000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:02,matpower/case14/2017-08-24,matpower/case14,basic,1,OPTIMAL,2,0.154000,0.02738981,393418.834205,382643.168864,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,8910.730000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:03,matpower/case14/2017-08-25,matpower/case14,basic,1,OPTIMAL,2,0.115000,0.04401304,371118.732693,354784.668561,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,4490.520000,11.983495,11983.495114,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:04,matpower/case14/2017-08-26,matpower/case14,basic,1,OPTIMAL,2,0.132000,0.00000000,320817.698752,320817.698752,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:05,matpower/case14/2017-08-27,matpower/case14,basic,1,OPTIMAL,2,0.128000,0.00000000,347759.673015,347759.673015,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:06,matpower/case14/2017-08-28,matpower/case14,basic,1,OPTIMAL,2,0.110000,0.00000000,369750.420089,369750.420089,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:07,matpower/case14/2017-08-29,matpower/case14,basic,1,OPTIMAL,2,0.145000,0.00000000,361754.608952,361754.608952,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:08,matpower/case14/2017-08-30,matpower/case14,basic,1,OPTIMAL,2,0.099000,0.00000000,370434.000739,370434.000739,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:09,matpower/case14/2017-08-31,matpower/case14,basic,1,OPTIMAL,2,0.124000,0.02239595,414527.477974,405243.742022,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,6601.560000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:10,matpower/case14/2017-09-01,matpower/case14,basic,1,OPTIMAL,2,0.103000,0.00000000,340422.810527,340422.810527,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:11,matpower/case14/2017-09-02,matpower/case14,basic,1,OPTIMAL,2,0.127000,0.00000000,292867.689849,292867.689849,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:12,matpower/case14/2017-09-03,matpower/case14,basic,1,OPTIMAL,2,0.110000,0.00124764,303812.364173,303433.316113,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:13,matpower/case14/2017-09-04,matpower/case14,basic,1,OPTIMAL,2,0.108000,0.01899728,342777.148439,336265.313631,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,4328.600000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:14,matpower/case14/2017-09-05,matpower/case14,basic,1,OPTIMAL,2,0.127000,0.00832967,398435.159321,395116.325558,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,1551.380000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:15,matpower/case14/2017-09-06,matpower/case14,basic,1,OPTIMAL,2,0.130000,0.00000000,336729.235952,336729.235952,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:16,matpower/case14/2017-09-07,matpower/case14,basic,1,OPTIMAL,2,0.111000,0.00010517,313837.244509,313804.236721,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:17,matpower/case14/2017-09-08,matpower/case14,basic,1,OPTIMAL,2,0.123000,0.00000000,316150.362566,316150.362566,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:18,matpower/case14/2017-09-09,matpower/case14,basic,1,OPTIMAL,2,0.124000,0.00000000,309428.910441,309428.910441,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:19,matpower/case14/2017-09-10,matpower/case14,basic,1,OPTIMAL,2,0.093000,0.00000000,297017.762017,297017.762017,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:20,matpower/case14/2017-09-11,matpower/case14,basic,1,OPTIMAL,2,0.121000,0.00010221,322501.551162,322468.586764,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:21,matpower/case14/2017-09-12,matpower/case14,basic,1,OPTIMAL,2,0.101000,0.00000000,327360.561597,327360.561597,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:22,matpower/case14/2017-09-13,matpower/case14,basic,1,OPTIMAL,2,0.110000,0.00534761,420295.691672,418048.115945,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,1558.890000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:23,matpower/case14/2017-09-14,matpower/case14,basic,1,OPTIMAL,2,0.170000,0.00000000,430105.996556,430105.996556,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.689025,689.024564,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:24,matpower/case14/2017-09-15,matpower/case14,basic,1,OPTIMAL,2,0.118000,0.01279762,442183.862683,436524.960181,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,3130.840000,1.988024,1988.023566,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:25,matpower/case14/2017-09-16,matpower/case14,basic,1,OPTIMAL,2,0.124000,0.00007856,398964.690259,398933.347743,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:26,matpower/case14/2017-09-17,matpower/case14,basic,1,OPTIMAL,2,0.121000,0.00000000,401424.921478,401424.921478,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:27,matpower/case14/2017-09-18,matpower/case14,basic,1,OPTIMAL,2,0.103000,0.00000000,414484.466429,414484.466429,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:28,matpower/case14/2017-09-19,matpower/case14,basic,1,OPTIMAL,2,0.140000,0.00000000,414942.293628,414942.293628,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,1557.220000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:29,matpower/case14/2017-09-20,matpower/case14,basic,1,OPTIMAL,2,0.133000,0.04689768,469203.971918,447199.396487,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,4.0000,8853.940000,2.306571,2306.570550,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:30,matpower/case14/2017-09-21,matpower/case14,basic,1,OPTIMAL,2,0.136000,0.00657014,482089.176466,478921.781927,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,1565.610000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:31,matpower/case14/2017-09-22,matpower/case14,basic,1,OPTIMAL,2,0.178000,0.02855767,480734.692247,467006.030541,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,3518.190000,6.089455,6089.455435,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:32,matpower/case14/2017-09-23,matpower/case14,basic,1,OPTIMAL,2,0.147000,0.03790632,479644.287300,461462.736615,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,5425.990000,5.735697,5735.697431,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:33,matpower/case14/2017-09-24,matpower/case14,basic,1,OPTIMAL,2,0.143000,0.04012311,432441.213154,415090.328282,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,6480.130000,6.127501,6127.501444,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:35,matpower/case14/2017-09-25,matpower/case14,basic,1,OPTIMAL,2,0.121000,0.03412910,427929.948562,413325.083687,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,6123.170000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:36,matpower/case14/2017-09-26,matpower/case14,basic,1,OPTIMAL,2,0.168000,0.04543016,414527.937690,395695.865167,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,5384.450000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:37,matpower/case14/2017-09-27,matpower/case14,basic,1,OPTIMAL,2,0.138000,0.03620477,425827.189400,410410.214939,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,3826.000000,5.036292,5036.292442,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:38,matpower/case14/2017-09-28,matpower/case14,basic,1,OPTIMAL,2,0.118000,0.02565290,369771.668250,360285.950906,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,2760.900000,3.104441,3104.441444,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:39,matpower/case14/2017-09-29,matpower/case14,basic,1,OPTIMAL,2,0.139000,0.00000000,317909.087228,317909.087228,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:40,matpower/case14/2017-09-30,matpower/case14,basic,1,OPTIMAL,2,0.136000,0.00000000,261726.601403,261726.601403,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:41,matpower/case14/2017-10-01,matpower/case14,basic,1,OPTIMAL,2,0.111000,0.00573266,270071.879450,268523.649021,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:42,matpower/case14/2017-10-02,matpower/case14,basic,1,OPTIMAL,2,0.110000,0.00284163,296287.113109,295445.174280,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:43,matpower/case14/2017-10-03,matpower/case14,basic,1,OPTIMAL,2,0.121000,0.00308510,297464.497391,296546.789487,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:44,matpower/case14/2017-10-04,matpower/case14,basic,1,OPTIMAL,2,0.127000,0.00132308,310338.079911,309927.476543,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:45,matpower/case14/2017-10-05,matpower/case14,basic,1,OPTIMAL,2,0.126000,0.00006623,343220.066703,343197.334633,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:46,matpower/case14/2017-10-06,matpower/case14,basic,1,OPTIMAL,2,0.102000,0.00016857,344485.918641,344427.849367,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:47,matpower/case14/2017-10-07,matpower/case14,basic,1,OPTIMAL,2,0.140000,0.00000000,342775.958248,342775.958248,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:48,matpower/case14/2017-10-08,matpower/case14,basic,1,OPTIMAL,2,0.125000,0.00000000,364261.817030,364261.817030,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:49,matpower/case14/2017-10-09,matpower/case14,basic,1,OPTIMAL,2,0.124000,0.00338789,370755.946090,369499.864182,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,1.280508,1280.507557,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:50,matpower/case14/2017-10-10,matpower/case14,basic,1,OPTIMAL,2,0.151000,0.01575417,378318.349711,372358.259560,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,1541.930000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:51,matpower/case14/2017-10-11,matpower/case14,basic,1,OPTIMAL,2,0.137000,0.00071747,407587.535948,407295.103291,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:52,matpower/case14/2017-10-12,matpower/case14,basic,1,OPTIMAL,2,0.142000,0.00000000,405601.920905,405601.920905,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:53,matpower/case14/2017-10-13,matpower/case14,basic,1,OPTIMAL,2,0.143000,0.00000000,388073.049699,388073.049699,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:54,matpower/case14/2017-10-14,matpower/case14,basic,1,OPTIMAL,2,0.128000,0.00053755,306587.615085,306422.809782,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:55,matpower/case14/2017-10-15,matpower/case14,basic,1,OPTIMAL,2,0.129000,0.00000000,333403.693311,333403.693311,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:56,matpower/case14/2017-10-16,matpower/case14,basic,1,OPTIMAL,2,0.125000,0.00000000,314423.960481,314423.960481,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:57,matpower/case14/2017-10-17,matpower/case14,basic,1,OPTIMAL,2,0.126000,0.00000000,311939.996644,311939.996644,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:58,matpower/case14/2017-10-18,matpower/case14,basic,1,OPTIMAL,2,0.137000,0.00000000,305180.795030,305180.795030,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:20:59,matpower/case14/2017-10-19,matpower/case14,basic,1,OPTIMAL,2,0.132000,0.00000000,331839.169095,331839.169095,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:00,matpower/case14/2017-10-20,matpower/case14,basic,1,OPTIMAL,2,0.138000,0.00000000,300785.664020,300785.664020,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:01,matpower/case14/2017-10-21,matpower/case14,basic,1,OPTIMAL,2,0.126000,0.00000000,273563.194615,273563.194615,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:02,matpower/case14/2017-10-22,matpower/case14,basic,1,OPTIMAL,2,0.118000,0.00538851,299413.969551,297800.575609,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:03,matpower/case14/2017-10-23,matpower/case14,basic,1,OPTIMAL,2,0.114000,0.00247433,356710.384242,355827.763411,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:04,matpower/case14/2017-10-24,matpower/case14,basic,1,OPTIMAL,2,0.121000,0.00000000,363491.946054,363491.946054,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:05,matpower/case14/2017-10-25,matpower/case14,basic,1,OPTIMAL,2,0.106000,0.00657417,355732.646627,353393.999697,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:06,matpower/case14/2017-10-26,matpower/case14,basic,1,OPTIMAL,2,0.164000,0.00485720,352056.517381,350346.508469,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:07,matpower/case14/2017-10-27,matpower/case14,basic,1,OPTIMAL,2,0.111000,0.00914870,342416.411513,339283.745448,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:08,matpower/case14/2017-10-28,matpower/case14,basic,1,OPTIMAL,2,0.110000,0.01310516,312190.534349,308099.228135,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:09,matpower/case14/2017-10-29,matpower/case14,basic,1,OPTIMAL,2,0.121000,0.00791352,331259.800171,328638.369163,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:10,matpower/case14/2017-10-30,matpower/case14,basic,1,OPTIMAL,2,0.125000,0.00000000,337872.014125,337872.014125,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:11,matpower/case14/2017-10-31,matpower/case14,basic,1,OPTIMAL,2,0.137000,0.00000000,346952.123131,346952.123131,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:12,matpower/case14/2017-11-01,matpower/case14,basic,1,OPTIMAL,2,0.130000,0.00521505,333510.240712,331770.968697,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:13,matpower/case14/2017-11-02,matpower/case14,basic,1,OPTIMAL,2,0.131000,0.00520845,343627.398024,341837.632328,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:14,matpower/case14/2017-11-03,matpower/case14,basic,1,OPTIMAL,2,0.117000,0.00644415,323348.047886,321264.344207,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:15,matpower/case14/2017-11-04,matpower/case14,basic,1,OPTIMAL,2,0.142000,0.00000000,301988.129915,301988.129915,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:16,matpower/case14/2017-11-05,matpower/case14,basic,1,OPTIMAL,2,0.112000,0.00650631,312350.822524,310318.572694,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:17,matpower/case14/2017-11-06,matpower/case14,basic,1,OPTIMAL,2,0.113000,0.00378742,328910.996531,327665.272193,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:18,matpower/case14/2017-11-07,matpower/case14,basic,1,OPTIMAL,2,0.152000,0.00000000,355732.926898,355732.926898,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:19,matpower/case14/2017-11-08,matpower/case14,basic,1,OPTIMAL,2,0.148000,0.01161835,380233.199170,375815.515832,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,0.0000,0.000000,4.617467,4617.467315,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:20,matpower/case14/2017-11-09,matpower/case14,basic,1,OPTIMAL,2,0.115000,0.01824420,400249.167984,392946.940116,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,1.0000,2816.740000,3.628941,3628.941306,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:21,matpower/case14/2017-11-10,matpower/case14,basic,1,OPTIMAL,2,0.118000,0.01379518,432372.330954,426407.676375,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,1.0000,2822.410000,0.196506,196.506317,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:22,matpower/case14/2017-11-11,matpower/case14,basic,1,OPTIMAL,2,0.114000,0.00259107,402763.799946,401720.210291,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:23,matpower/case14/2017-11-12,matpower/case14,basic,1,OPTIMAL,2,0.138000,0.00567047,382132.321360,379965.453367,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,0.0000,0.000000,2.021037,2021.036625,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:24,matpower/case14/2017-11-13,matpower/case14,basic,1,OPTIMAL,2,0.125000,0.00986957,413863.932803,409779.273355,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,1.0000,1573.890000,1.667279,1667.278620,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:25,matpower/case14/2017-11-14,matpower/case14,basic,1,OPTIMAL,2,0.120000,0.01809556,417335.124007,409783.209697,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,1.0000,1573.890000,5.466661,5466.661306,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:26,matpower/case14/2017-11-15,matpower/case14,basic,1,OPTIMAL,2,0.124000,0.00747522,392664.303994,389729.051900,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,1.0000,1984.930000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:27,matpower/case14/2017-11-16,matpower/case14,basic,1,OPTIMAL,2,0.126000,0.01019902,404529.925549,400404.117573,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,1.0000,2276.290000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:28,matpower/case14/2017-11-17,matpower/case14,basic,1,OPTIMAL,2,0.117000,0.00946819,404249.116556,400421.610054,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,1.0000,2030.410000,0.290169,290.169310,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:29,matpower/case14/2017-11-18,matpower/case14,basic,1,OPTIMAL,2,0.113000,0.00768144,350190.170668,347500.206824,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:30,matpower/case14/2017-11-19,matpower/case14,basic,1,OPTIMAL,2,0.112000,0.01624847,376316.220966,370201.656765,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,1.0000,2822.010000,2.344699,2344.699310,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:31,matpower/case14/2017-11-20,matpower/case14,basic,1,OPTIMAL,2,0.125000,0.02111202,443792.612203,434423.255589,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,2.0000,3925.550000,2.698457,2698.457315,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:32,matpower/case14/2017-11-21,matpower/case14,basic,1,OPTIMAL,2,0.114000,0.00367923,402510.477093,401029.550134,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:33,matpower/case14/2017-11-22,matpower/case14,basic,1,OPTIMAL,2,0.130000,0.00000000,406482.203223,406482.203223,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:34,matpower/case14/2017-11-23,matpower/case14,basic,1,OPTIMAL,2,0.120000,0.00468159,412352.386828,410421.922089,1,5076,540,540,4536,28836,100770,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:35,matpower/case14/2017-11-24,matpower/case14,basic,1,OPTIMAL,2,0.136000,0.00000000,393973.862069,393973.862069,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:36,matpower/case14/2017-11-25,matpower/case14,basic,1,OPTIMAL,2,0.161000,0.00000000,361751.705310,361751.705310,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:37,matpower/case14/2017-11-26,matpower/case14,basic,1,OPTIMAL,2,0.143000,0.00000000,387095.554374,387095.554374,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:38,matpower/case14/2017-11-27,matpower/case14,basic,1,OPTIMAL,2,0.161000,0.00669672,432628.739567,429731.544252,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,3.806416,3806.416163,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:39,matpower/case14/2017-11-28,matpower/case14,basic,1,OPTIMAL,2,0.153000,0.00000000,407023.066932,407023.066932,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:41,matpower/case14/2017-11-29,matpower/case14,basic,1,OPTIMAL,2,0.132000,0.00000000,408426.226934,408426.226934,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:42,matpower/case14/2017-11-30,matpower/case14,basic,1,OPTIMAL,2,0.131000,0.00000000,426103.182373,426103.182373,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:43,matpower/case14/2017-12-01,matpower/case14,basic,1,OPTIMAL,2,0.142000,0.00000000,401962.502031,401962.502031,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:44,matpower/case14/2017-12-02,matpower/case14,basic,1,OPTIMAL,2,0.120000,0.00000000,393050.022669,393050.022669,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:45,matpower/case14/2017-12-03,matpower/case14,basic,1,OPTIMAL,2,0.101000,0.00000020,414875.555392,414875.473294,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:46,matpower/case14/2017-12-04,matpower/case14,basic,1,OPTIMAL,2,0.099000,0.00000000,428938.317838,428938.317838,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:47,matpower/case14/2017-12-05,matpower/case14,basic,1,OPTIMAL,2,0.107000,0.00000000,412526.156702,412526.156702,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:48,matpower/case14/2017-12-06,matpower/case14,basic,1,OPTIMAL,2,0.105000,0.00000000,431884.309749,431884.309749,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:49,matpower/case14/2017-12-07,matpower/case14,basic,1,OPTIMAL,2,0.104000,0.00000000,471315.550228,471315.550228,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:50,matpower/case14/2017-12-08,matpower/case14,basic,1,OPTIMAL,2,0.123000,0.01553446,533062.785292,524781.940390,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,1717.960000,5.982310,5982.309896,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:51,matpower/case14/2017-12-09,matpower/case14,basic,1,OPTIMAL,2,0.115000,0.03001690,526495.198347,510691.444510,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,4772.640000,5.805431,5805.430894,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:52,matpower/case14/2017-12-10,matpower/case14,basic,1,OPTIMAL,2,0.124000,0.03168578,526621.412719,509935.004773,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,6454.080000,6.548382,6548.381661,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:52,matpower/case14/2017-12-11,matpower/case14,basic,1,OPTIMAL,2,0.134000,0.02829532,618608.093299,601104.381622,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,7774.520000,4.790795,4790.794885,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:53,matpower/case14/2017-12-12,matpower/case14,basic,1,OPTIMAL,2,0.123000,0.02962533,642655.249810,623616.375242,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,7774.520000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:55,matpower/case14/2017-12-13,matpower/case14,basic,1,OPTIMAL,2,0.141000,0.02674082,669369.034101,651469.556104,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,2161.690000,3.157434,3157.433783,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:56,matpower/case14/2017-12-14,matpower/case14,basic,1,OPTIMAL,2,0.144000,0.03780001,689111.392461,663062.975573,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,6.0000,14978.440000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:57,matpower/case14/2017-12-15,matpower/case14,basic,1,OPTIMAL,2,0.136000,0.02378331,709911.943752,693027.884727,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,2.0000,5430.210000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:58,matpower/case14/2017-12-16,matpower/case14,basic,1,OPTIMAL,2,0.122000,0.00000273,571401.335403,571399.776576,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,1186.970000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:58,matpower/case14/2017-12-17,matpower/case14,basic,1,OPTIMAL,2,0.117000,0.00119924,564419.668709,563742.793986,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,4797.790000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:21:59,matpower/case14/2017-12-18,matpower/case14,basic,1,OPTIMAL,2,0.126000,0.00543442,572954.751943,569841.075429,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,1.0000,2207.580000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:22:00,matpower/case14/2017-12-19,matpower/case14,basic,1,OPTIMAL,2,0.121000,0.00153691,543715.067952,542879.424451,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:22:01,matpower/case14/2017-12-20,matpower/case14,basic,1,OPTIMAL,2,0.135000,0.01609757,577285.025726,567992.139058,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:22:02,matpower/case14/2017-12-21,matpower/case14,basic,1,OPTIMAL,2,0.119000,0.00688690,615160.723459,610924.170218,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,4.338090,4338.090000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:22:03,matpower/case14/2017-12-22,matpower/case14,basic,1,OPTIMAL,2,0.125000,0.00067838,516586.761073,516236.318322,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:22:04,matpower/case14/2017-12-23,matpower/case14,basic,1,OPTIMAL,2,0.115000,0.00425634,425749.741501,423937.604275,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:22:05,matpower/case14/2017-12-24,matpower/case14,basic,1,OPTIMAL,2,0.122000,0.00256086,440842.755206,439713.818279,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:22:07,matpower/case14/2017-12-25,matpower/case14,basic,1,OPTIMAL,2,0.172000,0.00000000,475765.149686,475765.149686,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:22:08,matpower/case14/2017-12-26,matpower/case14,basic,1,OPTIMAL,2,0.130000,0.03115208,560677.986966,543211.703444,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,6510.760000,1.719665,1719.665011,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:22:09,matpower/case14/2017-12-27,matpower/case14,basic,1,OPTIMAL,2,0.141000,0.03676480,616855.322648,594176.761282,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,5.0000,10466.930000,2.430494,2430.494013,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:22:10,matpower/case14/2017-12-28,matpower/case14,basic,1,OPTIMAL,2,0.140000,0.02483626,634114.964822,618365.923436,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,6327.180000,1.396859,1396.859024,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:22:11,matpower/case14/2017-12-29,matpower/case14,basic,1,OPTIMAL,2,0.153000,0.00000000,965650.042927,965650.042927,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,3.0000,15648.000000,13.449190,13449.190067,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:22:12,matpower/case14/2017-12-30,matpower/case14,basic,1,OPTIMAL,2,0.121000,0.02717162,591797.798127,575717.695922,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,4.0000,7681.180000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
2025-08-18 19:22:13,matpower/case14/2017-12-31,matpower/case14,basic,1,OPTIMAL,2,0.129000,0.03754186,619072.262114,595831.136401,1,5076,540,540,4536,28836,100877,14,5,20,1,36,60,248,362,4.0000,7710.030000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000

```

### File: `src/data/output/matpower/case57/compare_matpower_case57_20250903_153640.csv`

```
timestamp,instance_name,split,method,status,status_code,runtime_sec,mip_gap,obj_val,obj_bound,nodes,feasible_ok,warm_start_applied_vars
2025-09-03 15:36:40,matpower/case57/2017-01-01,train,raw,OPTIMAL,2,0.725000,0.02062814,2056588.391226,2014164.796889,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-02,train,raw,OPTIMAL,2,0.638000,0.02399929,2174581.900301,2122393.488426,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-03,train,raw,OPTIMAL,2,0.699000,0.02616955,2199183.906250,2141632.261987,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-05,train,raw,OPTIMAL,2,0.726000,0.01109917,2945017.718250,2912330.465919,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-10,train,raw,OPTIMAL,2,0.769000,0.00188743,2996827.351898,2991171.056871,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-11,train,raw,OPTIMAL,2,0.681000,0.01964361,2415279.970568,2367835.156416,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-14,train,raw,OPTIMAL,2,0.607000,0.01623662,2455042.731110,2415181.136457,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-16,train,raw,OPTIMAL,2,0.675000,0.01807608,2385957.272846,2342828.524608,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-17,train,raw,OPTIMAL,2,0.695000,0.00767116,2344641.194804,2326655.070384,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-19,train,raw,OPTIMAL,2,0.665000,0.00512605,2354225.285467,2342157.410954,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-20,train,raw,OPTIMAL,2,0.651000,0.00681734,2391631.131114,2375326.557077,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-21,train,raw,OPTIMAL,2,0.604000,0.01672485,2001771.125244,1968291.806796,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-22,train,raw,OPTIMAL,2,0.585000,0.01083113,2078069.200775,2055561.363150,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-25,train,raw,OPTIMAL,2,0.687000,0.00871766,2325948.398803,2305671.574742,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-26,train,raw,OPTIMAL,2,0.667000,0.01585769,2180561.082480,2145982.420884,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-27,train,raw,OPTIMAL,2,0.645000,0.02147767,2159956.786299,2113565.939202,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-29,train,raw,OPTIMAL,2,0.720000,0.02147662,2466305.064585,2413337.157398,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-30,train,raw,OPTIMAL,2,0.608000,0.01079241,2860235.073481,2829366.245877,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-31,train,raw,OPTIMAL,2,0.597000,0.01291695,2608944.563812,2575244.962552,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-01,train,raw,OPTIMAL,2,0.647000,0.01361463,2585518.232760,2550317.367025,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-02,train,raw,OPTIMAL,2,0.616000,0.01017775,2814606.802235,2785960.436569,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-03,train,raw,OPTIMAL,2,0.653000,0.00800757,2827105.997392,2804467.753508,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-05,train,raw,OPTIMAL,2,0.631000,0.02268048,2404864.927137,2350321.428244,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-06,train,raw,OPTIMAL,2,0.622000,0.02180625,2437443.586057,2384292.077552,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-09,train,raw,OPTIMAL,2,0.591000,0.01319498,3008652.441716,2968953.325754,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-10,train,raw,OPTIMAL,2,0.569000,0.00287852,2999752.770866,2991117.907809,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-11,train,raw,OPTIMAL,2,0.592000,0.01299975,2224046.805374,2195134.752619,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-12,train,raw,OPTIMAL,2,0.600000,0.01101470,2217860.509118,2193431.436125,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-13,train,raw,OPTIMAL,2,0.584000,0.00352868,2560811.206241,2551774.918487,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-15,train,raw,OPTIMAL,2,0.589000,0.00319463,2628331.629599,2619935.092884,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-16,train,raw,OPTIMAL,2,0.600000,0.01598578,2723047.798491,2679517.758934,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-17,train,raw,OPTIMAL,2,0.728000,0.02119981,2374212.141860,2323879.298559,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-18,train,raw,OPTIMAL,2,0.758000,0.03193472,1718960.811577,1664066.273399,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-19,train,raw,OPTIMAL,2,0.815000,0.03275908,1720927.565321,1664551.569979,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-20,train,raw,OPTIMAL,2,0.723000,0.02393549,2118152.464535,2067453.454552,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-21,train,raw,OPTIMAL,2,0.753000,0.02162421,2141038.512195,2094740.241265,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-27,train,raw,OPTIMAL,2,0.711000,0.01885527,2032540.306441,1994216.213026,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-28,train,raw,OPTIMAL,2,0.719000,0.02077496,1859086.192516,1820463.743246,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-01,train,raw,OPTIMAL,2,0.717000,0.02848273,1796333.092245,1745168.622648,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-05,train,raw,OPTIMAL,2,0.720000,0.02365565,2273861.566171,2220071.895271,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-06,train,raw,OPTIMAL,2,0.728000,0.02711472,2044568.918984,1989130.996781,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-07,train,raw,OPTIMAL,2,0.718000,0.02271029,1867067.063871,1824665.437482,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-09,train,raw,OPTIMAL,2,0.719000,0.02253151,1961354.897994,1917162.616061,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-10,train,raw,OPTIMAL,2,0.719000,0.01586239,2548866.052213,2508434.953657,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-12,train,raw,OPTIMAL,2,0.808000,0.02095507,2656716.292675,2601044.607479,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-16,train,raw,OPTIMAL,2,0.720000,0.01543768,3006931.697172,2960511.657358,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-17,train,raw,OPTIMAL,2,0.730000,0.01640754,2353054.335015,2314446.503029,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-18,train,raw,OPTIMAL,2,0.703000,0.02904529,1931867.909624,1875756.236746,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-19,train,raw,OPTIMAL,2,0.713000,0.02138113,2130277.516682,2084729.780377,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-20,train,raw,OPTIMAL,2,0.695000,0.01944975,2311160.258789,2266208.768433,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-21,train,raw,OPTIMAL,2,0.717000,0.02753090,2115146.145995,2056914.269404,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-24,train,raw,OPTIMAL,2,0.709000,0.02945839,1986507.056525,1927987.765552,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-26,train,raw,OPTIMAL,2,0.695000,0.03525020,1687045.071365,1627576.398322,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-27,train,raw,OPTIMAL,2,0.692000,0.03394087,1753177.990842,1693673.607305,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-28,train,raw,OPTIMAL,2,0.699000,0.03414526,1742629.076251,1683126.559552,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-29,train,raw,OPTIMAL,2,0.712000,0.02958769,1950097.692235,1892398.797995,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-30,train,raw,OPTIMAL,2,0.711000,0.02640293,2175896.925394,2118446.881189,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-31,train,raw,OPTIMAL,2,0.752000,0.02233391,2123221.563898,2075801.730904,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-01,train,raw,OPTIMAL,2,0.784000,0.03495678,1663474.962535,1605325.229269,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-02,train,raw,OPTIMAL,2,0.709000,0.03444341,1677649.849086,1619865.867421,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-03,train,raw,OPTIMAL,2,0.698000,0.03090449,1842214.166168,1785281.484291,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-04,train,raw,OPTIMAL,2,0.712000,0.03080535,1793929.493134,1738666.871633,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-07,train,raw,OPTIMAL,2,0.696000,0.02792551,1829254.224181,1778171.365146,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-09,train,raw,OPTIMAL,2,0.679000,0.03350995,1598302.929894,1544743.878548,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-11,train,raw,OPTIMAL,2,0.711000,0.02388951,1886803.314894,1841728.510661,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-12,train,raw,OPTIMAL,2,0.717000,0.02620265,1801408.626584,1754206.945600,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-13,train,raw,OPTIMAL,2,0.703000,0.02975574,1762136.413893,1709702.732680,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-14,train,raw,OPTIMAL,2,0.715000,0.03342581,1632516.841098,1577948.637974,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-16,train,raw,OPTIMAL,2,0.729000,0.03173923,1719989.797896,1665398.653205,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-17,train,raw,OPTIMAL,2,0.703000,0.03096111,1757958.439072,1703530.099060,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-18,train,raw,OPTIMAL,2,0.713000,0.03296883,1651936.393001,1597473.975765,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-19,train,raw,OPTIMAL,2,0.718000,0.01969913,1600800.205188,1569265.831726,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-25,train,raw,OPTIMAL,2,0.770000,0.00955855,1632158.225144,1616557.160678,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-29,train,raw,OPTIMAL,2,0.703000,0.01108809,1433401.592126,1417507.899666,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-30,train,raw,OPTIMAL,2,0.703000,0.01816814,1278310.547807,1255086.028467,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-02,train,raw,OPTIMAL,2,0.701000,0.00790351,1499436.225639,1487585.414010,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-04,train,raw,OPTIMAL,2,0.713000,0.00537889,1543171.426126,1534870.872866,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-06,train,raw,OPTIMAL,2,0.712000,0.02575393,1164155.584469,1134174.002966,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-07,train,raw,OPTIMAL,2,0.699000,0.01871442,1332912.264881,1307967.584053,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-08,train,raw,OPTIMAL,2,0.698000,0.00947052,1565807.886586,1550978.876005,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-09,train,raw,OPTIMAL,2,0.687000,0.01718857,1596401.378995,1568961.519217,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-10,train,raw,OPTIMAL,2,0.719000,0.01764437,1573467.835813,1545704.984859,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-11,train,raw,OPTIMAL,2,0.801000,0.01223222,1668693.686085,1648281.866056,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-12,train,raw,OPTIMAL,2,0.758000,0.01587889,1586318.015353,1561129.041452,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-14,train,raw,OPTIMAL,2,0.693000,0.02811795,1372054.936982,1333475.567998,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-15,train,raw,OPTIMAL,2,0.752000,0.01589195,1589150.671935,1563895.975042,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-16,train,raw,OPTIMAL,2,0.712000,0.00514349,1836601.461694,1827154.918672,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-17,train,raw,OPTIMAL,2,0.713000,0.00125512,2223470.638297,2220679.911065,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-18,train,raw,OPTIMAL,2,0.719000,0.00052660,2441066.437004,2439780.974412,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-20,train,raw,OPTIMAL,2,0.696000,0.02833107,1316701.732894,1279398.165437,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-21,train,raw,OPTIMAL,2,0.709000,0.02643618,1279564.300524,1245737.503312,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-22,train,raw,OPTIMAL,2,0.726000,0.01417550,1535395.289383,1513630.294212,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-23,train,raw,OPTIMAL,2,0.710000,0.01379500,1547481.709174,1526134.193099,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-24,train,raw,OPTIMAL,2,0.744000,0.01063520,1654698.522859,1637100.473133,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-25,train,raw,OPTIMAL,2,0.713000,0.01315768,1595636.659591,1574641.781260,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-27,train,raw,OPTIMAL,2,0.753000,0.02850557,1228473.761959,1193455.417535,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-28,train,raw,OPTIMAL,2,0.704000,0.03077199,1186094.879672,1149596.377281,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-29,train,raw,OPTIMAL,2,0.696000,0.02519021,1260938.455565,1229175.150848,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-30,train,raw,OPTIMAL,2,0.725000,0.01508121,1535133.765034,1511982.088346,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-31,train,raw,OPTIMAL,2,0.745000,0.01209622,1677760.662301,1657466.100951,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-02,train,raw,OPTIMAL,2,0.707000,0.01378506,1655772.020035,1632947.107061,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-04,train,raw,OPTIMAL,2,0.741000,0.00754908,1737338.296254,1724222.989651,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-05,train,raw,OPTIMAL,2,0.713000,0.00809668,2010636.455303,1994356.965882,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-06,train,raw,OPTIMAL,2,0.694000,0.01255699,1816169.029497,1793363.416326,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-08,train,raw,OPTIMAL,2,0.712000,0.01180595,1684308.592702,1664423.726166,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-09,train,raw,OPTIMAL,2,0.720000,0.00727814,1825951.312636,1812661.779591,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-10,train,raw,OPTIMAL,2,0.710000,0.00492227,2121811.591219,2111367.471962,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-11,train,raw,OPTIMAL,2,0.734000,0.00089683,2745151.421610,2742689.486109,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-13,train,raw,OPTIMAL,2,0.728000,0.00016073,3476966.004893,3476407.167520,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-14,train,raw,OPTIMAL,2,0.792000,0.00224391,2733269.920306,2727136.715784,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-15,train,raw,OPTIMAL,2,0.745000,0.00328357,2326979.987566,2319339.186850,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-16,train,raw,OPTIMAL,2,0.718000,0.00363878,2381623.192317,2372956.985614,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-17,train,raw,OPTIMAL,2,0.785000,0.00308556,2434591.193291,2427079.124071,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-18,train,raw,OPTIMAL,2,0.710000,0.00055258,2898472.212485,2896870.567751,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-19,train,raw,OPTIMAL,2,0.744000,0.00070235,2970463.334820,2968377.043988,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-20,train,raw,OPTIMAL,2,0.719000,0.00105340,2858295.315827,2855284.379659,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-21,train,raw,OPTIMAL,2,0.714000,0.00065659,2831185.015651,2829326.077021,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-22,train,raw,OPTIMAL,2,0.733000,0.00004821,3229187.273703,3229031.589252,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-23,train,raw,OPTIMAL,2,0.761000,0.00062574,3208880.991928,3206873.067624,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-24,train,raw,OPTIMAL,2,0.761000,0.00242281,2759869.406574,2753182.768805,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-26,train,raw,OPTIMAL,2,0.712000,0.00277949,2471619.169205,2464749.330774,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-27,train,raw,OPTIMAL,2,0.718000,0.00426384,2398769.262565,2388541.287002,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-29,train,raw,OPTIMAL,2,0.736000,0.00054037,3127425.956850,3125735.999998,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-30,train,raw,OPTIMAL,2,0.754000,0.00255908,3311657.995397,3303183.186982,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-02,train,raw,OPTIMAL,2,0.727000,0.00436083,2998932.190373,2985854.366421,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-04,train,raw,OPTIMAL,2,0.719000,0.00523681,2778941.368776,2764388.587402,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-05,train,raw,OPTIMAL,2,0.718000,0.00488804,2852581.630061,2838638.102708,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-07,train,raw,OPTIMAL,2,0.713000,0.00471832,2882216.829016,2868617.603179,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-08,train,raw,OPTIMAL,2,0.745000,0.00815610,2619392.553177,2598028.526828,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-09,train,raw,OPTIMAL,2,0.710000,0.02333635,1845675.279593,1802603.955620,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-11,train,raw,OPTIMAL,2,0.716000,0.00091690,3373326.657967,3370233.660112,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-12,train,raw,OPTIMAL,2,0.730000,0.00033038,3591684.899232,3590498.289262,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-13,train,raw,OPTIMAL,2,0.734000,0.00092227,3643913.303598,3640552.630237,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-17,train,raw,OPTIMAL,2,0.734000,0.00002006,3160356.953488,3160293.555824,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-18,train,raw,OPTIMAL,2,0.732000,0.00000000,3305980.219103,3305980.219103,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-19,train,raw,OPTIMAL,2,0.713000,0.00065549,2530177.730225,2528519.218203,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-20,train,raw,OPTIMAL,2,0.756000,0.00000000,3654319.551434,3654319.551434,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-21,train,raw,OPTIMAL,2,0.761000,0.00025964,3524115.000070,3523199.983474,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-22,train,raw,OPTIMAL,2,0.728000,0.00127856,3144393.267876,3140372.972445,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-23,train,raw,OPTIMAL,2,0.713000,0.00464702,2014121.709752,2004762.037531,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-25,train,raw,OPTIMAL,2,0.717000,0.01028070,1732172.045146,1714364.103378,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-26,train,raw,OPTIMAL,2,0.727000,0.01021651,1730303.435831,1712625.769330,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-28,train,raw,OPTIMAL,2,0.711000,0.00464332,1700076.724139,1692182.725322,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-29,train,raw,OPTIMAL,2,0.729000,0.01067436,1400469.812333,1385520.687799,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-30,train,raw,OPTIMAL,2,0.729000,0.00482053,2014753.846970,2005041.659659,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-31,train,raw,OPTIMAL,2,0.701000,0.00088699,2544607.477837,2542350.431010,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-08-01,train,raw,OPTIMAL,2,0.790000,0.00012577,3005462.431129,3005084.432380,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-08-02,train,raw,OPTIMAL,2,0.715000,0.00019332,3079748.288810,3079152.915926,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-08-03,train,raw,OPTIMAL,2,0.719000,0.00016651,3036868.319127,3036362.650171,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-08-04,train,raw,INFEASIBLE,3,0.281000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-05,train,raw,INFEASIBLE,3,0.273000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-06,train,raw,INFEASIBLE,3,0.293000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-07,train,raw,INFEASIBLE,3,0.263000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-09,train,raw,INFEASIBLE,3,0.283000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-11,train,raw,INFEASIBLE,3,0.253000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-12,train,raw,INFEASIBLE,3,0.311000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-13,train,raw,INFEASIBLE,3,0.288000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-15,train,raw,INFEASIBLE,3,0.282000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-16,train,raw,INFEASIBLE,3,0.269000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-17,train,raw,INFEASIBLE,3,0.289000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-18,train,raw,INFEASIBLE,3,0.261000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-19,train,raw,INFEASIBLE,3,0.258000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-20,train,raw,INFEASIBLE,3,0.205000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-21,train,raw,INFEASIBLE,3,0.184000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-22,train,raw,INFEASIBLE,3,0.200000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-24,train,raw,INFEASIBLE,3,0.193000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-26,train,raw,OPTIMAL,2,0.535000,0.00165150,1429566.322589,1427205.396500,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-08-27,train,raw,INFEASIBLE,3,0.185000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-30,train,raw,INFEASIBLE,3,0.191000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-09-01,train,raw,OPTIMAL,2,0.509000,0.00661928,1547880.585799,1537634.736278,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-09-02,train,raw,OPTIMAL,2,0.544000,0.00901274,1331406.887525,1319407.265100,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-09-03,train,raw,OPTIMAL,2,0.518000,0.00689631,1389078.521163,1379499.004432,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-09-04,train,raw,INFEASIBLE,3,0.210000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-09-06,train,raw,INFEASIBLE,3,0.192000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-09-07,train,raw,OPTIMAL,2,0.508000,0.00209914,1511440.719976,1508267.987118,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-09-09,train,raw,OPTIMAL,2,0.526000,0.04688221,1267251.424281,1207839.881246,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-09-10,train,raw,OPTIMAL,2,0.514000,0.03907534,1304791.797171,1253806.614648,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-09-11,train,raw,OPTIMAL,2,0.525000,0.02181275,1469299.993626,1437250.516366,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-09-12,train,raw,OPTIMAL,2,0.530000,0.00000000,1677950.659099,1677950.659099,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-09-14,train,raw,INFEASIBLE,3,0.203000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-09-15,train,raw,INFEASIBLE,3,0.194000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-09-16,train,raw,OPTIMAL,2,0.541000,0.00085774,1652815.833752,1651398.139996,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-09-17,train,raw,INFEASIBLE,3,0.193000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-09-19,train,raw,INFEASIBLE,3,0.193000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-09-21,train,raw,INFEASIBLE,3,0.250000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-09-22,train,raw,INFEASIBLE,3,0.199000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-09-24,train,raw,INFEASIBLE,3,0.192000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-09-25,train,raw,INFEASIBLE,3,0.206000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-09-26,train,raw,INFEASIBLE,3,0.200000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-09-27,train,raw,INFEASIBLE,3,0.206000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-09-28,train,raw,INFEASIBLE,3,0.207000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-09-29,train,raw,OPTIMAL,2,0.529000,0.01481910,1279312.470934,1260354.206183,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-09-30,train,raw,OPTIMAL,2,0.516000,0.02125855,1090058.023769,1066884.973172,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-02,train,raw,OPTIMAL,2,0.507000,0.00587101,1364714.466114,1356702.211570,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-03,train,raw,OPTIMAL,2,0.536000,0.00591694,1548726.104301,1539562.377862,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-04,train,raw,OPTIMAL,2,0.519000,0.00234880,1620406.885227,1616600.878397,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-05,train,raw,INFEASIBLE,3,0.197000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-10-07,train,raw,INFEASIBLE,3,0.197000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-10-08,train,raw,INFEASIBLE,3,0.201000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-10-09,train,raw,INFEASIBLE,3,0.191000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-10-10,train,raw,INFEASIBLE,3,0.194000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-10-11,train,raw,OPTIMAL,2,0.534000,0.00040387,1738245.536507,1737543.519816,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-12,train,raw,OPTIMAL,2,0.526000,0.00193952,1825606.036150,1822065.235407,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-15,train,raw,OPTIMAL,2,0.566000,0.00514777,1735630.172769,1726695.541478,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-16,train,raw,OPTIMAL,2,0.537000,0.00395698,1697034.690749,1690319.561300,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-17,train,raw,OPTIMAL,2,0.521000,0.00486462,1675321.853871,1667172.047707,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-18,train,raw,OPTIMAL,2,0.550000,0.00702580,1576394.787386,1565319.345774,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-20,train,raw,OPTIMAL,2,0.517000,0.00744007,1461635.737320,1450761.067988,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-21,train,raw,OPTIMAL,2,0.505000,0.00961514,1353068.004779,1340058.070415,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-22,train,raw,OPTIMAL,2,0.509000,0.01212377,1440373.188583,1422910.434552,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-23,train,raw,OPTIMAL,2,0.527000,0.00241118,1709820.816703,1705698.137008,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-24,train,raw,OPTIMAL,2,0.518000,0.00864190,1762412.548327,1747181.962512,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-25,train,raw,OPTIMAL,2,0.526000,0.01116751,1689085.666849,1670222.784308,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-26,train,raw,OPTIMAL,2,0.506000,0.00393496,1597170.534658,1590885.732787,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-27,train,raw,OPTIMAL,2,0.519000,0.01457275,1470283.084819,1448857.015311,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-28,train,raw,OPTIMAL,2,0.509000,0.01705434,1303158.296822,1280933.791492,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-29,train,raw,OPTIMAL,2,0.511000,0.01302874,1428920.023659,1410303.000110,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-30,train,raw,OPTIMAL,2,0.535000,0.00311981,1670777.328330,1665564.812587,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-01,train,raw,OPTIMAL,2,0.529000,0.00335143,1862188.562263,1855947.575584,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-03,train,raw,OPTIMAL,2,0.529000,0.00952180,1481495.287255,1467388.790652,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-04,train,raw,OPTIMAL,2,0.528000,0.02465820,1350360.411888,1317062.950948,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-05,train,raw,OPTIMAL,2,0.529000,0.01792817,1387479.585086,1362604.610587,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-06,train,raw,OPTIMAL,2,0.541000,0.00912152,1483312.572242,1469782.506742,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-08,train,raw,INFEASIBLE,3,0.208000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-11-10,train,raw,INFEASIBLE,3,0.200000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-11-11,train,raw,OPTIMAL,2,0.550000,0.00701889,1712088.317895,1700071.352766,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-13,train,raw,INFEASIBLE,3,0.192000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-11-14,train,raw,INFEASIBLE,3,0.185000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-11-18,train,raw,OPTIMAL,2,0.558000,0.01864898,1556231.499022,1527209.365617,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-19,train,raw,INFEASIBLE,3,0.209000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-11-20,train,raw,INFEASIBLE,3,0.204000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-11-21,train,raw,OPTIMAL,2,0.539000,0.00577701,1842430.250026,1831786.512630,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-22,train,raw,OPTIMAL,2,0.583000,0.00364175,1876749.665792,1869915.015135,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-23,train,raw,OPTIMAL,2,0.544000,0.00980897,1860615.688037,1842364.962521,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-24,train,raw,OPTIMAL,2,0.597000,0.00787250,1866197.556007,1851505.909150,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-26,train,raw,OPTIMAL,2,0.595000,0.00790571,1847066.141029,1832463.769337,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-27,train,raw,INFEASIBLE,3,0.280000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-11-28,train,raw,OPTIMAL,2,0.645000,0.00176490,1968728.884843,1965254.281459,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-29,train,raw,OPTIMAL,2,0.598000,0.00100197,2015952.839764,2013932.919356,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-30,train,raw,INFEASIBLE,3,0.216000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-12-01,train,raw,INFEASIBLE,3,0.246000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-12-02,train,raw,OPTIMAL,2,0.631000,0.00476354,1763536.152193,1755135.482011,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-12-04,train,raw,OPTIMAL,2,0.557000,0.00079047,2108825.371290,2107158.403476,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-12-05,train,raw,OPTIMAL,2,0.620000,0.00291612,2014561.133303,2008686.435471,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-12-06,train,raw,OPTIMAL,2,0.561000,0.00103473,2139738.978259,2137524.935079,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-12-07,train,raw,INFEASIBLE,3,0.248000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-12-10,train,raw,INFEASIBLE,3,0.264000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-12-11,train,raw,INFEASIBLE,3,0.281000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-12-12,train,raw,INFEASIBLE,3,0.240000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-12-15,train,raw,INFEASIBLE,3,0.270000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-12-16,train,raw,INFEASIBLE,3,0.254000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-12-17,train,raw,INFEASIBLE,3,0.270000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-12-19,train,raw,OPTIMAL,2,0.597000,0.00451376,2544863.706703,2533376.814188,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-12-21,train,raw,INFEASIBLE,3,0.239000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-12-22,train,raw,INFEASIBLE,3,0.255000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-12-23,train,raw,OPTIMAL,2,0.604000,0.02149293,2075494.691668,2030886.219552,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-12-27,train,raw,INFEASIBLE,3,0.233000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-12-28,train,raw,INFEASIBLE,3,0.271000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-12-31,train,raw,INFEASIBLE,3,0.266000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-01-04,test,raw,OPTIMAL,2,0.657000,0.02244690,2479065.607097,2423418.261871,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-04,test,warm,OPTIMAL,2,0.644000,0.02244690,2479065.607097,2423418.261871,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-01-08,test,raw,OPTIMAL,2,0.620000,0.00002903,3402028.085171,3401929.309791,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-08,test,warm,OPTIMAL,2,0.627000,0.00002903,3402028.085171,3401929.309791,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-01-09,test,raw,OPTIMAL,2,0.573000,0.00000149,3559595.752563,3559590.460239,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-09,test,warm,OPTIMAL,2,0.652000,0.00000149,3559595.752563,3559590.460239,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-01-13,test,raw,OPTIMAL,2,0.637000,0.01802134,2389217.859453,2346160.948701,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-13,test,warm,OPTIMAL,2,0.618000,0.01802134,2389217.859453,2346160.948701,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-01-18,test,raw,OPTIMAL,2,0.616000,0.00779925,2284050.568398,2266236.677202,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-18,test,warm,OPTIMAL,2,0.619000,0.00779925,2284050.568398,2266236.677202,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-01-23,test,raw,OPTIMAL,2,0.653000,0.00345448,2474176.584501,2465629.579114,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-01-23,test,warm,OPTIMAL,2,0.678000,0.00345448,2474176.584501,2465629.579114,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-02-07,test,raw,OPTIMAL,2,0.651000,0.02464411,2163628.127507,2110307.436961,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-07,test,warm,OPTIMAL,2,0.633000,0.02464411,2163628.127507,2110307.436961,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-02-08,test,raw,OPTIMAL,2,0.575000,0.02378033,2227384.891848,2174416.941087,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-08,test,warm,OPTIMAL,2,0.613000,0.02378033,2227384.891848,2174416.941087,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-02-14,test,raw,OPTIMAL,2,0.602000,0.00306512,2664570.267719,2656403.051503,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-14,test,warm,OPTIMAL,2,0.619000,0.00306512,2664570.267719,2656403.051503,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-02-26,test,raw,OPTIMAL,2,0.614000,0.02972103,1816591.719791,1762600.747811,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-02-26,test,warm,OPTIMAL,2,0.634000,0.02972103,1816591.719791,1762600.747811,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-03-02,test,raw,OPTIMAL,2,0.662000,0.01665252,2046831.394004,2012746.487149,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-02,test,warm,OPTIMAL,2,0.660000,0.01665252,2046831.394004,2012746.487149,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-03-04,test,raw,OPTIMAL,2,0.652000,0.02392930,2225060.653827,2171816.517044,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-04,test,warm,OPTIMAL,2,0.629000,0.02392930,2225060.653827,2171816.517044,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-03-11,test,raw,OPTIMAL,2,0.644000,0.01371428,2673601.652370,2636935.136784,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-11,test,warm,OPTIMAL,2,0.713000,0.01371428,2673601.652370,2636935.136784,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-03-14,test,raw,OPTIMAL,2,0.602000,0.01505447,2921478.091885,2877496.793627,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-14,test,warm,OPTIMAL,2,0.655000,0.01505447,2921478.091885,2877496.793627,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-03-23,test,raw,OPTIMAL,2,0.544000,0.02226620,2546563.188734,2489860.903974,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-23,test,warm,OPTIMAL,2,0.606000,0.02226620,2546563.188734,2489860.903974,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-03-25,test,raw,OPTIMAL,2,0.605000,0.03835518,1549674.466609,1490236.430705,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-03-25,test,warm,OPTIMAL,2,0.643000,0.03835518,1549674.466609,1490236.430705,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-04-06,test,raw,OPTIMAL,2,0.657000,0.02882996,1861011.439030,1807358.558897,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-06,test,warm,OPTIMAL,2,0.642000,0.02882996,1861011.439030,1807358.558897,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-04-08,test,raw,OPTIMAL,2,0.632000,0.03133508,1646678.360645,1595079.559992,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-08,test,warm,OPTIMAL,2,0.609000,0.03133508,1646678.360645,1595079.559992,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-04-10,test,raw,OPTIMAL,2,0.616000,0.02528808,1814224.028226,1768345.794433,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-10,test,warm,OPTIMAL,2,0.627000,0.02528808,1814224.028226,1768345.794433,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-04-20,test,raw,OPTIMAL,2,0.636000,0.01057119,1530655.194963,1514474.343388,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-20,test,warm,OPTIMAL,2,0.677000,0.01057119,1530655.194963,1514474.343388,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-04-21,test,raw,OPTIMAL,2,0.643000,0.01543289,1492571.223236,1469536.529136,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-21,test,warm,OPTIMAL,2,0.638000,0.01543289,1492571.223236,1469536.529136,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-04-22,test,raw,OPTIMAL,2,0.636000,0.02862883,1266454.121089,1230197.019119,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-22,test,warm,OPTIMAL,2,0.646000,0.02862883,1266454.121089,1230197.019119,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-04-23,test,raw,OPTIMAL,2,0.633000,0.02271668,1356209.338690,1325400.763394,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-23,test,warm,OPTIMAL,2,0.631000,0.02271668,1356209.338690,1325400.763394,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-04-24,test,raw,OPTIMAL,2,0.586000,0.01254579,1588638.649690,1568707.917712,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-04-24,test,warm,OPTIMAL,2,0.649000,0.01254579,1588638.649690,1568707.917712,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-05-01,test,raw,OPTIMAL,2,0.628000,0.00680814,1529827.557129,1519412.274767,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-01,test,warm,OPTIMAL,2,0.658000,0.00680814,1529827.557129,1519412.274767,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-05-13,test,raw,OPTIMAL,2,0.610000,0.03063952,1353594.179746,1312120.702893,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-13,test,warm,OPTIMAL,2,0.617000,0.03063952,1353594.179746,1312120.702893,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-05-19,test,raw,OPTIMAL,2,0.614000,0.00642231,2107420.003723,2093885.497875,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-19,test,warm,OPTIMAL,2,0.700000,0.00642231,2107420.003723,2093885.497875,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-05-26,test,raw,OPTIMAL,2,0.627000,0.01833017,1441345.173291,1414925.064470,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-05-26,test,warm,OPTIMAL,2,0.661000,0.01833017,1441345.173291,1414925.064470,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-06-01,test,raw,OPTIMAL,2,0.622000,0.01182831,1682854.868385,1662949.547274,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-01,test,warm,OPTIMAL,2,0.676000,0.01182831,1682854.868385,1662949.547274,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-06-12,test,raw,OPTIMAL,2,0.652000,0.00003488,3463506.833303,3463386.040158,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-12,test,warm,OPTIMAL,2,0.635000,0.00003488,3463506.833303,3463386.040158,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-06-25,test,raw,OPTIMAL,2,0.627000,0.00261048,2440958.053056,2434585.975660,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-25,test,warm,OPTIMAL,2,0.665000,0.00261048,2440958.053056,2434585.975660,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-06-28,test,raw,OPTIMAL,2,0.629000,0.00252992,2517016.506402,2510648.661984,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-06-28,test,warm,OPTIMAL,2,0.658000,0.00252992,2517016.506402,2510648.661984,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-07-01,test,raw,OPTIMAL,2,0.612000,0.00408899,2969573.800701,2957431.247180,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-01,test,warm,OPTIMAL,2,0.653000,0.00408899,2969573.800701,2957431.247180,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-07-10,test,raw,OPTIMAL,2,0.636000,0.00286763,2991103.517460,2982526.144912,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-10,test,warm,OPTIMAL,2,0.670000,0.00286763,2991103.517460,2982526.144912,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-07-14,test,raw,OPTIMAL,2,0.617000,0.00394247,3053262.276343,3041224.884699,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-14,test,warm,OPTIMAL,2,0.716000,0.00394247,3053262.276343,3041224.884699,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-07-16,test,raw,OPTIMAL,2,0.656000,0.00602134,2044252.629630,2031943.486866,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-16,test,warm,OPTIMAL,2,0.667000,0.00602134,2044252.629630,2031943.486866,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-07-24,test,raw,OPTIMAL,2,0.620000,0.00611799,2001975.590601,1989727.529999,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-24,test,warm,OPTIMAL,2,0.607000,0.00611799,2001975.590601,1989727.529999,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-07-27,test,raw,OPTIMAL,2,0.597000,0.00605206,1930233.816418,1918551.922171,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-07-27,test,warm,OPTIMAL,2,0.644000,0.00605206,1930233.816418,1918551.922171,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-08-14,test,raw,INFEASIBLE,3,0.287000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-14,test,warm,INFEASIBLE,3,0.287000,inf,,inf,0,,16452
2025-09-03 15:36:40,matpower/case57/2017-08-25,test,raw,INFEASIBLE,3,0.255000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-25,test,warm,INFEASIBLE,3,0.319000,inf,,inf,0,,16452
2025-09-03 15:36:40,matpower/case57/2017-08-28,test,raw,INFEASIBLE,3,0.301000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-28,test,warm,INFEASIBLE,3,0.288000,inf,,inf,0,,16452
2025-09-03 15:36:40,matpower/case57/2017-08-29,test,raw,INFEASIBLE,3,0.270000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-29,test,warm,INFEASIBLE,3,0.368000,inf,,inf,0,,16452
2025-09-03 15:36:40,matpower/case57/2017-08-31,test,raw,INFEASIBLE,3,0.263000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-08-31,test,warm,INFEASIBLE,3,0.264000,inf,,inf,0,,16452
2025-09-03 15:36:40,matpower/case57/2017-09-05,test,raw,INFEASIBLE,3,0.282000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-09-05,test,warm,INFEASIBLE,3,0.264000,inf,,inf,0,,16452
2025-09-03 15:36:40,matpower/case57/2017-09-18,test,raw,INFEASIBLE,3,0.284000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-09-18,test,warm,INFEASIBLE,3,0.299000,inf,,inf,0,,16452
2025-09-03 15:36:40,matpower/case57/2017-09-23,test,raw,INFEASIBLE,3,0.262000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-09-23,test,warm,INFEASIBLE,3,0.297000,inf,,inf,0,,16452
2025-09-03 15:36:40,matpower/case57/2017-10-01,test,raw,OPTIMAL,2,0.604000,0.02060652,1175482.042180,1151259.451666,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-01,test,warm,OPTIMAL,2,0.594000,0.02060652,1175482.042180,1151259.451666,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-10-06,test,raw,INFEASIBLE,3,0.271000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-10-06,test,warm,INFEASIBLE,3,0.303000,inf,,inf,0,,16452
2025-09-03 15:36:40,matpower/case57/2017-10-13,test,raw,OPTIMAL,2,0.589000,0.00657172,1768790.749287,1757166.755926,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-13,test,warm,OPTIMAL,2,0.642000,0.00657172,1768790.749287,1757166.755926,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-10-14,test,raw,OPTIMAL,2,0.638000,0.00539283,1591149.964356,1582569.164019,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-14,test,warm,OPTIMAL,2,0.630000,0.00539283,1591149.964356,1582569.164019,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-10-19,test,raw,OPTIMAL,2,0.618000,0.00790004,1564876.576107,1552513.994482,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-19,test,warm,OPTIMAL,2,0.617000,0.00790004,1564876.576107,1552513.994482,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-10-31,test,raw,OPTIMAL,2,0.686000,0.00264387,1700411.992518,1695916.328673,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-10-31,test,warm,OPTIMAL,2,0.721000,0.00264387,1700411.992518,1695916.328673,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-11-02,test,raw,OPTIMAL,2,0.706000,0.00419582,1822564.747120,1814917.592447,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-02,test,warm,OPTIMAL,2,0.729000,0.00419582,1822564.747120,1814917.592447,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-11-07,test,raw,OPTIMAL,2,0.670000,0.00526344,1583685.199302,1575349.571571,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-07,test,warm,OPTIMAL,2,0.717000,0.00526344,1583685.199302,1575349.571571,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-11-09,test,raw,INFEASIBLE,3,0.276000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-11-09,test,warm,INFEASIBLE,3,0.369000,inf,,inf,0,,16452
2025-09-03 15:36:40,matpower/case57/2017-11-17,test,raw,INFEASIBLE,3,0.406000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-11-17,test,warm,INFEASIBLE,3,0.286000,inf,,inf,0,,16452
2025-09-03 15:36:40,matpower/case57/2017-11-25,test,raw,OPTIMAL,2,0.788000,0.01695401,1672110.943179,1643761.955364,1,OK,
2025-09-03 15:36:40,matpower/case57/2017-11-25,test,warm,OPTIMAL,2,0.779000,0.01695401,1672110.943179,1643761.955364,1,OK,16452
2025-09-03 15:36:40,matpower/case57/2017-12-14,test,raw,INFEASIBLE,3,0.296000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-12-14,test,warm,INFEASIBLE,3,0.313000,inf,,inf,0,,16452
2025-09-03 15:36:40,matpower/case57/2017-12-20,test,raw,INFEASIBLE,3,0.293000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-12-20,test,warm,INFEASIBLE,3,0.293000,inf,,inf,0,,16452
2025-09-03 15:36:40,matpower/case57/2017-12-25,test,raw,INFEASIBLE,3,0.270000,inf,,inf,0,,
2025-09-03 15:36:40,matpower/case57/2017-12-25,test,warm,INFEASIBLE,3,0.305000,inf,,inf,0,,16452

```

### File: `src/data/output/test/case14-fixed-status/perf_test_case14_fixed_status_basic_01.csv`

```
timestamp,instance_name,case_folder,technique,run_id,status,status_code,runtime_sec,mip_gap,obj_val,obj_bound,nodes,num_vars,num_bin_vars,num_int_vars,num_cont_vars,num_constrs,num_nzs,buses,units,lines,reserves,time_steps,time_step_min,ptdf_nnz,lodf_nnz,startup_count,startup_cost,reserve_shortfall_mw,reserve_shortfall_penalty,base_overflow_mw,base_overflow_penalty,cont_overflow_mw,cont_overflow_penalty
2025-08-18 19:22:13,test/case14-fixed-status,test/case14-fixed-status,basic,1,INFEASIBLE,3,0.019000,inf,,inf,1,548,72,72,476,3182,10380,14,6,20,1,4,60,248,362,0.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000

```

### File: `src/data/output/test/case14-profiled/perf_test_case14_profiled_basic_01.csv`

```
timestamp,instance_name,case_folder,technique,run_id,status,status_code,runtime_sec,mip_gap,obj_val,obj_bound,nodes,num_vars,num_bin_vars,num_int_vars,num_cont_vars,num_constrs,num_nzs,buses,units,lines,reserves,time_steps,time_step_min,ptdf_nnz,lodf_nnz,startup_count,startup_cost,reserve_shortfall_mw,reserve_shortfall_penalty,base_overflow_mw,base_overflow_penalty,cont_overflow_mw,cont_overflow_penalty
2025-08-18 19:22:13,test/case14-profiled,test/case14-profiled,basic,1,OPTIMAL,2,0.017000,0.01736596,37456.981338,36806.504925,1,548,72,72,476,3172,10370,14,6,20,1,4,60,248,362,5.0000,5000.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000

```

### File: `src/data/output/test/case14-storage/perf_test_case14_storage_basic_01.csv`

```
timestamp,instance_name,case_folder,technique,run_id,status,status_code,runtime_sec,mip_gap,obj_val,obj_bound,nodes,num_vars,num_bin_vars,num_int_vars,num_cont_vars,num_constrs,num_nzs,buses,units,lines,reserves,time_steps,time_step_min,ptdf_nnz,lodf_nnz,startup_count,startup_cost,reserve_shortfall_mw,reserve_shortfall_penalty,base_overflow_mw,base_overflow_penalty,cont_overflow_mw,cont_overflow_penalty
2025-08-18 19:22:13,test/case14-storage,test/case14-storage,basic,1,OPTIMAL,2,0.026000,0.01736596,37456.981338,36806.504925,1,548,72,72,476,3172,10370,14,6,20,1,4,60,248,362,5.0000,5000.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000

```

### File: `src/data/output/test/case14-sub-hourly/perf_test_case14_sub_hourly_basic_01.csv`

```
timestamp,instance_name,case_folder,technique,run_id,status,status_code,runtime_sec,mip_gap,obj_val,obj_bound,nodes,num_vars,num_bin_vars,num_int_vars,num_cont_vars,num_constrs,num_nzs,buses,units,lines,reserves,time_steps,time_step_min,ptdf_nnz,lodf_nnz,startup_count,startup_cost,reserve_shortfall_mw,reserve_shortfall_penalty,base_overflow_mw,base_overflow_penalty,cont_overflow_mw,cont_overflow_penalty
2025-08-18 19:22:13,test/case14-sub-hourly,test/case14-sub-hourly,basic,1,OPTIMAL,2,0.017000,0.01736596,37456.981338,36806.504925,1,524,72,72,452,3148,10276,14,6,20,0,4,30,248,362,5.0000,5000.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000

```

### File: `src/data/output/test/case14/perf_test_case14_basic_01.csv`

```
timestamp,instance_name,case_folder,technique,run_id,status,status_code,runtime_sec,mip_gap,obj_val,obj_bound,nodes,num_vars,num_bin_vars,num_int_vars,num_cont_vars,num_constrs,num_nzs,buses,units,lines,reserves,time_steps,time_step_min,ptdf_nnz,lodf_nnz,startup_count,startup_cost,reserve_shortfall_mw,reserve_shortfall_penalty,base_overflow_mw,base_overflow_penalty,cont_overflow_mw,cont_overflow_penalty
2025-08-18 19:22:13,test/case14,test/case14,basic,1,OPTIMAL,2,0.020000,0.01736596,37456.981338,36806.504925,1,548,72,72,476,3172,10370,14,6,20,1,4,60,248,362,5.0000,5000.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000

```

### File: `src/data_preparation/__init__.py`

```
from src.data_preparation import data_structure
from src.data_preparation import download_data
from src.data_preparation import params
from src.data_preparation import ptdf_lodf
from src.data_preparation import read_data
from src.data_preparation import utils

```

### File: `src/data_preparation/data_structure.py`

```
# src/data_preparation/data_structure.py
from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional
import scipy.sparse as sp

Number = Union[int, float]
Series = List[Number]


@dataclass
class CostSegment:
    amount: Series
    cost: Series


@dataclass
class StartupCategory:
    delay_steps: int
    cost: float


@dataclass
class Bus:
    name: str
    index: int
    load: Series
    thermal_units: List["ThermalUnit"] = field(default_factory=list)
    price_sensitive_loads: List["PriceSensitiveLoad"] = field(default_factory=list)
    profiled_units: List["ProfiledUnit"] = field(default_factory=list)
    storage_units: List["StorageUnit"] = field(default_factory=list)


@dataclass
class Reserve:
    name: str
    type: str
    amount: Series
    thermal_units: List["ThermalUnit"]
    shortfall_penalty: float


@dataclass
class ThermalUnit:
    name: str
    bus: Bus
    max_power: Series
    min_power: Series
    must_run: Series
    min_power_cost: Series
    segments: List[CostSegment]
    min_up: int
    min_down: int
    ramp_up: float
    ramp_down: float
    startup_limit: float
    shutdown_limit: float
    initial_status: Optional[int]
    initial_power: Optional[float]
    startup_categories: List[StartupCategory]
    reserves: List[Reserve]
    commitment_status: List[Optional[bool]]


@dataclass
class ProfiledUnit:
    name: str
    bus: Bus
    min_power: Series
    max_power: Series
    cost: Series


@dataclass
class StorageUnit:
    name: str
    bus: Bus
    min_level: Series
    max_level: Series
    simultaneous: Series
    charge_cost: Series
    discharge_cost: Series
    charge_eff: Series
    discharge_eff: Series
    loss_factor: Series
    min_charge: Series
    max_charge: Series
    min_discharge: Series
    max_discharge: Series
    initial_level: float
    last_min: float
    last_max: float


@dataclass
class TransmissionLine:
    name: str
    index: int
    source: Bus
    target: Bus
    susceptance: float
    normal_limit: Series
    emergency_limit: Series
    flow_penalty: Series


@dataclass
class Contingency:
    name: str
    lines: List[TransmissionLine]
    units: List[ThermalUnit]


@dataclass
class PriceSensitiveLoad:
    name: str
    bus: Bus
    demand: Series
    revenue: Series


@dataclass
class UnitCommitmentScenario:
    name: str
    probability: float
    buses_by_name: Dict[str, Bus]
    buses: List[Bus]
    contingencies_by_name: Dict[str, Contingency]
    contingencies: List[Contingency]
    lines_by_name: Dict[str, TransmissionLine]
    lines: List[TransmissionLine]
    power_balance_penalty: Series
    price_sensitive_loads_by_name: Dict[str, PriceSensitiveLoad]
    price_sensitive_loads: List[PriceSensitiveLoad]
    reserves: List[Reserve]
    reserves_by_name: Dict[str, Reserve]
    time: int
    time_step: int
    thermal_units_by_name: Dict[str, ThermalUnit]
    thermal_units: List[ThermalUnit]
    profiled_units_by_name: Dict[str, ProfiledUnit]
    profiled_units: List[ProfiledUnit]
    storage_units_by_name: Dict[str, StorageUnit]
    storage_units: List[StorageUnit]
    isf: sp.spmatrix
    lodf: sp.spmatrix


@dataclass
class UnitCommitmentInstance:
    time: int
    scenarios: List[UnitCommitmentScenario]

    # convenient alias
    @property
    def deterministic(self) -> UnitCommitmentScenario:
        if len(self.scenarios) != 1:
            raise ValueError("Instance is stochastic; pick a scenario explicitly")
        return self.scenarios[0]

```

### File: `src/data_preparation/download_data.py`

```
# src/data_preparation/download_data.py
import requests
from pathlib import Path
from tqdm import tqdm


def download(url: str, dst: Path, chunk: int = 1 << 20) -> None:
    """Stream a file to *dst* with a progress bar."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            dst.open("wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True, disable=total == 0) as bar,
        ):
            for chunk_data in r.iter_content(chunk_size=chunk):
                size = fh.write(chunk_data)
                bar.update(size)

```

### File: `src/data_preparation/params.py`

```
from pathlib import Path
from gurobipy import GRB


class DataParams:
    """
    Central place for:
      - External URLs
      - Cache/log directories
      - Solver status mapping
      - Default values used by the loader when input fields are missing

    Notes on defaults
    - These defaults describe loader behavior when a specific field is missing.
      In many places (e.g., bus load, reserve amount), the loader still enforces
      presence of key fields and will raise if missing. Defaults listed here are
      therefore either:
        • documentation of expected scales, or
        • actual fallback values when the loader allows omission.
    """

    # Remote sources
    # Base URL hosting UnitCommitment.jl benchmark JSON instances
    INSTANCES_URL = "https://axavier.org/UnitCommitment.jl/0.4/instances"
    # Reference solutions URL (informational; not used programmatically here)
    REFERENCE_SOLUTIONS_URL = (
        "https://github.com/ANL-CEEESA/UnitCommitment.jl/tree/dev/src"
    )

    # Cache directory for downloaded instances: src/data/input/<name>.json.gz
    _CACHE = Path(__file__).resolve().parent.parent / "data" / "input"
    _CACHE.mkdir(parents=True, exist_ok=True)

    # Output directory for JSON solutions mirroring input hierarchy:
    # src/data/output/<name>.json (note: without .gz)
    _OUTPUT = Path(__file__).resolve().parent.parent / "data" / "output"
    _OUTPUT.mkdir(parents=True, exist_ok=True)

    # Intermediate directory (for ML artifacts, warm starts, etc.)
    _INTERMEDIATE = Path(__file__).resolve().parent.parent / "data" / "intermediate"
    _INTERMEDIATE.mkdir(parents=True, exist_ok=True)

    # Warm start directory under intermediate
    _WARM_START = _INTERMEDIATE / "warm_start"
    _WARM_START.mkdir(parents=True, exist_ok=True)

    # Logs directory for solution/verification: src/data/logs
    _LOGS = Path(__file__).resolve().parent.parent / "data" / "logs"
    _LOGS.mkdir(parents=True, exist_ok=True)

    # Solver status → string (minimal mapping we print in logs)
    SOLVER_STATUS_STR = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
    }

    # Time grid defaults
    # Default time step in minutes (if "Time step (min)" is missing)
    DEFAULT_TIME_STEP_MIN = 60

    # System-level penalty
    # Penalty for power balance mismatch ($/MW); used if missing in Parameters
    DEFAULT_POWER_BALANCE_PENALTY_USD_PER_MW = 1000

    # Bus defaults
    # Default bus load if a loader path were to allow a missing entry (we still require it)
    DEFAULT_BUS_LOAD_MW = 100

    # Reserve defaults
    # Default requirement (MW) if allowed to be omitted (we still require Amount (MW))
    DEFAULT_RESERVE_REQUIREMENT_MW = 0
    # Shortfall penalty ($/MW) if missing on a reserve product
    DEFAULT_RESERVE_SHORTFALL_PENALTY_USD_PER_MW = 1000

    # Thermal generator defaults
    # Startup categories (delays in hours and cost in $) — used if Startup delays/costs missing
    DEFAULT_STARTUP_DELAY_H = 0
    DEFAULT_STARTUP_COST_USD = 0.0
    # Fixed commitment status per time step; None means "free" (not fixed)
    DEFAULT_COMMITMENT_STATUS = None
    # Must-run default flag if missing per time step
    DEFAULT_MUST_RUN = False

    # Ramp and start/stop limits (MW) if missing
    DEFAULT_RAMP_UP_MW = 100
    DEFAULT_RAMP_DOWN_MW = 100
    DEFAULT_STARTUP_RAMP_LIMIT_MW = 100
    DEFAULT_SHUTDOWN_RAMP_LIMIT_MW = 100

    # Profiled/minimum-power-style inputs (used by profiled units or when needed)
    DEFAULT_MINIMUM_POWER_MW = 0
    DEFAULT_INCREMENTAL_COST_USD_PER_MW = 100
    DEFAULT_MIN_UPTIME_H = 0
    DEFAULT_MIN_DOWNTIME_H = 0

    # Transmission line defaults (MW and penalty $/MW); used if missing
    DEFAULT_LINE_NORMAL_LIMIT_MW = 300
    DEFAULT_LINE_EMERGENCY_LIMIT_MW = 400
    DEFAULT_LINE_FLOW_PENALTY_USD_PER_MW = 500.0

    # Price-sensitive load defaults (if missing)
    DEFAULT_PSL_DEMAND_MW = 1000
    DEFAULT_PSL_REVENUE_USD_PER_MW = 1000.0

    # Storage defaults (used when omitted)
    DEFAULT_STORAGE_MIN_LEVEL_MWH = 0.0
    DEFAULT_STORAGE_MAX_LEVEL_MWH = 1e4
    DEFAULT_STORAGE_CHARGE_COST_USD_PER_MW = DEFAULT_INCREMENTAL_COST_USD_PER_MW
    DEFAULT_STORAGE_DISCHARGE_COST_USD_PER_MW = 0.0
    DEFAULT_STORAGE_CHARGE_EFFICIENCY = 1.0
    DEFAULT_STORAGE_DISCHARGE_EFFICIENCY = 1.0
    DEFAULT_STORAGE_LOSS_FACTOR = 0.0
    DEFAULT_STORAGE_MIN_CHARGE_RATE_MW = 0.0
    DEFAULT_STORAGE_MAX_CHARGE_RATE_MW = None
    DEFAULT_STORAGE_MIN_DISCHARGE_RATE_MW = 0.0
    DEFAULT_STORAGE_MAX_DISCHARGE_RATE_MW = None
    DEFAULT_STORAGE_INITIAL_LEVEL_MWH = 0.0
    DEFAULT_STORAGE_LAST_PERIOD_MIN_LEVEL_MWH = 0.0
    DEFAULT_STORAGE_LAST_PERIOD_MAX_LEVEL_MWH = DEFAULT_STORAGE_MAX_LEVEL_MWH

```

### File: `src/data_preparation/prepare_data.py`

```

```

### File: `src/data_preparation/ptdf_lodf.py`

```
import numpy as np
from scipy.sparse import csr_matrix
from .data_structure import UnitCommitmentScenario


def compute_ptdf_lodf(scenario: UnitCommitmentScenario) -> None:
    """
    Compute PTDF (ISF) and LODF matrices from network topology and line susceptances.

    Conventions
    - scenario.buses have 1-based indices (b.index ∈ {1..N}).
    - A reference bus is chosen (default: first bus in scenario.buses by its index).
    - ISF shape is (n_lines, n_buses - 1): columns correspond to non-reference buses.
    - LODF shape is (n_lines, n_lines).
    - Orientation follows the line's source->target direction.
    """
    buses = scenario.buses
    lines = scenario.lines
    n_buses = len(buses)
    n_lines = len(lines)

    if n_buses == 0 or n_lines == 0:
        scenario.isf = csr_matrix((0, 0), dtype=float)
        scenario.lodf = csr_matrix((0, 0), dtype=float)
        return

    bus_indices_1b = sorted([b.index for b in buses])
    pos_by_bus1b = {bidx: (i0) for i0, bidx in enumerate(bus_indices_1b)}
    ref_bus_1b = bus_indices_1b[0]
    ref_pos = pos_by_bus1b[ref_bus_1b]

    B = np.zeros((n_buses, n_buses), dtype=float)
    for ln in lines:
        i0 = pos_by_bus1b[ln.source.index]
        j0 = pos_by_bus1b[ln.target.index]
        b = float(ln.susceptance)
        B[i0, i0] += b
        B[j0, j0] += b
        B[i0, j0] -= b
        B[j0, i0] -= b

    keep = [k for k in range(n_buses) if k != ref_pos]
    B_red = B[np.ix_(keep, keep)]

    try:
        X = np.linalg.inv(B_red)
    except np.linalg.LinAlgError:
        X = np.linalg.pinv(B_red)

    non_ref_positions = [k for k in range(n_buses) if k != ref_pos]
    col_by_pos = {pos: c for c, pos in enumerate(non_ref_positions)}

    isf = np.zeros((n_lines, n_buses - 1), dtype=float)
    for l_idx, ln in enumerate(lines):
        i0 = pos_by_bus1b[ln.source.index]
        j0 = pos_by_bus1b[ln.target.index]
        b_l = float(ln.susceptance)
        row = np.zeros(n_buses - 1, dtype=float)
        if i0 != ref_pos:
            row += b_l * X[col_by_pos[i0], :]
        if j0 != ref_pos:
            row -= b_l * X[col_by_pos[j0], :]
        isf[l_idx, :] = row

    isf_full = np.zeros((n_lines, n_buses), dtype=float)
    for c, pos in enumerate(non_ref_positions):
        isf_full[:, pos] = isf[:, c]

    lodf = np.zeros((n_lines, n_lines), dtype=float)
    for c_idx, c_line in enumerate(lines):
        mpos = pos_by_bus1b[c_line.source.index]
        npos = pos_by_bus1b[c_line.target.index]
        ptdf_c_mn = isf_full[c_idx, mpos] - isf_full[c_idx, npos]
        denom = 1.0 - ptdf_c_mn
        if abs(denom) < 1e-9:
            continue
        for l_idx in range(n_lines):
            ptdf_l_mn = isf_full[l_idx, mpos] - isf_full[l_idx, npos]
            lodf[l_idx, c_idx] = ptdf_l_mn / denom

    # Set diagonal to -1 (outaged line post-flow is zero)
    np.fill_diagonal(lodf, -1.0)

    # Numerics: drop extremely small entries
    tol = 1e-10
    isf[np.abs(isf) < tol] = 0.0
    lodf[np.abs(lodf) < tol] = 0.0

    scenario.isf = csr_matrix(isf)  # shape (n_lines, n_buses - 1)
    scenario.lodf = csr_matrix(lodf)  # shape (n_lines, n_lines)
    scenario.__dict__["ptdf_ref_bus_index"] = ref_bus_1b

```

### File: `src/data_preparation/read_data.py`

```
from typing import Union, Sequence, Optional
import gzip
import json
import re
from pathlib import Path

from src.data_preparation.params import DataParams
from src.data_preparation.download_data import download
from src.data_preparation.data_structure import (
    UnitCommitmentInstance,
    UnitCommitmentScenario,
)
from src.data_preparation.utils import (
    from_json,
    repair_scenario_names_and_probabilities,
    migrate,
)


def _sanitize_identifier(s: str) -> str:
    """
    Turn an instance name like 'matpower/case300/2017-06-24'
    into a filesystem/log-friendly id: 'matpower_case300_2017-06-24'.
    """
    s = s.strip().strip("/\\")
    s = s.replace("\\", "/")
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")


def read_benchmark(name: str, *, quiet: bool = False) -> UnitCommitmentInstance:
    """
    Download (if necessary) a benchmark instance and load it.

    Example
    -------
    inst = read_benchmark("matpower/case3375wp/2017-02-01")
    """
    gz_name = f"{name}.json.gz"
    local_path = DataParams._CACHE / gz_name
    url = f"{DataParams.INSTANCES_URL}/{gz_name}"

    if not local_path.is_file():
        if not quiet:
            print(f"Downloading  {url}")
        download(url, local_path)

    instance = _read(str(local_path), scenario_id_hint=name)

    print(f"→ Loaded instance '{name}' with {len(instance.scenarios)} scenarios.")
    print("Path to instance:", local_path)

    return instance


def _read(
    path_or_paths: Union[str, Sequence[str]],
    scenario_id_hint: Optional[str] = None,
) -> UnitCommitmentInstance:
    """
    Generic loader.  Accepts:
      • single path (JSON or JSON.GZ) ➜ deterministic instance
      • list / tuple of paths           ➜ stochastic instance

    scenario_id_hint:
      When a single path is passed (deterministic case), use this hint to
      label the scenario name in a log-friendly way, so logs clearly show
      which dataset was solved.
    """
    if isinstance(path_or_paths, (list, tuple)):
        scenarios = [_read_scenario(p) for p in path_or_paths if isinstance(p, str)]
        repair_scenario_names_and_probabilities(scenarios, list(path_or_paths))
    else:
        scenarios = [_read_scenario(path_or_paths)]
        # Name scenario using the original "name" hint if available; otherwise
        # fall back to a sanitized path-based id.
        if scenario_id_hint:
            scenarios[0].name = _sanitize_identifier(scenario_id_hint)
        else:
            try:
                rel = (
                    Path(path_or_paths)
                    .resolve()
                    .relative_to(DataParams._CACHE.resolve())
                )
                base = rel.as_posix()
                if base.endswith(".json.gz"):
                    base = base[: -len(".json.gz")]
                elif base.endswith(".json"):
                    base = base[: -len(".json")]
                scenarios[0].name = _sanitize_identifier(base)
            except Exception:
                scenarios[0].name = "scenario"
        scenarios[0].probability = 1.0

    return UnitCommitmentInstance(time=scenarios[0].time, scenarios=scenarios)


def _read_scenario(path: str) -> UnitCommitmentScenario:
    raw = _read_json(path)
    migrate(raw)
    return from_json(raw)


def _read_json(path: str) -> dict:
    """Open JSON or JSON.GZ transparently."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

```

### File: `src/data_preparation/utils.py`

```
from pathlib import Path
from typing import List, Dict

import numpy as np
from scipy import sparse
import logging

from src.data_preparation.data_structure import (
    UnitCommitmentScenario,
    ThermalUnit,
    ProfiledUnit,
    StorageUnit,
    PriceSensitiveLoad,
    TransmissionLine,
    Contingency,
    Bus,
    Reserve,
    CostSegment,
    StartupCategory,
)

from .ptdf_lodf import compute_ptdf_lodf

from .params import DataParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def ts(x, T, default=None):
    return _timeseries(x, T, default=default)


def _timeseries(val, T: int, *, default=None):
    """
    * if val is missing ➜ default
    * if val is array  ➜ keep
    * if val is scalar ➜ replicate T times
    """
    if val is None:
        return default if default is not None else [None] * T
    if isinstance(val, list):
        if len(val) != T:
            raise ValueError(f"Time-series length {len(val)} does not match T={T}")
        return val
    return [val] * T


def _scalar(val, default=None):
    return default if val is None else val


def _parse_version(v):
    """Return (major, minor) tuple; treat malformed strings as (0, 0)."""
    try:
        return tuple(int(x) for x in str(v).split(".")[:2])
    except Exception:
        return (0, 0)


def migrate(json_: dict) -> None:
    """
    Bring legacy (< 0.4) files up to date:
        * v0.2 → v0.3:  restructure reserves & generator flags
        * v0.3 → v0.4:  ensure every generator has `"Type": "Thermal"`
    """
    params = json_.get("Parameters", {})
    ver_raw = params.get("Version")
    if ver_raw is None:
        raise ValueError(
            "Input file has no Parameters['Version'] entry – please add it "
            '(e.g. {"Parameters": {"Version": "0.3"}}).'
        )

    ver = _parse_version(ver_raw)
    if ver < (0, 3):
        _migrate_to_v03(json_)
    if ver < (0, 4):
        _migrate_to_v04(json_)


def _migrate_to_v03(json_: dict) -> None:
    """Match Julia’s _migrate_to_v03: create r1 spinning reserve, map flags."""
    reserves = json_.get("Reserves")
    if reserves and "Spinning (MW)" in reserves:
        amount = reserves["Spinning (MW)"]
        json_["Reserves"] = {
            "r1": {
                "Type": "spinning",
                "Amount (MW)": amount,
            }
        }
        for gen in json_.get("Generators", {}).values():
            if gen.get("Provides spinning reserves?") is True:
                gen["Reserve eligibility"] = ["r1"]


def _migrate_to_v04(json_: dict) -> None:
    """Match Julia’s _migrate_to_v04: default missing types to Thermal."""
    for gen in json_.get("Generators", {}).values():
        gen.setdefault("Type", "Thermal")


def from_json(j: dict) -> UnitCommitmentScenario:
    # -- Time grid ---------------------------------------------------------- #
    par = j["Parameters"]
    time_horizon = (
        par.get("Time horizon (min)")
        or par.get("Time (h)")
        or par.get("Time horizon (h)")
    )
    if time_horizon is None:
        raise ValueError("Missing parameter: Time horizon")
    if "Time (h)" in par or "Time horizon (h)" in par:
        time_horizon *= 60  # convert hours → minutes

    time_horizon = int(time_horizon)
    time_step = int(
        _scalar(par.get("Time step (min)"), default=DataParams.DEFAULT_TIME_STEP_MIN)
    )
    if 60 % time_step or time_horizon % time_step:
        raise ValueError("Time step must divide 60 and the horizon")

    time_multiplier = 60 // time_step
    T = time_horizon // time_step

    # ---------------------------------------------------------------------- #
    #  Look-up tables                                                        #
    # ---------------------------------------------------------------------- #
    buses: List[Bus] = []
    lines: List[TransmissionLine] = []
    thermal_units: List[ThermalUnit] = []
    profiled_units: List[ProfiledUnit] = []
    storage_units: List[StorageUnit] = []
    reserves: List[Reserve] = []
    contingencies: List[Contingency] = []
    loads: List[PriceSensitiveLoad] = []

    name_to_bus: Dict[str, Bus] = {}
    name_to_line: Dict[str, TransmissionLine] = {}
    name_to_unit: Dict[str, ThermalUnit] = {}  # Only thermal for contingencies
    name_to_reserve: Dict[str, Reserve] = {}

    def ts(x, *, default=None):
        return _timeseries(x, T, default=default)

    # ---------------------------------------------------------------------- #
    #  Penalties                                                             #
    # ---------------------------------------------------------------------- #
    power_balance_penalty = ts(
        par.get("Power balance penalty ($/MW)"),
        default=[DataParams.DEFAULT_POWER_BALANCE_PENALTY_USD_PER_MW] * T,
    )

    # ---------------------------------------------------------------------- #
    #  Buses                                                                 #
    # ---------------------------------------------------------------------- #
    for idx, (bname, bdict) in enumerate(j.get("Buses", {}).items(), start=1):
        if "Load (MW)" not in bdict:
            raise ValueError(f"Bus '{bname}' missing 'Load (MW)'")
        bus = Bus(
            name=bname,
            index=idx,
            load=ts(
                bdict.get("Load (MW)"), default=[DataParams.DEFAULT_BUS_LOAD_MW] * T
            ),
        )
        name_to_bus[bname] = bus
        buses.append(bus)

    # ---------------------------------------------------------------------- #
    #  Reserves                                                              #
    # ---------------------------------------------------------------------- #
    if "Reserves" in j:
        for rname, rdict in j["Reserves"].items():
            if "Amount (MW)" not in rdict:
                raise ValueError(f"Reserve '{rname}' missing 'Amount (MW)'")
            r_type = rdict.get("Type", "spinning").lower()
            if r_type != "spinning":
                raise ValueError(f"Unsupported reserve type '{r_type}' for '{rname}'")
            r = Reserve(
                name=rname,
                type=r_type,
                amount=ts(
                    rdict.get("Amount (MW)"),
                    default=[DataParams.DEFAULT_RESERVE_REQUIREMENT_MW] * T,
                ),
                thermal_units=[],
                shortfall_penalty=_scalar(
                    rdict.get("Shortfall penalty ($/MW)"),
                    default=DataParams.DEFAULT_RESERVE_SHORTFALL_PENALTY_USD_PER_MW,
                ),
            )
            name_to_reserve[rname] = r
            reserves.append(r)

    # ---------------------------------------------------------------------- #
    #  Generators                                                            #
    # ---------------------------------------------------------------------- #
    for uname, udict in j.get("Generators", {}).items():
        utype = udict.get("Type", "Thermal")
        bus_name = udict["Bus"]
        if bus_name not in name_to_bus:
            raise ValueError(f"Unknown bus '{bus_name}' for generator '{uname}'")
        bus = name_to_bus[bus_name]

        if utype.lower() == "thermal":
            # Production cost curve validation and parsing
            if (
                "Production cost curve (MW)" not in udict
                or "Production cost curve ($)" not in udict
            ):
                raise ValueError(f"Generator '{uname}' missing production cost curve")
            curve_mw_list = udict["Production cost curve (MW)"]
            curve_cost_list = udict["Production cost curve ($)"]

            K = len(curve_mw_list)
            if len(curve_cost_list) != K:
                raise ValueError(
                    f"Generator '{uname}' production cost curve lengths mismatch (MW: {len(curve_mw_list)}, $: {len(curve_cost_list)})"
                )
            if K == 0:
                raise ValueError(f"Generator '{uname}' has no break-points (K=0)")

            # Convert to time series arrays
            curve_mw = np.column_stack([ts(curve_mw_list[k]) for k in range(K)])
            curve_cost = np.column_stack([ts(curve_cost_list[k]) for k in range(K)])

            # Validate monotonicity
            if not np.all(np.diff(curve_mw, axis=1) > 0):
                raise ValueError(
                    f"Generator '{uname}' production cost curve MW must be strictly increasing"
                )
            if not np.all(np.diff(curve_cost, axis=1) >= 0):
                raise ValueError(
                    f"Generator '{uname}' production cost curve $ must be non-decreasing"
                )

            min_power = curve_mw[:, 0].tolist()
            max_power = curve_mw[:, -1].tolist()
            min_power_cost = curve_cost[:, 0].tolist()

            # Build segmented increments above minimum
            segments: List[CostSegment] = []
            if K > 1:
                for k in range(1, K):
                    amount = curve_mw[:, k] - curve_mw[:, k - 1]
                    # slope = (C_k - C_{k-1}) / (P_k - P_{k-1})
                    with np.errstate(divide="ignore", invalid="ignore"):
                        marginal = np.divide(
                            curve_cost[:, k] - curve_cost[:, k - 1],
                            amount,
                            out=np.zeros_like(amount, dtype=float),
                            where=amount != 0,
                        )
                    segments.append(
                        CostSegment(amount=amount.tolist(), cost=marginal.tolist())
                    )
            # If K == 1, there are no variable segments; only min_power_cost applies when committed.

            # Startup categories validation and parsing
            delays = udict.get(
                "Startup delays (h)", [DataParams.DEFAULT_STARTUP_DELAY_H]
            )
            scost = udict.get(
                "Startup costs ($)", [DataParams.DEFAULT_STARTUP_COST_USD]
            )
            if len(delays) != len(scost):
                raise ValueError(f"Startup delays/costs mismatch for '{uname}'")
            startup_categories = sorted(
                [
                    StartupCategory(
                        delay_steps=int(delays[k] * time_multiplier), cost=scost[k]
                    )
                    for k in range(len(delays))
                ],
                key=lambda cat: cat.delay_steps,
            )

            # Reserve eligibility
            unit_reserves = [
                name_to_reserve[n]
                for n in udict.get("Reserve eligibility", [])
                if n in name_to_reserve
            ]

            # Initial conditions validation
            init_p = udict.get("Initial power (MW)")
            init_s = udict.get("Initial status (h)")
            if init_p is not None and init_s is None:
                raise ValueError(f"{uname} has initial power but no status")
            if init_s is not None:
                init_s = int(init_s * time_multiplier)
                if init_p and init_p > 0 and init_s <= 0:
                    raise ValueError(f"{uname} initial power >0 but status <=0")

            # Ramp and limits validation
            ramp_up = _scalar(
                udict.get("Ramp up limit (MW)"), DataParams.DEFAULT_RAMP_UP_MW
            )
            ramp_down = _scalar(
                udict.get("Ramp down limit (MW)"), DataParams.DEFAULT_RAMP_DOWN_MW
            )
            if ramp_up <= 0 or ramp_down <= 0:
                raise ValueError(f"Invalid ramp limits for '{uname}'")

            commitment_status = ts(
                udict.get("Commitment status"),
                default=[DataParams.DEFAULT_COMMITMENT_STATUS] * T,
            )

            # Robustly read startup/shutdown limits; ensure they are at least Pmin_max
            def _get_first(d, keys, default=None):
                for k in keys:
                    if k in d:
                        return d[k]
                return default

            pmin_max = float(max(min_power)) if min_power else 0.0

            su_raw = _get_first(
                udict,
                ["Startup limit (MW)", "Startup ramp limit (MW)"],
                None,
            )
            sd_raw = _get_first(
                udict,
                ["Shutdown limit (MW)", "Shutdown ramp limit (MW)"],
                None,
            )

            # Defaults: if missing, set to at least Pmin_max to guarantee feasible start/stop
            startup_limit = su_raw if su_raw is not None else pmin_max
            shutdown_limit = sd_raw if sd_raw is not None else pmin_max

            # Enforce minimum feasibility: SU, SD >= max_t Pmin
            startup_limit = max(float(startup_limit), pmin_max)
            shutdown_limit = max(float(shutdown_limit), pmin_max)

            tu = ThermalUnit(
                name=uname,
                bus=bus,
                max_power=max_power,
                min_power=min_power,
                must_run=ts(udict.get("Must run?", [DataParams.DEFAULT_MUST_RUN] * T)),
                min_power_cost=min_power_cost,
                segments=segments,
                min_up=int(
                    _scalar(
                        udict.get("Minimum uptime (h)"), DataParams.DEFAULT_MIN_UPTIME_H
                    )
                    * time_multiplier
                ),
                min_down=int(
                    _scalar(
                        udict.get("Minimum downtime (h)"),
                        DataParams.DEFAULT_MIN_DOWNTIME_H,
                    )
                    * time_multiplier
                ),
                ramp_up=ramp_up,
                ramp_down=ramp_down,
                startup_limit=startup_limit,
                shutdown_limit=shutdown_limit,
                initial_status=init_s,
                initial_power=init_p,
                startup_categories=startup_categories,
                reserves=unit_reserves,
                commitment_status=commitment_status,
            )
            bus.thermal_units.append(tu)
            thermal_units.append(tu)
            name_to_unit[uname] = tu
            for r in unit_reserves:
                r.thermal_units.append(tu)

        elif utype.lower() == "profiled":
            pu = ProfiledUnit(
                name=uname,
                bus=bus,
                min_power=ts(
                    _scalar(
                        udict.get("Minimum power (MW)"),
                        DataParams.DEFAULT_MINIMUM_POWER_MW,
                    )
                ),
                max_power=ts(udict.get("Maximum power (MW)")),
                cost=ts(
                    udict.get("Cost ($/MW)"),
                    default=[DataParams.DEFAULT_INCREMENTAL_COST_USD_PER_MW] * T,
                ),
            )
            bus.profiled_units.append(pu)
            profiled_units.append(pu)

        else:
            raise ValueError(f"Unit {uname} has invalid type '{utype}'")

    # ---------------------------------------------------------------------- #
    #  Lines                                                                 #
    # ---------------------------------------------------------------------- #
    if "Transmission lines" in j:
        for idx, (lname, ldict) in enumerate(j["Transmission lines"].items(), start=1):
            source_name = ldict["Source bus"]
            target_name = ldict["Target bus"]
            if source_name not in name_to_bus or target_name not in name_to_bus:
                raise ValueError(f"Unknown bus in line '{lname}'")
            source = name_to_bus[source_name]
            target = name_to_bus[target_name]
            if "Susceptance (S)" not in ldict:
                raise ValueError(f"Line '{lname}' missing susceptance")
            line = TransmissionLine(
                name=lname,
                index=idx,
                source=source,
                target=target,
                susceptance=float(ldict["Susceptance (S)"]),
                normal_limit=ts(
                    ldict.get("Normal flow limit (MW)"),
                    default=[DataParams.DEFAULT_LINE_NORMAL_LIMIT_MW] * T,
                ),
                emergency_limit=ts(
                    ldict.get("Emergency flow limit (MW)"),
                    default=[DataParams.DEFAULT_LINE_EMERGENCY_LIMIT_MW] * T,
                ),
                flow_penalty=ts(
                    ldict.get("Flow limit penalty ($/MW)"),
                    default=[DataParams.DEFAULT_LINE_FLOW_PENALTY_USD_PER_MW] * T,
                ),
            )
            lines.append(line)
            name_to_line[lname] = line

    # ---------------------------------------------------------------------- #
    #  Contingencies                                                         #
    # ---------------------------------------------------------------------- #
    if "Contingencies" in j:
        for cname, cdict in j["Contingencies"].items():
            affected_lines = [
                name_to_line[line]
                for line in cdict.get("Affected lines", [])
                if line in name_to_line
            ]
            affected_units = [
                name_to_unit[u]
                for u in cdict.get("Affected units", [])
                if u in name_to_unit
            ]
            cont = Contingency(name=cname, lines=affected_lines, units=affected_units)
            contingencies.append(cont)

    # ---------------------------------------------------------------------- #
    #  Price-sensitive loads                                                 #
    # ---------------------------------------------------------------------- #
    if "Price-sensitive loads" in j:
        for lname, ldict in j["Price-sensitive loads"].items():
            bus_name = ldict["Bus"]
            if bus_name not in name_to_bus:
                raise ValueError(f"Unknown bus '{bus_name}' for load '{lname}'")
            bus = name_to_bus[bus_name]
            load = PriceSensitiveLoad(
                name=lname,
                bus=bus,
                demand=ts(
                    ldict.get("Demand (MW)"),
                    default=[DataParams.DEFAULT_PSL_DEMAND_MW] * T,
                ),
                revenue=ts(
                    ldict.get("Revenue ($/MW)"),
                    default=[DataParams.DEFAULT_PSL_REVENUE_USD_PER_MW] * T,
                ),
            )
            loads.append(load)
            bus.price_sensitive_loads.append(load)

    # ---------------------------------------------------------------------- #
    #  Storage units                                                         #
    # ---------------------------------------------------------------------- #
    if "Storage units" in j:
        for sname, sdict in j["Storage units"].items():
            bus_name = sdict["Bus"]
            if bus_name not in name_to_bus:
                raise ValueError(f"Unknown bus '{bus_name}' for storage '{sname}'")
            bus = name_to_bus[bus_name]
            min_level = ts(
                _scalar(
                    sdict.get("Minimum level (MWh)"),
                    DataParams.DEFAULT_STORAGE_MIN_LEVEL_MWH,
                )
            )
            max_level = ts(sdict.get("Maximum level (MWh)"))
            su = StorageUnit(
                name=sname,
                bus=bus,
                min_level=min_level,
                max_level=max_level,
                simultaneous=ts(
                    _scalar(
                        sdict.get("Allow simultaneous charging and discharging"), True
                    )
                ),
                charge_cost=ts(
                    sdict.get("Charge cost ($/MW)"),
                    default=[DataParams.DEFAULT_STORAGE_CHARGE_COST_USD_PER_MW] * T,
                ),
                discharge_cost=ts(
                    sdict.get("Discharge cost ($/MW)"),
                    default=[DataParams.DEFAULT_STORAGE_DISCHARGE_COST_USD_PER_MW] * T,
                ),
                charge_eff=ts(
                    _scalar(
                        sdict.get("Charge efficiency"),
                        DataParams.DEFAULT_STORAGE_CHARGE_EFFICIENCY,
                    )
                ),
                discharge_eff=ts(
                    _scalar(
                        sdict.get("Discharge efficiency"),
                        DataParams.DEFAULT_STORAGE_DISCHARGE_EFFICIENCY,
                    )
                ),
                loss_factor=ts(
                    _scalar(
                        sdict.get("Loss factor"), DataParams.DEFAULT_STORAGE_LOSS_FACTOR
                    )
                ),
                min_charge=ts(
                    _scalar(
                        sdict.get("Minimum charge rate (MW)"),
                        DataParams.DEFAULT_STORAGE_MIN_CHARGE_RATE_MW,
                    )
                ),
                max_charge=ts(sdict.get("Maximum charge rate (MW)")),
                min_discharge=ts(
                    _scalar(
                        sdict.get("Minimum discharge rate (MW)"),
                        DataParams.DEFAULT_STORAGE_MIN_DISCHARGE_RATE_MW,
                    )
                ),
                max_discharge=ts(sdict.get("Maximum discharge rate (MW)")),
                initial_level=_scalar(
                    sdict.get("Initial level (MWh)"),
                    DataParams.DEFAULT_STORAGE_INITIAL_LEVEL_MWH,
                ),
                last_min=_scalar(
                    sdict.get("Last period minimum level (MWh)"), min_level[-1]
                ),
                last_max=_scalar(
                    sdict.get("Last period maximum level (MWh)"), max_level[-1]
                ),
            )
            storage_units.append(su)
            bus.storage_units.append(su)

    # ---------------------------------------------------------------------- #
    #  Sparse matrices defaults                                              #
    # ---------------------------------------------------------------------- #
    isf = sparse.csr_matrix((len(lines), len(buses) - 1 if buses else 0), dtype=float)
    lodf = sparse.csr_matrix((len(lines), len(lines)), dtype=float)

    scenario = UnitCommitmentScenario(
        name=_scalar(par.get("Scenario name"), ""),
        probability=float(_scalar(par.get("Scenario weight"), 1)),
        buses_by_name={b.name: b for b in buses},
        buses=buses,
        contingencies_by_name={c.name: c for c in contingencies},
        contingencies=contingencies,
        lines_by_name={line.name: line for line in lines},
        lines=lines,
        power_balance_penalty=power_balance_penalty,
        price_sensitive_loads_by_name={pl.name: pl for pl in loads},
        price_sensitive_loads=loads,
        reserves=reserves,
        reserves_by_name=name_to_reserve,
        time=T,
        time_step=time_step,
        thermal_units_by_name={tu.name: tu for tu in thermal_units},
        thermal_units=thermal_units,
        profiled_units_by_name={pu.name: pu for pu in profiled_units},
        profiled_units=profiled_units,
        storage_units_by_name={su.name: su for su in storage_units},
        storage_units=storage_units,
        isf=isf,
        lodf=lodf,
    )

    _repair(scenario)
    print(
        f"Parsed scenario: {len(buses)} buses, {len(thermal_units)} thermal units, {len(lines)} lines, {len(reserves)} reserves"
    )
    return scenario


def _repair(scenario: UnitCommitmentScenario) -> None:
    """
    • fills commitment_status for must-run units
    • builds ISF/LODF if lines present
    """
    for gen in scenario.thermal_units:
        for t, must_run_flag in enumerate(gen.must_run):
            if must_run_flag:
                gen.commitment_status[t] = True

    if scenario.lines:
        compute_ptdf_lodf(scenario)


def repair_scenario_names_and_probabilities(
    scenarios: List[UnitCommitmentScenario], paths: List[str]
) -> None:
    """Normalize names and probabilities so they sum to 1."""
    total = sum(sc.probability for sc in scenarios)
    for sc, p in zip(scenarios, paths):
        if not sc.name:
            sc.name = Path(p).stem.split(".")[0]
        sc.probability /= total

```

### File: `src/ml_models/__init__.py`

```
from src.ml_models.warm_start import WarmStartProvider

```

### File: `src/ml_models/fix_warm_start.py`

```
"""
Warm-start "fixer" that repairs a warm JSON without using a solver (no Gurobi).

Goal
- Start from a warm-start JSON produced from a neighbor solution (k-NN).
- Make it solver-friendly and as-feasible-as-possible by:
  • fixing commitment to enforce must-run and min up/down (given initial status),
  • enforcing ramp limits (with startup/shutdown limits) on total generation,
  • clipping segment power to available segment capacity when committed,
  • balancing total production to match total load at each time step exactly
    (this addresses power_balance equalities),
  • setting reserves to a safe feasible pattern (zero provision with shortfall covering requirement),
  • recomputing base-case line flows via PTDF and setting overflow slacks.

Notes
- Power balance (system equality) is enforced in this repair by construction,
  provided that the load lies within the system's feasible envelope
  sum_i p_lb_i(t) <= Load(t) <= sum_i p_ub_i(t) for each t,
  where p_lb/p_ub include min output and ramp/SU/SD limits.
  If the envelope does not contain the load, exact balancing is impossible for that commitment,
  and we fall back to the nearest feasible bound (p_lb or p_ub). We report any residual gap.
- Min up/down and ramp feasibility are enforced by construction (lock-based commitment repair and
  ramp-aware bounds for p_t).
- Reserves are set conservatively: provision r[k,g,t] = 0 for all, shortfall = requirement,
  which always satisfies reserve constraints (feasible; not optimal).
- Line limits are respected by setting overflow slacks (feasible via slacks).
- No Gurobi import is used anywhere in this fixer.

Usage (CLI):
  python -m src.ml_models.fix_warm_start --instance matpower/case57/2017-06-24

Optional flags:
  --warm-file PATH       Use an existing warm JSON. If omitted, try to find the default
                         warm_<instance>.json. If missing, auto-generate via neighbor DB.
  --require-pretrained   Fail if no pre-trained index is found (when auto-generating).
  --use-train-db         Restrict neighbor search to the training split (when auto-generating).

This writes:
  src/data/intermediate/warm_start/warm_fixed_<instance>.json
and prints a short report (balance feasibility, max balance gap, etc.).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math

from src.data_preparation.params import DataParams
from src.data_preparation.read_data import read_benchmark


# ------------------------------ helpers ------------------------------ #


def _sanitize_name(s: str) -> str:
    s = (s or "").strip().strip("/\\").replace("\\", "/")
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    t = "".join(out)
    while "__" in t:
        t = t.replace("__", "_")
    return t.strip("_").lower()


def _warm_path_for_instance(instance_name: str) -> Path:
    tag = _sanitize_name(instance_name)
    return (DataParams._WARM_START / f"warm_{tag}.json").resolve()


def fixed_warm_path_for_instance(instance_name: str) -> Path:
    tag = _sanitize_name(instance_name)
    return (DataParams._WARM_START / f"warm_fixed_{tag}.json").resolve()


def _ensure_list_length(vals, T: int, pad_with_last: bool = True, default=0):
    if vals is None:
        return [default] * T
    vals = list(vals)
    if len(vals) < T:
        pad_val = vals[-1] if (pad_with_last and vals) else default
        vals = vals + [pad_val] * (T - len(vals))
    elif len(vals) > T:
        vals = vals[:T]
    return vals


def _b01(x) -> int:
    try:
        return 1 if float(x) >= 0.5 else 0
    except Exception:
        return 0


def _startup_shutdown_from_commit(
    commit: List[int], initial_u: int
) -> Tuple[List[int], List[int]]:
    T = len(commit)
    v = [0] * T
    w = [0] * T
    prev = int(initial_u)
    for t in range(T):
        u = int(commit[t])
        if u > prev:
            v[t] = 1
        elif u < prev:
            w[t] = 1
        prev = u
    return v, w


def _enforce_min_up_down(
    u_raw: List[int],
    must_run: List[bool],
    min_up: int,
    min_down: int,
    initial_status_steps: Optional[int],
) -> List[int]:
    """
    Return a commitment series that enforces:
      - must_run[t] -> 1
      - min up/down constraints using a simple lock mechanism
      - initial_status boundary condition

    Lock policy:
      - When we change state at t (0->1 or 1->0), lock the next (L-1) periods to that state,
        where L is min_up for ON and min_down for OFF.
      - If initial_status is provided and its absolute value is less than L for the
        corresponding state, start with a lock for the remaining steps.
    """
    T = len(u_raw)
    u = [1 if must_run[t] or _b01(u_raw[t]) else 0 for t in range(T)]

    # Initial state & initial lock from initial_status
    if initial_status_steps is None or initial_status_steps == 0:
        prev_state = u[0]
        lock_remaining = 0
        locked_state = prev_state
    else:
        if initial_status_steps > 0:
            prev_state = 1
            need = max(0, min_up - int(initial_status_steps))
            lock_remaining = need
            locked_state = 1
        else:
            prev_state = 0
            s_off = -int(initial_status_steps)
            need = max(0, min_down - s_off)
            lock_remaining = need
            locked_state = 0

    u_fixed: List[int] = [0] * T

    for t in range(T):
        desired = u[t]

        # Apply lock first
        if lock_remaining > 0:
            desired = locked_state

        # Must-run overrides desired
        if must_run[t]:
            desired = 1
            # starting an ON run implies lock for min_up-1 (next periods)
            lock_remaining = max(lock_remaining, max(0, min_up - 1))
            locked_state = 1

        # Detect change vs previous state
        if desired != prev_state:
            # New state implies starting a new lock
            if desired == 1:
                lock_remaining = max(lock_remaining, max(0, min_up - 1))
                locked_state = 1
            else:
                lock_remaining = max(lock_remaining, max(0, min_down - 1))
                locked_state = 0

        u_fixed[t] = desired
        prev_state = desired
        lock_remaining = max(0, lock_remaining - 1)

    return u_fixed


def _add_to_segments(row: List[float], caps: List[float], add: float) -> float:
    """
    Add energy 'add' to row subject to per-segment caps (row[s] <= caps[s]).
    Returns leftover that could not be added (0 if fully added).
    """
    remaining = float(add)
    nS = len(row)
    for s in range(nS):
        if remaining <= 1e-12:
            break
        cap_s = max(0.0, float(caps[s]))
        cur = max(0.0, float(row[s]))
        free = max(0.0, cap_s - cur)
        if free <= 0.0:
            continue
        take = min(free, remaining)
        row[s] = cur + take
        remaining -= take
    return remaining


def _remove_from_segments(row: List[float], remove: float) -> float:
    """
    Remove energy 'remove' from row (prefer removing from higher segments first).
    Returns leftover that could not be removed (0 if fully removed).
    """
    remaining = float(remove)
    nS = len(row)
    for s in reversed(range(nS)):
        if remaining <= 1e-12:
            break
        cur = max(0.0, float(row[s]))
        if cur <= 0.0:
            continue
        take = min(cur, remaining)
        row[s] = cur - take
        remaining -= take
    return remaining


# --------------------------- core fixing logic --------------------------- #


def _fix_generators_and_reserves(
    scenario, warm: Dict
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """
    Fix generator commitment, segment power with capacity and ramping, balance to exact load,
    and set a safe reserve pattern.

    Returns
    -------
    (gen_out, reserves_out, p_total, p_above_min, balance_info)
      - gen_out:  dict[name] -> {"commit": [...], "segment_power": [[...], ...]}
      - reserves_out: dict[rname] -> provided_by_gen, total_provided, requirement, shortfall
      - p_total: dict[name] -> [p_t]
      - p_above_min: dict[name] -> [above_min_t]
      - balance_info: {"feasible": bool, "max_abs_gap": float, "gaps": List[float]}
    """
    sc = scenario
    T = sc.time

    # System total load per time
    total_load = [sum(float(b.load[t]) for b in sc.buses) for t in range(T)]

    warm_gen = warm.get("generators", {}) or {}
    gen_out: Dict[str, Dict] = {}
    p_total: Dict[str, List[float]] = {}
    p_above_min: Dict[str, List[float]] = {}

    # 1) Fix commitment with must-run + min up/down
    for gen in sc.thermal_units:
        gw = warm_gen.get(gen.name, {}) or {}
        u_raw = _ensure_list_length(gw.get("commit", []), T, True, 0)
        u_raw = [_b01(x) for x in u_raw]
        must = [bool(x) for x in gen.must_run]
        u_fixed = _enforce_min_up_down(
            u_raw=u_raw,
            must_run=must,
            min_up=int(getattr(gen, "min_up", 0) or 0),
            min_down=int(getattr(gen, "min_down", 0) or 0),
            initial_status_steps=getattr(gen, "initial_status", None),
        )
        gen_out[gen.name] = {
            "commit": u_fixed,
            "segment_power": [
                [0.0] * (len(gen.segments) if gen.segments else 0) for _ in range(T)
            ],
        }

    # 2) Compute startup/shutdown from commitment
    start_flags: Dict[str, List[int]] = {}
    stop_flags: Dict[str, List[int]] = {}
    init_u_map: Dict[str, int] = {}
    for gen in sc.thermal_units:
        init_u = 1 if (gen.initial_status is not None and gen.initial_status > 0) else 0
        init_u_map[gen.name] = init_u
        v, w = _startup_shutdown_from_commit(gen_out[gen.name]["commit"], init_u)
        start_flags[gen.name] = v
        stop_flags[gen.name] = w

    # 3) Forward pass per time: ramp-aware feasible ranges and balancing to exact total load
    balance_gaps: List[float] = [0.0] * T
    feasible_all = True

    # Keep p_prev and u_prev per generator
    p_prev_map: Dict[str, float] = {}
    u_prev_map: Dict[str, int] = {}
    for gen in sc.thermal_units:
        p_prev_map[gen.name] = (
            float(gen.initial_power) if (gen.initial_power is not None) else 0.0
        )
        u_prev_map[gen.name] = init_u_map[gen.name]

    for t in range(T):
        # Pre-collect bounds and warm desired rows
        per_gen_data = []
        sum_lb = 0.0
        sum_ub = 0.0

        for gen in sc.thermal_units:
            name = gen.name
            u_t = int(gen_out[name]["commit"][t])
            v_t = int(start_flags[name][t])
            w_t = int(stop_flags[name][t])

            min_t = float(gen.min_power[t]) * u_t
            max_t = float(gen.max_power[t]) * u_t

            # Segment caps for above-min power at this t
            nS = len(gen.segments) if gen.segments else 0
            caps = [
                (float(gen.segments[s].amount[t]) if nS > 0 else 0.0) * u_t
                for s in range(nS)
            ]

            # Warm desired above-min row
            warm_seg = warm_gen.get(name, {}).get("segment_power", None)
            if isinstance(warm_seg, list):
                warm_row = warm_seg[t] if t < len(warm_seg) else []
                warm_row = (warm_row + [0.0] * nS)[:nS]
                warm_row = [max(0.0, float(warm_row[s])) for s in range(nS)]
            else:
                warm_row = [0.0] * nS
            # Clip to caps
            warm_row = [min(warm_row[s], max(0.0, caps[s])) for s in range(nS)]
            desired_above = sum(warm_row)

            # Ramp-aware bounds
            p_prev = float(p_prev_map[name])
            u_prev = int(u_prev_map[name])
            ru = float(gen.ramp_up)
            rd = float(gen.ramp_down)
            SU = float(gen.startup_limit)
            SD = float(gen.shutdown_limit)

            up_ub = p_prev + ru * float(u_prev) + SU * float(v_t)
            dn_lb = p_prev - (rd * float(u_t) + SD * float(w_t))

            p_lb = max(min_t, dn_lb)
            p_ub = min(max_t, up_ub)
            if p_lb > p_ub:
                # Inconsistent; collapse to a point (still feasible from model's viewpoint)
                p_ub = p_lb

            # Initial pick within bounds
            target = min(max(min_t + desired_above, p_lb), p_ub)
            # Convert to row consistent with 'target'
            above_need = max(0.0, target - min_t)
            row = [0.0] * nS
            if above_need > 0.0 and nS > 0:
                leftover = _add_to_segments(row, caps, above_need)
                if leftover > 1e-9:
                    # Should not happen if above_need <= sum(caps), but keep robust
                    pass

            per_gen_data.append(
                {
                    "gen": gen,
                    "name": name,
                    "u_t": u_t,
                    "min_t": min_t,
                    "max_t": max_t,
                    "p_lb": p_lb,
                    "p_ub": p_ub,
                    "row": row,
                    "caps": caps,
                }
            )
            sum_lb += p_lb
            sum_ub += p_ub

        # Balance to exact total load if possible
        L = float(total_load[t])
        if L < sum_lb - 1e-8:
            # Cannot meet load with current commitment/ramp; choose p = p_lb (closest)
            feasible_all = False
            balance_gaps[t] = L - sum_lb
            # finalize rows as per p_lb
            for item in per_gen_data:
                gen = item["gen"]
                min_t = item["min_t"]
                row = item["row"]
                p_lb = item["p_lb"]
                current = min_t + sum(row)
                if current > p_lb + 1e-12:
                    # remove surplus
                    need_remove = current - p_lb
                    _remove_from_segments(row, need_remove)
                elif current < p_lb - 1e-12:
                    # add deficit (limited by caps)
                    need_add = p_lb - current
                    _add_to_segments(row, item["caps"], need_add)
        elif L > sum_ub + 1e-8:
            # Cannot meet load; choose p = p_ub (closest)
            feasible_all = False
            balance_gaps[t] = L - sum_ub
            for item in per_gen_data:
                gen = item["gen"]
                min_t = item["min_t"]
                row = item["row"]
                p_ub = item["p_ub"]
                current = min_t + sum(row)
                if current < p_ub - 1e-12:
                    need_add = p_ub - current
                    _add_to_segments(row, item["caps"], need_add)
                elif current > p_ub + 1e-12:
                    need_remove = current - p_ub
                    _remove_from_segments(row, need_remove)
        else:
            # Feasible to hit exact load
            delta = L - sum(
                min(item["min_t"] + sum(item["row"]), item["p_ub"])
                for item in per_gen_data
            )
            # The above formula already uses our current target; recompute from scratch
            current_sum = sum(item["min_t"] + sum(item["row"]) for item in per_gen_data)
            delta = L - current_sum
            if abs(delta) > 1e-9:
                if delta > 0:
                    # Distribute upward within p_ub bounds
                    remaining = delta
                    # Multi-pass greedy water-filling
                    # First pass in natural order, then repeat if needed
                    for _pass in range(2):
                        for item in per_gen_data:
                            if remaining <= 1e-12:
                                break
                            min_t = item["min_t"]
                            row = item["row"]
                            caps = item["caps"]
                            p_ub = item["p_ub"]
                            current = min_t + sum(row)
                            up_slack = max(0.0, p_ub - current)
                            if up_slack <= 0.0:
                                continue
                            take = min(up_slack, remaining)
                            leftover = _add_to_segments(row, caps, take)
                            # leftover should be ~0; but if not, reduce effective take
                            effective = take - leftover
                            remaining -= effective
                        if remaining <= 1e-12:
                            break
                    # Any remaining small residual can be distributed again (should be ~0)
                    balance_gaps[t] = remaining if remaining != 0 else 0.0
                else:
                    # Distribute downward within p_lb bounds
                    remaining = -delta
                    for _pass in range(2):
                        for item in per_gen_data:
                            if remaining <= 1e-12:
                                break
                            min_t = item["min_t"]
                            row = item["row"]
                            p_lb = item["p_lb"]
                            current = min_t + sum(row)
                            dn_slack = max(0.0, current - p_lb)
                            if dn_slack <= 0.0:
                                continue
                            take = min(dn_slack, remaining)
                            leftover = _remove_from_segments(row, take)
                            effective = take - leftover
                            remaining -= effective
                        if remaining <= 1e-12:
                            break
                    balance_gaps[t] = -remaining if remaining != 0 else 0.0

        # Finalize rows and p_t; update prevs
        for item in per_gen_data:
            gen = item["gen"]
            name = item["name"]
            row = [max(0.0, float(x)) for x in item["row"]]
            gen_out[name]["segment_power"][t] = row
            p_t = float(item["min_t"]) + sum(row)
            p_total.setdefault(name, [])
            p_total[name].append(float(p_t))
            p_above_min.setdefault(name, [])
            p_above_min[name].append(float(sum(row)))
            p_prev_map[name] = p_t
            u_prev_map[name] = int(gen_out[name]["commit"][t])

    max_abs_gap = max(abs(float(g)) for g in balance_gaps) if balance_gaps else 0.0
    balance_info = {
        "feasible": bool(feasible_all and max_abs_gap <= 1e-6),
        "max_abs_gap": float(max_abs_gap),
        "gaps": [float(x) for x in balance_gaps],
    }

    # 4) Reserves — set to minimal safe feasible pattern: all provision zero; shortfall = requirement
    reserves_out: Dict[str, Dict] = {}
    if getattr(sc, "reserves", None):
        for r in sc.reserves:
            requirement = [float(r.amount[t]) for t in range(T)]
            shortfall = [float(r.amount[t]) for t in range(T)]
            provided_by_gen = {g.name: [0.0 for _ in range(T)] for g in r.thermal_units}
            total_provided = [0.0 for _ in range(T)]
            reserves_out[r.name] = {
                "provided_by_gen": provided_by_gen,
                "total_provided": total_provided,
                "requirement": requirement,
                "shortfall": shortfall,
            }

    return gen_out, reserves_out, p_total, p_above_min, balance_info


def _fix_network(sc, p_total: Dict[str, List[float]]) -> Dict:
    """
    Compute flows with PTDF and set overflow slacks (base-case).
    Returns a dict 'network' with { "lines": { line_name: {...} } }
    """
    T = sc.time
    if not sc.lines or sc.isf is None:
        return {}

    isf = sc.isf.tocsr()
    buses = sc.buses
    lines = sc.lines

    ref_1b = getattr(sc, "ptdf_ref_bus_index", buses[0].index)
    non_ref_bus_indices = sorted([b.index for b in buses if b.index != ref_1b])

    # Precompute p_total at buses
    inj: Dict[int, List[float]] = {b.index: [0.0] * T for b in buses}
    for b in buses:
        for gen in b.thermal_units:
            pt = p_total[gen.name]
            for t in range(T):
                inj[b.index][t] += float(pt[t])
        for t in range(T):
            inj[b.index][t] -= float(b.load[t])

    # Compute flows and overflow slacks
    lines_out = {}
    for line in lines:
        flow = [0.0 for _ in range(T)]
        ovp = [0.0 for _ in range(T)]
        ovn = [0.0 for _ in range(T)]
        lim = [float(line.normal_limit[t]) for t in range(T)]
        pen = [float(line.flow_penalty[t]) for t in range(T)]

        row = isf.getrow(line.index - 1)
        cols = row.indices.tolist()
        coeffs = row.data.tolist()

        for t in range(T):
            fval = 0.0
            for col, coeff in zip(cols, coeffs):
                bus_1b = non_ref_bus_indices[col]
                fval += float(coeff) * float(inj[bus_1b][t])
            flow[t] = fval
            # Overflow
            ovp[t] = max(0.0, fval - lim[t])
            ovn[t] = max(0.0, -fval - lim[t])

        lines_out[line.name] = {
            "source": line.source.name,
            "target": line.target.name,
            "flow": list(map(float, flow)),
            "limit": lim,
            "overflow_pos": list(map(float, ovp)),
            "overflow_neg": list(map(float, ovn)),
            "penalty": pen,
        }

    return {"lines": lines_out}


def _power_balance_gap(
    sc, p_total: Dict[str, List[float]]
) -> Tuple[float, List[float]]:
    T = sc.time
    load_t = [sum(b.load[t] for b in sc.buses) for t in range(T)]
    prod_t = [0.0] * T
    for gen in sc.thermal_units:
        pt = p_total[gen.name]
        for t in range(T):
            prod_t[t] += float(pt[t])
    gap = [float(prod_t[t] - load_t[t]) for t in range(T)]
    worst = max(abs(x) for x in gap) if gap else 0.0
    return worst, gap


def fix_warm_payload(instance_name: str, sc, warm: Dict) -> Dict:
    """
    Produce a fixed-warm payload for 'instance_name' and scenario 'sc'.
    """
    gen_out, reserves_out, p_total, _p_above_min, balance_info = (
        _fix_generators_and_reserves(sc, warm)
    )
    network_out = _fix_network(sc, p_total)

    fixed = {
        "instance_name": instance_name,
        "case_folder": "/".join(instance_name.strip().split("/")[:2]),
        "neighbor": warm.get("neighbor", "unknown_neighbor"),
        "distance": float(warm.get("distance", 0.0)),
        "coverage": float(warm.get("coverage", 0.0)),
        "generators": gen_out,
        "reserves": reserves_out,
        "network": network_out,
        "balance": {
            "feasible": bool(balance_info.get("feasible", False)),
            "max_abs_gap": float(balance_info.get("max_abs_gap", 0.0)),
        },
    }
    return fixed


def fix_warm_file(
    instance_name: str,
    warm_file: Optional[Path] = None,
    *,
    require_pretrained: bool = False,
    use_train_db: bool = False,
) -> Optional[Path]:
    """
    Fix an existing warm JSON (or auto-generate it if missing) and write warm_fixed_<instance>.json.

    Returns the path to the fixed warm JSON or None on failure.
    """
    name = instance_name
    inst = read_benchmark(name, quiet=True)
    sc = inst.deterministic

    # Find/generate warm JSON
    warm_path = Path(warm_file) if warm_file else _warm_path_for_instance(name)
    if not warm_path.exists():
        # auto-generate via neighbor DB (no solver)
        try:
            from src.ml_models.warm_start import (
                WarmStartProvider,
            )  # lazy import to avoid cycles

            cf = "/".join(name.strip().split("/")[:2])
            wsp = WarmStartProvider(case_folder=cf)
            trained, cov = wsp.ensure_trained(
                cf, allow_build_if_missing=not require_pretrained
            )
            if not trained:
                print(
                    f"[fix_warm] No warm-start index available for {cf} (coverage={cov:.3f})."
                )
                return None
            warm_path = wsp.generate_and_save_warm_start(
                name,
                use_train_index_only=use_train_db,
                exclude_self=True,
                auto_fix=False,
            )
            if not warm_path or not Path(warm_path).exists():
                print("[fix_warm] Failed to generate warm-start JSON.")
                return None
            print(f"[fix_warm] Generated warm JSON: {warm_path}")
        except Exception as e:
            print(f"[fix_warm] Auto-generation failed: {e}")
            return None

    # Load warm JSON
    try:
        warm = json.loads(Path(warm_path).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[fix_warm] Could not read warm file '{warm_path}': {e}")
        return None

    # Fix payload
    fixed = fix_warm_payload(name, sc, warm)

    # Save fixed
    out_path = fixed_warm_path_for_instance(name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixed, indent=2), encoding="utf-8")

    # Report
    # Recompute power balance gaps from the fixed payload for clarity
    p_total_map = {
        g: [
            fixed["generators"][g]["commit"][t]
            * float(sc.thermal_units_by_name[g].min_power[t])
            + sum(fixed["generators"][g]["segment_power"][t])
            for t in range(sc.time)
        ]
        for g in fixed["generators"]
    }
    worst_gap, gap = _power_balance_gap(sc, p_total_map)

    print(f"[fix_warm] Wrote fixed warm: {out_path}")
    feas_flag = fixed.get("balance", {}).get("feasible", False)
    print(
        f"[fix_warm] Power balance feasibility: {'OK' if feas_flag and worst_gap <= 1e-6 else 'NEAR/NOT OK'}, "
        f"Max |production - load| = {worst_gap:.6f} MW"
    )

    return out_path


# ------------------------------- CLI ------------------------------- #


def main():
    ap = argparse.ArgumentParser(
        description="Fix a warm-start JSON without using Gurobi (min-up/down, ramp, segment caps, reserves, PTDF slacks, and balance to load)."
    )
    ap.add_argument(
        "--instance",
        required=True,
        help="Dataset name, e.g., matpower/case57/2017-06-24",
    )
    ap.add_argument(
        "--warm-file",
        default=None,
        help="Path to an existing warm JSON. If omitted, use default warm_<instance>.json (auto-generate if missing).",
    )
    ap.add_argument(
        "--require-pretrained",
        action="store_true",
        default=False,
        help="If auto-generating the warm JSON, require a pre-trained index.",
    )
    ap.add_argument(
        "--use-train-db",
        action="store_true",
        default=False,
        help="If auto-generating, restrict neighbor DB to the training split.",
    )
    args = ap.parse_args()

    out = fix_warm_file(
        instance_name=args.instance,
        warm_file=Path(args.warm_file) if args.warm_file else None,
        require_pretrained=args.require_pretrained,
        use_train_db=args.use_train_db,
    )
    if out is None:
        print("[fix_warm] Failed.")
    else:
        print(f"[fix_warm] Fixed warm saved to: {out}")


if __name__ == "__main__":
    main()

```

### File: `src/ml_models/pretrain_warm_start.py`

```
"""
Pretrain warm-start k-NN indexes once and save them to:
  src/data/intermediate/warm_start/ws_index_<case_tag>.json

Usage examples:
  - Pretrain for specific case folders:
      python -m src.ml_models.pretrain_warm_start --cases matpower/case14 matpower/case30

  - Auto-detect case folders from solved outputs:
      python -m src.ml_models.pretrain_warm_start --auto-cases

Options:
  --min-coverage  Minimum coverage to consider index 'trained' (default 0.70)
  --force         Rebuild and overwrite even if index file exists
"""

import argparse
from pathlib import Path
from typing import List, Set, Optional
from src.data_preparation.params import DataParams
from src.ml_models.warm_start import WarmStartProvider


def _discover_case_folders_from_outputs() -> List[str]:
    base = DataParams._OUTPUT.resolve()
    cases: Set[str] = set()
    for p in base.rglob("*.json"):
        try:
            rel = p.resolve().relative_to(base).as_posix()
        except Exception:
            continue
        parts = rel.split("/")
        if len(parts) >= 3:
            cases.add("/".join(parts[:2]))
    return sorted(cases)


def main():
    ap = argparse.ArgumentParser(
        description="Pretrain and persist warm-start indexes per case."
    )
    ap.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Explicit case folders, e.g., matpower/case14 matpower/case30",
    )
    ap.add_argument(
        "--auto-cases",
        action="store_true",
        help="Auto-detect case folders from solved outputs",
    )
    ap.add_argument(
        "--min-coverage",
        type=float,
        default=0.70,
        help="Min coverage to mark index as trained",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Rebuild and overwrite index even if it exists",
    )
    args = ap.parse_args()

    cases: Optional[List[str]] = args.cases
    if (not cases) and args.auto_cases:
        cases = _discover_case_folders_from_outputs()

    if not cases:
        print(
            "No cases provided and --auto-cases not used (or found none). Nothing to do."
        )
        return

    print(f"Pretraining warm-start indexes for {len(cases)} case folder(s) ...")
    ok_count = 0
    for cf in cases:
        print(f"- {cf}: building index (force={args.force}) ...", end="", flush=True)
        wsp = WarmStartProvider(case_folder=cf, coverage_threshold=args.min_coverage)
        idx_path = wsp.pretrain(force=args.force)
        if not idx_path:
            print(" no data (missing outputs) -> skipped")
            continue
        trained, cov = wsp.ensure_trained(cf, allow_build_if_missing=False)
        status = "TRAINED" if trained else "NOT TRAINED"
        print(f" done. coverage={cov:.3f}, status={status}, index={idx_path}")
        ok_count += 1

    print(f"Finished. Indexes written: {ok_count}/{len(cases)}")


if __name__ == "__main__":
    main()

```

### File: `src/ml_models/redundant_constraints.py`

```
"""
RedundancyProvider: k-NN constraint-pruning for contingencies.

Goal
- Learn from historical solutions (src/data/output) which contingency constraints
  (line-outage, gen-outage) are always slack with a comfortable margin, so we can
  skip adding them to the SCUC model for similar instances and reduce model size.

Approach
- Per case folder (e.g., matpower/case57), build an index over solved outputs:
  Item = {instance name, feature vector = zscored system load, per-constraint margins}
- For each solved instance i and scenario sc(i):
  * For each contingency c and time t:
    - Line-outage: for each monitored line l and outaged line m with LODF[l,m] = α,
      estimate post-contingency flow = f_l[t] + α * f_m[t], margin s = F_em(l,t) - |post|.
    - Gen-outage: for each gen g at bus b with ISF[l,b] = β,
      estimate post flow = f_l[t] - β * p_g[t], margin s = F_em(l,t) - |post|.
  * Store s per (l,m,t) and (l,g,t). Positive s = safe slack (redundant candidate).
- At inference time for a target instance j:
  * Find nearest neighbor i* by Euclidean distance on zscored load vector.
  * Build a filter predicate that skips a constraint if neighbor margin s >= thr_abs + thr_rel*F_em.
    Otherwise, keep the constraint.

Notes
- Our SCUC contingency constraints use these same base-case expressions (no explicit redispatch),
  so using neighbor’s margins is a strong heuristic to prune constraints safely.
- You can restrict neighbors to the TRAIN split to emulate train/test data usage.
- If no trained index or no neighbor found, the provider returns None (no pruning).

Artifacts
- Index is saved to: src/data/intermediate/redundancy/rc_index_<case_tag>.json
"""

import json
import gzip
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Callable

from scipy.sparse import csr_matrix, csc_matrix

from src.data_preparation.params import DataParams
from src.data_preparation.read_data import read_benchmark
from src.ml_models.warm_start import _hash01  # deterministic split helper


def _sanitize_name(s: str) -> str:
    s = (s or "").strip().strip("/\\").replace("\\", "/")
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    t = "".join(out)
    while "__" in t:
        t = t.replace("__", "_")
    return t.strip("_").lower()


def _case_folder_from_instance(instance_name: str) -> str:
    parts = instance_name.strip().strip("/\\").replace("\\", "/").split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] if parts else "case"


def _case_tag(case_folder: str) -> str:
    return _sanitize_name(case_folder)


def _zscore(vec: List[float]) -> List[float]:
    if not vec:
        return []
    mean = sum(vec) / len(vec)
    var = sum((x - mean) ** 2 for x in vec) / max(1, len(vec) - 1)
    std = math.sqrt(var) if var > 0 else 1.0
    return [(x - mean) / std for x in vec]


def _l2(v1: List[float], v2: List[float]) -> float:
    if len(v1) != len(v2):
        L = max(len(v1), len(v2))
        v1 = v1 + [v1[-1] if v1 else 0.0] * (L - len(v1))
        v2 = v2 + [v2[-1] if v2 else 0.0] * (L - len(v2))
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def _ensure_list_length(vals: List, T: int, pad_with_last: bool = True, default=0.0):
    if vals is None:
        return [default] * T
    vals = list(vals)
    if len(vals) < T:
        pad_val = vals[-1] if (pad_with_last and vals) else default
        vals = vals + [pad_val] * (T - len(vals))
    elif len(vals) > T:
        vals = vals[:T]
    return vals


def _load_input_system_load(input_path: Path) -> Optional[List[float]]:
    try:
        if input_path.suffix == ".gz":
            with gzip.open(input_path, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            with input_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        buses = data.get("Buses", {})
        T = None
        for b in buses.values():
            ld = b.get("Load (MW)", 0.0)
            if isinstance(ld, list):
                T = len(ld)
                break
        if T is None:
            T = 1
        sys_load = [0.0] * T
        for b in buses.values():
            ld = b.get("Load (MW)", 0.0)
            if isinstance(ld, list):
                for t in range(T):
                    sys_load[t] += float(ld[t])
            else:
                sys_load[0] += float(ld)
        return sys_load
    except Exception:
        return None


def _load_output_solution(output_path: Path) -> Optional[dict]:
    try:
        with output_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


class RedundancyProvider:
    """
    k-NN based predictor to skip (prune) contingency constraints safely.

    Data saved per instance:
      - features: z-scored system load vector
      - line_pairs:  dict "l|m" -> [margin_t] where margin_t = F_em(l,t) - |f_l + α f_m|
      - gen_pairs:   dict "l|g" -> [margin_t] where margin_t = F_em(l,t) - |f_l - β p_g|
    """

    def __init__(
        self,
        case_folder: Optional[str] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        split_seed: int = 42,
    ):
        self.base_input = DataParams._CACHE
        self.base_output = DataParams._OUTPUT
        self.base_idx = DataParams._INTERMEDIATE / "redundancy"
        self.base_idx.mkdir(parents=True, exist_ok=True)

        self.case_folder = case_folder
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        if self.train_ratio < 0.0:
            self.train_ratio = 0.0
        if self.val_ratio < 0.0:
            self.val_ratio = 0.0
        if self.train_ratio + self.val_ratio > 1.0:
            self.val_ratio = max(0.0, 1.0 - self.train_ratio)
        self.split_seed = int(split_seed)

        # Runtime state
        self._index: Dict[str, dict] = {}
        self._coverage: float = 0.0
        self._available: bool = False
        self._inputs_list: List[str] = []
        self._outputs_list: List[str] = []
        self._splits: Dict[str, Set[str]] = {
            "train": set(),
            "val": set(),
            "test": set(),
        }

    def _list_inputs(self, case_folder: str) -> List[Path]:
        case_dir = (self.base_input / case_folder).resolve()
        if not case_dir.exists():
            return []
        return sorted(case_dir.glob("*.json.gz"))

    def _list_outputs(self, case_folder: str) -> List[Path]:
        case_dir = (self.base_output / case_folder).resolve()
        if not case_dir.exists():
            return []
        return sorted(case_dir.glob("*.json"))

    def _dataset_name_from_output(self, p: Path) -> str:
        rel = p.resolve().relative_to(self.base_output.resolve()).as_posix()
        if rel.endswith(".json"):
            rel = rel[: -len(".json")]
        return rel

    def _dataset_name_from_input(self, p: Path) -> str:
        rel = p.resolve().relative_to(self.base_input.resolve()).as_posix()
        if rel.endswith(".json.gz"):
            rel = rel[: -len(".json.gz")]
        return rel

    def _index_path(self, case_folder: str) -> Path:
        tag = _case_tag(case_folder)
        return (self.base_idx / f"rc_index_{tag}.json").resolve()

    def _compute_splits_from_names(self, names: List[str]) -> None:
        tr: Set[str] = set()
        va: Set[str] = set()
        te: Set[str] = set()
        for nm in sorted(names):
            r = _hash01(nm, self.split_seed)
            if r < self.train_ratio:
                tr.add(nm)
            elif r < self.train_ratio + self.val_ratio:
                va.add(nm)
            else:
                te.add(nm)
        self._splits = {"train": tr, "val": va, "test": te}

    def _load_index_file(self, case_folder: str) -> bool:
        path = self._index_path(case_folder)
        if not path.exists():
            return False
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if obj.get("case_folder") != case_folder:
                return False
            items = obj.get("items", [])
            idx: Dict[str, dict] = {}
            for it in items:
                name = it.get("instance")
                feats = it.get("features", [])
                lpairs = it.get("line_pairs", {})
                gpairs = it.get("gen_pairs", {})
                if not name or feats is None:
                    continue
                idx[name] = {
                    "features": feats,
                    "line_pairs": lpairs,
                    "gen_pairs": gpairs,
                }
            self._index = idx
            self._coverage = float(obj.get("coverage", 0.0))
            self._available = len(self._index) > 0

            self._inputs_list = list(obj.get("inputs", []))
            self._outputs_list = list(obj.get("outputs", []))
            split = obj.get("split", None)
            if isinstance(split, dict):
                self._splits = {
                    "train": set(split.get("train", [])),
                    "val": set(split.get("val", [])),
                    "test": set(split.get("test", [])),
                }
            else:
                self._compute_splits_from_names(list(self._index.keys()))
            return True
        except Exception:
            return False

    def _save_index_file(self, case_folder: str) -> Path:
        path = self._index_path(case_folder)
        path.parent.mkdir(parents=True, exist_ok=True)
        items = []
        for name, it in self._index.items():
            items.append(
                {
                    "instance": name,
                    "features": list(it.get("features", [])),
                    "line_pairs": it.get("line_pairs", {}),
                    "gen_pairs": it.get("gen_pairs", {}),
                }
            )
        payload = {
            "case_folder": case_folder,
            "built_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "coverage": float(self._coverage),
            "inputs": list(self._inputs_list),
            "outputs": list(self._outputs_list),
            "split": {
                "train": sorted(self._splits["train"]),
                "val": sorted(self._splits["val"]),
                "test": sorted(self._splits["test"]),
            },
            "items": items,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _compute_margins_for_instance(
        self, instance_name: str, out_json: dict
    ) -> Optional[Tuple[Dict[str, List[float]], Dict[str, List[float]], List[float]]]:
        """
        Compute per-constraint margins (line-outage, gen-outage) for one instance.
        Returns (line_pairs, gen_pairs, feats) or None if not computable.
        """
        # Load scenario (to get ISF, LODF, limits)
        try:
            inst = read_benchmark(instance_name, quiet=True)
            sc = inst.deterministic
        except Exception:
            return None
        if not sc.lines:
            return None

        T = sc.time
        # Features: system load (z-scored)
        try:
            feats = out_json.get("system", {}).get("load", None)
            feats = _ensure_list_length([float(x) for x in feats], T, True, 0.0)
        except Exception:
            feats = [sum(b.load[t] for b in sc.buses) for t in range(T)]
        feats = _zscore(feats)

        # Base flows and gen power from output JSON
        net = out_json.get("network", {}) or {}
        lines_out = net.get("lines", {}) or {}
        if not lines_out:
            return None
        flows: Dict[str, List[float]] = {}
        for ln in sc.lines:
            obj = lines_out.get(ln.name)
            if not obj:
                return None
            flows[ln.name] = _ensure_list_length(obj.get("flow", []), T, True, 0.0)

        gens_out = out_json.get("generators", {}) or {}
        pgen: Dict[str, List[float]] = {}
        for g in sc.thermal_units:
            gobj = gens_out.get(g.name, {}) or {}
            p_list = gobj.get("total_power", None)
            if p_list is None:
                # reconstruct as min + sum(segments)
                commit = _ensure_list_length(gobj.get("commit", []), T, True, 0)
                seg2d = gobj.get("segment_power", None)
                nS = len(g.segments) if g.segments else 0
                if isinstance(seg2d, list):
                    seg2d = _ensure_list_length(seg2d, T, True, [0.0] * nS)
                else:
                    seg2d = [[0.0] * nS for _ in range(T)]
                total = []
                for t in range(T):
                    val = float(commit[t]) * float(g.min_power[t])
                    for s in range(nS):
                        val += float(seg2d[t][s])
                    total.append(val)
                pgen[g.name] = total
            else:
                pgen[g.name] = _ensure_list_length(p_list, T, True, 0.0)

        # Precompute sparse matrices and mappings
        lodf_csc: csc_matrix = sc.lodf.tocsc()
        isf_csc: csc_matrix = sc.isf.tocsc()
        buses = sc.buses
        ref_1b = getattr(sc, "ptdf_ref_bus_index", buses[0].index)
        non_ref_bus_indices = sorted([b.index for b in buses if b.index != ref_1b])
        col_by_bus_1b = {bus_1b: col for col, bus_1b in enumerate(non_ref_bus_indices)}
        line_by_row = {ln.index - 1: ln for ln in sc.lines}
        row_by_line = {ln.name: ln.index - 1 for ln in sc.lines}

        # Compute margins
        line_pairs: Dict[str, List[float]] = {}
        gen_pairs: Dict[str, List[float]] = {}

        # Line-outage
        for cont in sc.contingencies or []:
            if not cont.lines:
                continue
            for out_line in cont.lines:
                mcol = out_line.index - 1
                col = lodf_csc.getcol(mcol)
                for l_row, alpha in zip(col.indices.tolist(), col.data.tolist()):
                    if l_row == mcol:
                        continue
                    line_l = line_by_row.get(l_row)
                    if line_l is None:
                        continue
                    # Build per-t margin
                    key = f"{line_l.name}|{out_line.name}"
                    s_vec = line_pairs.get(key)
                    if s_vec is None:
                        s_vec = [0.0 for _ in range(T)]
                        line_pairs[key] = s_vec
                    for t in range(T):
                        f_l = float(flows[line_l.name][t])
                        f_m = float(flows[out_line.name][t])
                        post = f_l + float(alpha) * f_m
                        F_em = float(line_l.emergency_limit[t])
                        s_vec[t] = F_em - abs(post)

        # Gen-outage
        for cont in sc.contingencies or []:
            if not getattr(cont, "units", None):
                continue
            for gen in cont.units:
                bidx = gen.bus.index
                if bidx == ref_1b or bidx not in col_by_bus_1b:
                    # Effect zero or undefined; margin = F_em - |f_l|
                    col = None
                else:
                    col = isf_csc.getcol(col_by_bus_1b[bidx])

                # Build margin per line,time
                if col is None:
                    # No ISF column => post = f_l (no change)
                    for line_l in sc.lines:
                        key = f"{line_l.name}|{gen.name}"
                        s_vec = gen_pairs.get(key)
                        if s_vec is None:
                            s_vec = [0.0 for _ in range(T)]
                            gen_pairs[key] = s_vec
                        for t in range(T):
                            f_l = float(flows[line_l.name][t])
                            F_em = float(line_l.emergency_limit[t])
                            s_vec[t] = F_em - abs(f_l)
                else:
                    rows = col.indices.tolist()
                    vals = col.data.tolist()
                    isf_map = {r: v for r, v in zip(rows, vals)}
                    for line_l in sc.lines:
                        key = f"{line_l.name}|{gen.name}"
                        s_vec = gen_pairs.get(key)
                        if s_vec is None:
                            s_vec = [0.0 for _ in range(T)]
                            gen_pairs[key] = s_vec
                        beta = float(isf_map.get(row_by_line[line_l.name], 0.0))
                        for t in range(T):
                            f_l = float(flows[line_l.name][t])
                            p_g = float(pgen[gen.name][t])
                            post = f_l - beta * p_g
                            F_em = float(line_l.emergency_limit[t])
                            s_vec[t] = F_em - abs(post)

        return line_pairs, gen_pairs, feats

    def _build_index(self, case_folder: str) -> None:
        inputs = self._list_inputs(case_folder)
        outputs = self._list_outputs(case_folder)
        total_inputs = len(inputs)
        total_outputs = len(outputs)
        self._coverage = (total_outputs / total_inputs) if total_inputs > 0 else 0.0

        self._inputs_list = [self._dataset_name_from_input(p) for p in inputs]
        outputs_names = [self._dataset_name_from_output(p) for p in outputs]

        idx: Dict[str, dict] = {}
        for out_path in outputs:
            out = _load_output_solution(out_path)
            if not out:
                continue
            instance_name = self._dataset_name_from_output(out_path)
            computed = self._compute_margins_for_instance(instance_name, out)
            if computed is None:
                continue
            line_pairs, gen_pairs, feats = computed
            idx[instance_name] = {
                "features": feats,
                "line_pairs": line_pairs,
                "gen_pairs": gen_pairs,
            }

        self._index = idx
        self._available = len(self._index) > 0
        self._compute_splits_from_names(sorted(self._index.keys()))

    def pretrain(
        self, case_folder: Optional[str] = None, force: bool = False
    ) -> Optional[Path]:
        cf = case_folder or self.case_folder
        if not cf:
            return None
        path = self._index_path(cf)
        if path.exists() and not force:
            if self._load_index_file(cf):
                return path
        self._build_index(cf)
        if not self._index:
            return None
        return self._save_index_file(cf)

    def ensure_trained(
        self, case_folder: Optional[str] = None, allow_build_if_missing: bool = True
    ) -> Tuple[bool, float]:
        cf = case_folder or self.case_folder
        if not cf:
            return False, 0.0
        if self._load_index_file(cf):
            return (self._available, self._coverage)
        if allow_build_if_missing:
            self._build_index(cf)
            return (self._available, self._coverage)
        return False, 0.0

    def _nearest_neighbor(
        self,
        target_feats: List[float],
        restrict_to_names: Optional[Set[str]] = None,
        exclude_names: Optional[Set[str]] = None,
    ) -> Optional[Tuple[str, dict, float]]:
        best_name = None
        best_item = None
        best_dist = float("inf")
        for name, item in self._index.items():
            if restrict_to_names is not None and name not in restrict_to_names:
                continue
            if exclude_names is not None and name in exclude_names:
                continue
            d = _l2(target_feats, item.get("features", []))
            if d < best_dist:
                best_dist = d
                best_item = item
                best_name = name
        if best_item is None:
            return None
        return best_name, best_item, best_dist

    def make_filter_for_instance(
        self,
        scenario,
        instance_name: str,
        *,
        thr_abs: float = 20.0,
        thr_rel: float = 0.10,
        use_train_index_only: bool = True,
        exclude_self: bool = True,
    ) -> Optional[Tuple[Callable, Dict]]:
        """
        Build a filter predicate closure for contingencies.add_constraints.

        A constraint is pruned (skipped) if neighbor margin s >= thr_abs + thr_rel * F_em.

        Returns (predicate, stats) or None if no trained index/neighbor.
        stats = {"neighbor": str, "distance": float, "skipped_line": int, "skipped_gen": int}
        """
        case_folder = _case_folder_from_instance(instance_name)
        if not self._index:
            self.ensure_trained(case_folder, allow_build_if_missing=False)
        if not self._index:
            return None

        # Target features from input JSON system load
        input_path = (DataParams._CACHE / (instance_name + ".json.gz")).resolve()
        target_load = _load_input_system_load(input_path)
        if not target_load:
            return None
        T = scenario.time
        target_feats = _zscore(
            _ensure_list_length([float(x) for x in target_load], T, True, 0.0)
        )

        restrict = self._splits["train"] if use_train_index_only else None
        exclude = {instance_name} if exclude_self else None
        nn = self._nearest_neighbor(
            target_feats, restrict_to_names=restrict, exclude_names=exclude
        )
        if nn is None:
            return None
        nn_name, nn_item, nn_dist = nn

        # Build skip sets
        line_pairs = nn_item.get("line_pairs", {}) or {}
        gen_pairs = nn_item.get("gen_pairs", {}) or {}

        thr = lambda F_em: float(thr_abs) + float(thr_rel) * float(F_em)

        # Pre-generate skip masks as dict[(lname, oname, t)] -> True to skip
        skip_line = {}
        skip_gen = {}
        # Line-outage keys l|m
        for key, s_list in line_pairs.items():
            try:
                l_name, m_name = key.split("|", 1)
            except Exception:
                continue
            s_list = _ensure_list_length([float(x) for x in s_list], T, True, 0.0)
            # We'll compute threshold at call-time (depends on F_em of target scenario)
            # so here we only store margins per t.
            skip_line[(l_name, m_name)] = s_list

        # Gen-outage keys l|g
        for key, s_list in gen_pairs.items():
            try:
                l_name, g_name = key.split("|", 1)
            except Exception:
                continue
            s_list = _ensure_list_length([float(x) for x in s_list], T, True, 0.0)
            skip_gen[(l_name, g_name)] = s_list

        stats = {
            "neighbor": nn_name,
            "distance": nn_dist,
            "skipped_line": 0,
            "skipped_gen": 0,
        }

        def _predicate(
            kind: str, line_l, out_obj, t: int, coeff: float, F_em: float
        ) -> bool:
            """
            Return True to include the constraint, False to prune.
            kind: "line" for line-outage; "gen" for gen-outage
            """
            Fthr = thr(F_em)
            if kind == "line":
                key = (line_l.name, out_obj.name)
                s_vec = skip_line.get(key)
                if s_vec is None:
                    return True
                s = float(s_vec[min(max(0, int(t)), T - 1)])
                if s >= Fthr:
                    stats["skipped_line"] += 1
                    return False
                return True
            else:
                key = (line_l.name, out_obj.name)
                s_vec = skip_gen.get(key)
                if s_vec is None:
                    return True
                s = float(s_vec[min(max(0, int(t)), T - 1)])
                if s >= Fthr:
                    stats["skipped_gen"] += 1
                    return False
                return True

        return _predicate, stats

```

### File: `src/ml_models/warm_start.py`

```
import json
import gzip
import math
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from src.data_preparation.params import DataParams


def _sanitize_name(s: str) -> str:
    s = (s or "").strip().strip("/\\").replace("\\", "/")
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    t = "".join(out)
    while "__" in t:
        t = t.replace("__", "_")
    return t.strip("_").lower()


def _case_folder_from_instance(instance_name: str) -> str:
    parts = instance_name.strip().strip("/\\").replace("\\", "/").split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] if parts else "case"


def _case_tag(case_folder: str) -> str:
    return _sanitize_name(case_folder)


def _load_input_system_load(input_path: Path) -> Optional[List[float]]:
    try:
        if input_path.suffix == ".gz":
            with gzip.open(input_path, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            with input_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        buses = data.get("Buses", {})
        T = None
        for b in buses.values():
            ld = b.get("Load (MW)", 0.0)
            if isinstance(ld, list):
                T = len(ld)
                break
        if T is None:
            T = 1
        sys_load = [0.0] * T
        for b in buses.values():
            ld = b.get("Load (MW)", 0.0)
            if isinstance(ld, list):
                for t in range(T):
                    sys_load[t] += float(ld[t])
            else:
                sys_load[0] += float(ld)
        return sys_load
    except Exception:
        return None


def _load_output_solution(output_path: Path) -> Optional[dict]:
    try:
        with output_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _zscore(vec: List[float]) -> List[float]:
    if not vec:
        return []
    mean = sum(vec) / len(vec)
    var = sum((x - mean) ** 2 for x in vec) / max(1, len(vec) - 1)
    std = math.sqrt(var) if var > 0 else 1.0
    return [(x - mean) / std for x in vec]


def _l2(v1: List[float], v2: List[float]) -> float:
    if len(v1) != len(v2):
        L = max(len(v1), len(v2))
        v1 = v1 + [v1[-1] if v1 else 0.0] * (L - len(v1))
        v2 = v2 + [v2[-1] if v2 else 0.0] * (L - len(v2))
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def _ensure_list_length(vals: List, T: int, pad_with_last: bool = True, default=0):
    if vals is None:
        return [default] * T
    vals = list(vals)
    if len(vals) < T:
        pad_val = vals[-1] if (pad_with_last and vals) else default
        vals = vals + [pad_val] * (T - len(vals))
    elif len(vals) > T:
        vals = vals[:T]
    return vals


def _hash01(name: str, seed: int) -> float:
    """
    Deterministic pseudo-random in [0,1) from (name, seed).
    """
    h = hashlib.md5(f"{name}::{int(seed)}".encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / float(0xFFFFFFFF)
    return min(max(v, 0.0), math.nextafter(1.0, 0.0))


class WarmStartProvider:
    """
    k-NN warm start over historical solutions (persisted 'index' per case).

    Training (pretrain): see original docstring.

    Inference: generate_and_save_warm_start() writes a warm JSON capturing a neighbor's
    solution, and apply_warm_start_to_model() sets Start values on the current model.

    New in this version:
      - generate_and_save_warm_start(..., auto_fix=True) will automatically build a
        'warm_fixed_<instance>.json' using src.ml_models.fix_warm_start (no Gurobi).
      - apply_warm_start_to_model(...) now prefers a 'warm_fixed_<instance>.json' file
        if present, falling back to 'warm_<instance>.json'.
    """

    def __init__(
        self,
        case_folder: Optional[str] = None,
        coverage_threshold: float = 0.70,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        split_seed: int = 42,
    ):
        self.base_input = DataParams._CACHE
        self.base_output = DataParams._OUTPUT
        self.base_warm = DataParams._WARM_START
        self.case_folder = case_folder  # e.g., "matpower/case14"
        self.coverage_threshold = float(coverage_threshold)

        # Split config
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        if self.train_ratio < 0.0:
            self.train_ratio = 0.0
        if self.val_ratio < 0.0:
            self.val_ratio = 0.0
        if self.train_ratio + self.val_ratio > 1.0:
            self.val_ratio = max(0.0, 1.0 - self.train_ratio)
        self.split_seed = int(split_seed)

        # Runtime state
        self._trained_index: Dict[str, dict] = {}
        self._coverage: float = 0.0
        self._available: bool = False
        self._trained: bool = False

        # Dataset bookkeeping
        self._inputs_list: List[str] = []
        self._outputs_list: List[str] = []
        self._splits: Dict[str, Set[str]] = {
            "train": set(),
            "val": set(),
            "test": set(),
        }

    def _list_inputs(self, case_folder: str) -> List[Path]:
        case_dir = (self.base_input / case_folder).resolve()
        if not case_dir.exists():
            return []
        return sorted(case_dir.glob("*.json.gz"))

    def _list_outputs(self, case_folder: str) -> List[Path]:
        case_dir = (self.base_output / case_folder).resolve()
        if not case_dir.exists():
            return []
        return sorted(case_dir.glob("*.json"))

    def _dataset_name_from_output(self, p: Path) -> str:
        rel = p.resolve().relative_to(self.base_output.resolve()).as_posix()
        if rel.endswith(".json"):
            rel = rel[: -len(".json")]
        return rel

    def _dataset_name_from_input(self, p: Path) -> str:
        rel = p.resolve().relative_to(self.base_input.resolve()).as_posix()
        if rel.endswith(".json.gz"):
            rel = rel[: -len(".json.gz")]
        return rel

    def _index_path(self, case_folder: str) -> Path:
        tag = _case_tag(case_folder)
        return (self.base_warm / f"ws_index_{tag}.json").resolve()

    def _load_index_file(self, case_folder: str) -> bool:
        path = self._index_path(case_folder)
        if not path.exists():
            return False
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if obj.get("case_folder") != case_folder:
                return False

            items = obj.get("items", [])
            idx: Dict[str, dict] = {}
            for it in items:
                name = it.get("instance")
                feats = it.get("features", [])
                commit = it.get("generators", {})
                if not name or not feats or commit is None:
                    continue
                commit_simple = {}
                for g, v in commit.items():
                    try:
                        commit_simple[g] = list(v.get("commit", []))
                    except Exception:
                        commit_simple[g] = []
                idx[name] = {"features": feats, "commit": commit_simple}
            self._trained_index = idx
            self._coverage = float(obj.get("coverage", 0.0))
            self._available = len(self._trained_index) > 0
            self._trained = self._available and (
                self._coverage >= self.coverage_threshold
            )

            self._inputs_list = list(obj.get("inputs", []))
            self._outputs_list = list(obj.get("outputs", []))

            ratios = obj.get("ratios", {})
            self.train_ratio = float(ratios.get("train", self.train_ratio))
            self.val_ratio = float(ratios.get("val", self.val_ratio))
            self.split_seed = int(obj.get("split_seed", self.split_seed))

            split = obj.get("split", None)
            if isinstance(split, dict):
                self._splits = {
                    "train": set(split.get("train", [])),
                    "val": set(split.get("val", [])),
                    "test": set(split.get("test", [])),
                }
            else:
                self._compute_splits_from_index()
            return True
        except Exception:
            return False

    def _save_index_file(self, case_folder: str) -> Path:
        path = self._index_path(case_folder)
        path.parent.mkdir(parents=True, exist_ok=True)
        items = []
        for name, item in self._trained_index.items():
            items.append(
                {
                    "instance": name,
                    "features": list(item["features"]),
                    "generators": {
                        g: {"commit": list(v)} for g, v in item["commit"].items()
                    },
                }
            )
        payload = {
            "case_folder": case_folder,
            "built_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "coverage": float(self._coverage),
            "threshold": float(self.coverage_threshold),
            "ratios": {
                "train": float(self.train_ratio),
                "val": float(self.val_ratio),
                "test": float(max(0.0, 1.0 - self.train_ratio - self.val_ratio)),
            },
            "split_seed": int(self.split_seed),
            "inputs": list(self._inputs_list),
            "outputs": list(self._outputs_list),
            "split": {
                "train": sorted(self._splits["train"]),
                "val": sorted(self._splits["val"]),
                "test": sorted(self._splits["test"]),
            },
            "items": items,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _compute_splits_from_names(self, names: List[str]) -> None:
        tr: Set[str] = set()
        va: Set[str] = set()
        te: Set[str] = set()
        for nm in names:
            r = _hash01(nm, self.split_seed)
            if r < self.train_ratio:
                tr.add(nm)
            elif r < self.train_ratio + self.val_ratio:
                va.add(nm)
            else:
                te.add(nm)
        self._splits = {"train": tr, "val": va, "test": te}

    def _compute_splits_from_index(self) -> None:
        names = sorted(self._trained_index.keys())
        self._compute_splits_from_names(names)

    def _build_index(self, case_folder: str) -> None:
        inputs = self._list_inputs(case_folder)
        outputs = self._list_outputs(case_folder)
        total_inputs = len(inputs)
        total_outputs = len(outputs)
        self._coverage = (total_outputs / total_inputs) if total_inputs > 0 else 0.0

        self._inputs_list = [self._dataset_name_from_input(p) for p in inputs]
        outputs_names = [self._dataset_name_from_output(p) for p in outputs]
        self._outputs_list = outputs_names

        index: Dict[str, dict] = {}
        for out_path in outputs:
            out = _load_output_solution(out_path)
            if not out:
                continue
            meta = out.get("meta", {})
            instance_name = meta.get("instance_name") or self._dataset_name_from_output(
                out_path
            )
            sys_load = None
            try:
                sys_load = out.get("system", {}).get("load", None)
            except Exception:
                sys_load = None
            if not sys_load:
                in_path = (self.base_input / (instance_name + ".json.gz")).resolve()
                sys_load = _load_input_system_load(in_path)
            if not sys_load:
                continue

            gens = out.get("generators", {}) or {}
            commits = {}
            for gname, gsol in gens.items():
                commits[gname] = list(gsol.get("commit", []))
            feats = _zscore([float(x) for x in sys_load])
            index[instance_name] = {"features": feats, "commit": commits}

        self._trained_index = index
        self._available = len(self._trained_index) > 0
        self._trained = self._available and (self._coverage >= self.coverage_threshold)

        self._compute_splits_from_index()

    def pretrain(
        self, case_folder: Optional[str] = None, force: bool = False
    ) -> Optional[Path]:
        cf = case_folder or self.case_folder
        if not cf:
            return None
        idx_path = self._index_path(cf)
        if idx_path.exists() and not force:
            if self._load_index_file(cf):
                return idx_path
        self._build_index(cf)
        if not self._trained_index:
            return None
        return self._save_index_file(cf)

    def ensure_trained(
        self, case_folder: Optional[str] = None, allow_build_if_missing: bool = True
    ) -> Tuple[bool, float]:
        cf = case_folder or self.case_folder
        if not cf:
            return False, 0.0
        if self._load_index_file(cf):
            return (self._available or self._trained), self._coverage
        if allow_build_if_missing:
            self._build_index(cf)
            return (self._available or self._trained), self._coverage
        return False, 0.0

    def _nearest_neighbor(
        self,
        target_feats: List[float],
        restrict_to_names: Optional[Set[str]] = None,
        exclude_names: Optional[Set[str]] = None,
    ) -> Optional[Tuple[str, dict, float]]:
        best_name = None
        best_item = None
        best_dist = float("inf")
        for name, item in self._trained_index.items():
            if restrict_to_names is not None and name not in restrict_to_names:
                continue
            if exclude_names is not None and name in exclude_names:
                continue
            d = _l2(target_feats, item["features"])
            if d < best_dist:
                best_dist = d
                best_item = item
                best_name = name
        if best_item is None:
            return None
        return best_name, best_item, best_dist

    def generate_and_save_warm_start(
        self,
        instance_name: str,
        use_train_index_only: bool = False,
        exclude_self: bool = True,
        auto_fix: bool = True,
    ) -> Optional[Path]:
        case_folder = _case_folder_from_instance(instance_name)

        if not self._trained_index:
            self.ensure_trained(case_folder, allow_build_if_missing=False)
        if not self._trained_index:
            return None

        input_path = (DataParams._CACHE / (instance_name + ".json.gz")).resolve()
        target_load = _load_input_system_load(input_path)
        if not target_load:
            return None
        target_feats = _zscore([float(x) for x in target_load])

        restrict = self._splits["train"] if use_train_index_only else None
        exclude = {instance_name} if exclude_self else None

        nn = self._nearest_neighbor(
            target_feats, restrict_to_names=restrict, exclude_names=exclude
        )
        if nn is None:
            return None
        nn_name, nn_item, nn_dist = nn

        neighbor_json_path = (self.base_output / (nn_name + ".json")).resolve()
        neighbor = _load_output_solution(neighbor_json_path)
        if neighbor is None:
            return None

        gens = neighbor.get("generators", {}) or {}
        reserves = neighbor.get("reserves", {}) or {}
        network = neighbor.get("network", {}) or {}

        payload = {
            "instance_name": instance_name,
            "case_folder": case_folder,
            "neighbor": nn_name,
            "distance": float(nn_dist),
            "coverage": float(self._coverage),
            "generators": gens,
            "reserves": reserves,
            "network": network,
        }

        fname = f"warm_{_sanitize_name(instance_name)}.json"
        out_path = (self.base_warm / fname).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Automatically produce a fixed warm JSON (no Gurobi), if requested
        if auto_fix:
            try:
                from src.ml_models.fix_warm_start import fix_warm_file

                fix_warm_file(instance_name, warm_file=out_path)
            except Exception as e:
                # non-fatal: fall back to raw warm if fixer fails
                print(f"[warm_start] Auto-fix failed for {instance_name}: {e}")

        return out_path

    @staticmethod
    def _startup_shutdown_from_commit(
        commit: List[int], initial_u: int
    ) -> Tuple[List[int], List[int]]:
        T = len(commit)
        v = [0] * T
        w = [0] * T
        u_prev = initial_u
        for t in range(T):
            u_t = int(round(commit[t]))
            delta = u_t - u_prev
            if delta > 0:
                v[t] = 1
            elif delta < 0:
                w[t] = 1
            u_prev = u_t
        return v, w

    def apply_warm_start_to_model(
        self, model, scenario, instance_name: str, mode: str = "repair"
    ) -> int:
        """
        Set Start on model vars using a warm JSON for 'instance_name'.

        Preference order for input file:
          1) warm_fixed_<instance>.json (if exists)
          2) warm_<instance>.json

        mode:
          - "repair"      -> apply and repair to satisfy easy constraints
          - "commit-only" -> set only u, v, w
          - "as-is"       -> raw application
        """
        # Normalize mode
        mode = (mode or "repair").strip().lower()
        if mode not in ("repair", "commit-only", "as-is"):
            mode = "repair"

        # Prefer fixed warm file if present
        tag = _sanitize_name(instance_name)
        fixed_path = (self.base_warm / f"warm_fixed_{tag}.json").resolve()
        if fixed_path.exists():
            fpath = fixed_path
        else:
            fpath = (self.base_warm / f"warm_{tag}.json").resolve()
            if not fpath.exists():
                alt = (
                    self.base_warm / f"warm_{_sanitize_name(scenario.name)}.json"
                ).resolve()
                if not alt.exists():
                    return 0
                fpath = alt

        try:
            warm = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            return 0

        # Pull model var containers (support SCUC and ED)
        commit_td = getattr(model, "commit", None)
        seg_td = getattr(model, "gen_segment_power", None) or getattr(
            model, "seg_power", None
        )
        startup_td = getattr(model, "startup", None)
        shutdown_td = getattr(model, "shutdown", None)
        reserve_td = getattr(model, "reserve", None)
        shortfall_td = getattr(model, "reserve_shortfall", None)
        line_flow_td = getattr(model, "line_flow", None)
        line_ovp_td = getattr(model, "line_overflow_pos", None)
        line_ovn_td = getattr(model, "line_overflow_neg", None)
        cont_ovp_td = getattr(model, "contingency_overflow_pos", None)
        cont_ovn_td = getattr(model, "contingency_overflow_neg", None)

        applied = 0
        T = scenario.time

        # Helper: binary
        def _b(x):
            return int(1 if float(x) >= 0.5 else 0)

        warm_gen = warm.get("generators", {}) or {}

        # 1) Commitment and startup/shutdown
        gen_u: Dict[str, List[int]] = {}
        for gen in scenario.thermal_units:
            gw = warm_gen.get(gen.name, {}) or {}
            if commit_td is not None:
                u_list = _ensure_list_length(gw.get("commit", []), T, True, 0)
                u_list = [_b(x) for x in u_list]
                for t in range(T):
                    try:
                        commit_td[gen.name, t].Start = u_list[t]
                        applied += 1
                    except Exception:
                        pass
            else:
                u_list = _ensure_list_length(gw.get("commit", []), T, True, 0)
                u_list = [_b(x) for x in u_list]
            gen_u[gen.name] = u_list

            # startup/shutdown from commitment and initial status
            init_u = (
                1 if (gen.initial_status is not None and gen.initial_status > 0) else 0
            )
            v_list, w_list = self._startup_shutdown_from_commit(gen_u[gen.name], init_u)
            if startup_td is not None:
                for t in range(T):
                    try:
                        startup_td[gen.name, t].Start = int(v_list[t])
                        applied += 1
                    except Exception:
                        pass
            if shutdown_td is not None:
                for t in range(T):
                    try:
                        shutdown_td[gen.name, t].Start = int(w_list[t])
                        applied += 1
                    except Exception:
                        pass

        if mode == "commit-only":
            return applied

        # 2) Segment power — basic application (repair mode clips caps)
        gen_above_min: Dict[str, List[float]] = {
            g.name: [0.0] * T for g in scenario.thermal_units
        }
        if seg_td is not None:
            for gen in scenario.thermal_units:
                gw = warm_gen.get(gen.name, {}) or {}
                seg2d = gw.get("segment_power", None)
                nS = len(gen.segments) if gen.segments else 0
                if isinstance(seg2d, list):
                    seg2d = _ensure_list_length(seg2d, T, True, [0.0] * nS)
                else:
                    seg2d = [[0.0] * nS for _ in range(T)]

                for t in range(T):
                    row = seg2d[t] if isinstance(seg2d[t], list) else []
                    row = (row + [0.0] * nS)[:nS]
                    u_t = gen_u[gen.name][t]
                    for s in range(nS):
                        cap = max(0.0, float(gen.segments[s].amount[t])) * float(u_t)
                        val = float(row[s])
                        if mode == "repair":
                            val = max(0.0, min(val, cap))
                        try:
                            seg_td[gen.name, t, s].Start = float(val)
                            applied += 1
                        except Exception:
                            pass
                        gen_above_min[gen.name][t] += float(val)

        # Helper: total production for gen,t
        def _p_total(gen, t: int) -> float:
            u = float(gen_u[gen.name][t])
            p = u * float(gen.min_power[t])
            p += float(gen_above_min[gen.name][t])
            return p

        # 3) Reserves — repair to headroom and shortfall (same as previous version)
        warm_res = warm.get("reserves", {}) or {}
        if reserve_td is not None and scenario.reserves:
            # Prepare warm reserve by product and gen
            pb_map: Dict[str, Dict[str, List[float]]] = {}
            for r in scenario.reserves:
                rw = warm_res.get(r.name, {}) or {}
                pb = rw.get("provided_by_gen", {}) or {}
                inner: Dict[str, List[float]] = {}
                for g in r.thermal_units:
                    gl = _ensure_list_length(pb.get(g.name, []), T, True, 0.0)
                    inner[g.name] = [max(0.0, float(x)) for x in gl]
                pb_map[r.name] = inner

            for gen in scenario.thermal_units:
                for t in range(T):
                    u_t = float(gen_u[gen.name][t])
                    headroom = (float(gen.max_power[t]) - float(gen.min_power[t])) * u_t
                    used = float(gen_above_min[gen.name][t])
                    allow_res = max(0.0, headroom - used)
                    r_sum = 0.0
                    for r in scenario.reserves:
                        r_sum += pb_map.get(r.name, {}).get(gen.name, [0.0] * T)[t]
                    scale = (
                        (allow_res / r_sum)
                        if (r_sum > allow_res + 1e-9 and r_sum > 0)
                        else 1.0
                    )
                    for r in scenario.reserves:
                        val = pb_map.get(r.name, {}).get(gen.name, [0.0] * T)[t] * scale
                        try:
                            reserve_td[r.name, gen.name, t].Start = float(val)
                            applied += 1
                        except Exception:
                            pass

            if shortfall_td is not None:
                for r in scenario.reserves:
                    req = [float(r.amount[t]) for t in range(T)]
                    for t in range(T):
                        provided_t = 0.0
                        for g in r.thermal_units:
                            try:
                                provided_t += float(reserve_td[r.name, g.name, t].Start)
                            except Exception:
                                try:
                                    provided_t += float(reserve_td[r.name, g.name, t].X)
                                except Exception:
                                    pass
                        short = max(0.0, float(req[t]) - provided_t)
                        try:
                            shortfall_td[r.name, t].Start = float(short)
                            applied += 1
                        except Exception:
                            pass

        # 4) Network base-case repair (same as previous version)
        if scenario.lines and line_flow_td is not None:
            try:
                from scipy.sparse import csr_matrix, csc_matrix  # noqa: F401

                isf = scenario.isf.tocsr()
                buses = scenario.buses
                lines = scenario.lines
                ref_1b = getattr(scenario, "ptdf_ref_bus_index", buses[0].index)
                non_ref_bus_indices = sorted(
                    [b.index for b in buses if b.index != ref_1b]
                )
                flows: Dict[Tuple[str, int], float] = {}
                for t in range(T):
                    inj_by_bus: Dict[int, float] = {}
                    for b in buses:
                        inj = 0.0
                        for gen in b.thermal_units:
                            inj += _p_total(gen, t)
                        inj -= float(b.load[t])
                        inj_by_bus[b.index] = float(inj)

                    for line in lines:
                        row = isf.getrow(line.index - 1)
                        f_val = 0.0
                        for col, coeff in zip(row.indices.tolist(), row.data.tolist()):
                            bus_1b = non_ref_bus_indices[col]
                            f_val += float(coeff) * float(inj_by_bus[bus_1b])
                        flows[(line.name, t)] = float(f_val)
                        try:
                            line_flow_td[line.name, t].Start = float(f_val)
                            applied += 1
                        except Exception:
                            pass
                        if line_ovp_td is not None and line_ovn_td is not None:
                            F = float(line.normal_limit[t])
                            ovp = max(0.0, f_val - F)
                            ovn = max(0.0, -f_val - F)
                            try:
                                line_ovp_td[line.name, t].Start = float(ovp)
                                applied += 1
                            except Exception:
                                pass
                            try:
                                line_ovn_td[line.name, t].Start = float(ovn)
                                applied += 1
                            except Exception:
                                pass

                # Contingency slacks (shared) remain as in previous version
                if (
                    cont_ovp_td is not None
                    and cont_ovn_td is not None
                    and scenario.contingencies
                ):
                    lodf_csc = scenario.lodf.tocsc()
                    line_by_row = {ln.index - 1: ln for ln in lines}
                    isf_csc = scenario.isf.tocsc()
                    col_by_bus_1b = {
                        b.index: col for col, b in enumerate(buses) if b.index != ref_1b
                    }
                    req_ovp: Dict[Tuple[str, int], float] = {
                        (ln.name, t): 0.0 for ln in lines for t in range(T)
                    }
                    req_ovn: Dict[Tuple[str, int], float] = {
                        (ln.name, t): 0.0 for ln in lines for t in range(T)
                    }
                    _LODF_TOL = 1e-4
                    for cont in scenario.contingencies:
                        if not cont.lines:
                            continue
                        for out_line in cont.lines:
                            mcol = out_line.index - 1
                            col = lodf_csc.getcol(mcol)
                            for l_row, alpha in zip(
                                col.indices.tolist(), col.data.tolist()
                            ):
                                if l_row == mcol or abs(alpha) < _LODF_TOL:
                                    continue
                                line_l = line_by_row.get(l_row)
                                if line_l is None:
                                    continue
                                for t in range(T):
                                    base_l = float(flows[(line_l.name, t)])
                                    base_m = float(flows[(out_line.name, t)])
                                    post = base_l + float(alpha) * base_m
                                    F_em = float(line_l.emergency_limit[t])
                                    req_ovp[(line_l.name, t)] = max(
                                        req_ovp[(line_l.name, t)], max(0.0, post - F_em)
                                    )
                                    req_ovn[(line_l.name, t)] = max(
                                        req_ovn[(line_l.name, t)],
                                        max(0.0, -post - F_em),
                                    )

                    _ISF_TOL = 1e-8
                    for cont in scenario.contingencies:
                        if not getattr(cont, "units", None):
                            continue
                        for gen in cont.units:
                            bidx = gen.bus.index
                            if bidx == ref_1b or bidx not in col_by_bus_1b:
                                continue
                            col = isf_csc.getcol(col_by_bus_1b[bidx])
                            for t in range(T):
                                pg = _p_total(gen, t)
                                for l_row, coeff in zip(
                                    col.indices.tolist(), col.data.tolist()
                                ):
                                    if abs(coeff) < _ISF_TOL:
                                        continue
                                    line_l = line_by_row.get(l_row)
                                    if line_l is None:
                                        continue
                                    base_l = float(flows[(line_l.name, t)])
                                    post = base_l - float(coeff) * float(pg)
                                    F_em = float(line_l.emergency_limit[t])
                                    req_ovp[(line_l.name, t)] = max(
                                        req_ovp[(line_l.name, t)], max(0.0, post - F_em)
                                    )
                                    req_ovn[(line_l.name, t)] = max(
                                        req_ovn[(line_l.name, t)],
                                        max(0.0, -post - F_em),
                                    )

                    for ln in lines:
                        for t in range(T):
                            try:
                                cont_ovp_td[ln.name, t].Start = float(
                                    req_ovp[(ln.name, t)]
                                )
                                applied += 1
                            except Exception:
                                pass
                            try:
                                cont_ovn_td[ln.name, t].Start = float(
                                    req_ovn[(ln.name, t)]
                                )
                                applied += 1
                            except Exception:
                                pass
            except Exception:
                pass

        return applied

    def get_train_instances(self) -> List[str]:
        return sorted(self._splits.get("train", set()))

    def get_val_instances(self) -> List[str]:
        return sorted(self._splits.get("val", set()))

    def get_test_instances(self) -> List[str]:
        return sorted(self._splits.get("test", set()))

    def get_nontrain_instances(self) -> List[str]:
        return sorted(
            (self._splits.get("val", set()) | self._splits.get("test", set()))
        )

    def get_all_inputs(self) -> List[str]:
        return list(self._inputs_list)

    def get_all_outputs(self) -> List[str]:
        return list(self._outputs_list)

    def report_splits(self) -> None:
        tr = self.get_train_instances()
        va = self.get_val_instances()
        te = self.get_test_instances()
        print("Warm-start index split report")
        print(f"- Train ({len(tr)}):")
        for nm in tr:
            print(f"  {nm}")
        print(f"- Verification/Val ({len(va)}):")
        for nm in va:
            print(f"  {nm}")
        print(f"- Test ({len(te)}):")
        for nm in te:
            print(f"  {nm}")

    def generate_warm_for_split(
        self,
        split: str = "test",
        use_train_db: bool = True,
        limit: Optional[int] = None,
    ) -> List[Tuple[str, Optional[Path]]]:
        split = (split or "test").strip().lower()
        if split in ("val", "verification", "valid", "validation"):
            names = self.get_val_instances()
        elif split in ("train",):
            names = self.get_train_instances()
        else:
            names = self.get_test_instances()
        out: List[Tuple[str, Optional[Path]]] = []
        count = 0
        for nm in names:
            if limit is not None and count >= limit:
                break
            count += 1
            p = self.generate_and_save_warm_start(
                nm, use_train_index_only=use_train_db, exclude_self=True, auto_fix=True
            )
            out.append((nm, p))
        return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Warm-start index inspector with train/val/test split."
    )
    ap.add_argument(
        "--case",
        required=True,
        help="Case folder, e.g., 'matpower/case14'",
    )
    ap.add_argument(
        "--pretrain",
        action="store_true",
        help="Build and persist the index (if missing)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild of index (overwrite existing)",
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help="Print the train/val/test split membership",
    )
    ap.add_argument(
        "--generate-for",
        choices=["train", "val", "test"],
        default=None,
        help="Generate warm-start files for the chosen split (auto-fix on)",
    )
    ap.add_argument(
        "--use-train-db",
        action="store_true",
        default=False,
        help="Restrict neighbor DB to training items when generating warm starts",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="Limit number of items for generation"
    )
    ap.add_argument(
        "--train-ratio", type=float, default=0.70, help="Train ratio (default 0.70)"
    )
    ap.add_argument(
        "--val-ratio", type=float, default=0.15, help="Val ratio (default 0.15)"
    )
    ap.add_argument("--seed", type=int, default=42, help="Split seed (default 42)")
    args = ap.parse_args()

    wsp = WarmStartProvider(
        case_folder=args.case,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_seed=args.seed,
    )
    if args.pretrain or args.force:
        p = wsp.pretrain(force=args.force)
        if p:
            print(f"Index written to: {p}")
        else:
            print("No data to build index.")
    else:
        wsp.ensure_trained(args.case, allow_build_if_missing=True)

    if args.report:
        wsp.report_splits()

    if args.generate_for:
        lim = args.limit if args.limit and args.limit > 0 else None
        results = wsp.generate_warm_for_split(
            split=args.generate_for, use_train_db=args.use_train_db, limit=lim
        )
        ok = sum(1 for _, p in results if p is not None)
        print(
            f"Warm-start files generated for split '{args.generate_for}': {ok}/{len(results)}"
        )

```

### File: `src/optimization_model/SCUC_solver/__init__.py`

```
from src.optimization_model.SCUC_solver import ed_model_builder
from src.optimization_model.SCUC_solver import scuc_model_builder
from src.optimization_model.SCUC_solver import optimizer
from src.optimization_model.SCUC_solver import solve_instances

```

### File: `src/optimization_model/SCUC_solver/compare_ml_raw.py`

```
"""
Benchmark pipeline: compare raw Gurobi vs ML-based (warm-start + redundant-constraint pruning).

What it does:
- Stage 1 (TRAIN): solve with raw Gurobi, save outputs (skip if --skip-existing).
- Stage 2: pretrain warm-start index from TRAIN outputs.
- Stage 2.5 (optional): pretrain redundancy index from TRAIN outputs.
- Stage 3 (TEST): for each instance, solve RAW and WARM(+optional pruning);
  verify feasibility and write:
    • a per-case results CSV under src/data/output/<case>/compare_<tag>_<timestamp>.csv
    • a general logs CSV under src/data/logs/compare_logs_<tag>_<timestamp>.csv
      with (instance, split, method, num vars, num constrs, runtime, max constraint violation, etc.)

Notes:
- Redundancy pruning controls (all OFF unless you pass --rc-enable):
    --rc-enable               Enable pruning
    --rc-thr-abs 20.0         Absolute margin threshold [MW]
    --rc-thr-rel 0.10         Relative margin threshold (fraction of emergency limit)
    --rc-use-train-db         Restrict redundancy k-NN to TRAIN split (default True)
"""

from __future__ import annotations

import argparse
import csv
import math
import time
import requests
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gurobipy import GRB

from src.data_preparation.params import DataParams
from src.data_preparation.read_data import read_benchmark
from src.optimization_model.SCUC_solver.scuc_model_builder import build_model
from src.optimization_model.SCUC_solver.solve_instances import (
    list_remote_instances,
    list_local_cached_instances,
)
from src.optimization_model.helpers.save_json_solution import (
    save_solution_as_json,
    compute_output_path,
)
from src.optimization_model.helpers.verify_solution import (
    verify_solution,
    verify_solution_to_log,
)
from src.optimization_model.helpers.run_utils import allocate_run_id, make_log_filename
from src.ml_models.warm_start import WarmStartProvider, _hash01
from src.ml_models.redundant_constraints import (
    RedundancyProvider as RCProvider,
)  # ML-based constraint pruning provider


def _status_str(code: int) -> str:
    return DataParams.SOLVER_STATUS_STR.get(code, f"STATUS_{code}")


def _case_tag(case_folder: str) -> str:
    t = case_folder.strip().strip("/\\").replace("\\", "/")
    return "".join(ch if ch.isalnum() else "_" for ch in t).strip("_").lower()


def _list_case_instances(case_folder: str) -> List[str]:
    items = list_remote_instances(
        include_filters=[case_folder], roots=["matpower", "test"], max_depth=4
    )
    items = [x for x in items if x.startswith(case_folder)]
    if not items:
        items = list_local_cached_instances(include_filters=[case_folder])
        items = [x for x in items if x.startswith(case_folder)]
    return sorted(set(items))


def _split_instances(
    names: List[str], train_ratio: float, val_ratio: float, seed: int
) -> Tuple[List[str], List[str], List[str]]:
    tr: List[str] = []
    va: List[str] = []
    te: List[str] = []
    for nm in sorted(names):
        r = _hash01(nm, seed)
        if r < train_ratio:
            tr.append(nm)
        elif r < train_ratio + val_ratio:
            va.append(nm)
        else:
            te.append(nm)
    return tr, va, te


@dataclass
class SolveResult:
    instance_name: str
    split: str
    method: str  # raw | warm
    status: str
    status_code: int
    runtime_sec: float
    mip_gap: Optional[float]
    obj_val: Optional[float]
    obj_bound: Optional[float]
    nodes: Optional[float]
    feasible_ok: Optional[bool]
    warm_start_applied_vars: Optional[int] = None
    # New fields for logging/analysis
    num_vars: Optional[int] = None
    num_bin_vars: Optional[int] = None
    num_int_vars: Optional[int] = None
    num_constrs: Optional[int] = None
    max_constraint_violation: Optional[float] = None


def _metrics_from_model(
    model,
) -> Tuple[
    int, float, Optional[float], Optional[float], Optional[float], Optional[float]
]:
    code = int(getattr(model, "Status", -1))
    runtime = float(getattr(model, "Runtime", 0.0) or 0.0)
    try:
        mip_gap = float(model.MIPGap)
    except Exception:
        mip_gap = None
    try:
        obj_val = float(model.ObjVal)
    except Exception:
        obj_val = None
    try:
        obj_bound = float(model.ObjBound)
    except Exception:
        obj_bound = None
    try:
        nodes = float(getattr(model, "NodeCount", 0.0))
    except Exception:
        nodes = None
    return code, runtime, mip_gap, obj_val, obj_bound, nodes


def _model_size(model) -> Tuple[int, int, int, int]:
    """
    Return (num_vars, num_bin_vars, num_int_vars, num_constrs)
    """
    try:
        nvars = int(getattr(model, "NumVars", 0))
    except Exception:
        nvars = 0
    try:
        nbin = int(getattr(model, "NumBinVars", 0))
    except Exception:
        nbin = 0
    try:
        nint = int(getattr(model, "NumIntVars", 0))
    except Exception:
        nint = 0
    try:
        ncons = int(getattr(model, "NumConstrs", 0))
    except Exception:
        ncons = 0
    return nvars, nbin, nint, ncons


def _max_constraint_violation(checks) -> Optional[float]:
    """
    Compute max violation among constraint checks (IDs starting with 'C-').
    Returns None if checks missing; 0.0 if all within tolerance.
    """
    if not checks:
        return None
    vals: List[float] = []
    for c in checks:
        try:
            if isinstance(c.idx, str) and c.idx.startswith("C-"):
                v = float(c.value)
                if math.isfinite(v):
                    vals.append(v)
        except Exception:
            continue
    return max(vals) if vals else 0.0


def _save_logs_if_requested(sc, model, save_logs: bool) -> None:
    if not save_logs:
        return
    run_id = allocate_run_id(sc.name or "scenario")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        sol_fname = make_log_filename("solution", sc.name, run_id, ts)
        _ = Path(DataParams._LOGS / sol_fname)
        from src.optimization_model.helpers.save_solution import (
            save_solution_to_log as _save,
        )

        _save(sc, model, filename=sol_fname)
    except Exception:
        pass
    try:
        ver_fname = make_log_filename("verify", sc.name, run_id, ts)
        verify_solution_to_log(sc, model, filename=ver_fname)
    except Exception:
        pass


def _instance_cache_path_and_url(instance_name: str) -> Tuple[Path, str]:
    gz_name = f"{instance_name}.json.gz"
    local_path = (DataParams._CACHE / gz_name).resolve()
    url = f"{DataParams.INSTANCES_URL.rstrip('/')}/{gz_name}"
    return local_path, url


def _robust_download(instance_name: str, attempts: int, timeout: int) -> bool:
    """
    Try to download instance_name.json.gz to the input cache with retry and timeout.
    Returns True on success, False on final failure.
    """
    local_path, url = _instance_cache_path_and_url(instance_name)
    if local_path.is_file():
        return True

    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")

    backoff_base = 2.0
    for k in range(1, max(1, attempts) + 1):
        try:
            with requests.get(url, stream=True, timeout=max(1, int(timeout))) as r:
                r.raise_for_status()
                with tmp_path.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
            tmp_path.replace(local_path)
            return True
        except Exception:
            if k < attempts:
                sleep_s = min(60.0, backoff_base ** (k - 1))
                time.sleep(sleep_s)
            else:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                return False
    return False


def _prepare_results_csv(case_folder: str) -> Path:
    """
    Create CSV file (under src/data/output/<case>/) and write header. Return path.
    """
    tag = _case_tag(case_folder)
    out_dir = (DataParams._OUTPUT / case_folder).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"compare_{tag}_{ts}.csv"

    header = [
        "timestamp",
        "instance_name",
        "split",
        "method",
        "status",
        "status_code",
        "runtime_sec",
        "mip_gap",
        "obj_val",
        "obj_bound",
        "nodes",
        "feasible_ok",
        "warm_start_applied_vars",
        "num_vars",
        "num_bin_vars",
        "num_int_vars",
        "num_constrs",
        "max_constraint_violation",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
    return path


def _append_result_to_csv(csv_path: Path, r: SolveResult) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                now,
                r.instance_name,
                r.split,
                r.method,
                r.status,
                r.status_code,
                f"{r.runtime_sec:.6f}",
                "" if r.mip_gap is None else f"{r.mip_gap:.8f}",
                "" if r.obj_val is None else f"{r.obj_val:.6f}",
                "" if r.obj_bound is None else f"{r.obj_bound:.6f}",
                "" if r.nodes is None else f"{r.nodes:.0f}",
                "" if r.feasible_ok is None else ("OK" if r.feasible_ok else "FAIL"),
                ""
                if r.warm_start_applied_vars is None
                else int(r.warm_start_applied_vars),
                "" if r.num_vars is None else int(r.num_vars),
                "" if r.num_bin_vars is None else int(r.num_bin_vars),
                "" if r.num_int_vars is None else int(r.num_int_vars),
                "" if r.num_constrs is None else int(r.num_constrs),
                ""
                if r.max_constraint_violation is None
                else f"{float(r.max_constraint_violation):.8f}",
            ]
        )


def _prepare_logs_csv(case_folder: str) -> Path:
    """
    Create a general logs CSV under src/data/logs and write header. Return path.
    Includes general info that the user requested (instance, method, sizes, times, violations).
    """
    tag = _case_tag(case_folder)
    logs_dir = DataParams._LOGS.resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = logs_dir / f"compare_logs_{tag}_{ts}.csv"
    header = [
        "timestamp",
        "instance_name",
        "split",
        "method",
        "status",
        "runtime_sec",
        "obj_val",
        "num_vars",
        "num_constrs",
        "num_bin_vars",
        "num_int_vars",
        "max_constraint_violation",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(header)
    return path


def _append_result_to_logs_csv(csv_path: Path, r: SolveResult) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                now,
                r.instance_name,
                r.split,
                r.method,
                r.status,
                f"{r.runtime_sec:.6f}",
                "" if r.obj_val is None else f"{r.obj_val:.6f}",
                "" if r.num_vars is None else int(r.num_vars),
                "" if r.num_constrs is None else int(r.num_constrs),
                "" if r.num_bin_vars is None else int(r.num_bin_vars),
                "" if r.num_int_vars is None else int(r.num_int_vars),
                ""
                if r.max_constraint_violation is None
                else f"{float(r.max_constraint_violation):.8f}",
            ]
        )


def _build_contingency_filter(
    rc: Optional[RCProvider],
    sc,
    instance_name: str,
    enable: bool,
    thr_abs: float,
    thr_rel: float,
    use_train_db: bool,
) -> Optional[callable]:
    if not enable or rc is None:
        return None
    try:
        res = rc.make_filter_for_instance(
            sc,
            instance_name,
            thr_abs=thr_abs,
            thr_rel=thr_rel,
            use_train_index_only=use_train_db,
            exclude_self=True,
        )
        if res is None:
            return None
        predicate, stats = res
        print(
            f"[redundancy] Using neighbor {stats.get('neighbor')} (dist={stats.get('distance'):.3f}); "
            f"constraints skipped will be counted in builder logs."
        )
        return predicate
    except Exception:
        return None


def _solve_raw(
    instance_name: str,
    time_limit: int,
    mip_gap: float,
    split: str,
    save_logs: bool,
    skip_existing: bool,
    download_attempts: int,
    download_timeout: int,
    *,
    rc_provider: Optional[RCProvider] = None,
    rc_enable: bool = False,
    rc_thr_abs: float = 20.0,
    rc_thr_rel: float = 0.10,
    rc_use_train_db: bool = True,
) -> SolveResult:
    out_json_path = compute_output_path(instance_name)
    if skip_existing and out_json_path.is_file():
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="raw",
            status="SKIPPED_EXISTING",
            status_code=-2,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )

    ok_dl = _robust_download(
        instance_name=instance_name,
        attempts=max(1, int(download_attempts)),
        timeout=max(1, int(download_timeout)),
    )
    if not ok_dl:
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="raw",
            status="DOWNLOAD_FAIL",
            status_code=-3,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )

    try:
        inst = read_benchmark(instance_name, quiet=True)
    except Exception:
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="raw",
            status="READ_FAIL",
            status_code=-4,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )

    sc = inst.deterministic
    cont_filter = _build_contingency_filter(
        rc_provider,
        sc,
        instance_name,
        rc_enable,
        rc_thr_abs,
        rc_thr_rel,
        rc_use_train_db,
    )
    model = build_model(scenario=sc, contingency_filter=cont_filter)
    try:
        model.Params.OutputFlag = 1
        model.Params.MIPGap = mip_gap
        model.Params.TimeLimit = time_limit
        model.Params.NumericFocus = 1
    except Exception:
        pass
    model.optimize()

    try:
        save_solution_as_json(sc, model, instance_name=instance_name)
    except Exception:
        pass

    feasible_ok = None
    max_viol = None
    try:
        feasible_ok, checks, _ = verify_solution(sc, model)
        max_viol = _max_constraint_violation(checks)
    except Exception:
        feasible_ok = None
        max_viol = None

    _save_logs_if_requested(sc, model, save_logs)

    code, runtime, gap, obj, bound, nodes = _metrics_from_model(model)
    nvars, nbin, nint, ncons = _model_size(model)
    return SolveResult(
        instance_name=instance_name,
        split=split,
        method="raw",
        status=_status_str(code),
        status_code=code,
        runtime_sec=runtime,
        mip_gap=gap,
        obj_val=obj,
        obj_bound=bound,
        nodes=nodes,
        feasible_ok=feasible_ok,
        warm_start_applied_vars=None,
        num_vars=nvars,
        num_bin_vars=nbin,
        num_int_vars=nint,
        num_constrs=ncons,
        max_constraint_violation=max_viol,
    )


def _solve_warm(
    instance_name: str,
    wsp: WarmStartProvider,
    time_limit: int,
    mip_gap: float,
    split: str,
    warm_mode: str,
    save_logs: bool,
    skip_existing: bool,
    download_attempts: int,
    download_timeout: int,
    *,
    rc_provider: Optional[RCProvider] = None,
    rc_enable: bool = False,
    rc_thr_abs: float = 20.0,
    rc_thr_rel: float = 0.10,
    rc_use_train_db: bool = True,
) -> SolveResult:
    out_json_path = compute_output_path(instance_name)
    if skip_existing and out_json_path.is_file():
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="warm",
            status="SKIPPED_EXISTING",
            status_code=-2,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )

    ok_dl = _robust_download(
        instance_name=instance_name,
        attempts=max(1, int(download_attempts)),
        timeout=max(1, int(download_timeout)),
    )
    if not ok_dl:
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="warm",
            status="DOWNLOAD_FAIL",
            status_code=-3,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )

    try:
        inst = read_benchmark(instance_name, quiet=True)
    except Exception:
        return SolveResult(
            instance_name=instance_name,
            split=split,
            method="warm",
            status="READ_FAIL",
            status_code=-4,
            runtime_sec=0.0,
            mip_gap=None,
            obj_val=None,
            obj_bound=None,
            nodes=None,
            feasible_ok=None,
            warm_start_applied_vars=None,
        )
    sc = inst.deterministic

    try:
        wsp.generate_and_save_warm_start(
            instance_name, use_train_index_only=True, exclude_self=True, auto_fix=True
        )
    except Exception:
        pass

    cont_filter = _build_contingency_filter(
        rc_provider,
        sc,
        instance_name,
        rc_enable,
        rc_thr_abs,
        rc_thr_rel,
        rc_use_train_db,
    )
    model = build_model(scenario=sc, contingency_filter=cont_filter)

    mode = warm_mode.strip().lower()
    if mode == "fixed":
        mode = "repair"
    applied = 0
    try:
        applied = wsp.apply_warm_start_to_model(model, sc, instance_name, mode=mode)
    except Exception:
        applied = 0

    try:
        model.Params.OutputFlag = 1
        model.Params.MIPGap = mip_gap
        model.Params.TimeLimit = time_limit
        model.Params.NumericFocus = 1
    except Exception:
        pass

    model.optimize()

    try:
        save_solution_as_json(sc, model, instance_name=instance_name)
    except Exception:
        pass

    feasible_ok = None
    max_viol = None
    try:
        feasible_ok, checks, _ = verify_solution(sc, model)
        max_viol = _max_constraint_violation(checks)
    except Exception:
        feasible_ok = None
        max_viol = None

    _save_logs_if_requested(sc, model, save_logs)

    code, runtime, gap, obj, bound, nodes = _metrics_from_model(model)
    nvars, nbin, nint, ncons = _model_size(model)
    return SolveResult(
        instance_name=instance_name,
        split=split,
        method="warm",
        status=_status_str(code),
        status_code=code,
        runtime_sec=runtime,
        mip_gap=gap,
        obj_val=obj,
        obj_bound=bound,
        nodes=nodes,
        feasible_ok=feasible_ok,
        warm_start_applied_vars=applied,
        num_vars=nvars,
        num_bin_vars=nbin,
        num_int_vars=nint,
        num_constrs=ncons,
        max_constraint_violation=max_viol,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Compare raw Gurobi vs warm-start (+ optional redundancy pruning) on TRAIN/TEST for a case."
    )
    ap.add_argument("--case", required=True, help="Case folder, e.g., matpower/case118")
    ap.add_argument(
        "--time-limit",
        type=int,
        default=600,
        help="Time limit [s] for both RAW and WARM",
    )
    ap.add_argument(
        "--mip-gap",
        type=float,
        default=0.05,
        help="Relative MIP gap for both RAW and WARM",
    )
    ap.add_argument(
        "--train-ratio", type=float, default=0.70, help="Train ratio for split"
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation ratio; remainder is test",
    )
    ap.add_argument("--seed", type=int, default=42, help="Split seed")
    ap.add_argument(
        "--limit-train",
        type=int,
        default=0,
        help="Limit number of TRAIN instances (0 => no limit)",
    )
    ap.add_argument(
        "--limit-test",
        type=int,
        default=0,
        help="Limit number of TEST instances (0 => no limit)",
    )
    ap.add_argument(
        "--save-logs",
        action="store_true",
        default=False,
        help="Save human logs to src/data/logs (solution and verification reports)",
    )
    ap.add_argument(
        "--warm-mode",
        choices=["fixed", "commit-only", "as-is"],
        default="fixed",
        help="Warm-start application mode for evaluation",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="TRAIN ONLY: skip instances that already have a JSON solution in src/data/output",
    )
    ap.add_argument(
        "--download-attempts",
        type=int,
        default=3,
        help="Number of attempts to download a missing instance (default: 3)",
    )
    ap.add_argument(
        "--download-timeout",
        type=int,
        default=60,
        help="Per-attempt HTTP timeout for instance download in seconds (default: 60)",
    )
    # Redundancy pruning
    ap.add_argument(
        "--rc-enable",
        action="store_true",
        default=False,
        help="Enable ML-based redundant contingency pruning (OFF by default)",
    )
    ap.add_argument(
        "--rc-thr-abs",
        type=float,
        default=20.0,
        help="Absolute margin threshold [MW] to prune a constraint (default 20 MW)",
    )
    ap.add_argument(
        "--rc-thr-rel",
        type=float,
        default=0.50,
        help="Relative margin threshold (fraction of F_em) to prune (default 0.50)",
    )
    ap.add_argument(
        "--rc-use-train-db",
        action="store_true",
        default=True,
        help="Restrict redundancy k-NN to TRAIN split (default True)",
    )

    args = ap.parse_args()

    case_folder = args.case.strip().strip("/\\").replace("\\", "/")
    all_instances = _list_case_instances(case_folder)
    if not all_instances:
        print(f"No instances found for {case_folder}.")
        return

    train, val, test = _split_instances(
        all_instances, args.train_ratio, args.val_ratio, args.seed
    )
    if not train:
        print("Empty TRAIN split; cannot pretrain warm-start index. Aborting.")
        return
    if not test:
        print("Empty TEST split; nothing to compare. Aborting.")
        return

    if args.limit_train and args.limit_train > 0:
        train = train[: args.limit_train]
    if args.limit_test and args.limit_test > 0:
        test = test[: args.limit_test]

    print(f"Case: {case_folder}")
    print(f"- Train: {len(train)}")
    print(f"- Val  : {len(val)}")
    print(f"- Test : {len(test)}")
    if args.skip_existing:
        print(
            "Note: --skip-existing applies to TRAIN stage only. TEST runs will re-solve even if outputs exist."
        )

    # Prepare CSVs
    csv_path = _prepare_results_csv(case_folder)
    logs_csv_path = _prepare_logs_csv(case_folder)
    print(f"Results will be appended to: {csv_path}")
    print(f"General logs CSV will be appended to: {logs_csv_path}")

    results: List[SolveResult] = []

    # Stage 1: RAW Train (allow skipping existing)
    print("Stage 1: Solving TRAIN with raw Gurobi (no warm start) ...")
    for nm in train:
        r = _solve_raw(
            nm,
            args.time_limit,
            args.mip_gap,
            split="train",
            save_logs=args.save_logs,
            skip_existing=args.skip_existing,
            download_attempts=args.download_attempts,
            download_timeout=args.download_timeout,
            rc_provider=None,
            rc_enable=False,  # Do not prune while building train outputs
        )
        _append_result_to_csv(csv_path, r)
        _append_result_to_logs_csv(logs_csv_path, r)
        results.append(r)

    # Stage 2: Pretrain warm-start index from outputs (existing solutions are used)
    print("Stage 2: Pretraining warm-start index from TRAIN outputs ...")
    wsp = WarmStartProvider(case_folder=case_folder)
    wsp.pretrain(force=True)
    trained, cov = wsp.ensure_trained(case_folder, allow_build_if_missing=False)
    print(f"- Warm-start index: trained={trained}, coverage={cov:.3f}")
    if not trained:
        print(
            "Index not trained (insufficient outputs). Evaluation will still try warm, but coverage may be low."
        )

    # Stage 2.5: Pretrain redundancy index from TRAIN outputs (if enabled)
    rc_provider = None
    if args.rc_enable:
        print("Stage 2.5: Pretraining redundancy index from TRAIN outputs ...")
        rc_provider = RCProvider(case_folder=case_folder)
        rc_provider.pretrain(force=True)
        ok, rc_cov = rc_provider.ensure_trained(
            case_folder, allow_build_if_missing=False
        )
        print(f"- Redundancy index: available={ok}, coverage={rc_cov:.3f}")
        if not ok:
            print("Redundancy index not available. Pruning will be OFF.")
            rc_provider = None

    # Stage 3: Compare on TEST (RAW vs WARM) — always re-solve; ignore skip-existing
    print("Stage 3: Comparing on TEST (RAW vs WARM) ...")
    for nm in test:
        # RAW
        r_raw = _solve_raw(
            nm,
            args.time_limit,
            args.mip_gap,
            split="test",
            save_logs=args.save_logs,
            skip_existing=False,
            download_attempts=args.download_attempts,
            download_timeout=args.download_timeout,
            rc_provider=rc_provider,
            rc_enable=args.rc_enable,
            rc_thr_abs=args.rc_thr_abs,
            rc_thr_rel=args.rc_thr_rel,
            rc_use_train_db=args.rc_use_train_db,
        )
        _append_result_to_csv(csv_path, r_raw)
        _append_result_to_logs_csv(logs_csv_path, r_raw)
        results.append(r_raw)

        # WARM
        r_warm = _solve_warm(
            nm,
            wsp,
            args.time_limit,
            args.mip_gap,
            split="test",
            warm_mode=args.warm_mode,
            save_logs=args.save_logs,
            skip_existing=False,
            download_attempts=args.download_attempts,
            download_timeout=args.download_timeout,
            rc_provider=rc_provider,
            rc_enable=args.rc_enable,
            rc_thr_abs=args.rc_thr_abs,
            rc_thr_rel=args.rc_thr_rel,
            rc_use_train_db=args.rc_use_train_db,
        )
        _append_result_to_csv(csv_path, r_warm)
        _append_result_to_logs_csv(logs_csv_path, r_warm)
        results.append(r_warm)

    print(f"Results appended to: {csv_path}")
    print(f"General logs appended to: {logs_csv_path}")

    # Simple summary
    test_pairs = {}
    for r in results:
        if r.split != "test":
            continue
        test_pairs.setdefault(r.instance_name, {})[r.method] = r
    speed_wins = 0
    obj_wins = 0
    count_pairs = 0
    for nm, pair in test_pairs.items():
        if "raw" in pair and "warm" in pair:
            if pair["raw"].status not in (
                "SKIPPED_EXISTING",
                "DOWNLOAD_FAIL",
                "READ_FAIL",
            ) and pair["warm"].status not in (
                "SKIPPED_EXISTING",
                "DOWNLOAD_FAIL",
                "READ_FAIL",
            ):
                count_pairs += 1
                if pair["warm"].runtime_sec < pair["raw"].runtime_sec:
                    speed_wins += 1
                if (pair["warm"].obj_val is not None) and (
                    pair["raw"].obj_val is not None
                ):
                    if pair["warm"].obj_val <= pair["raw"].obj_val + 1e-6:
                        obj_wins += 1
    if count_pairs > 0:
        print(
            f"Warm faster on {speed_wins}/{count_pairs} test instances; objective <= raw on {obj_wins}/{count_pairs}."
        )


if __name__ == "__main__":
    main()

```

### File: `src/optimization_model/SCUC_solver/ed_model_builder.py`

```
"""Simple Economic Dispatch with segmented costs and commitment.

This file now only orchestrates the model building by delegating:
  - data preparation (total load)
  - variable creation (commitment, segment power)
  - constraints (commitment fixing, linking, power balance)
  - objective (production cost)

All logic lives under:
  src/optimization_model/solver/economic_dispatch/
"""

from __future__ import annotations

import gurobipy as gp

from src.data_preparation.data_structure import UnitCommitmentScenario
from src.optimization_model.solver.economic_dispatch.data.load import compute_total_load
from src.optimization_model.solver.economic_dispatch import (
    vars as ed_vars,
    constraints as ed_cons,
    objectives as ed_obj,
)


def build_model(scenario: UnitCommitmentScenario) -> gp.Model:
    """Build a segmented ED model with commitment using modular components."""
    model = gp.Model("SimpleEconomicDispatch_Segmented")

    units = scenario.thermal_units
    time_periods = range(scenario.time)

    # Data preparation
    total_load = compute_total_load(scenario.buses, scenario.time)

    # Variables
    commit = ed_vars.commitment.add_variables(model, units, time_periods)
    seg_power = ed_vars.segment_power.add_variables(model, units, time_periods)

    # Keep attributes used by downstream code (e.g., optimizer pretty print)
    model.__dict__["_commit"] = commit
    model.__dict__["_seg_power"] = seg_power

    # Constraints
    ed_cons.commitment_fixing.add_constraints(model, units, commit, time_periods)
    ed_cons.linking.add_constraints(model, units, commit, seg_power, time_periods)
    ed_cons.power_balance_segmented.add_constraints(
        model, total_load, units, commit, seg_power, time_periods
    )

    # Objective
    ed_obj.power_cost_segmented.set_objective(
        model, units, commit, seg_power, time_periods
    )

    return model

```

### File: `src/optimization_model/SCUC_solver/optimizer.py`

```
import argparse
import logging
from datetime import datetime
from typing import List, Optional, Tuple

from gurobipy import GRB

from src.data_preparation.read_data import read_benchmark
from src.optimization_model.SCUC_solver.scuc_model_builder import build_model
from src.optimization_model.helpers.save_solution import save_solution_to_log
from src.optimization_model.helpers.verify_solution import verify_solution_to_log
from src.optimization_model.helpers.run_utils import allocate_run_id, make_log_filename
from src.optimization_model.helpers.save_json_solution import (
    save_solution_as_json,
    compute_output_path,
)
from src.data_preparation.params import DataParams

# Warm-start provider and instance listers
from src.ml_models.warm_start import WarmStartProvider
from src.optimization_model.SCUC_solver.solve_instances import (
    list_remote_instances,
    list_local_cached_instances,
)

logging.basicConfig(level=logging.INFO)

logging.getLogger("gurobipy").setLevel(logging.WARNING)
logging.getLogger("gurobipy").propagate = False

logger = logging.getLogger(__name__)

# Default remote listing scope (kept internal; no CLI noise)
_DEFAULT_ROOTS = ["matpower", "test"]
_DEFAULT_MAX_DEPTH = 4


def _status_str(code: int) -> str:
    mapping = DataParams.SOLVER_STATUS_STR
    return mapping.get(code, f"STATUS_{code}")


def _solve_one(
    name: str,
    time_limit: int,
    mip_gap: float,
    use_warm_start: bool,
    require_pretrained: bool,
    skip_existing: bool,
    warm_cache: dict,
    warm_start_mode: str,
    warm_use_train_db: bool,
    save_logs: bool,
) -> Tuple[bool, Optional[str]]:
    """
    Solve one dataset by name; optionally apply a warm start if available.
    Returns (success, output_json_path_if_any).
    """
    out_json_path = compute_output_path(name)
    if skip_existing and out_json_path.is_file():
        logger.info("Skipping existing solution: %s", out_json_path)
        return True, str(out_json_path)

    inst = read_benchmark(name, quiet=True)
    sc = inst.deterministic
    logger.info(
        "Build '%s' (units=%d, buses=%d, lines=%d, T=%d, dt=%dmin)",
        name,
        len(sc.thermal_units),
        len(sc.buses),
        len(sc.lines),
        sc.time,
        sc.time_step,
    )

    # Warm-start prepare (only if explicitly requested)
    wsp = None
    if use_warm_start:
        cf = "/".join(name.strip().split("/")[:2])
        wsp = warm_cache.get(cf)
        if wsp is None:
            wsp = WarmStartProvider(case_folder=cf)
            trained, cov = wsp.ensure_trained(
                cf, allow_build_if_missing=not require_pretrained
            )
            if not trained:
                logger.info(
                    "Warm-start: no trained index for %s (coverage=%.3f; require_pretrained=%s)",
                    cf,
                    cov,
                    require_pretrained,
                )
                wsp = None
            else:
                warm_cache[cf] = wsp
                logger.info(
                    "Warm-start: using pre-trained index for %s (coverage=%.3f)",
                    cf,
                    cov,
                )

    # If trained index present, write warm file for this instance
    if wsp is not None:
        try:
            warm_path = wsp.generate_and_save_warm_start(
                name, use_train_index_only=warm_use_train_db, exclude_self=True
            )
            if warm_path:
                logger.info("Warm-start file generated: %s", warm_path)
            else:
                logger.info(
                    "Warm-start not generated for '%s' (no neighbor or missing features).",
                    name,
                )
        except Exception as e:
            logger.warning("Warm-start generation failed for '%s': %s", name, e)

    # Build model
    model = build_model(sc)

    # Apply warm start if available
    if wsp is not None:
        try:
            # map CLI term 'fixed' to internal 'repair'
            mode = warm_start_mode.strip().lower()
            if mode == "fixed":
                mode = "repair"
            assigned = wsp.apply_warm_start_to_model(model, sc, name, mode=mode)
            if assigned > 0:
                logger.info(
                    "Warm-start applied (%s): Start set on %d variable(s).",
                    mode,
                    assigned,
                )
            else:
                logger.info("No warm-start variables set.")
        except Exception as e:
            logger.warning("Failed to apply warm-start to model: %s", e)

    # Solver params
    try:
        model.Params.OutputFlag = 1
        model.Params.MIPGap = mip_gap
        model.Params.TimeLimit = time_limit
        model.Params.NumericFocus = 1
    except Exception:
        pass

    # Optimize
    model.optimize()
    logger.info("Solver status: %s", _status_str(model.Status))

    # Save JSON solution (used as training data and for later analysis)
    if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        try:
            out_path = save_solution_as_json(sc, model, instance_name=name)
            logger.info("Saved JSON solution: %s", out_path)
        except Exception as e:
            logger.warning("Failed to save JSON solution for '%s': %s", name, e)

        if save_logs:
            # Also write human-readable logs and verification
            run_id = allocate_run_id(sc.name)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                sol_fname = make_log_filename(
                    kind="solution", scenario=sc.name, run_id=run_id, ts=ts
                )
                sol_path = save_solution_to_log(sc, model, filename=sol_fname)
                logger.info("Full solution saved to: %s", sol_path)
            except Exception:
                pass
            try:
                ver_fname = make_log_filename(
                    kind="verify", scenario=sc.name, run_id=run_id, ts=ts
                )
                ver_path = verify_solution_to_log(sc, model, filename=ver_fname)
                logger.info("Verification report saved to: %s", ver_path)
            except Exception:
                pass
        return True, str(out_json_path)
    else:
        return False, None


def main():
    ap = argparse.ArgumentParser(
        description="Solve UnitCommitment.jl SCUC instances. Defaults keep everything OFF (no warm start, no logs)."
    )
    # What to solve
    ap.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Explicit dataset names (e.g., matpower/case14/2017-01-01).",
    )
    ap.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Tokens to include in dataset name (e.g., case57). If omitted, --instances is required.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Solve at most N instances (0 => unlimited)",
    )

    # Solver controls
    ap.add_argument("--time-limit", type=int, default=600, help="Gurobi time limit [s]")
    ap.add_argument("--mip-gap", type=float, default=0.05, help="Gurobi MIP gap")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Do not re-solve instances that already have a JSON solution.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List instances and exit (do not solve)",
    )

    # Warm start (all OFF by default)
    ap.add_argument(
        "--warm",
        dest="use_warm_start",
        action="store_true",
        default=False,
        help="Use warm start (generate from pre-trained DB if available).",
    )
    ap.add_argument(
        "--require-pretrained",
        action="store_true",
        default=False,
        help="Only use pre-trained warm-start index (do not auto-build from outputs).",
    )
    ap.add_argument(
        "--warm-mode",
        choices=["fixed", "commit-only", "as-is"],
        default="fixed",
        help="Warm-start application mode. 'fixed' = repaired for feasibility (recommended).",
    )
    ap.add_argument(
        "--warm-use-train-db",
        action="store_true",
        default=False,
        help="Restrict neighbor search to the training split of the index.",
    )

    # Logs (OFF by default)
    ap.add_argument(
        "--save-logs",
        action="store_true",
        default=False,
        help="Write human-readable solution and verification logs to src/data/logs",
    )

    args = ap.parse_args()

    # Build instance list
    instances: List[str] = []
    if args.instances:
        instances = args.instances
    elif args.include:
        logger.info("Listing remote instances from: %s", DataParams.INSTANCES_URL)
        instances = list_remote_instances(
            include_filters=args.include,
            roots=_DEFAULT_ROOTS,
            max_depth=_DEFAULT_MAX_DEPTH,
        )
        if not instances:
            logger.warning("Remote listing returned none. Falling back to local cache.")
            instances = list_local_cached_instances(include_filters=args.include)
        if not instances:
            logger.error("No instances found for include tokens: %s", args.include)
            return
    else:
        logger.error("Please pass either --instances or --include filters.")
        return

    if args.dry_run:
        for n in instances:
            print(n)
        return

    # Solve
    warm_cache = {}  # case_folder -> WarmStartProvider (loaded index)
    limit = args.limit if args.limit and args.limit > 0 else None
    solved = 0
    total = 0
    for name in instances:
        if limit is not None and total >= limit:
            break
        total += 1
        logger.info("[%d/%s] Solving %s", total, str(limit) if limit else "-", name)
        ok, out_json = _solve_one(
            name=name,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            use_warm_start=args.use_warm_start,
            require_pretrained=args.require_pretrained,
            skip_existing=args.skip_existing,
            warm_cache=warm_cache,
            warm_start_mode=args.warm_mode,
            warm_use_train_db=args.warm_use_train_db,
            save_logs=args.save_logs,
        )
        if ok:
            solved += 1

    logger.info("Done. Solved %d/%d", solved, total)


if __name__ == "__main__":
    main()

```

### File: `src/optimization_model/SCUC_solver/scuc_model_builder.py`

```
"""Simple SCUC with segmented costs and commitment.

This file orchestrates model building by delegating:
  - data preparation (total load)
  - variable creation (commitment, segment power, reserve, startup/shutdown, line flows)
  - constraints (commitment fixing, linking, power balance, reserve, initial conditions/ramping, line flow PTDF, line limits, min up/down, line/generator-contingency limits)
  - objective (production cost + reserve shortfall penalty + line overflow penalty)
"""

from __future__ import annotations

import gurobipy as gp
import logging
from typing import Optional, Callable

from src.data_preparation.data_structure import UnitCommitmentScenario
from src.optimization_model.solver.scuc.data.load import compute_total_load
from src.optimization_model.solver.scuc import (
    vars as scuc_vars,
    constraints as scuc_cons,
    objectives as scuc_obj,
)

logger = logging.getLogger(__name__)


def build_model(
    scenario: UnitCommitmentScenario,
    *,
    contingency_filter: Optional[Callable] = None,
) -> gp.Model:
    """Build a segmented SCUC model with commitment, startup/shutdown, reserves, and line constraints.

    Parameters
    ----------
    scenario : UnitCommitmentScenario
    contingency_filter : Optional[Callable]
        Optional predicate(kind, line_l, out_obj, t, coeff, F_em) -> bool.
        If provided, contingency constraints (+/- pairs) for which this returns False are pruned.
    """
    model = gp.Model("SCUC_Segmented")

    units = scenario.thermal_units
    time_periods = range(scenario.time)

    # Data preparation
    total_load = compute_total_load(scenario.buses, scenario.time)

    # Scenario summary
    logger.info(
        "SCUC build: units=%d, buses=%d, lines=%d, reserves=%d, contingencies=%d, T=%d, dt(min)=%d",
        len(scenario.thermal_units),
        len(scenario.buses),
        len(scenario.lines),
        len(scenario.reserves),
        len(scenario.contingencies),
        scenario.time,
        scenario.time_step,
    )
    if scenario.lines:
        try:
            isf = scenario.isf
            lodf = scenario.lodf
            ref_bus = getattr(scenario, "ptdf_ref_bus_index", scenario.buses[0].index)
            logger.info(
                "PTDF/ISF shape=%s, nnz=%d; LODF shape=%s, nnz=%d; ref_bus_index(1-based)=%s",
                getattr(isf, "shape", None),
                getattr(isf, "nnz", None),
                getattr(lodf, "shape", None),
                getattr(lodf, "nnz", None),
                ref_bus,
            )
        except Exception:
            pass

    # Variables

    gen_commit = scuc_vars.commitment.add_variables(model, units, time_periods)
    gen_segment_power = scuc_vars.segment_power.add_variables(
        model, units, time_periods
    )
    gen_startup, gen_shutdown = scuc_vars.startup_shutdown.add_variables(
        model, units, time_periods
    )

    reserve = None
    reserve_shortfall = None
    if scenario.reserves:
        reserve, reserve_shortfall = scuc_vars.reserve.add_variables(
            model, scenario.reserves, units, time_periods
        )

    line_flow = None
    line_over_pos = None
    line_over_neg = None
    if scenario.lines:
        line_flow, line_over_pos, line_over_neg = scuc_vars.line_flow.add_variables(
            model, scenario.lines, time_periods
        )

    # Shared contingency overflow slacks (per line/time)
    cont_over_pos = None
    cont_over_neg = None
    if scenario.lines and scenario.contingencies:
        from src.optimization_model.solver.scuc.vars import (
            contingency_overflow as covars,
        )

        cont_over_pos, cont_over_neg = covars.add_variables(
            model, scenario.lines, time_periods
        )

    # Constraints
    scuc_cons.commitment_fixing.add_constraints(model, units, gen_commit, time_periods)

    scuc_cons.linking.add_constraints(
        model, units, gen_commit, gen_segment_power, time_periods
    )

    scuc_cons.power_balance_segmented.add_constraints(
        model, total_load, units, gen_commit, gen_segment_power, time_periods
    )

    scuc_cons.initial_conditions.add_constraints(
        model,
        units,
        gen_commit,
        gen_segment_power,
        gen_startup,
        gen_shutdown,
        time_periods,
    )

    scuc_cons.min_up_down.add_constraints(
        model, units, gen_commit, gen_startup, gen_shutdown, time_periods
    )

    if scenario.reserves:
        scuc_cons.reserve.add_constraints(
            model,
            scenario.reserves,
            gen_commit,
            gen_segment_power,
            reserve,
            time_periods,
        )

        scuc_cons.reserve_requirement.add_constraints(
            model, scenario.reserves, reserve, reserve_shortfall, time_periods
        )

    if scenario.lines:
        scuc_cons.line_flow_ptdf.add_constraints(
            model, scenario, gen_commit, gen_segment_power, line_flow, time_periods
        )

        scuc_cons.line_limits.add_constraints(
            model, scenario.lines, line_flow, line_over_pos, line_over_neg, time_periods
        )

        # Line and generator contingency limits with shared slacks
        if scenario.contingencies:
            scuc_cons.contingencies.add_constraints(
                model,
                scenario,
                gen_commit,
                gen_segment_power,
                line_flow,
                time_periods,
                cont_over_pos,
                cont_over_neg,
                filter_predicate=contingency_filter,
            )

    # Objective with reserve penalty and line overflow penalty (base case and contingencies)
    scuc_obj.power_cost_segmented.set_objective(
        model,
        units,
        gen_commit,
        gen_segment_power,
        time_periods,
        scenario.reserves,
        scenario.lines,
    )

    # Expose key data for downstream tooling (solution dump, verification)
    model.__dict__["_total_load"] = total_load
    model.__dict__["_scenario_name"] = scenario.name or "scenario"

    return model

```

### File: `src/optimization_model/SCUC_solver/solve_instances.py`

```
"""
Batch solver for UnitCommitment.jl instances hosted at:
  https://axavier.org/UnitCommitment.jl/0.4/instances/

Features
- Discover remote .json.gz instances by parsing directory listings (HTML index pages).
- Filter which instances/folders to solve (e.g., case14, case30, case57).
- Download (cache), build SCUC model, optimize, and save solution as JSON under src/data/output.
- Skip instances that already have a saved JSON (resume-friendly).
- Log unsolved instances (with reasons) to src/data/logs.
- Append per-instance performance rows to ONE CSV per case folder:
  src/data/output/<case_folder>/perf_<case_tag>_<technique>_<run>.csv
  e.g., for all items under matpower/case14:
        src/data/output/matpower/case14/perf_matpower_case14_basic_01.csv

Usage
  python -m src.optimization_model.SCUC_solver.solve_instances --include case57 case30 case14

Useful flags:
  --option basic         # technique tag embedded in CSV filename (basic, warm_start, ...)
  --max-depth 4
  --time-limit 600
  --mip-gap 0.05
  --roots matpower test
  --limit 0
  --skip-existing
  --dry-run
"""

from __future__ import annotations

import argparse
import logging
import re
from urllib.parse import urljoin, urlparse
from typing import Iterable, List, Set, Tuple, Optional

import requests
from gurobipy import GRB

from src.data_preparation.params import DataParams
from src.data_preparation.read_data import read_benchmark
from src.optimization_model.SCUC_solver.scuc_model_builder import build_model
from src.optimization_model.helpers.save_json_solution import (
    save_solution_as_json,
    compute_output_path,
)
from src.optimization_model.helpers.perf_logger import PerfCSVLogger

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _fetch_links(url: str, timeout: int = 30) -> List[str]:
    """
    Fetch an HTML index page and return list of href-links (as strings).
    Non-HTML responses return empty list.
    """
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        text = r.text
    except Exception as e:
        logger.warning("Failed to fetch listing '%s': %s", url, e)
        return []

    # crude anchor parser; sufficient for simple directory index listings
    hrefs = re.findall(r'href=[\'"]([^\'"]+)[\'"]', text, flags=re.IGNORECASE)
    return hrefs


def _normalize_href(base_url: str, href: str) -> Optional[str]:
    """
    Make href absolute and canonical. Ignore fragments and query.
    """
    if not href or href.startswith("#"):
        return None
    abs_url = urljoin(base_url if base_url.endswith("/") else base_url + "/", href)
    return abs_url


def list_remote_instances(
    include_filters: Optional[List[str]] = None,
    roots: Optional[List[str]] = None,
    max_depth: int = 4,
) -> List[str]:
    """
    Recursively list remote .json.gz instances and return dataset names
    relative to DataParams.INSTANCES_URL without extension.

    Example return values:
      - 'test/case14'
      - 'matpower/case30/2017-06-24'
      - 'matpower/case57/2017-01-01'
    """
    base = DataParams.INSTANCES_URL.rstrip("/") + "/"
    root_urls: List[str] = []
    if roots:
        for r in roots:
            r = str(r).strip().strip("/\\")
            root_urls.append(urljoin(base, r + "/"))
    else:
        root_urls = [base]

    visited: Set[str] = set()
    results: List[str] = []

    base_path = urlparse(base).path

    def _walk(url: str, depth: int):
        if url in visited or depth > max_depth:
            return
        visited.add(url)
        links = _fetch_links(url)
        if not links:
            return

        for href in links:
            abs_url = _normalize_href(url, href)
            if abs_url is None:
                continue
            if abs_url in visited:
                continue
            # Only traverse under the same host/base
            if not abs_url.startswith(base):
                continue

            if abs_url.endswith("/"):
                _walk(abs_url, depth + 1)
            elif abs_url.lower().endswith(".json.gz"):
                # Derive dataset name relative to base path without extension
                path = urlparse(abs_url).path
                if not path.startswith(base_path):
                    continue
                rel = path[len(base_path) :].lstrip("/")
                if rel.endswith(".json.gz"):
                    rel = rel[: -len(".json.gz")]
                dataset_name = rel.replace("\\", "/")
                # Apply include filters (if any)
                if include_filters:
                    if not any(tok in dataset_name for tok in include_filters):
                        continue
                results.append(dataset_name)

    for ru in root_urls:
        _walk(ru, depth=0)

    # Make unique and sorted for reproducibility
    return sorted(set(results))


def list_local_cached_instances(
    include_filters: Optional[List[str]] = None,
) -> List[str]:
    """
    Fallback: list locally cached .json.gz under src/data/input and return dataset
    names without extension, filtered by include_filters if provided.
    """
    base = DataParams._CACHE.resolve()
    out: List[str] = []
    for p in base.rglob("*.json.gz"):
        try:
            rel = p.resolve().relative_to(base).as_posix()
        except Exception:
            continue
        if rel.endswith(".json.gz"):
            rel = rel[: -len(".json.gz")]
        if include_filters:
            if not any(tok in rel for tok in include_filters):
                continue
        out.append(rel)
    return sorted(set(out))


def solve_instance(
    name: str,
    time_limit: int,
    mip_gap: float,
    skip_existing: bool = True,
    perf_logger: Optional[PerfCSVLogger] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Solve a single instance by name (e.g., 'matpower/case30/2017-06-24').

    Returns:
      (success, output_json_path, error_message_if_any)
    """
    try:
        out_json_path = compute_output_path(name)
        if skip_existing and out_json_path.is_file():
            logger.info("Skipping existing solution: %s", out_json_path)
            return True, str(out_json_path), None

        inst = read_benchmark(name, quiet=True)
        sc = inst.deterministic
        logger.info(
            "Building model: '%s' (units=%d, lines=%d, T=%d, dt=%dmin)",
            name,
            len(sc.thermal_units),
            len(sc.lines),
            sc.time,
            sc.time_step,
        )
        model = build_model(sc)

        try:
            model.Params.OutputFlag = 1
            model.Params.MIPGap = mip_gap
            model.Params.TimeLimit = time_limit
            model.Params.NumericFocus = 1
        except Exception:
            pass

        model.optimize()
        status = model.Status

        # Always log performance for this attempt (even if infeasible or interrupted)
        if perf_logger is not None:
            try:
                csv_path = perf_logger.append_result(name, sc, model)
                logger.info("Appended performance row to: %s", csv_path)
            except Exception as e:
                logger.warning("Failed to append performance CSV row: %s", e)

        if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            out_path = save_solution_as_json(sc, model, instance_name=name)
            logger.info("Saved JSON solution: %s", out_path)
            return True, str(out_path), None
        else:
            msg = f"Unsolved: status={status}"
            logger.warning("%s for '%s'", msg, name)
            return False, None, msg
    except Exception as e:
        emsg = f"Exception while solving '{name}': {e}"
        logger.exception(emsg)
        return False, None, emsg


def _append_to_log(filename: str, lines: Iterable[str]) -> None:
    path = DataParams._LOGS / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for ln in lines:
            fh.write(ln.rstrip("\n") + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch solve UnitCommitment.jl instances and save JSON solutions."
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=["case57", "case30", "case14"],
        help="Filter tokens to include (e.g., case57 case30 case14)",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=["matpower", "test"],
        help="Top-level folders to search under the base URL",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum recursion depth for remote listing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of instances to solve (0 => unlimited)",
    )
    parser.add_argument(
        "--time-limit", type=int, default=600, help="Gurobi time limit in seconds"
    )
    parser.add_argument("--mip-gap", type=float, default=0.05, help="Gurobi MIP gap")
    parser.add_argument(
        "--option",
        type=str,
        default="basic",
        help="Technique/option tag placed into CSV filename (e.g., basic, warm_start, redundant_constraints)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip instances that already have an output JSON",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Do not skip existing outputs",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List instances but do not solve"
    )

    args = parser.parse_args()

    # 1) List remote; if empty, fall back to locally cached
    logger.info("Listing remote instances from: %s", DataParams.INSTANCES_URL)
    instances = list_remote_instances(
        include_filters=args.include, roots=args.roots, max_depth=args.max_depth
    )
    if not instances:
        logger.warning(
            "Remote listing returned no instances. Falling back to local cache."
        )
        instances = list_local_cached_instances(include_filters=args.include)

    if not instances:
        logger.error("No instances found matching filters %s", args.include)
        return

    logger.info("Found %d instance(s) to consider.", len(instances))

    if args.dry_run:
        for n in instances:
            print(n)
        return

    # Create a performance CSV logger for this technique
    perf_logger = PerfCSVLogger(
        technique=args.option, base_output_dir=DataParams._OUTPUT
    )

    # 2) Solve
    unsolved_log_name = "unsolved_instances.log"
    solved_log_name = "solved_instances.log"
    solved_count = 0
    unsolved_count = 0

    limit = args.limit if args.limit and args.limit > 0 else None
    count = 0

    for name in instances:
        if limit is not None and count >= limit:
            break
        count += 1
        logger.info("[%d/%s] Solving %s", count, str(limit) if limit else "-", name)
        ok, out_json, err = solve_instance(
            name=name,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            skip_existing=args.skip_existing,
            perf_logger=perf_logger,
        )
        if ok:
            solved_count += 1
            _append_to_log(solved_log_name, [f"{name} -> {out_json}"])
        else:
            unsolved_count += 1
            _append_to_log(unsolved_log_name, [f"{name} : {err}"])

    logger.info("Done. Solved=%d, Unsolved=%d", solved_count, unsolved_count)
    if unsolved_count > 0:
        logger.info("See unsolved log: %s", DataParams._LOGS / unsolved_log_name)
    logger.info("Solved items log: %s", DataParams._LOGS / solved_log_name)


if __name__ == "__main__":
    main()

```

### File: `src/optimization_model/helpers/__init__.py`

```
from src.optimization_model.helpers import run_utils
from src.optimization_model.helpers import save_solution
from src.optimization_model.helpers import verify_solution
from src.optimization_model.helpers import restore_solution
from src.optimization_model.helpers import save_json_solution
from src.optimization_model.helpers import perf_logger

```

### File: `src/optimization_model/helpers/perf_logger.py`

```
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from gurobipy import Model, GRB

from src.data_preparation.data_structure import UnitCommitmentScenario
from src.data_preparation.params import DataParams

# Try to import contingency penalty factor used by objective to compute consistent penalty contribution
try:
    from src.optimization_model.solver.scuc.objectives.power_cost_segmented import (
        _CONTINGENCY_PENALTY_FACTOR as CONTINGENCY_PENALTY_FACTOR,
    )
except Exception:
    CONTINGENCY_PENALTY_FACTOR = 10.0


def _status_str(code: int) -> str:
    mapping = DataParams.SOLVER_STATUS_STR
    return mapping.get(code, f"STATUS_{code}")


def _sanitize_token(s: str) -> str:
    s = (s or "").strip()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    # normalize repeated underscores
    t = "".join(out)
    while "__" in t:
        t = t.replace("__", "_")
    return t.strip("_").lower()


def case_folder_from_instance(instance_name: str) -> str:
    """
    For 'matpower/case300/2017-06-24' -> 'matpower/case300'
    For 'test/case14'                 -> 'test/case14'
    For 'matpower/case14'             -> 'matpower/case14'
    """
    parts = instance_name.strip().strip("/\\").replace("\\", "/").split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] if parts else "case"


def _sanitized_case_tag(case_folder: str) -> str:
    """
    'matpower/case14' -> 'matpower_case14'
    """
    return _sanitize_token(case_folder)


@dataclass
class PerfRow:
    # identity/meta
    timestamp: str
    instance_name: str
    case_folder: str
    technique: str
    run_id: int
    # solver info
    status: str
    status_code: int
    runtime_sec: float
    mip_gap: Optional[float]
    obj_val: Optional[float]
    obj_bound: Optional[float]
    nodes: Optional[float]
    # model size
    num_vars: int
    num_bin_vars: int
    num_int_vars: int
    num_constrs: int
    num_cont_vars: int
    num_nzs: Optional[int]
    # data size
    buses: int
    units: int
    lines: int
    reserves: int
    time_steps: int
    time_step_min: int
    ptdf_nnz: Optional[int]
    lodf_nnz: Optional[int]
    # penalties and counts
    startup_count: Optional[float]
    startup_cost: Optional[float]
    reserve_shortfall_mw: Optional[float]
    reserve_shortfall_penalty: Optional[float]
    base_overflow_mw: Optional[float]
    base_overflow_penalty: Optional[float]
    cont_overflow_mw: Optional[float]
    cont_overflow_penalty: Optional[float]


class PerfCSVLogger:
    """
    Create and append rows to one CSV per case folder (e.g., matpower/case14):

      src/data/output/<case_folder>/perf_<case_tag>_<technique>_<run_id>.csv

    - case_folder is typically 'matpower/case14'
    - case_tag is that folder with slashes replaced: 'matpower_case14'
    - technique is a short user label ('basic', 'warm_start', etc.)
    - run_id increments per (case, technique)
    """

    HEADER = [
        "timestamp",
        "instance_name",
        "case_folder",
        "technique",
        "run_id",
        "status",
        "status_code",
        "runtime_sec",
        "mip_gap",
        "obj_val",
        "obj_bound",
        "nodes",
        "num_vars",
        "num_bin_vars",
        "num_int_vars",
        "num_cont_vars",
        "num_constrs",
        "num_nzs",
        "buses",
        "units",
        "lines",
        "reserves",
        "time_steps",
        "time_step_min",
        "ptdf_nnz",
        "lodf_nnz",
        "startup_count",
        "startup_cost",
        "reserve_shortfall_mw",
        "reserve_shortfall_penalty",
        "base_overflow_mw",
        "base_overflow_penalty",
        "cont_overflow_mw",
        "cont_overflow_penalty",
    ]

    def __init__(self, technique: str, base_output_dir: Optional[Path] = None):
        self.technique = _sanitize_token(technique) or "basic"
        self.base_dir = base_output_dir or DataParams._OUTPUT
        self._case_to_csv: Dict[str, Path] = {}
        self._case_to_runid: Dict[str, int] = {}

    def _allocate_run_id(self, case_folder: str) -> int:
        # Scan existing perf_ files to get max id, then +1
        case_dir = self.base_dir / case_folder
        case_dir.mkdir(parents=True, exist_ok=True)
        tag = _sanitized_case_tag(case_folder)
        prefix = f"perf_{tag}_{self.technique}_"
        max_id = 0
        if case_dir.exists():
            for p in case_dir.glob(f"{prefix}*.csv"):
                name = p.name
                try:
                    # Expect suffix before .csv to be nn or nnn (digits)
                    stem = name[: -len(".csv")] if name.endswith(".csv") else name
                    run_part = stem.split("_")[-1]
                    rid = int(run_part)
                    if rid > max_id:
                        max_id = rid
                except Exception:
                    continue
        return max_id + 1

    def _get_csv_path(self, case_folder: str) -> Tuple[Path, int]:
        if case_folder in self._case_to_csv:
            return self._case_to_csv[case_folder], self._case_to_runid[case_folder]
        rid = self._allocate_run_id(case_folder)
        tag = _sanitized_case_tag(case_folder)
        filename = f"perf_{tag}_{self.technique}_{rid:02d}.csv"
        path = (self.base_dir / case_folder / filename).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write header if new file
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(self.HEADER)
        self._case_to_csv[case_folder] = path
        self._case_to_runid[case_folder] = rid
        return path, rid

    def _sum_over_vars(self, var_td, index_pairs) -> float:
        if var_td is None:
            return 0.0
        s = 0.0
        for idx in index_pairs:
            try:
                s += float(var_td[idx].X)
            except Exception:
                continue
        return s

    def _compute_penalties_and_counts(
        self, sc: UnitCommitmentScenario, model: Model
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        # Startup
        startup = getattr(model, "startup", None)
        startup_cost_total = 0.0
        startup_count_total = 0.0
        if startup is not None:
            for g in sc.thermal_units:
                # chosen policy: 'hot start' cost = min startup category cost (0 if none)
                cost = 0.0
                try:
                    if g.startup_categories:
                        cost = float(min(cat.cost for cat in g.startup_categories))
                except Exception:
                    cost = 0.0
                for t in range(sc.time):
                    try:
                        v = float(startup[g.name, t].X)
                    except Exception:
                        v = 0.0
                    startup_count_total += v
                    startup_cost_total += cost * v

        # Reserve shortfall and penalty
        shortfall_total = 0.0
        shortfall_penalty_total = 0.0
        short = getattr(model, "reserve_shortfall", None)
        if short is not None and sc.reserves:
            for r in sc.reserves:
                pen = float(r.shortfall_penalty)
                for t in range(sc.time):
                    try:
                        s = float(short[r.name, t].X)
                    except Exception:
                        s = 0.0
                    shortfall_total += s
                    shortfall_penalty_total += pen * s

        # Base overflow: sum(MW) and penalty
        base_ovp_total = 0.0
        base_ovn_total = 0.0
        base_penalty_total = 0.0
        ovp = getattr(model, "line_overflow_pos", None)
        ovn = getattr(model, "line_overflow_neg", None)
        if sc.lines and ovp is not None and ovn is not None:
            for ln in sc.lines:
                for t in range(sc.time):
                    try:
                        vpos = float(ovp[ln.name, t].X)
                    except Exception:
                        vpos = 0.0
                    try:
                        vneg = float(ovn[ln.name, t].X)
                    except Exception:
                        vneg = 0.0
                    base_ovp_total += vpos
                    base_ovn_total += vneg
                    base_penalty_total += (vpos + vneg) * float(ln.flow_penalty[t])

        # Contingency overflow: use shared slacks and same line penalty scaled by factor
        cont_ovp_total = 0.0
        cont_ovn_total = 0.0
        cont_penalty_total = 0.0
        covp = getattr(model, "contingency_overflow_pos", None)
        covn = getattr(model, "contingency_overflow_neg", None)
        if sc.lines and covp is not None and covn is not None:
            for ln in sc.lines:
                for t in range(sc.time):
                    try:
                        vpos = float(covp[ln.name, t].X)
                    except Exception:
                        vpos = 0.0
                    try:
                        vneg = float(covn[ln.name, t].X)
                    except Exception:
                        vneg = 0.0
                    cont_ovp_total += vpos
                    cont_ovn_total += vneg
                    cont_penalty_total += (
                        (vpos + vneg)
                        * float(ln.flow_penalty[t])
                        * float(CONTINGENCY_PENALTY_FACTOR)
                    )

        base_overflow_mw = base_ovp_total + base_ovn_total
        cont_overflow_mw = cont_ovp_total + cont_ovn_total
        return (
            startup_count_total,
            startup_cost_total,
            shortfall_total,
            shortfall_penalty_total,
            base_overflow_mw,
            base_penalty_total,
            cont_overflow_mw,
            cont_penalty_total,
        )

    def _make_row(
        self, instance_name: str, sc: UnitCommitmentScenario, model: Model, run_id: int
    ) -> PerfRow:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_code = getattr(model, "Status", -1)
        status = _status_str(status_code)
        runtime = getattr(model, "Runtime", None)
        mip_gap = None
        try:
            mip_gap = float(model.MIPGap)
        except Exception:
            mip_gap = None
        try:
            obj_val = float(model.ObjVal)
        except Exception:
            obj_val = None
        try:
            obj_bound = float(model.ObjBound)
        except Exception:
            obj_bound = None
        try:
            nodes = float(getattr(model, "NodeCount", 0.0))
        except Exception:
            nodes = None

        num_vars = int(getattr(model, "NumVars", 0))
        num_bin = int(getattr(model, "NumBinVars", 0))
        num_int = int(getattr(model, "NumIntVars", 0))
        num_constrs = int(getattr(model, "NumConstrs", 0))
        try:
            num_nzs = int(getattr(model, "NumNZs"))
        except Exception:
            num_nzs = None
        num_cont = num_vars - num_int  # Gurobi counts integer (incl bin) as int vars

        ptdf_nnz = None
        lodf_nnz = None
        try:
            ptdf_nnz = int(getattr(sc.isf, "nnz", 0)) if sc.isf is not None else 0
        except Exception:
            pass
        try:
            lodf_nnz = int(getattr(sc.lodf, "nnz", 0)) if sc.lodf is not None else 0
        except Exception:
            pass

        (
            startup_count,
            startup_cost,
            r_short_mw,
            r_short_penalty,
            base_ov_mw,
            base_penalty,
            cont_ov_mw,
            cont_penalty,
        ) = self._compute_penalties_and_counts(sc, model)

        case_folder = case_folder_from_instance(instance_name)

        return PerfRow(
            timestamp=now,
            instance_name=instance_name,
            case_folder=case_folder,
            technique=self.technique,
            run_id=run_id,
            status=status,
            status_code=int(status_code),
            runtime_sec=float(runtime) if runtime is not None else 0.0,
            mip_gap=mip_gap,
            obj_val=obj_val,
            obj_bound=obj_bound,
            nodes=nodes,
            num_vars=num_vars,
            num_bin_vars=num_bin,
            num_int_vars=num_int,
            num_constrs=num_constrs,
            num_cont_vars=num_cont,
            num_nzs=num_nzs,
            buses=len(sc.buses),
            units=len(sc.thermal_units),
            lines=len(sc.lines),
            reserves=len(sc.reserves) if sc.reserves else 0,
            time_steps=int(sc.time),
            time_step_min=int(sc.time_step),
            ptdf_nnz=ptdf_nnz,
            lodf_nnz=lodf_nnz,
            startup_count=startup_count,
            startup_cost=startup_cost,
            reserve_shortfall_mw=r_short_mw,
            reserve_shortfall_penalty=r_short_penalty,
            base_overflow_mw=base_ov_mw,
            base_overflow_penalty=base_penalty,
            cont_overflow_mw=cont_ov_mw,
            cont_overflow_penalty=cont_penalty,
        )

    def append_result(
        self, instance_name: str, sc: UnitCommitmentScenario, model: Model
    ) -> Path:
        """
        Build a row from model+scenario and append it to the correct per-case CSV.
        Returns the CSV path.
        """
        case_folder = case_folder_from_instance(instance_name)
        csv_path, run_id = self._get_csv_path(case_folder)
        row = self._make_row(instance_name, sc, model, run_id)

        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(
                [
                    row.timestamp,
                    row.instance_name,
                    row.case_folder,
                    row.technique,
                    row.run_id,
                    row.status,
                    row.status_code,
                    f"{row.runtime_sec:.6f}",
                    "" if row.mip_gap is None else f"{row.mip_gap:.8f}",
                    "" if row.obj_val is None else f"{row.obj_val:.6f}",
                    "" if row.obj_bound is None else f"{row.obj_bound:.6f}",
                    "" if row.nodes is None else f"{row.nodes:.0f}",
                    row.num_vars,
                    row.num_bin_vars,
                    row.num_int_vars,
                    row.num_cont_vars,
                    row.num_constrs,
                    "" if row.num_nzs is None else row.num_nzs,
                    row.buses,
                    row.units,
                    row.lines,
                    row.reserves,
                    row.time_steps,
                    row.time_step_min,
                    "" if row.ptdf_nnz is None else row.ptdf_nnz,
                    "" if row.lodf_nnz is None else row.lodf_nnz,
                    "" if row.startup_count is None else f"{row.startup_count:.4f}",
                    "" if row.startup_cost is None else f"{row.startup_cost:.6f}",
                    ""
                    if row.reserve_shortfall_mw is None
                    else f"{row.reserve_shortfall_mw:.6f}",
                    ""
                    if row.reserve_shortfall_penalty is None
                    else f"{row.reserve_shortfall_penalty:.6f}",
                    ""
                    if row.base_overflow_mw is None
                    else f"{row.base_overflow_mw:.6f}",
                    ""
                    if row.base_overflow_penalty is None
                    else f"{row.base_overflow_penalty:.6f}",
                    ""
                    if row.cont_overflow_mw is None
                    else f"{row.cont_overflow_mw:.6f}",
                    ""
                    if row.cont_overflow_penalty is None
                    else f"{row.cont_overflow_penalty:.6f}",
                ]
            )
        return csv_path

```

### File: `src/optimization_model/helpers/restore_solution.py`

```
from typing import Dict, List, Any
from gurobipy import Model, GRB
from src.data_preparation.data_structure import UnitCommitmentScenario


def restore_solution(scenario: UnitCommitmentScenario, model: Model) -> Dict[str, Any]:
    """
    Restore a full SCUC solution from a solved Gurobi model into Python structures.

    Returns a nested dict with:
      - objective: float
      - status: str
      - generators: dict[name] with
          initial_status_steps: int or None
          initial_u: int (0/1)
          initial_power: float
          commit: List[int] (0/1 over time)
          startup: List[int] (0/1 over time)
          shutdown: List[int] (0/1 over time)
          min_power_output: List[float]  (u * min_power)
          segment_power: List[List[float]] (per t: per-segment power)
          total_power: List[float]  (min + sum(segments))
      - reserves: dict[reserve_name] with
          requirement: List[float]
          shortfall: List[float]
          provided_by_gen: dict[gen_name] -> List[float]
          total_provided: List[float]
      - network:
          lines: dict[line_name] with
            source: str
            target: str
            flow: List[float]
            limit: List[float]  (normal limit)
            overflow_pos: List[float]
            overflow_neg: List[float]
            penalty: List[float] (per-time penalty $/MW)
      - system:
          load: List[float]
          total_production: List[float]
    """
    T = scenario.time
    generators = scenario.thermal_units

    # Helper to convert solver status to string
    def _status_str(code: int) -> str:
        mapping = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.TIME_LIMIT: "TIME_LIMIT",
        }
        return mapping.get(code, f"STATUS_{code}")

    solution: Dict[str, Any] = {
        "objective": float(getattr(model, "ObjVal", float("nan"))),
        "status": _status_str(getattr(model, "Status", -1)),
        "generators": {},
        "system": {
            "load": [sum(b.load[t] for b in scenario.buses) for t in range(T)],
            "total_production": [0.0 for _ in range(T)],
        },
    }

    # Model attributes created by SCUC builders
    commit_vars = getattr(model, "commit", None)
    seg_vars = getattr(model, "gen_segment_power", None)
    startup_vars = getattr(model, "startup", None)
    shutdown_vars = getattr(model, "shutdown", None)

    for gen in generators:
        n_segments = len(gen.segments) if gen.segments else 0
        gen_commit_list: List[int] = []
        gen_startup_list: List[int] = []
        gen_shutdown_list: List[int] = []
        gen_min_output_list: List[float] = []
        gen_segment_power_list: List[List[float]] = []
        gen_total_power_list: List[float] = []

        initial_status_steps = (
            gen.initial_status if gen.initial_status is not None else None
        )
        initial_u = (
            1 if (gen.initial_status is not None and gen.initial_status > 0) else 0
        )
        initial_power = (
            float(gen.initial_power) if gen.initial_power is not None else 0.0
        )

        for t in range(T):
            u_val = 0.0
            if commit_vars is not None:
                u_val = commit_vars[gen.name, t].X
            gen_commit_list.append(int(round(u_val)))

            st = startup_vars[gen.name, t].X if startup_vars is not None else 0.0
            sh = shutdown_vars[gen.name, t].X if shutdown_vars is not None else 0.0
            gen_startup_list.append(int(round(st)))
            gen_shutdown_list.append(int(round(sh)))

            min_output = u_val * float(gen.min_power[t])
            gen_min_output_list.append(min_output)

            segs_t: List[float] = []
            if n_segments > 0 and seg_vars is not None:
                for s in range(n_segments):
                    segs_t.append(float(seg_vars[gen.name, t, s].X))
            gen_segment_power_list.append(segs_t)

            total = min_output + (sum(segs_t) if segs_t else 0.0)
            gen_total_power_list.append(total)

        solution["generators"][gen.name] = {
            "initial_status_steps": initial_status_steps,
            "initial_u": initial_u,
            "initial_power": initial_power,
            "commit": gen_commit_list,
            "startup": gen_startup_list,
            "shutdown": gen_shutdown_list,
            "min_power_output": gen_min_output_list,
            "segment_power": gen_segment_power_list,
            "total_power": gen_total_power_list,
        }

        # accumulate system production
        for t in range(T):
            solution["system"]["total_production"][t] += solution["generators"][
                gen.name
            ]["total_power"][t]

    # Reserves (if present)
    reserve_defs = scenario.reserves
    if reserve_defs:
        reserve_vars = getattr(model, "reserve", None)
        shortfall_vars = getattr(model, "reserve_shortfall", None)
        reserves_out: Dict[str, Any] = {}

        for reserve_def in reserve_defs:
            req = [float(reserve_def.amount[t]) for t in range(T)]
            short = [0.0 for _ in range(T)]
            provided_by_gen: Dict[str, List[float]] = {}
            total_provided = [0.0 for _ in range(T)]

            # Shortfall per t
            if shortfall_vars is not None:
                for t in range(T):
                    short[t] = float(shortfall_vars[reserve_def.name, t].X)

            # Provision by eligible generators
            if reserve_vars is not None:
                for gen in reserve_def.thermal_units:
                    gvals: List[float] = []
                    for t in range(T):
                        gvals.append(
                            float(reserve_vars[reserve_def.name, gen.name, t].X)
                        )
                        total_provided[t] += gvals[-1]
                    provided_by_gen[gen.name] = gvals

            reserves_out[reserve_def.name] = {
                "requirement": req,
                "shortfall": short,
                "provided_by_gen": provided_by_gen,
                "total_provided": total_provided,
            }

        solution["reserves"] = reserves_out

    # Network: line flows and overflows
    lines = scenario.lines
    flow_vars = getattr(model, "line_flow", None)
    ovp_vars = getattr(model, "line_overflow_pos", None)
    ovn_vars = getattr(model, "line_overflow_neg", None)
    if lines:
        net_out: Dict[str, Any] = {"lines": {}}
        for line in lines:
            flow = [0.0 for _ in range(T)]
            ovp = [0.0 for _ in range(T)]
            ovn = [0.0 for _ in range(T)]
            limit = [float(line.normal_limit[t]) for t in range(T)]
            penalty = [float(line.flow_penalty[t]) for t in range(T)]
            if flow_vars is not None:
                for t in range(T):
                    flow[t] = float(flow_vars[line.name, t].X)
            if ovp_vars is not None:
                for t in range(T):
                    ovp[t] = float(ovp_vars[line.name, t].X)
            if ovn_vars is not None:
                for t in range(T):
                    ovn[t] = float(ovn_vars[line.name, t].X)
            net_out["lines"][line.name] = {
                "source": line.source.name,
                "target": line.target.name,
                "flow": flow,
                "limit": limit,
                "overflow_pos": ovp,
                "overflow_neg": ovn,
                "penalty": penalty,
            }
        solution["network"] = net_out

    return solution

```

### File: `src/optimization_model/helpers/run_utils.py`

```
import re
from typing import Optional
from src.data_preparation.params import DataParams


def allocate_run_id(scenario: Optional[str]) -> int:
    """
    Scan the logs directory and allocate the next run id for the given scenario.
    Filenames are expected to follow: SCUC_{scenario}_runNNN_*.{solution|verify}.<ext>
    """
    out_dir = DataParams._LOGS
    out_dir.mkdir(parents=True, exist_ok=True)
    scen = (scenario or "scenario").replace("/", "_")
    pattern = re.compile(
        rf"^SCUC_{re.escape(scen)}_run(\d+)_.*\.(?:solution|verify)\.[^.]+$"
    )
    max_id = 0
    for p in out_dir.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if m:
            try:
                rid = int(m.group(1))
                if rid > max_id:
                    max_id = rid
            except Exception:
                pass
    return max_id + 1


def make_log_filename(kind: str, scenario: Optional[str], run_id: int, ts: str) -> str:
    """
    Create a standardized filename for logs.

    kind: 'solution' or 'verify'
    scenario: scenario/case name
    run_id: integer
    ts: timestamp string like YYYYmmdd_HHMMSS

    We now save:
      - solution as JSON:  SCUC_<scenario>_runNNN_<ts>.solution.json
      - verify as text:    SCUC_<scenario>_runNNN_<ts>.verify.log
    """
    scen = (scenario or "scenario").replace("/", "_")
    if kind == "solution":
        return f"SCUC_{scen}_run{run_id:03d}_{ts}.solution.log"
    else:
        return f"SCUC_{scen}_run{run_id:03d}_{ts}.verify.log"

```

### File: `src/optimization_model/helpers/save_json_solution.py`

```
import json
from pathlib import Path
from typing import Optional, Dict, Any

from gurobipy import Model
from src.data_preparation.data_structure import UnitCommitmentScenario
from src.optimization_model.helpers.restore_solution import restore_solution
from src.data_preparation.params import DataParams


def _instance_name_to_output_path(
    instance_name: str, base_dir: Optional[Path] = None
) -> Path:
    """
    Map an instance name like:
        'matpower/case300/2017-06-24'
    to output path:
        <base_dir>/matpower/case300/2017-06-24.json

    base_dir defaults to DataParams._OUTPUT.
    """
    base = base_dir or DataParams._OUTPUT
    rel = instance_name.strip().strip("/\\").replace("\\", "/")
    return base / rel / ".."  # placeholder to simplify next line


def compute_output_path(instance_name: str, base_dir: Optional[Path] = None) -> Path:
    """
    Compute output JSON path mirroring the input hierarchy, with .json extension.
    """
    base = base_dir or DataParams._OUTPUT
    rel = instance_name.strip().strip("/\\").replace("\\", "/")
    # If rel contains multiple segments, parent directories will be created.
    # Replace trailing ".json" or ".json.gz" if mistakenly passed with extension.
    if rel.endswith(".json.gz"):
        rel = rel[: -len(".json.gz")]
    elif rel.endswith(".json"):
        rel = rel[: -len(".json")]
    out_path = base / rel
    # ensure the filename ends with .json (not a directory)
    if out_path.suffix != ".json":
        out_path = out_path.with_suffix(".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def save_solution_as_json(
    scenario: UnitCommitmentScenario,
    model: Model,
    instance_name: str,
    out_base_dir: Optional[Path] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a SCUC solution as JSON in src/data/output mirroring the input hierarchy.

    Example mapping:
      input:  src/data/input/matpower/case300/2017-06-24.json.gz
      output: src/data/output/matpower/case300/2017-06-24.json

    Returns the path to the written JSON file.
    """
    out_path = compute_output_path(instance_name, base_dir=out_base_dir)

    sol = restore_solution(scenario, model)

    # Minimal metadata to facilitate ML usage and traceability
    meta: Dict[str, Any] = {
        "instance_name": instance_name,
        "scenario_name": scenario.name,
        "time_steps": scenario.time,
        "time_step_min": scenario.time_step,
    }
    if extra_meta:
        meta.update(extra_meta)

    payload: Dict[str, Any] = {
        "meta": meta,
        "objective": sol.get("objective"),
        "status": sol.get("status"),
        "system": sol.get("system"),
        "generators": sol.get("generators"),
        "reserves": sol.get("reserves", {}),
        "network": sol.get("network", {}),
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path

```

### File: `src/optimization_model/helpers/save_solution.py`

```
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from src.data_preparation.data_structure import UnitCommitmentScenario, ThermalUnit
from src.optimization_model.helpers.restore_solution import restore_solution
from src.data_preparation.params import DataParams


def _fmt_float(x: float, nd: int = 6) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _generator_segment_costs(
    gen: ThermalUnit, t: int, seg_prod_t: List[float]
) -> List[float]:
    """Compute cost per segment produced at time t."""
    costs: List[float] = []
    n_segments = len(gen.segments) if gen.segments else 0
    for s in range(n_segments):
        c = float(gen.segments[s].cost[t]) * float(seg_prod_t[s])
        costs.append(c)
    return costs


def _avg_incremental_cost_at_t(
    gens: List[ThermalUnit], t: int, seg_prod: Dict[str, List[List[float]]]
) -> Optional[float]:
    """Weighted average marginal cost at time t across all produced segments."""
    total_mw = 0.0
    total_cost = 0.0
    for gen in gens:
        n_segments = len(gen.segments) if gen.segments else 0
        if n_segments == 0:
            continue
        segs_t = seg_prod[gen.name][t]
        for s in range(n_segments):
            mw = float(segs_t[s])
            if mw > 0:
                price = float(gen.segments[s].cost[t])
                total_mw += mw
                total_cost += price * mw
    if total_mw <= 0:
        return None
    return total_cost / total_mw


def _startup_cost_for_gen(gen: ThermalUnit) -> float:
    """Hot-start policy: min of startup category costs; 0 if none."""
    try:
        if gen.startup_categories:
            return float(min(cat.cost for cat in gen.startup_categories))
    except Exception:
        pass
    return 0.0


def save_solution_to_log(
    scenario: UnitCommitmentScenario,
    model,
    out_dir: Optional[Path] = None,
    filename: Optional[str] = None,
) -> Path:
    """
    Write a detailed solution log with improved readability.

    Contents
    - Solver/instance summary
    - System load and production by time
    - Reserve details by product and time (requirement, provided, shortfall)
    - Line flows by time (flow, limit, overflow+, overflow-)
    - Generator-by-generator details per time (aligned columns):
        * initial status and initial power
        * commitment, startup/shutdown, min/max/headroom
        * produced min power, segment-by-segment: produced, capacity, price, cost
        * energy totals, reserve provided per product, remaining headroom
        * generator total cost components (min, segments, startup)
    - Objective breakdown (incl. reserve penalty, base and contingency line overflow penalty, startup)
    - Simple feasibility checks (energy balance)
    """
    out_dir = out_dir or DataParams._LOGS
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        scen_name = scenario.name or "scenario"
        filename = f"SCUC_{scen_name}_run001_{ts}.solution.log"
    out_path = out_dir / filename

    # Restore solution in structured form
    sol = restore_solution(scenario, model)

    T = scenario.time
    generators = scenario.thermal_units
    buses = scenario.buses

    # Unpack for convenience
    sys_load = sol["system"]["load"]  # length T
    sys_prod = sol["system"]["total_production"]  # length T

    gen_sol = sol["generators"]  # dict[gen_name] -> dict with lists over time

    # Precompute costs per generator and time
    gen_min_cost: Dict[str, List[float]] = {}
    gen_segs_cost: Dict[str, List[List[float]]] = {}
    gen_startup_cost: Dict[str, List[float]] = {}
    gen_total_cost: Dict[str, List[float]] = {}

    startup_vars = getattr(model, "startup", None)

    for gen in generators:
        gsol = gen_sol[gen.name]
        min_cost_t: List[float] = []
        segs_cost_t: List[List[float]] = []
        startup_cost_t: List[float] = []
        total_cost_t: List[float] = []
        s_cost_param = _startup_cost_for_gen(gen)
        for t in range(T):
            # cost for min power if on
            minc = float(gsol["commit"][t]) * float(gen.min_power_cost[t])
            min_cost_t.append(minc)
            # per-segment costs
            seg_costs = _generator_segment_costs(gen, t, gsol["segment_power"][t])
            segs_cost_t.append(seg_costs)
            # startup cost this period
            st_cost = 0.0
            if startup_vars is not None and s_cost_param > 0.0:
                try:
                    st = float(startup_vars[gen.name, t].X)
                except Exception:
                    st = 0.0
                st_cost = st * s_cost_param
            startup_cost_t.append(st_cost)

            total_cost_t.append(minc + sum(seg_costs) + st_cost)

        gen_min_cost[gen.name] = min_cost_t
        gen_segs_cost[gen.name] = segs_cost_t
        gen_startup_cost[gen.name] = startup_cost_t
        gen_total_cost[gen.name] = total_cost_t

    # Average incremental energy cost at each time (over produced segments)
    from typing import Optional as _OptionalTypeHint  # local alias

    avg_incr_cost_t: List[_OptionalTypeHint[float]] = []
    seg_prod_by_gen = {
        gen.name: gen_sol[gen.name]["segment_power"] for gen in generators
    }
    for t in range(T):
        avg_incr_cost_t.append(
            _avg_incremental_cost_at_t(generators, t, seg_prod_by_gen)
        )

    # Reserve unpack
    reserves_sol = sol.get("reserves", {})
    reserve_names = list(reserves_sol.keys())

    # Network unpack
    network_sol = sol.get("network", {})
    line_sol = network_sol.get("lines", {}) if network_sol else {}
    line_names_sorted = list(line_sol.keys())

    # Compose log text with aligned columns
    lines: List[str] = []
    lines.append("===== SCUC Solution Report =====")
    lines.append(f"Scenario         : {scenario.name}")
    lines.append(f"Time steps       : {T} (step = {scenario.time_step} min)")
    lines.append(
        f"Counts           : {len(generators)} thermal units, {len(buses)} buses, {len(scenario.lines)} lines, {len(reserve_names)} reserve products"
    )
    lines.append(f"Solver status    : {sol['status']}")
    # Objective
    obj_val = sol.get("objective", float("nan"))
    try:
        lines.append(f"Objective value  : {obj_val:.6f}")
    except Exception:
        lines.append(f"Objective value  : {obj_val}")

    # MIP gap etc. if present
    try:
        mip_gap = getattr(model, "MIPGap", None)
        if mip_gap is not None:
            lines.append(f"MIP gap          : {mip_gap:.6f}")
    except Exception:
        pass
    lines.append("")

    # System load and production
    lines.append("== System load and production by time ==")
    header = f"{'t':>3}  {'load(MW)':>12}  {'production(MW)':>16}  {'balance(MW)':>13}  {'avg_incremental_cost($/MW)':>28}"
    lines.append(header)
    for t in range(T):
        bal = float(sys_prod[t]) - float(sys_load[t])
        aic = avg_incr_cost_t[t]
        aic_str = _fmt_float(aic, 6) if aic is not None else "n/a"
        lines.append(
            f"{t:>3}  {_fmt_float(sys_load[t], 4):>12}  {_fmt_float(sys_prod[t], 4):>16}  {_fmt_float(bal, 4):>13}  {aic_str:>28}"
        )
    lines.append("")

    # Reserve details
    if reserve_names:
        lines.append("== Reserves by product and time ==")
        for rname in reserve_names:
            rsol = reserves_sol[rname]
            lines.append(f"[Reserve '{rname}']")
            lines.append(
                f"{'t':>3}  {'requirement(MW)':>18}  {'provided(MW)':>14}  {'shortfall(MW)':>14}"
            )
            for t in range(T):
                lines.append(
                    f"{t:>3}  {_fmt_float(rsol['requirement'][t], 4):>18}  {_fmt_float(rsol['total_provided'][t], 4):>14}  {_fmt_float(rsol['shortfall'][t], 4):>14}"
                )
            lines.append("")
    else:
        lines.append("== Reserves: none defined in this scenario ==")
        lines.append("")

    # Line flows
    if line_names_sorted:
        lines.append("== Line flows (base case) by time ==")
        for lname in line_names_sorted:
            lsol = line_sol[lname]
            src = lsol.get("source", "?")
            tgt = lsol.get("target", "?")
            lines.append(f"[Line '{lname}' {src}->{tgt}]")
            lines.append(
                f"{'t':>3}  {'flow(MW)':>12}  {'limit(MW)':>12}  {'overflow+(MW)':>15}  {'overflow-(MW)':>15}"
            )
            for t in range(T):
                lines.append(
                    f"{t:>3}  {_fmt_float(lsol['flow'][t], 4):>12}  {_fmt_float(lsol['limit'][t], 4):>12}  {_fmt_float(lsol['overflow_pos'][t], 4):>15}  {_fmt_float(lsol['overflow_neg'][t], 4):>15}"
                )
            lines.append("")
    else:
        lines.append("== Lines: none defined in this scenario ==")
        lines.append("")

    # Generator details: per generator, per time, include segments and prices
    lines.append("== Generator details (per time step) ==")
    for gen in generators:
        gsol = gen_sol[gen.name]
        lines.append(f"[Generator '{gen.name}' | Bus='{gen.bus.name}']")
        # Initial conditions
        lines.append(
            f"   initial: status_steps={gsol['initial_status_steps']}, u0={gsol['initial_u']}, p0={_fmt_float(gsol['initial_power'], 4)}"
        )
        n_segments = len(gen.segments) if gen.segments else 0

        # Table header for time rows
        lines.append(
            f"{'t':>3}  {'commit':>6}  {'start':>6}  {'stop':>6}  "
            f"{'min(MW)':>10}  {'max(MW)':>10}  {'min_out(MW)':>13}  {'energy>min(MW)':>16}  {'total(MW)':>11}"
        )
        for t in range(T):
            committed = int(gsol["commit"][t])
            started = int(gsol["startup"][t])
            stopped = int(gsol["shutdown"][t])
            min_p = float(gen.min_power[t])
            max_p = float(gen.max_power[t])
            min_out = float(gsol["min_power_output"][t])
            segs_prod = gsol["segment_power"][t]
            energy_above_min = sum(segs_prod) if segs_prod else 0.0
            total_power = float(gsol["total_power"][t])

            lines.append(
                f"{t:>3}  {committed:>6}  {started:>6}  {stopped:>6}  "
                f"{_fmt_float(min_p, 4):>10}  {_fmt_float(max_p, 4):>10}  {_fmt_float(min_out, 4):>13}  {_fmt_float(energy_above_min, 4):>16}  {_fmt_float(total_power, 4):>11}"
            )

            # Headroom and reserves per product
            headroom_if_on = (max_p - min_p) if committed == 1 else 0.0
            reserves_by_product: Dict[str, float] = {}
            for rname in reserve_names:
                provided_map = reserves_sol[rname]["provided_by_gen"]
                reserves_by_product[rname] = float(
                    provided_map.get(gen.name, [0.0] * T)[t]
                    if isinstance(provided_map.get(gen.name), list)
                    else 0.0
                )
            total_reserve_t = sum(reserves_by_product.values())
            remaining_headroom = headroom_if_on - energy_above_min - total_reserve_t

            lines.append(
                f"       headroom_if_on={_fmt_float(headroom_if_on, 4)}, reserve_total={_fmt_float(total_reserve_t, 4)}, remaining_headroom={_fmt_float(remaining_headroom, 4)}"
            )

            # Segment details
            if n_segments > 0:
                segs_costs = _generator_segment_costs(gen, t, segs_prod)
                lines.append(
                    "       segments: idx, produced(MW), capacity(MW), price($/MW), cost($)"
                )
                for s in range(n_segments):
                    capacity = float(gen.segments[s].amount[t])
                    price = float(gen.segments[s].cost[t])
                    produced = float(segs_prod[s])
                    scost = float(segs_costs[s])
                    lines.append(
                        f"         {s:02d}, {_fmt_float(produced, 4)}, {_fmt_float(capacity, 4)}, {_fmt_float(price, 4)}, {_fmt_float(scost, 4)}"
                    )
            else:
                lines.append("       segments: none")

            # Reserve breakdown per product
            if reserve_names:
                parts = ", ".join(
                    [
                        f"{rn}={_fmt_float(reserves_by_product[rn], 4)}"
                        for rn in reserve_names
                    ]
                )
                lines.append(f"       reserves_by_product: {parts}")

            # Costs
            min_c = float(gen_min_cost[gen.name][t])
            seg_c = (
                sum(_generator_segment_costs(gen, t, segs_prod))
                if n_segments > 0
                else 0.0
            )
            su_cost = float(gen_startup_cost[gen.name][t])
            tot_c = min_c + seg_c + su_cost
            lines.append(
                f"       cost: min_out_cost={_fmt_float(min_c, 4)} | segments_cost={_fmt_float(seg_c, 4)} | startup_cost={_fmt_float(su_cost, 4)} | generator_total_cost={_fmt_float(tot_c, 4)}"
            )
        lines.append("")  # blank line between generators

    # Objective breakdown
    lines.append("== Objective breakdown ==")
    total_min = sum(sum(gen_min_cost[gen.name]) for gen in generators)
    total_seg = sum(sum(map(sum, gen_segs_cost[gen.name])) for gen in generators)
    total_su = sum(sum(gen_startup_cost[gen.name]) for gen in generators)

    total_res_shortfall_penalty = 0.0
    if reserve_names:
        for r in scenario.reserves:
            rname = r.name
            rsol = reserves_sol.get(rname)
            if not rsol:
                continue
            penalty = float(r.shortfall_penalty)
            total_res_shortfall_penalty += sum(
                penalty * float(x) for x in rsol["shortfall"]
            )

    total_line_penalty = 0.0
    if line_names_sorted:
        for lname in line_names_sorted:
            lsol = line_sol[lname]
            for t in range(T):
                total_line_penalty += float(lsol["penalty"][t]) * (
                    float(lsol["overflow_pos"][t]) + float(lsol["overflow_neg"][t])
                )

    # Contingency penalties (if any)
    cont_ovp = getattr(model, "contingency_overflow_pos", None)
    cont_ovn = getattr(model, "contingency_overflow_neg", None)
    CONT_FACTOR = 10.0
    total_cont_penalty = 0.0
    if line_names_sorted and cont_ovp is not None and cont_ovn is not None:
        for lname in line_names_sorted:
            line = next((L for L in scenario.lines if L.name == lname), None)
            if line is None:
                continue
            for t in range(T):
                pen = float(line.flow_penalty[t]) * CONT_FACTOR
                try:
                    total_cont_penalty += pen * (
                        float(cont_ovp[lname, t].X) + float(cont_ovn[lname, t].X)
                    )
                except Exception:
                    pass

    lines.append(f"Total min-output cost               = {_fmt_float(total_min, 4)}")
    lines.append(f"Total segment (energy) cost         = {_fmt_float(total_seg, 4)}")
    lines.append(f"Total startup cost                  = {_fmt_float(total_su, 4)}")
    if reserve_names:
        lines.append(
            f"Total reserve shortfall penalty     = {_fmt_float(total_res_shortfall_penalty, 4)}"
        )
    if line_names_sorted:
        lines.append(
            f"Total line overflow penalty         = {_fmt_float(total_line_penalty, 4)}"
        )
        lines.append(
            f"Total contingency overflow penalty  = {_fmt_float(total_cont_penalty, 4)}"
        )
    try:
        lines.append(f"Reported objective (solver)         = {_fmt_float(obj_val, 4)}")
        lines.append(
            f"Sum of above components             = {_fmt_float(total_min + total_seg + total_su + total_res_shortfall_penalty + total_line_penalty + total_cont_penalty, 4)}"
        )
    except Exception:
        lines.append(f"Reported objective (solver)         = {obj_val}")
        lines.append(
            f"Sum of above components             = {total_min + total_seg + total_su + total_res_shortfall_penalty + total_line_penalty + total_cont_penalty}"
        )
    lines.append("")

    # Feasibility checks
    lines.append("== Checks ==")
    max_balance_abs = max(
        abs(float(sys_prod[t]) - float(sys_load[t])) for t in range(T)
    )
    lines.append(
        f"Max |production - load| over time = {_fmt_float(max_balance_abs, 6)}"
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path

```

### File: `src/optimization_model/helpers/verify_solution.py`

```
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, List

import math

from src.data_preparation.data_structure import (
    UnitCommitmentScenario,
    ThermalUnit,
    Reserve,
)
from src.data_preparation.params import DataParams

# Reuse tolerances and penalty factor from the same modules used by the model
from src.optimization_model.solver.scuc.constraints.contingencies import (
    _LODF_TOL as CONT_LODF_TOL,
    _ISF_TOL as CONT_ISF_TOL,
)
from src.optimization_model.solver.scuc.objectives.power_cost_segmented import (
    _CONTINGENCY_PENALTY_FACTOR as CONT_PENALTY_FACTOR,
)

# Unique indices for verification items
# Constraints
ID_C_FIX = "C-101"  # Commitment fixing
ID_C_LINK = "C-102"  # Segment capacity linking
ID_C_BAL = "C-103"  # Power balance
ID_C_RES_HEAD = "C-104"  # Reserve headroom linking (shared across products)
ID_C_RES_REQ = "C-105"  # Reserve requirement
ID_C_SU_DEF = "C-106"  # Startup/shutdown definition
ID_C_RAMP = "C-107"  # Ramping with startup/shutdown limits
ID_C_MIN_UP = "C-110"  # Minimum up-time
ID_C_MIN_DOWN = "C-111"  # Minimum down-time
ID_C_MIN_INIT = "C-112"  # Initial condition enforcement (min up/down)
ID_C_FLOW_DEF = "C-108"  # Line flow PTDF equality
ID_C_FLOW_LIM = "C-109"  # Line flow limits with overflow slacks
ID_C_CONT_LINE = "C-120"  # Post-contingency limits (line outages via LODF)
ID_C_CONT_GEN = "C-121"  # Post-contingency limits (generator outages via ISF)

# Variables
ID_V_COMMIT = "V-201"  # Commitment u[gen,t] in {0,1}
ID_V_PSEG = "V-202"  # Segment power pseg[gen,t,s] >= 0
ID_V_R = "V-203"  # Reserve provision r[k,gen,t] >= 0
ID_V_S = "V-204"  # Reserve shortfall s[k,t] >= 0
ID_V_SU = "V-205"  # Startup v[gen,t] in {0,1}
ID_V_SD = "V-206"  # Shutdown w[gen,t] in {0,1}
ID_V_FLOW = "V-207"  # Line flows f[line,t] finite
ID_V_OVP = "V-208"  # Line overflow+ >= 0
ID_V_OVN = "V-209"  # Line overflow- >= 0

# Objective
ID_O_TOTAL = "O-301"  # TotalCostWithReservePenalty consistency

EPS = 1e-6  # numerical tolerance


@dataclass
class CheckItem:
    idx: str
    name: str
    value: float  # 0 means OK; otherwise worst violation


def _get_model_vars(model):
    commit = getattr(model, "commit", None)
    seg = getattr(model, "gen_segment_power", None)
    r = getattr(model, "reserve", None)
    s = getattr(model, "reserve_shortfall", None)
    su = getattr(model, "startup", None)
    sd = getattr(model, "shutdown", None)
    f = getattr(model, "line_flow", None)
    ovp = getattr(model, "line_overflow_pos", None)
    ovn = getattr(model, "line_overflow_neg", None)
    covp = getattr(model, "contingency_overflow_pos", None)
    covn = getattr(model, "contingency_overflow_neg", None)
    return commit, seg, r, s, su, sd, f, ovp, ovn, covp, covn


def _compute_p_of_gen_t(gen: ThermalUnit, t: int, commit, seg) -> float:
    u = float(commit[gen.name, t].X) if commit is not None else 0.0
    p = u * float(gen.min_power[t])
    n_segments = len(gen.segments) if gen.segments else 0
    if n_segments > 0 and seg is not None:
        for s in range(n_segments):
            p += float(seg[gen.name, t, s].X)
    return p


def _compute_production_at_t(
    units: Sequence[ThermalUnit], t: int, commit, seg
) -> float:
    prod = 0.0
    for gen in units:
        prod += _compute_p_of_gen_t(gen, t, commit, seg)
    return prod


def _initial_u(gen: ThermalUnit) -> float:
    try:
        return (
            1.0 if (gen.initial_status is not None and gen.initial_status > 0) else 0.0
        )
    except Exception:
        return 0.0


def _initial_p(gen: ThermalUnit) -> float:
    try:
        return float(gen.initial_power) if gen.initial_power is not None else 0.0
    except Exception:
        return 0.0


def _startup_cost_for_gen(gen: ThermalUnit) -> float:
    """
    Same policy as objective: 'hot start' = min of startup category costs; 0 if none.
    """
    try:
        if gen.startup_categories:
            return float(min(cat.cost for cat in gen.startup_categories))
    except Exception:
        pass
    return 0.0


def _objective_from_vars(
    units: Sequence[ThermalUnit],
    reserves: Sequence[Reserve],
    lines,
    T: int,
    commit,
    seg,
    r_short,
    ovp,
    ovn,
    covp=None,
    covn=None,
    su=None,
    contingency_penalty_factor: float = CONT_PENALTY_FACTOR,
) -> float:
    total = 0.0
    # Production cost
    for gen in units:
        n_segments = len(gen.segments) if gen.segments else 0
        s_cost = _startup_cost_for_gen(gen)
        for t in range(T):
            u = float(commit[gen.name, t].X) if commit is not None else 0.0
            total += u * float(gen.min_power_cost[t])
            for sidx in range(n_segments):
                total += float(seg[gen.name, t, sidx].X) * float(
                    gen.segments[sidx].cost[t]
                )
            if su is not None and s_cost > 0.0:
                total += float(su[gen.name, t].X) * s_cost
    # Reserve shortfall penalty
    if reserves and r_short is not None:
        for reserve_def in reserves:
            pen = float(reserve_def.shortfall_penalty)
            for t in range(T):
                total += float(r_short[reserve_def.name, t].X) * pen
    # Base-case line overflow penalty
    if lines and ovp is not None and ovn is not None:
        for line in lines:
            for t in range(T):
                pen = float(line.flow_penalty[t])
                total += pen * (float(ovp[line.name, t].X) + float(ovn[line.name, t].X))
    # Contingency overflow penalty (shared slacks)
    if lines and covp is not None and covn is not None:
        for line in lines:
            for t in range(T):
                pen = float(line.flow_penalty[t]) * contingency_penalty_factor
                total += pen * (
                    float(covp[line.name, t].X) + float(covn[line.name, t].X)
                )
    return total


def verify_solution(
    scenario: UnitCommitmentScenario, model
) -> Tuple[bool, List[CheckItem], str]:
    """
    In-memory verification of a solved model.
    Returns: (overall_ok, checks, report_text)
    """
    T = scenario.time
    units = scenario.thermal_units
    reserves = scenario.reserves or []
    lines = scenario.lines or []

    total_load = getattr(model, "_total_load", None)
    if total_load is None:
        total_load = [sum(b.load[t] for b in scenario.buses) for t in range(T)]

    (
        commit,
        seg,
        r,
        s,
        su,
        sd,
        f,
        ovp,
        ovn,
        covp,
        covn,
    ) = _get_model_vars(model)

    checks: List[CheckItem] = []

    # Variables: V-201 (commitment domain & integrality)
    worst = 0.0
    if commit is not None:
        for gen in units:
            for t in range(T):
                val = float(commit[gen.name, t].X)
                worst = max(worst, max(0.0, -val))
                worst = max(worst, max(0.0, val - 1.0))
                worst = max(worst, abs(val - round(val)))
    checks.append(CheckItem(ID_V_COMMIT, "Commitment u[gen,t] in {0,1}", worst))

    # Variables: V-202 (segment power >= 0)
    worst = 0.0
    if seg is not None:
        for gen in units:
            n_segments = len(gen.segments) if gen.segments else 0
            for t in range(T):
                for s_idx in range(n_segments):
                    val = float(seg[gen.name, t, s_idx].X)
                    worst = max(worst, max(0.0, -val))
    checks.append(CheckItem(ID_V_PSEG, "Segment power pseg[gen,t,s] >= 0", worst))

    # Variables: V-203 (reserve provision >= 0)
    worst = 0.0
    if reserves and r is not None:
        for reserve_def in reserves:
            for gen in reserve_def.thermal_units:
                for t in range(T):
                    val = float(r[reserve_def.name, gen.name, t].X)
                    worst = max(worst, max(0.0, -val))
    checks.append(CheckItem(ID_V_R, "Reserve provision r[k,gen,t] >= 0", worst))

    # Variables: V-204 (shortfall >= 0)
    worst = 0.0
    if reserves and s is not None:
        for reserve_def in reserves:
            for t in range(T):
                val = float(s[reserve_def.name, t].X)
                worst = max(worst, max(0.0, -val))
    checks.append(CheckItem(ID_V_S, "Reserve shortfall s[k,t] >= 0", worst))

    # Variables: V-205 (startup binary)
    worst = 0.0
    if su is not None:
        for gen in units:
            for t in range(T):
                val = float(su[gen.name, t].X)
                worst = max(worst, max(0.0, -val))
                worst = max(worst, max(0.0, val - 1.0))
                worst = max(worst, abs(val - round(val)))
    checks.append(CheckItem(ID_V_SU, "Startup v[gen,t] in {0,1}", worst))

    # Variables: V-206 (shutdown binary)
    worst = 0.0
    if sd is not None:
        for gen in units:
            for t in range(T):
                val = float(sd[gen.name, t].X)
                worst = max(worst, max(0.0, -val))
                worst = max(worst, max(0.0, val - 1.0))
                worst = max(worst, abs(val - round(val)))
    checks.append(CheckItem(ID_V_SD, "Shutdown w[gen,t] in {0,1}", worst))

    # Variables: V-207 (line flow – finite)
    worst = 0.0
    if f is not None and lines:
        for line in lines:
            for t in range(T):
                val = float(f[line.name, t].X)
                if not math.isfinite(val):
                    worst = float("inf")
                    break
    checks.append(CheckItem(ID_V_FLOW, "Line flow f[line,t] finite", worst))

    # Variables: V-208 (overflow+ >= 0)
    worst = 0.0
    if ovp is not None and lines:
        for line in lines:
            for t in range(T):
                worst = max(worst, max(0.0, -float(ovp[line.name, t].X)))
    checks.append(CheckItem(ID_V_OVP, "Line overflow+ ov_pos[line,t] >= 0", worst))

    # Variables: V-209 (overflow- >= 0)
    worst = 0.0
    if ovn is not None and lines:
        for line in lines:
            for t in range(T):
                worst = max(worst, max(0.0, -float(ovn[line.name, t].X)))
    checks.append(CheckItem(ID_V_OVN, "Line overflow- ov_neg[line,t] >= 0", worst))

    # Constraints: C-101 (commitment fixing)
    worst = 0.0
    for gen in units:
        if not gen.commitment_status:
            continue
        for t in range(T):
            st = gen.commitment_status[t]
            if st is None or commit is None:
                continue
            val = float(commit[gen.name, t].X)
            worst = max(worst, abs(val - (1.0 if st else 0.0)))
    checks.append(CheckItem(ID_C_FIX, "Commitment fixing (fix_commit_on/off)", worst))

    # Constraints: C-102 (linking: seg <= amount * u)
    worst = 0.0
    if seg is not None and commit is not None:
        for gen in units:
            n_segments = len(gen.segments) if gen.segments else 0
            for t in range(T):
                u = float(commit[gen.name, t].X)
                for s_idx in range(n_segments):
                    amount = float(gen.segments[s_idx].amount[t])
                    if amount < 0:
                        amount = 0.0
                    val = float(seg[gen.name, t, s_idx].X)
                    viol = val - amount * u
                    if viol > worst:
                        worst = viol
    checks.append(CheckItem(ID_C_LINK, "Segment capacity linking", max(0.0, worst)))

    # Constraints: C-103 (power balance: equality)
    worst = 0.0
    for t in range(T):
        lhs = _compute_production_at_t(units, t, commit, seg)
        rhs = float(total_load[t])
        worst = max(worst, abs(lhs - rhs))
    checks.append(CheckItem(ID_C_BAL, "System power balance", worst))

    # Constraints: C-104 (reserve headroom (shared across products))
    worst = 0.0
    if reserves and commit is not None and seg is not None and r is not None:
        eligible_by_gen: Dict[str, List] = {}
        for res in reserves:
            for g in res.thermal_units:
                eligible_by_gen.setdefault(g.name, []).append(res)
        for gen in units:
            n_segments = len(gen.segments) if gen.segments else 0
            for t in range(T):
                energy_above_min = 0.0
                for s_idx in range(n_segments):
                    energy_above_min += float(seg[gen.name, t, s_idx].X)
                total_reserve = 0.0
                for res in eligible_by_gen.get(gen.name, []):
                    total_reserve += float(r[res.name, gen.name, t].X)
                headroom = float(gen.max_power[t]) - float(gen.min_power[t])
                rhs = headroom * float(commit[gen.name, t].X)
                viol = energy_above_min + total_reserve - rhs
                if viol > worst:
                    worst = viol
    checks.append(
        CheckItem(ID_C_RES_HEAD, "Reserve headroom linking (shared)", max(0.0, worst))
    )

    # Constraints: C-105 (reserve requirement)
    worst = 0.0
    if reserves and r is not None and s is not None:
        for reserve_def in reserves:
            for t in range(T):
                provided = 0.0
                for gen in reserve_def.thermal_units:
                    provided += float(r[reserve_def.name, gen.name, t].X)
                left = provided + float(s[reserve_def.name, t].X)
                req = float(reserve_def.amount[t])
                viol = req - left
                if viol > worst:
                    worst = viol
    checks.append(CheckItem(ID_C_RES_REQ, "Reserve requirement", max(0.0, worst)))

    # Constraints: C-106 (startup/shutdown definition + exclusivity)
    worst = 0.0
    if commit is not None and su is not None and sd is not None:
        for gen in units:
            u0 = _initial_u(gen)
            for t in range(T):
                gen_commit_t = float(commit[gen.name, t].X)
                gen_startup_t = float(su[gen.name, t].X)
                gen_shutdown_t = float(sd[gen.name, t].X)
                u_prev = u0 if t == 0 else float(commit[gen.name, t - 1].X)
                eq_violation = abs(
                    (gen_commit_t - u_prev) - (gen_startup_t - gen_shutdown_t)
                )
                excl_violation = max(0.0, gen_startup_t + gen_shutdown_t - 1.0)
                worst = max(worst, eq_violation, excl_violation)
    checks.append(
        CheckItem(
            ID_C_SU_DEF,
            "Startup/shutdown definition (u[t]-u_prev == v-w) and exclusivity (v+w<=1)",
            worst,
        )
    )

    # Constraints: C-107 (ramping with startup/shutdown limits + initial power)
    worst = 0.0
    if commit is not None and seg is not None and su is not None and sd is not None:
        for gen in units:
            ru = float(gen.ramp_up)
            rd = float(gen.ramp_down)
            SU = float(gen.startup_limit)
            SD = float(gen.shutdown_limit)

            p0 = _initial_p(gen)
            u0 = _initial_u(gen)

            for t in range(T):
                p_t = _compute_p_of_gen_t(gen, t, commit, seg)
                v_t = float(su[gen.name, t].X)
                w_t = float(sd[gen.name, t].X)

                if t == 0:
                    viol_up = (p_t - p0) - (ru * u0 + SU * v_t)
                    u_t = float(commit[gen.name, t].X)
                    viol_dn = (p0 - p_t) - (rd * u_t + SD * w_t)
                else:
                    p_prev = _compute_p_of_gen_t(gen, t - 1, commit, seg)
                    u_prev = float(commit[gen.name, t - 1].X)
                    u_t = float(commit[gen.name, t].X)
                    viol_up = (p_t - p_prev) - (ru * u_prev + SU * v_t)
                    viol_dn = (p_prev - p_t) - (rd * u_t + SD * w_t)

                worst = max(worst, max(0.0, viol_up))
                worst = max(worst, max(0.0, viol_dn))
    checks.append(
        CheckItem(
            ID_C_RAMP,
            "Ramping with startup/shutdown limits (incl. initial status/power)",
            worst,
        )
    )

    # Constraints: C-110 (min up-time windows)
    worst = 0.0
    if su is not None and commit is not None:
        for gen in units:
            Lu = int(getattr(gen, "min_up", 0) or 0)
            if Lu > 0:
                for t in range(T):
                    start_k = max(0, t - Lu + 1)
                    if start_k <= t:
                        lhs = 0.0
                        for k in range(start_k, t + 1):
                            lhs += float(su[gen.name, k].X)
                        rhs = float(commit[gen.name, t].X)
                        viol = lhs - rhs
                        worst = max(worst, max(0.0, viol))
    checks.append(CheckItem(ID_C_MIN_UP, "Minimum up-time window", worst))

    # Constraints: C-111 (min down-time windows)
    worst = 0.0
    if sd is not None and commit is not None:
        for gen in units:
            Ld = int(getattr(gen, "min_down", 0) or 0)
            if Ld > 0:
                for t in range(T):
                    start_k = max(0, t - Ld + 1)
                    if start_k <= t:
                        lhs = 0.0
                        for k in range(start_k, t + 1):
                            lhs += float(sd[gen.name, k].X)
                        rhs = 1.0 - float(commit[gen.name, t].X)
                        viol = lhs - rhs
                        worst = max(worst, max(0.0, viol))
    checks.append(CheckItem(ID_C_MIN_DOWN, "Minimum down-time window", worst))

    # Constraints: C-112 (initial-condition enforcement at horizon start)
    worst = 0.0
    if commit is not None:
        for gen in units:
            Lu = int(getattr(gen, "min_up", 0) or 0)
            Ld = int(getattr(gen, "min_down", 0) or 0)
            s0 = getattr(gen, "initial_status", None)
            if s0 is None:
                continue

            if s0 > 0 and Lu > 0:
                remaining_on = max(0, Lu - int(s0))
                for t in range(min(remaining_on, T)):
                    val = float(commit[gen.name, t].X)
                    worst = max(worst, abs(val - 1.0))

            if s0 < 0 and Ld > 0:
                s_off = -int(s0)
                remaining_off = max(0, Ld - s_off)
                for t in range(min(remaining_off, T)):
                    val = float(commit[gen.name, t].X)
                    worst = max(worst, abs(val - 0.0))
    checks.append(CheckItem(ID_C_MIN_INIT, "Initial up/down enforcement", worst))

    # Constraints: C-108 (line flow PTDF equality)
    worst = 0.0
    if lines and f is not None:
        isf = scenario.isf.tocsr()
        ref_1b = getattr(scenario, "ptdf_ref_bus_index", scenario.buses[0].index)
        non_ref_bus_indices = sorted(
            [b.index for b in scenario.buses if b.index != ref_1b]
        )

        for t in range(T):
            inj_by_busidx: Dict[int, float] = {}
            for b in scenario.buses:
                gen_at_b = 0.0
                for gen in b.thermal_units:
                    gen_at_b += _compute_p_of_gen_t(gen, t, commit, seg)
                inj_by_busidx[b.index] = gen_at_b - float(b.load[t])

            for line in lines:
                row = isf.getrow(line.index - 1)
                fhat = 0.0
                for col, coeff in zip(row.indices.tolist(), row.data.tolist()):
                    bus_1b = non_ref_bus_indices[col]
                    fhat += float(coeff) * float(inj_by_busidx[bus_1b])
                actual = float(f[line.name, t].X)
                worst = max(worst, abs(fhat - actual))
    checks.append(CheckItem(ID_C_FLOW_DEF, "Line flow PTDF equality", worst))

    # Constraints: C-109 (flow limits with overflow slacks)
    worst = 0.0
    if lines and f is not None and ovp is not None and ovn is not None:
        for line in lines:
            for t in range(T):
                F = float(line.normal_limit[t])
                val = float(f[line.name, t].X)
                vpos = val - F - float(ovp[line.name, t].X)
                vneg = -val - F - float(ovn[line.name, t].X)
                if vpos > worst:
                    worst = vpos
                if vneg > worst:
                    worst = vneg
    checks.append(
        CheckItem(
            ID_C_FLOW_LIM, "Line flow limits with overflow slacks", max(0.0, worst)
        )
    )

    # C-120 — Post-contingency line limits (line outages via LODF), with shared slacks
    worst_line = 0.0
    if lines and f is not None and scenario.contingencies:
        lodf = scenario.lodf.tocsc()
        for cont in scenario.contingencies:
            if not cont.lines:
                continue
            for out_line in cont.lines:
                mcol = out_line.index - 1
                col = lodf.getcol(mcol)
                for l_row, alpha in zip(col.indices.tolist(), col.data.tolist()):
                    if l_row == mcol:
                        continue
                    if abs(alpha) < CONT_LODF_TOL:
                        continue
                    line_l = scenario.lines[l_row]
                    for t in range(T):
                        base_l = float(f[line_l.name, t].X)
                        base_m = float(f[out_line.name, t].X)
                        post = base_l + float(alpha) * base_m
                        ovp_shared = (
                            float(covp[line_l.name, t].X) if covp is not None else 0.0
                        )
                        ovn_shared = (
                            float(covn[line_l.name, t].X) if covn is not None else 0.0
                        )
                        vpos = post - float(line_l.emergency_limit[t]) - ovp_shared
                        vneg = -post - float(line_l.emergency_limit[t]) - ovn_shared
                        worst_line = max(worst_line, vpos, vneg)
    checks.append(
        CheckItem(
            ID_C_CONT_LINE,
            "Post-contingency line limits (line outage via LODF)",
            max(0.0, worst_line),
        )
    )

    # C-121 — Post-contingency line limits (generator outages via ISF), with shared slacks
    worst_gen = 0.0
    if lines and f is not None and scenario.contingencies:
        isf_csc = scenario.isf.tocsc()
        ref_1b = getattr(scenario, "ptdf_ref_bus_index", scenario.buses[0].index)
        non_ref_bus_indices = sorted(
            [b.index for b in scenario.buses if b.index != ref_1b]
        )
        col_by_bus_1b = {bus_1b: col for col, bus_1b in enumerate(non_ref_bus_indices)}

        for cont in scenario.contingencies:
            if not getattr(cont, "units", None):
                continue
            for gen in cont.units:
                bidx = gen.bus.index
                if bidx == ref_1b or bidx not in col_by_bus_1b:
                    continue
                col = isf_csc.getcol(col_by_bus_1b[bidx])
                for l_row, coeff in zip(col.indices.tolist(), col.data.tolist()):
                    if abs(coeff) < CONT_ISF_TOL:
                        continue
                    line_l = scenario.lines[l_row]
                    for t in range(T):
                        post = float(f[line_l.name, t].X) - float(
                            coeff
                        ) * _compute_p_of_gen_t(gen, t, commit, seg)
                        ovp_shared = (
                            float(covp[line_l.name, t].X) if covp is not None else 0.0
                        )
                        ovn_shared = (
                            float(covn[line_l.name, t].X) if covn is not None else 0.0
                        )
                        vpos = post - float(line_l.emergency_limit[t]) - ovp_shared
                        vneg = -post - float(line_l.emergency_limit[t]) - ovn_shared
                        worst_gen = max(worst_gen, vpos, vneg)
    checks.append(
        CheckItem(
            ID_C_CONT_GEN,
            "Post-contingency line limits (generator outage via ISF)",
            max(0.0, worst_gen),
        )
    )

    # Objective: O-301 (consistency)
    worst = 0.0
    try:
        obj_val = float(getattr(model, "ObjVal"))
    except Exception:
        obj_val = math.nan
    calc_val = _objective_from_vars(
        units,
        reserves,
        lines,
        T,
        commit,
        seg,
        s,
        ovp,
        ovn,
        covp,
        covn,
        su=su,
        contingency_penalty_factor=CONT_PENALTY_FACTOR,
    )
    if not (math.isnan(obj_val) or math.isinf(obj_val)):
        worst = abs(calc_val - obj_val)
    else:
        worst = float("nan")
    checks.append(
        CheckItem(ID_O_TOTAL, "Objective consistency (recomputed vs solver)", worst)
    )

    # Compose lines
    lines_out: List[str] = []
    lines_out.append("===== SCUC Solution Verification =====")
    lines_out.append(f"Scenario      : {scenario.name}")
    lines_out.append(f"Time steps    : {T} (step = {scenario.time_step} min)")
    lines_out.append(
        f"Counts        : {len(units)} units, {len(scenario.buses)} buses, {len(scenario.lines)} lines, {len(reserves)} reserve products"
    )
    lines_out.append("")
    lines_out.append("Index, Name, Result")
    for c in checks:
        if isinstance(c.value, float) and not math.isnan(c.value) and c.value <= EPS:
            res = "OK"
        else:
            res = (
                f"{c.value:.8f}"
                if isinstance(c.value, float) and not math.isnan(c.value)
                else str(c.value)
            )
        lines_out.append(f"{c.idx}, {c.name}, {res}")

    overall_ok = all(
        (isinstance(c.value, float) and c.value <= EPS)
        or (c.idx == ID_O_TOTAL and math.isnan(c.value))
        for c in checks
    )
    lines_out.append("")
    lines_out.append(f"Overall: {'OK' if overall_ok else 'VIOLATIONS DETECTED'}")

    return overall_ok, checks, "\n".join(lines_out)


def verify_solution_to_log(
    scenario: UnitCommitmentScenario,
    model,
    out_dir: Optional[Path] = None,
    filename: Optional[str] = None,
) -> Path:
    """
    Verify and write a report to src/data/logs (or a custom out_dir). Returns the report path.
    """
    out_dir = out_dir or DataParams._LOGS
    out_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = "SCUC_verification.log"
    out_path = out_dir / filename

    overall_ok, _checks, report_text = verify_solution(scenario, model)
    out_path.write_text(report_text, encoding="utf-8")
    return out_path

```

### File: `src/optimization_model/solver/economic_dispatch/constraints/__init__.py`

```
from src.optimization_model.solver.economic_dispatch.constraints import (
    commitment_fixing,
)
from src.optimization_model.solver.economic_dispatch.constraints import linking
from src.optimization_model.solver.economic_dispatch.constraints import (
    power_balance_segmented,
)

```

### File: `src/optimization_model/solver/economic_dispatch/constraints/commitment_fixing.py`

```
"""Fix commitment variables to known statuses when provided by data."""

import gurobipy as gp
from typing import Sequence


def add_constraints(
    model: gp.Model,
    generators: Sequence,
    commit,
    time_periods: range,
) -> None:
    for g in generators:
        for t in time_periods:
            status = g.commitment_status[t] if g.commitment_status else None
            if status is True:
                model.addConstr(
                    commit[g.name, t] == 1, name=f"fix_commit_on[{g.name},{t}]"
                )
            elif status is False:
                model.addConstr(
                    commit[g.name, t] == 0, name=f"fix_commit_off[{g.name},{t}]"
                )

```

### File: `src/optimization_model/solver/economic_dispatch/constraints/linking.py`

```
"""Link segment power to commitment: 0 <= pseg[g,t,s] <= amount[g,s,t] * u[g,t]."""

import gurobipy as gp
from typing import Sequence


def add_constraints(
    model: gp.Model,
    generators: Sequence,
    commit,
    seg_power,
    time_periods: range,
) -> None:
    for g in generators:
        Sg = len(g.segments) if g.segments else 0
        for t in time_periods:
            u_gt = commit[g.name, t]
            for s in range(Sg):
                amount = float(g.segments[s].amount[t])
                if amount < 0:
                    amount = 0.0
                model.addConstr(
                    seg_power[g.name, t, s] <= amount * u_gt,
                    name=f"seg_cap[{g.name},{t},{s}]",
                )

```

### File: `src/optimization_model/solver/economic_dispatch/constraints/power_balance_segmented.py`

```
"""System-wide power balance using commitment and segment power."""

import gurobipy as gp
from typing import Sequence


def add_constraints(
    model: gp.Model,
    total_load: Sequence[float],
    generators: Sequence,
    commit,
    seg_power,
    time_periods: range,
) -> None:
    for t in time_periods:
        production_t = gp.LinExpr()
        for g in generators:
            # Minimum output when committed
            production_t += commit[g.name, t] * float(g.min_power[t])

            # Incremental segments
            Sg = len(g.segments) if g.segments else 0
            if Sg > 0:
                production_t += gp.quicksum(seg_power[g.name, t, s] for s in range(Sg))

        model.addConstr(
            production_t == float(total_load[t]), name=f"power_balance[{t}]"
        )

```

### File: `src/optimization_model/solver/economic_dispatch/data/load.py`

```
def compute_total_load(buses, T):
    """Compute total system load at each time period."""

    total_load = [sum(b.load[t] for b in buses) for t in range(T)]
    return total_load

```

### File: `src/optimization_model/solver/economic_dispatch/objectives/__init__.py`

```
from src.optimization_model.solver.economic_dispatch.objectives import (
    power_cost_segmented,
)

```

### File: `src/optimization_model/solver/economic_dispatch/objectives/power_cost_segmented.py`

```
"""Production cost objective for segmented ED with commitment.

Minimize:
  sum_{g,t} u[g,t] * min_power_cost[g,t] +
  sum_{g,t,s} pseg[g,t,s] * seg_cost[g,s,t]
"""

import gurobipy as gp
from typing import Sequence

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def set_objective(
    model: gp.Model,
    generators: Sequence,
    commit,
    seg_power,
    time_periods: range,
) -> None:
    obj = gp.LinExpr()
    for g in generators:
        Sg = len(g.segments) if g.segments else 0
        for t in time_periods:
            # fixed cost part due to minimum output when on
            obj += commit[g.name, t] * float(g.min_power_cost[t])

            # marginal cost on each segment
            for s in range(Sg):
                obj += seg_power[g.name, t, s] * float(g.segments[s].cost[t])

    model.setObjective(obj, gp.GRB.MINIMIZE)
    try:
        model.getObjective().setAttr("ObjName", "TotalProductionCost")
    except Exception:
        logging.exception("Failed to set objective name 'TotalProductionCost'")

```

### File: `src/optimization_model/solver/economic_dispatch/vars/__init__.py`

```
from src.optimization_model.solver.economic_dispatch.vars import commitment
from src.optimization_model.solver.economic_dispatch.vars import segment_power

```

### File: `src/optimization_model/solver/economic_dispatch/vars/commitment.py`

```
"""Commitment variables u[g,t] in {0,1}."""

import gurobipy as gp
from gurobipy import GRB
from typing import Sequence


def add_variables(model: gp.Model, generators: Sequence, time_periods: range):
    """Add binary commitment variables to the model.

    Returns
    -------
    Var
        Gurobi Var dict keyed by (gen_name, t)
    """
    indices = [(gen.name, t) for gen in generators for t in time_periods]
    u = model.addVars(indices, vtype=GRB.BINARY, name="u")
    # Expose also as attribute if desired by callers
    model.__dict__["commit"] = u
    return u

```

### File: `src/optimization_model/solver/economic_dispatch/vars/segment_power.py`

```
"""Segment power variables pseg[g,t,s] >= 0."""

import gurobipy as gp
from typing import Sequence


def add_variables(model: gp.Model, generators: Sequence, time_periods: range):
    """Add segment power variables to the model.

    Returns
    -------
    Var
        Gurobi Var dict keyed by (gen_name, t, s)
    """
    idx = []
    for g in generators:
        Sg = len(g.segments) if g.segments else 0
        for t in time_periods:
            for s in range(Sg):
                idx.append((g.name, t, s))

    pseg = model.addVars(idx, lb=0.0, name="pseg")
    model.__dict__["seg_power"] = pseg
    return pseg

```

### File: `src/optimization_model/solver/scuc/constraints/__init__.py`

```
from src.optimization_model.solver.scuc.constraints import (
    commitment_fixing,
)
from src.optimization_model.solver.scuc.constraints import linking
from src.optimization_model.solver.scuc.constraints import (
    power_balance_segmented,
)
from src.optimization_model.solver.scuc.constraints import (
    reserve,
)
from src.optimization_model.solver.scuc.constraints import (
    reserve_requirement,
)
from src.optimization_model.solver.scuc.constraints import (
    initial_conditions,
)
from src.optimization_model.solver.scuc.constraints import (
    line_flow_ptdf,
)
from src.optimization_model.solver.scuc.constraints import (
    line_limits,
)
from src.optimization_model.solver.scuc.constraints import (
    min_up_down,
)
from src.optimization_model.solver.scuc.constraints import (
    contingencies,
)

```

### File: `src/optimization_model/solver/scuc/constraints/commitment_fixing.py`

```
"""ID: C-101 — Commitment fixing constraints.

Fix commitment variables to known statuses when provided by data:
- If status[gen,t] = True  -> u[gen,t] == 1
- If status[gen,t] = False -> u[gen,t] == 0
"""

import logging
import gurobipy as gp
from typing import Sequence

logger = logging.getLogger(__name__)


def add_constraints(
    model: gp.Model,
    generators: Sequence,
    commit,
    time_periods: range,
) -> None:
    n_on = 0
    n_off = 0
    for gen in generators:
        for t in time_periods:
            status = gen.commitment_status[t] if gen.commitment_status else None
            if status is True:
                model.addConstr(
                    commit[gen.name, t] == 1, name=f"fix_commit_on[{gen.name},{t}]"
                )
                n_on += 1
            elif status is False:
                model.addConstr(
                    commit[gen.name, t] == 0, name=f"fix_commit_off[{gen.name},{t}]"
                )
                n_off += 1
    logger.info(
        "Cons(C-101): commitment fixing on=%d, off=%d, total=%d",
        n_on,
        n_off,
        n_on + n_off,
    )

```

### File: `src/optimization_model/solver/scuc/constraints/contingencies.py`

```
"""ID: C-120/C-121 — Contingency constraints with slacks.

We enforce for every time t:

Line outage (C-120):
  For each contingency c that lists one or more outaged lines m, for every monitored
  line l != m:
      f_l,t + LODF[l,m] * f_m,t <= F_emergency[l,t] + cont_ov_pos[l,t]
     -f_l,t - LODF[l,m] * f_m,t <= F_emergency[l,t] + cont_ov_neg[l,t]

Generator outage (C-121):
  For each contingency c that lists one or more outaged generators g at bus b(g), for
  every monitored line l:
      f_l,t - ISF[l,b(g)] * p_g,t <= F_emergency[l,t] + cont_ov_pos[l,t]
     -f_l,t + ISF[l,b(g)] * p_g,t <= F_emergency[l,t] + cont_ov_neg[l,t]

Notes
- LODF matrix has shape (n_lines, n_lines).
- ISF (PTDF) matrix has shape (n_lines, n_buses-1) for all non-reference buses.
- Slack bus convention: the loss of p_g at bus b(g) is balanced by the reference bus.
  If b(g) equals the reference bus, ISF column is absent (treated as zero).
- Slacks are shared per (line,time) across all contingencies.

New:
- Optional filter_predicate(kind, line_l, out_obj, t, coeff, F_em) -> bool.
  If provided and returns False, the corresponding +/- constraints are not added.
  This enables ML-based redundancy pruning.
"""

import logging
import gurobipy as gp
from scipy.sparse import csc_matrix
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Drop extremely small coefficients to reduce constraint count
_LODF_TOL = 1e-4
_ISF_TOL = 1e-8


def _total_power_expr(gen, t: int, commit, seg_power) -> gp.LinExpr:
    """
    Build a linear expression for total power of generator gen at time t:
      p[gen,t] = u[gen,t] * min_power[gen,t] + sum_s pseg[gen,t,s]
    """
    expr = commit[gen.name, t] * float(gen.min_power[t])
    n_segments = len(gen.segments) if gen.segments else 0
    if n_segments > 0:
        expr += gp.quicksum(seg_power[gen.name, t, s] for s in range(n_segments))
    return expr


def add_constraints(
    model: gp.Model,
    scenario,
    commit,
    seg_power,
    line_flow,
    time_periods: range,
    cont_over_pos,
    cont_over_neg,
    filter_predicate: Optional[Callable] = None,
) -> None:
    contingencies = scenario.contingencies or []
    lines = scenario.lines or []
    if not contingencies or not lines or line_flow is None:
        return

    lodf_csc: csc_matrix = scenario.lodf.tocsc()  # (n_lines, n_lines)
    isf_csc: csc_matrix = scenario.isf.tocsc()  # (n_lines, n_buses-1)

    # Bus index mapping: non-reference columns in ascending 1-based bus index
    buses = scenario.buses
    ref_1b = getattr(scenario, "ptdf_ref_bus_index", buses[0].index)
    non_ref_bus_indices = sorted([b.index for b in buses if b.index != ref_1b])
    col_by_bus_1b = {bus_1b: col for col, bus_1b in enumerate(non_ref_bus_indices)}

    # Line row index to line object (index in data is 1-based)
    line_by_row = {ln.index - 1: ln for ln in lines}

    n_cons_line = 0
    n_cons_gen = 0
    n_conts_used_line = 0
    n_conts_used_gen = 0
    n_skipped_line = 0
    n_skipped_gen = 0

    for cont in contingencies:
        # Line-outage constraints (C-120)
        if cont.lines:
            n_conts_used_line += 1
            for out_line in cont.lines:
                mcol = out_line.index - 1
                col = lodf_csc.getcol(mcol)
                col_rows = col.indices.tolist()
                col_vals = col.data.tolist()

                for l_row, alpha_lm in zip(col_rows, col_vals):
                    # Skip the outaged line itself; skip tiny coefficients
                    if l_row == mcol:
                        continue
                    if abs(alpha_lm) < _LODF_TOL:
                        continue

                    line_l = line_by_row.get(l_row)
                    if line_l is None:
                        continue

                    for t in time_periods:
                        F_em = float(line_l.emergency_limit[t])
                        coeff = float(alpha_lm)

                        # ML-based pruning hook
                        if filter_predicate is not None:
                            try:
                                keep = bool(
                                    filter_predicate(
                                        "line", line_l, out_line, t, coeff, F_em
                                    )
                                )
                            except Exception:
                                keep = True
                            if not keep:
                                n_skipped_line += 2  # +/- pair
                                continue

                        # + direction
                        model.addConstr(
                            line_flow[line_l.name, t]
                            + coeff * line_flow[out_line.name, t]
                            <= F_em + cont_over_pos[line_l.name, t],
                            name=f"cont_line_pos[{cont.name},{line_l.name},{out_line.name},{t}]",
                        )
                        # - direction
                        model.addConstr(
                            -line_flow[line_l.name, t]
                            - coeff * line_flow[out_line.name, t]
                            <= F_em + cont_over_neg[line_l.name, t],
                            name=f"cont_line_neg[{cont.name},{line_l.name},{out_line.name},{t}]",
                        )
                        n_cons_line += 2

        # Generator-outage constraints (C-121)
        if getattr(cont, "units", None):
            n_conts_used_gen += 1
            for gen in cont.units:
                bus_1b = gen.bus.index
                # If the outaged generator is at the reference bus, ISF effect is zero
                if bus_1b == ref_1b or bus_1b not in col_by_bus_1b:
                    col = None
                else:
                    col = isf_csc.getcol(col_by_bus_1b[bus_1b])
                if col is None:
                    # ISF column absent => coefficient 0 for all lines (constraints reduce to +/- f_l <= F_em + slack)
                    # We still add them unless filtered (coeff=0).
                    for line_l in lines:
                        for t in time_periods:
                            F_em = float(line_l.emergency_limit[t])
                            coeff = 0.0
                            if filter_predicate is not None:
                                try:
                                    keep = bool(
                                        filter_predicate(
                                            "gen", line_l, gen, t, coeff, F_em
                                        )
                                    )
                                except Exception:
                                    keep = True
                                if not keep:
                                    n_skipped_gen += 2
                                    continue
                            model.addConstr(
                                line_flow[line_l.name, t]
                                <= F_em + cont_over_pos[line_l.name, t],
                                name=f"cont_gen_pos[{cont.name},{line_l.name},{gen.name},{t}]",
                            )
                            model.addConstr(
                                -line_flow[line_l.name, t]
                                <= F_em + cont_over_neg[line_l.name, t],
                                name=f"cont_gen_neg[{cont.name},{line_l.name},{gen.name},{t}]",
                            )
                            n_cons_gen += 2
                else:
                    col_rows = col.indices.tolist()
                    col_vals = col.data.tolist()

                    for l_row, isf_lb in zip(col_rows, col_vals):
                        if abs(isf_lb) < _ISF_TOL:
                            continue
                        line_l = line_by_row.get(l_row)
                        if line_l is None:
                            continue

                        for t in time_periods:
                            F_em = float(line_l.emergency_limit[t])
                            coeff = float(isf_lb)
                            if filter_predicate is not None:
                                try:
                                    keep = bool(
                                        filter_predicate(
                                            "gen", line_l, gen, t, coeff, F_em
                                        )
                                    )
                                except Exception:
                                    keep = True
                                if not keep:
                                    n_skipped_gen += 2
                                    continue
                            p_expr = _total_power_expr(gen, t, commit, seg_power)
                            # + direction: f_l - ISF * p_g <= F_em + slack
                            model.addConstr(
                                line_flow[line_l.name, t] - coeff * p_expr
                                <= F_em + cont_over_pos[line_l.name, t],
                                name=f"cont_gen_pos[{cont.name},{line_l.name},{gen.name},{t}]",
                            )
                            # - direction: -(f_l - ISF * p_g) <= F_em + slack
                            model.addConstr(
                                -line_flow[line_l.name, t] + coeff * p_expr
                                <= F_em + cont_over_neg[line_l.name, t],
                                name=f"cont_gen_neg[{cont.name},{line_l.name},{gen.name},{t}]",
                            )
                            n_cons_gen += 2

    logger.info(
        "Cons(C-120/C-121): line-out=%d (conts_used=%d, skipped=%d), gen-out=%d (conts_used=%d, skipped=%d); lines=%d, T=%d",
        n_cons_line,
        n_conts_used_line,
        n_skipped_line,
        n_cons_gen,
        n_conts_used_gen,
        n_skipped_gen,
        len(lines),
        len(time_periods),
    )

```

### File: `src/optimization_model/solver/scuc/constraints/initial_conditions.py`

```
"""ID: C-106/C-107 — Initial status and ramping with startup/shutdown limits.

Adds constraints that leverage:
  - Initial status (h)      -> to define previous commitment u_prev at t=0
  - Initial power (MW)      -> to define previous power p_prev at t=0
  - Startup limit (MW)      -> ramp up extra capability on startup
  - Shutdown limit (MW)     -> ramp down extra capability on shutdown

Define startup/shutdown indicators exactly:
  u[gen,t] - u_prev = v[gen,t] - w[gen,t]
  v[gen,t] + w[gen,t] <= 1

Ramping with startup/shutdown limits:
  p[gen,t] - p[gen,t-1] <= RU[gen] * u[gen,t-1] + SU[gen] * v[gen,t]
  p[gen,t-1] - p[gen,t] <= RD[gen] * u[gen,t]   + SD[gen] * w[gen,t]

For t=0, we use u_prev = u0 (from initial status) and p_prev = p0 (initial power).
"""

import logging
from typing import Sequence
import gurobipy as gp

logger = logging.getLogger(__name__)


def _total_power_expr(gen, t: int, commit, seg_power) -> gp.LinExpr:
    """
    Build a linear expression for total power of generator gen at time t:
      p[gen,t] = u[gen,t] * min_power[gen,t] + sum_s pseg[gen,t,s]
    """
    expr = commit[gen.name, t] * float(gen.min_power[t])
    n_segments = len(gen.segments) if gen.segments else 0
    if n_segments > 0:
        expr += gp.quicksum(seg_power[gen.name, t, s] for s in range(n_segments))
    return expr


def _initial_u(gen) -> float:
    """Return 1.0 if initial_status > 0 else 0.0 (None or <=0 => off)."""
    s = gen.initial_status
    try:
        return 1.0 if (s is not None and s > 0) else 0.0
    except Exception:
        return 0.0


def _initial_p(gen) -> float:
    """Return initial power if provided, else 0.0."""
    try:
        return float(gen.initial_power) if gen.initial_power is not None else 0.0
    except Exception:
        return 0.0


def add_constraints(
    model: gp.Model,
    generators: Sequence,
    commit,
    seg_power,
    startup,
    shutdown,
    time_periods: range,
) -> None:
    n_def = 0
    n_excl = 0
    n_ramp = 0

    # C-106: startup/shutdown definition + exclusivity
    for gen in generators:
        u0 = _initial_u(gen)
        for t in time_periods:
            if t == 0:
                model.addConstr(
                    commit[gen.name, t] - u0
                    == startup[gen.name, t] - shutdown[gen.name, t],
                    name=f"startstop_def[{gen.name},{t}]",
                )
            else:
                model.addConstr(
                    commit[gen.name, t] - commit[gen.name, t - 1]
                    == startup[gen.name, t] - shutdown[gen.name, t],
                    name=f"startstop_def[{gen.name},{t}]",
                )
            n_def += 1
            model.addConstr(
                startup[gen.name, t] + shutdown[gen.name, t] <= 1,
                name=f"startstop_exclusive[{gen.name},{t}]",
            )
            n_excl += 1

    # C-107: ramping with startup/shutdown limits (includes t=0 with initial conditions)
    for gen in generators:
        ru = float(gen.ramp_up)
        rd = float(gen.ramp_down)
        su = float(gen.startup_limit)
        sd = float(gen.shutdown_limit)

        p0 = _initial_p(gen)
        u0 = _initial_u(gen)

        for t in time_periods:
            p_t = _total_power_expr(gen, t, commit, seg_power)

            if t == 0:
                model.addConstr(
                    p_t - p0 <= ru * u0 + su * startup[gen.name, t],
                    name=f"ramp_up_init[{gen.name},{t}]",
                )
                model.addConstr(
                    p0 - p_t <= rd * commit[gen.name, t] + sd * shutdown[gen.name, t],
                    name=f"ramp_down_init[{gen.name},{t}]",
                )
                n_ramp += 2
            else:
                p_prev = _total_power_expr(gen, t - 1, commit, seg_power)
                model.addConstr(
                    p_t - p_prev
                    <= ru * commit[gen.name, t - 1] + su * startup[gen.name, t],
                    name=f"ramp_up[{gen.name},{t}]",
                )
                model.addConstr(
                    p_prev - p_t
                    <= rd * commit[gen.name, t] + sd * shutdown[gen.name, t],
                    name=f"ramp_down[{gen.name},{t}]",
                )
                n_ramp += 2

    logger.info(
        "Cons(C-106/C-107): start/stop def=%d, exclusivity=%d, ramping=%d (total=%d)",
        n_def,
        n_excl,
        n_ramp,
        n_def + n_excl + n_ramp,
    )

```

### File: `src/optimization_model/solver/scuc/constraints/line_flow_ptdf.py`

```
"""ID: C-108 — Line flow definition using PTDF.

Base-case DC flows:
  f[line,t] = sum_{b ∈ non_ref_buses} ISF[line, b] * inj[b,t]

where:
  inj[b,t] = sum_{gen at bus b} (u[gen,t]*min[gen,t] + sum_s pseg[gen,t,s]) - load[b,t]

Reference bus used is scenario.ptdf_ref_bus_index (1-based); ISF columns correspond
to all buses except the reference bus, in ascending 1-based index order.
"""

import logging
import gurobipy as gp
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def _total_power_expr(gen, t: int, commit, seg_power) -> gp.LinExpr:
    expr = commit[gen.name, t] * float(gen.min_power[t])
    n_segments = len(gen.segments) if gen.segments else 0
    if n_segments > 0:
        expr += gp.quicksum(seg_power[gen.name, t, s] for s in range(n_segments))
    return expr


def add_constraints(
    model: gp.Model,
    scenario,
    commit,
    seg_power,
    line_flow,
    time_periods: range,
) -> None:
    isf: csr_matrix = scenario.isf.tocsr()
    buses = scenario.buses
    lines = scenario.lines
    ref_1b = getattr(scenario, "ptdf_ref_bus_index", buses[0].index)
    non_ref_bus_indices = sorted([b.index for b in buses if b.index != ref_1b])

    n = 0
    for t in time_periods:
        inj_expr_by_busidx = {}
        for b in buses:
            expr = gp.LinExpr()
            for gen in b.thermal_units:
                expr += _total_power_expr(gen, t, commit, seg_power)
            expr += -float(b.load[t])
            inj_expr_by_busidx[b.index] = expr

        for line in lines:
            l_row = isf.getrow(line.index - 1)
            f_expr = gp.LinExpr()
            for col, coeff in zip(l_row.indices.tolist(), l_row.data.tolist()):
                bus_1b = non_ref_bus_indices[col]
                f_expr += float(coeff) * inj_expr_by_busidx[bus_1b]

            model.addConstr(
                line_flow[line.name, t] == f_expr,
                name=f"flow_def[{line.name},{t}]",
            )
            n += 1
    logger.info(
        "Cons(C-108): line flow PTDF equalities added=%d (ISF shape=%s nnz=%d, ref_bus=%s)",
        n,
        getattr(isf, "shape", None),
        getattr(isf, "nnz", None),
        ref_1b,
    )

```

### File: `src/optimization_model/solver/scuc/constraints/line_limits.py`

```
"""ID: C-109 — Line flow limits with overflow slacks.

For each line l and time t:
  f[l,t] <= F[l,t] + ov_pos[l,t]
 -f[l,t] <= F[l,t] + ov_neg[l,t]

with ov_pos, ov_neg >= 0 and penalty in the objective.
"""

import logging
from typing import Sequence
import gurobipy as gp

logger = logging.getLogger(__name__)


def add_constraints(
    model: gp.Model,
    lines: Sequence,
    line_flow,
    over_pos,
    over_neg,
    time_periods: range,
) -> None:
    n = 0
    for line in lines:
        for t in time_periods:
            F = float(line.normal_limit[t])
            model.addConstr(
                line_flow[line.name, t] <= F + over_pos[line.name, t],
                name=f"flow_pos_limit[{line.name},{t}]",
            )
            model.addConstr(
                -line_flow[line.name, t] <= F + over_neg[line.name, t],
                name=f"flow_neg_limit[{line.name},{t}]",
            )
            n += 2
    logger.info(
        "Cons(C-109): line limits added=%d (lines=%d, T=%d)",
        n,
        len(lines),
        len(time_periods),
    )

```

### File: `src/optimization_model/solver/scuc/constraints/linking.py`

```
"""ID: C-102 — Segment capacity linking.

Link segment power to commitment:
  0 <= pseg[gen,t,s] <= amount[gen,s,t] * u[gen,t].
If amount < 0 in data, it is treated as 0.
"""

import logging
import gurobipy as gp
from typing import Sequence

logger = logging.getLogger(__name__)


def add_constraints(
    model: gp.Model,
    generators: Sequence,
    commit,
    seg_power,
    time_periods: range,
) -> None:
    n = 0
    for gen in generators:
        n_segments = len(gen.segments) if gen.segments else 0
        for t in time_periods:
            u_gen_t = commit[gen.name, t]
            for s in range(n_segments):
                amount = float(gen.segments[s].amount[t])
                if amount < 0:
                    amount = 0.0
                model.addConstr(
                    seg_power[gen.name, t, s] <= amount * u_gen_t,
                    name=f"seg_cap[{gen.name},{t},{s}]",
                )
                n += 1
    logger.info("Cons(C-102): segment capacity linking added=%d", n)

```

### File: `src/optimization_model/solver/scuc/constraints/min_up_down.py`

```
"""ID: C-110/C-111/C-112 — Minimum up/down time constraints.

Implements minimum up and down times using startup/shutdown indicators (v, w):

Sliding-window constraints (classic UC formulation):
  - Minimum up-time L_u:
        sum_{k=t-L_u+1}^t v[g,k] <= u[g,t]
  - Minimum down-time L_d:
        sum_{k=t-L_d+1}^t w[g,k] <= 1 - u[g,t]

Initial-condition enforcement at the horizon boundary:
  Let s = initial_status (in steps); s > 0 means the unit has been ON for s steps,
  s < 0 means the unit has been OFF for |s| steps. Then, if s < L_u and the unit is
  ON initially, it must stay ON for (L_u - s) steps from t=0. Similarly, if s < L_d
  and the unit is OFF initially, it must stay OFF for (L_d - s) steps from t=0:

      if s > 0:  for t=0..(L_u - s - 1):  u[g,t] == 1
      if s < 0:  for t=0..(L_d - |s| - 1):  u[g,t] == 0
"""

import logging
from typing import Sequence
import gurobipy as gp

logger = logging.getLogger(__name__)


def _int_or_zero(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def add_constraints(
    model: gp.Model,
    generators: Sequence,
    commit,
    startup,
    shutdown,
    time_periods: range,
) -> None:
    T = len(time_periods)
    n_min_up = 0
    n_min_down = 0
    n_init_on = 0
    n_init_off = 0

    # Sliding-window minimum up/down time constraints
    for g in generators:
        Lu = _int_or_zero(getattr(g, "min_up", 0))
        Ld = _int_or_zero(getattr(g, "min_down", 0))

        if Lu and Lu > 0:
            for t in time_periods:
                start_k = max(0, t - Lu + 1)
                if start_k <= t:
                    lhs = gp.quicksum(startup[g.name, k] for k in range(start_k, t + 1))
                    model.addConstr(
                        lhs <= commit[g.name, t],
                        name=f"min_up_window[{g.name},{t}]",
                    )
                    n_min_up += 1

        if Ld and Ld > 0:
            for t in time_periods:
                start_k = max(0, t - Ld + 1)
                if start_k <= t:
                    lhs = gp.quicksum(
                        shutdown[g.name, k] for k in range(start_k, t + 1)
                    )
                    model.addConstr(
                        lhs <= 1 - commit[g.name, t],
                        name=f"min_down_window[{g.name},{t}]",
                    )
                    n_min_down += 1

    # Initial-condition enforcement across the left boundary of the horizon
    for g in generators:
        Lu = _int_or_zero(getattr(g, "min_up", 0))
        Ld = _int_or_zero(getattr(g, "min_down", 0))
        s = getattr(g, "initial_status", None)
        if s is None:
            continue

        # If initially on for s>0 steps, enforce remaining up-time
        if s > 0 and Lu > 0:
            remaining_on = max(0, Lu - int(s))
            for t in range(min(remaining_on, T)):
                model.addConstr(
                    commit[g.name, t] == 1, name=f"min_up_initial_enforce[{g.name},{t}]"
                )
                n_init_on += 1

        # If initially off for |s| steps, enforce remaining down-time
        if s < 0 and Ld > 0:
            s_off = -int(s)
            remaining_off = max(0, Ld - s_off)
            for t in range(min(remaining_off, T)):
                model.addConstr(
                    commit[g.name, t] == 0,
                    name=f"min_down_initial_enforce[{g.name},{t}]",
                )
                n_init_off += 1

    logger.info(
        "Cons(C-110/111/112): min_up=%d, min_down=%d, initial_on=%d, initial_off=%d, total=%d",
        n_min_up,
        n_min_down,
        n_init_on,
        n_init_off,
        n_min_up + n_min_down + n_init_on + n_init_off,
    )

```

### File: `src/optimization_model/solver/scuc/constraints/power_balance_segmented.py`

```
"""ID: C-103 — System-wide power balance.

Sum over generators of:
  u[gen,t] * min_power[gen,t] + sum_s pseg[gen,t,s]
must equal total system load at time t.
"""

import logging
import gurobipy as gp
from typing import Sequence

logger = logging.getLogger(__name__)


def add_constraints(
    model: gp.Model,
    total_load: Sequence[float],
    generators: Sequence,
    commit,
    seg_power,
    time_periods: range,
) -> None:
    for t in time_periods:
        production_t = gp.LinExpr()
        for gen in generators:
            # Minimum output when committed
            production_t += commit[gen.name, t] * float(gen.min_power[t])

            # Incremental segments
            n_segments = len(gen.segments) if gen.segments else 0
            if n_segments > 0:
                production_t += gp.quicksum(
                    seg_power[gen.name, t, s] for s in range(n_segments)
                )

        model.addConstr(
            production_t == float(total_load[t]), name=f"power_balance[{t}]"
        )
    logger.info("Cons(C-103): power balance equalities added=%d", len(time_periods))

```

### File: `src/optimization_model/solver/scuc/constraints/reserve.py`

```
"""ID: C-104 — Reserve headroom linking (shared across products).

For each generator gen and time t:
    sum_s pseg[gen,t,s] + sum_k r[k,gen,t] <= (max[gen,t] - min[gen,t]) * u[gen,t]

This shares the above-minimum headroom across all reserve products,
preventing double counting of headroom.
"""

import logging
from typing import Sequence
import gurobipy as gp

logger = logging.getLogger(__name__)


def add_constraints(
    model: gp.Model,
    reserves: Sequence,
    commit,
    seg_power,
    reserve,
    time_periods: range,
) -> None:
    """
    Shared headroom constraint across all reserve products per generator and time.
    """
    eligible_by_gen = {}
    for r in reserves:
        for g in r.thermal_units:
            eligible_by_gen.setdefault(g.name, []).append(r)

    n = 0
    for gen_name, rlist in eligible_by_gen.items():
        gen_obj = None
        for r in reserves:
            for g in r.thermal_units:
                if g.name == gen_name:
                    gen_obj = g
                    break
            if gen_obj is not None:
                break
        if gen_obj is None:
            continue

        n_segments = len(gen_obj.segments) if gen_obj.segments else 0
        for t in time_periods:
            energy_above_min = (
                gp.quicksum(seg_power[gen_obj.name, t, s] for s in range(n_segments))
                if n_segments > 0
                else 0.0
            )
            total_reserve = gp.quicksum(reserve[r.name, gen_obj.name, t] for r in rlist)
            headroom_coeff = float(gen_obj.max_power[t]) - float(gen_obj.min_power[t])

            model.addConstr(
                energy_above_min + total_reserve
                <= headroom_coeff * commit[gen_obj.name, t],
                name=f"reserve_headroom_shared[{gen_obj.name},{t}]",
            )
            n += 1
    logger.info("Cons(C-104): reserve headroom linking added=%d", n)

```

### File: `src/optimization_model/solver/scuc/constraints/reserve_requirement.py`

```
"""ID: C-105 — Reserve requirement.

For each reserve product k and time t:
  sum_{g eligible} r[k,g,t] + shortfall[k,t] >= requirement[k,t]
"""

import logging
from typing import Sequence
import gurobipy as gp

logger = logging.getLogger(__name__)


def add_constraints(
    model: gp.Model,
    reserves: Sequence,
    reserve,
    shortfall,
    time_periods: range,
) -> None:
    """
    For each reserve product k and time t:
        sum_{g eligible} r[k,g,t] + shortfall[k,t] >= requirement[k,t]
    """
    n = 0
    for r in reserves:
        for t in time_periods:
            provided = gp.quicksum(reserve[r.name, g.name, t] for g in r.thermal_units)
            model.addConstr(
                provided + shortfall[r.name, t] >= float(r.amount[t]),
                name=f"reserve_requirement[{r.name},{t}]",
            )
            n += 1
    logger.info("Cons(C-105): reserve requirement added=%d", n)

```

### File: `src/optimization_model/solver/scuc/data/load.py`

```
def compute_total_load(buses, T):
    """Compute total system load at each time period."""

    total_load = [sum(b.load[t] for b in buses) for t in range(T)]
    return total_load

```

### File: `src/optimization_model/solver/scuc/objectives/__init__.py`

```
from src.optimization_model.solver.scuc.objectives import minimum_output_cost
from src.optimization_model.solver.scuc.objectives import segment_power_cost
from src.optimization_model.solver.scuc.objectives import startup_cost
from src.optimization_model.solver.scuc.objectives import (
    reserve_shortfall_penalty,
)
from src.optimization_model.solver.scuc.objectives import base_overflow_penalty
from src.optimization_model.solver.scuc.objectives import (
    contingency_overflow_penalty,
)

```

### File: `src/optimization_model/solver/scuc/objectives/base_overflow_penalty.py`

```
import gurobipy as gp
from typing import Sequence


def build_expression(
    model: gp.Model, lines: Sequence, time_periods: range
) -> gp.LinExpr:
    """
    Base-case line overflow penalty:
      sum_{l,t} (ov_pos[l,t] + ov_neg[l,t]) * flow_penalty[l,t]
    """
    expr = gp.LinExpr()
    if not lines:
        return expr
    over_pos = getattr(model, "line_overflow_pos", None)
    over_neg = getattr(model, "line_overflow_neg", None)
    if over_pos is None or over_neg is None:
        return expr
    for line in lines:
        for t in time_periods:
            pen = float(line.flow_penalty[t])
            expr += pen * (over_pos[line.name, t] + over_neg[line.name, t])
    return expr

```

### File: `src/optimization_model/solver/scuc/objectives/contingency_overflow_penalty.py`

```
import gurobipy as gp
from typing import Sequence


def build_expression(
    model: gp.Model,
    lines: Sequence,
    time_periods: range,
    contingency_penalty_factor: float,
) -> gp.LinExpr:
    """
    Contingency overflow penalty with multiplier:
      sum_{l,t} (cont_ovp[l,t] + cont_ovn[l,t]) * flow_penalty[l,t] * factor
    """
    expr = gp.LinExpr()
    if not lines:
        return expr
    cont_ovp = getattr(model, "contingency_overflow_pos", None)
    cont_ovn = getattr(model, "contingency_overflow_neg", None)
    if cont_ovp is None or cont_ovn is None:
        return expr

    for line in lines:
        for t in time_periods:
            pen = float(line.flow_penalty[t]) * contingency_penalty_factor
            expr += pen * (cont_ovp[line.name, t] + cont_ovn[line.name, t])
    return expr

```

### File: `src/optimization_model/solver/scuc/objectives/minimum_output_cost.py`

```
import gurobipy as gp
from typing import Sequence


def build_expression(
    model: gp.Model, generators: Sequence, commit, time_periods: range
) -> gp.LinExpr:
    """
    Sum of minimum-output (no-load) cost:
      sum_{g,t} u[g,t] * min_power_cost[g,t]
    """
    expr = gp.LinExpr()
    for g in generators:
        for t in time_periods:
            expr += commit[g.name, t] * float(g.min_power_cost[t])
    return expr

```

### File: `src/optimization_model/solver/scuc/objectives/power_cost_segmented.py`

```
"""ID: O-301 — Total cost with reserve, startup, and line penalties.

This module assembles the SCUC objective from modular term expressions:
  - Minimum-output (no-load) cost
  - Energy (segment) cost
  - Startup cost
  - Reserve shortfall penalty
  - Base-case line overflow penalty
  - Contingency overflow penalty
"""

import logging
import gurobipy as gp
from typing import Sequence, Optional

from src.optimization_model.solver.scuc.objectives import (
    minimum_output_cost as oc_min,
    segment_power_cost as oc_seg,
    startup_cost as oc_su,
    reserve_shortfall_penalty as oc_res,
    base_overflow_penalty as oc_flow,
    contingency_overflow_penalty as oc_cflow,
)

logger = logging.getLogger(__name__)

# Penalty multiplier for contingency slacks vs. base-case overflow
_CONTINGENCY_PENALTY_FACTOR = 10.0


def set_objective(
    model: gp.Model,
    generators: Sequence,
    commit,
    seg_power,
    time_periods: range,
    reserves: Optional[Sequence] = None,
    lines: Optional[Sequence] = None,
) -> None:
    # Build separate expressions
    expr_min = oc_min.build_expression(model, generators, commit, time_periods)
    expr_seg = oc_seg.build_expression(model, generators, seg_power, time_periods)

    startup_vars = getattr(model, "startup", None)
    expr_su = gp.LinExpr()
    if startup_vars is not None:
        expr_su = oc_su.build_expression(model, generators, startup_vars, time_periods)

    expr_res = gp.LinExpr()
    if reserves:
        expr_res = oc_res.build_expression(model, reserves, time_periods)

    expr_flow = gp.LinExpr()
    expr_cflow = gp.LinExpr()
    if lines:
        expr_flow = oc_flow.build_expression(model, lines, time_periods)
        expr_cflow = oc_cflow.build_expression(
            model,
            lines,
            time_periods,
            contingency_penalty_factor=_CONTINGENCY_PENALTY_FACTOR,
        )

    # Sum expressions
    obj = gp.LinExpr()
    obj += expr_min
    obj += expr_seg
    obj += expr_su
    obj += expr_res
    obj += expr_flow
    obj += expr_cflow

    # Set the objective
    model.setObjective(obj, gp.GRB.MINIMIZE)
    try:
        model._objective_name = "TotalCostWithReserveStartupAndLinePenalty"
    except Exception:
        pass

    logger.info("Obj(O-301): objective assembled from modular terms.")

```

### File: `src/optimization_model/solver/scuc/objectives/reserve_shortfall_penalty.py`

```
import gurobipy as gp
from typing import Sequence


def build_expression(
    model: gp.Model, reserves: Sequence, time_periods: range
) -> gp.LinExpr:
    """
    Reserve shortfall penalty:
      sum_{k,t} shortfall[k,t] * shortfall_penalty_k
    """
    expr = gp.LinExpr()
    shortfall_vars = getattr(model, "reserve_shortfall", None)
    if reserves and shortfall_vars is not None:
        for r in reserves:
            penalty = float(r.shortfall_penalty)
            for t in time_periods:
                expr += shortfall_vars[r.name, t] * penalty
    return expr

```

### File: `src/optimization_model/solver/scuc/objectives/segment_power_cost.py`

```
import gurobipy as gp
from typing import Sequence


def build_expression(
    model: gp.Model, generators: Sequence, seg_power, time_periods: range
) -> gp.LinExpr:
    """
    Sum of energy cost on piecewise-linear segments:
      sum_{g,t,s} pseg[g,t,s] * seg_cost[g,s,t]
    """
    expr = gp.LinExpr()
    for g in generators:
        Sg = len(g.segments) if g.segments else 0
        for t in time_periods:
            for s in range(Sg):
                expr += seg_power[g.name, t, s] * float(g.segments[s].cost[t])
    return expr

```

### File: `src/optimization_model/solver/scuc/objectives/startup_cost.py`

```
import gurobipy as gp
from typing import Sequence


def _startup_cost_for_gen(g) -> float:
    """
    Chosen policy: 'hot start' cost = minimum of provided startup categories.
    If no categories, return 0.0.

    Rationale:
      - We currently model a single startup indicator per period (no downtime-dependent category selection).
      - Using the 'hot' (lowest) cost keeps consistency and avoids over-charging.
      - Can be extended later to downtime-dependent categories if needed.
    """
    try:
        if g.startup_categories:
            return float(min(cat.cost for cat in g.startup_categories))
    except Exception:
        pass
    return 0.0


def build_expression(
    model: gp.Model, generators: Sequence, startup, time_periods: range
) -> gp.LinExpr:
    """
    Sum of startup costs:
      sum_{g,t} startup[g,t] * StartupCost_g

    StartupCost_g is computed with _startup_cost_for_gen (currently 'hot' cost).
    """
    expr = gp.LinExpr()
    for g in generators:
        s_cost = _startup_cost_for_gen(g)
        if s_cost == 0.0:
            continue
        for t in time_periods:
            expr += startup[g.name, t] * s_cost
    return expr

```

### File: `src/optimization_model/solver/scuc/vars/__init__.py`

```
from src.optimization_model.solver.scuc.vars import commitment
from src.optimization_model.solver.scuc.vars import segment_power
from src.optimization_model.solver.scuc.vars import reserve
from src.optimization_model.solver.scuc.vars import startup_shutdown
from src.optimization_model.solver.scuc.vars import line_flow
from src.optimization_model.solver.scuc.vars import contingency_redispatch
from src.optimization_model.solver.scuc.vars import contingency_overflow

```

### File: `src/optimization_model/solver/scuc/vars/commitment.py`

```
"""ID: V-201 — Commitment variables u[g,t] in {0,1}."""

import logging
import gurobipy as gp
from gurobipy import GRB
from typing import Sequence

logger = logging.getLogger(__name__)


def add_variables(model: gp.Model, generators: Sequence, time_periods: range):
    """Add binary commitment variables to the model.

    Returns
    -------
    Var
        Gurobi Var dict keyed by (gen_name, t)
    """
    indices = [(gen.name, t) for gen in generators for t in time_periods]
    gen_commit = model.addVars(indices, vtype=GRB.BINARY, name="gen_commit")
    model.__dict__["commit"] = gen_commit
    logger.info("Vars(V-201): commitment u[g,t] added=%d", len(indices))
    return gen_commit

```

### File: `src/optimization_model/solver/scuc/vars/contingency_overflow.py`

```
"""ID: V-212/V-213 — Contingency overflow slacks.

- V-212: cont_overflow_pos[l,t] >= 0
- V-213: cont_overflow_neg[l,t] >= 0

These slacks are shared across all contingencies for a given (line,time).
"""

import logging
from typing import Sequence, Tuple
import gurobipy as gp

logger = logging.getLogger(__name__)


def add_variables(
    model: gp.Model, lines: Sequence, time_periods: range
) -> Tuple[gp.tupledict, gp.tupledict]:
    idx = [(line.name, t) for line in lines for t in time_periods]
    ovp = model.addVars(idx, lb=0.0, name="cont_overflow_pos")
    ovn = model.addVars(idx, lb=0.0, name="cont_overflow_neg")
    model.__dict__["contingency_overflow_pos"] = ovp
    model.__dict__["contingency_overflow_neg"] = ovn
    logger.info(
        "Vars(V-212/V-213): contingency overflow slacks added=%d (lines=%d, T=%d)",
        len(idx) * 2,
        len(lines),
        len(time_periods),
    )
    return ovp, ovn

```

### File: `src/optimization_model/solver/scuc/vars/contingency_redispatch.py`

```
"""ID: V-210/V-211 — Post-contingency redispatch variables.

- V-210: delta_up[c, g, t] >= 0  (upward deployment of reserves under contingency c)
- V-211: delta_down[c, g, t] >= 0 (downward curtailment of above-min energy under contingency c)

These variables are bounded in constraints/contingencies.py by:
- delta_up[c,g,t] <= sum_k r[k,g,t]
- delta_down[c,g,t] <= sum_s pseg[g,t,s]
and set to 0 for outaged units.
"""

import logging
from typing import Sequence, Tuple
import gurobipy as gp

logger = logging.getLogger(__name__)


def add_variables(
    model: gp.Model,
    contingencies: Sequence,
    generators: Sequence,
    time_periods: range,
) -> Tuple[gp.tupledict, gp.tupledict]:
    idx = [
        (c.name, g.name, t)
        for c in contingencies
        for g in generators
        for t in time_periods
    ]
    before = model.NumVars
    delta_up = model.addVars(idx, lb=0.0, name="cont_delta_up")
    delta_dn = model.addVars(idx, lb=0.0, name="cont_delta_down")

    model.__dict__["cont_delta_up"] = delta_up
    model.__dict__["cont_delta_down"] = delta_dn

    logger.info(
        "Vars(V-210/V-211): contingency redispatch added=%d (conts=%d, gens=%d, T=%d)",
        model.NumVars - before,
        len(contingencies),
        len(generators),
        len(time_periods),
    )
    return delta_up, delta_dn

```

### File: `src/optimization_model/solver/scuc/vars/line_flow.py`

```
"""ID: V-207/V-208/V-209 — Line flow variables and slack.

- V-207: Line flow variables f[l,t] (free continuous)
- V-208: Positive overflow slack ov_pos[l,t] >= 0
- V-209: Negative overflow slack ov_neg[l,t] >= 0
"""

import logging
from typing import Sequence, Tuple
import gurobipy as gp
from gurobipy import GRB

logger = logging.getLogger(__name__)


def add_variables(
    model: gp.Model, lines: Sequence, time_periods: range
) -> Tuple[gp.tupledict, gp.tupledict, gp.tupledict]:
    """
    Create:
      - line_flow[l,t] ∈ ℝ (free)
      - line_overflow_pos[l,t] ≥ 0
      - line_overflow_neg[l,t] ≥ 0
    """
    idx = [(line.name, t) for line in lines for t in time_periods]
    before = model.NumVars

    line_flow = model.addVars(idx, lb=-GRB.INFINITY, name="line_flow")
    over_pos = model.addVars(idx, lb=0.0, name="line_overflow_pos")
    over_neg = model.addVars(idx, lb=0.0, name="line_overflow_neg")

    model.__dict__["line_flow"] = line_flow
    model.__dict__["line_overflow_pos"] = over_pos
    model.__dict__["line_overflow_neg"] = over_neg

    logger.info(
        "Vars(V-207/208/209): line_flow=%d, overflow_pos=%d, overflow_neg=%d (lines=%d, T=%d)",
        len(idx),
        len(idx),
        len(idx),
        len(lines),
        len(time_periods),
    )
    return line_flow, over_pos, over_neg

```

### File: `src/optimization_model/solver/scuc/vars/reserve.py`

```
"""ID: V-203/V-204 — Reserve variables.

- V-203: Reserve provision variables r[k,g,t] >= 0
- V-204: Reserve shortfall variables s[k,t] >= 0
"""

import logging
from typing import Sequence
import gurobipy as gp

logger = logging.getLogger(__name__)


def add_variables(
    model: gp.Model,
    reserves: Sequence,
    generators: Sequence,  # kept for signature consistency; not used directly
    time_periods: range,
):
    """
    Create:
      - reserve provision variables r[k,g,t] for each reserve product k and eligible generator g
      - shortfall variables s[k,t] for each reserve product k

    Returns
    -------
    (Var, Var)
        (reserve, shortfall) Gurobi Var dicts keyed by (k, g, t) and (k, t)
    """
    prov_idx = []
    eligible_count = 0
    for r in reserves:
        for g in r.thermal_units:
            eligible_count += 1
            for t in time_periods:
                prov_idx.append((r.name, g.name, t))
    before = model.NumVars
    reserve = model.addVars(prov_idx, lb=0.0, name="r")

    short_idx = [(r.name, t) for r in reserves for t in time_periods]
    shortfall = model.addVars(short_idx, lb=0.0, name="r_shortfall")
    model.__dict__["reserve"] = reserve
    model.__dict__["reserve_shortfall"] = shortfall

    logger.info(
        "Vars(V-203/V-204): reserve r[k,g,t] added=%d (eligible_pairs=%d*T), shortfall s[k,t] added=%d",
        len(prov_idx),
        eligible_count,
        len(short_idx),
    )
    return reserve, shortfall

```

### File: `src/optimization_model/solver/scuc/vars/segment_power.py`

```
"""ID: V-202 — Segment power variables pseg[gen,t,s] >= 0."""

import logging
import gurobipy as gp
from typing import Sequence

logger = logging.getLogger(__name__)


def add_variables(model: gp.Model, generators: Sequence, time_periods: range):
    """Add segment power variables to the model.

    Returns
    -------
    Var
        Gurobi Var dict keyed by (gen_name, t, s)
    """
    idx = []
    total_segments = 0
    for gen in generators:
        n_segments = len(gen.segments) if gen.segments else 0
        total_segments += n_segments
        for t in time_periods:
            for s in range(n_segments):
                idx.append((gen.name, t, s))

    gen_segment_power = model.addVars(idx, lb=0.0, name="gen_segment_power")
    model.__dict__["gen_segment_power"] = gen_segment_power
    logger.info(
        "Vars(V-202): segment power pseg[g,t,s] added=%d (gens=%d, total_segments=%d, T=%d)",
        len(idx),
        len(generators),
        total_segments,
        len(time_periods),
    )
    return gen_segment_power

```

### File: `src/optimization_model/solver/scuc/vars/startup_shutdown.py`

```
"""ID: V-205/V-206 — Startup/Shutdown binary variables.

- V-205: Startup indicators v[g,t] in {0,1}
- V-206: Shutdown indicators w[g,t] in {0,1}
"""

import logging
from typing import Sequence, Tuple
import gurobipy as gp
from gurobipy import GRB

logger = logging.getLogger(__name__)


def add_variables(
    model: gp.Model, generators: Sequence, time_periods: range
) -> Tuple[gp.tupledict, gp.tupledict]:
    """
    Create binary variables:
      - startup[g,t]  ∈ {0,1}
      - shutdown[g,t] ∈ {0,1}
    """
    idx = [(g.name, t) for g in generators for t in time_periods]
    startup = model.addVars(idx, vtype=GRB.BINARY, name="gen_startup")
    shutdown = model.addVars(idx, vtype=GRB.BINARY, name="gen_shutdown")

    model.__dict__["startup"] = startup
    model.__dict__["shutdown"] = shutdown

    logger.info(
        "Vars(V-205/V-206): startup/shutdown added=%d (pairs=%d)",
        len(idx) * 2,
        len(idx),
    )
    return startup, shutdown

```

