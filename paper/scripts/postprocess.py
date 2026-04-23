"""Post-processing of already-computed revision results.

Reads pre-computed artifacts from `paper/data/` and produces additional
summaries requested by reviewers (CI-enriched speedup table, GNN vs 1-NN
ablation, case1354pegase runtime table, quantitative literature context).

This script performs NO new solver runs. It only re-aggregates existing
CSVs into LaTeX-ready tables and a machine-readable JSON summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TABLES = HERE.parent / "tables"
TABLES.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Mean speedup with 95% bootstrap CI for the key modes (R1-C9)
# ---------------------------------------------------------------------------
def speedup_with_ci() -> pd.DataFrame:
    df = pd.read_csv(DATA / "speedup_summary.csv")
    df = df[df["mode"] != "RAW"].copy()
    df["case_short"] = df["case_folder"].str.replace("matpower/", "", regex=False)
    df["speedup_ci"] = df.apply(
        lambda r: f"{r['speedup_mean']:.2f} [{r['speedup_mean_ci_lo']:.2f}, {r['speedup_mean_ci_hi']:.2f}]",
        axis=1,
    )
    wide = df.pivot(index="mode", columns="case_short", values="speedup_ci")

    key_modes = [
        "RAW+COMMIT",
        "WARM",
        "WARM+HINTS+COMMIT",
        "WARM+LAZY",
        "WARM+LAZY+BANDIT",
        "WARM+PRUNE-0.10",
        "WARM+PRUNE-0.10+GNN",
        "WARM+PRUNE-0.30",
        "WARM+PRUNE-0.50",
        "WARM+PRUNE-0.80",
    ]
    wide = wide.reindex(key_modes)
    case_order = ["case14", "case30", "case57", "case89pegase", "case118", "case300"]
    wide = wide[case_order]
    wide.to_csv(TABLES / "speedup_with_ci.csv")
    return wide


def render_speedup_ci_latex(wide: pd.DataFrame) -> str:
    """Produce a compact sideways LaTeX table with CI-enriched speedups."""
    cases = list(wide.columns)
    header = (
        "\\begin{sidewaystable*}\\centering\n"
        "\\caption{Speed-up factors with bootstrap 95\\% confidence intervals "
        "(mean [CI\\textsubscript{lo}, CI\\textsubscript{hi}]).}\n"
        "\\label{tab:speedup_ci}\n"
        "\\renewcommand{\\arraystretch}{1.2}\n"
        "\\footnotesize\n"
        "\\begin{tabular}{|l||" + "c|" * len(cases) + "}\\hline\n"
        "\\textbf{Mode} & "
        + " & ".join(f"\\textbf{{{c}}}" for c in cases)
        + " \\\\ \\hline \\hline\n"
    )
    rows = []
    for mode, row in wide.iterrows():
        cells = [str(v).replace("nan", "--") for v in row.values]
        rows.append(f"{mode} & " + " & ".join(cells) + " \\\\ \\hline")
    body = "\n".join(rows)
    footer = "\n\\end{tabular}\n\\end{sidewaystable*}\n"
    tex = header + body + footer
    (TABLES / "speedup_with_ci.tex").write_text(tex, encoding="utf-8")
    return tex


# ---------------------------------------------------------------------------
# 2. GNN vs 1-NN ablation (R1-C5): does adding GNN change retention / speedup?
# ---------------------------------------------------------------------------
def gnn_vs_1nn() -> pd.DataFrame:
    spd = pd.read_csv(DATA / "speedup_summary.csv")
    red = pd.read_csv(DATA / "constraint_reduction.csv")

    def canon_mode(m: str) -> tuple[str, bool]:
        has_gnn = m.endswith("+GNN")
        base = m.replace("+GNN", "") if has_gnn else m
        return base, has_gnn

    spd[["base_mode", "has_gnn"]] = spd["mode"].apply(
        lambda m: pd.Series(canon_mode(m))
    )
    red[["base_mode", "has_gnn"]] = red["mode"].apply(
        lambda m: pd.Series(canon_mode(m))
    )

    # keep only PRUNE-0.xx threshold families (both 1-NN and +GNN exist)
    spd = spd[spd["base_mode"].str.startswith("WARM+PRUNE-")]
    red = red[red["base_mode"].str.startswith("WARM+PRUNE-")]

    pivot_spd = spd.pivot_table(
        index=["case_folder", "base_mode"],
        columns="has_gnn",
        values="speedup_mean",
    ).rename(columns={False: "speedup_1nn", True: "speedup_gnn"})
    pivot_red = red.pivot_table(
        index=["case_folder", "base_mode"],
        columns="has_gnn",
        values="kept_ratio_mean",
    ).rename(columns={False: "kept_1nn", True: "kept_gnn"})

    merged = pivot_spd.join(pivot_red, how="inner").reset_index()
    merged["speedup_ratio_gnn_over_1nn"] = (
        merged["speedup_gnn"] / merged["speedup_1nn"]
    )
    merged["kept_delta"] = merged["kept_gnn"] - merged["kept_1nn"]
    merged["case_short"] = merged["case_folder"].str.replace(
        "matpower/", "", regex=False
    )
    merged = merged[
        [
            "case_short",
            "base_mode",
            "speedup_1nn",
            "speedup_gnn",
            "speedup_ratio_gnn_over_1nn",
            "kept_1nn",
            "kept_gnn",
            "kept_delta",
        ]
    ].sort_values(["base_mode", "case_short"])
    merged.to_csv(TABLES / "gnn_vs_1nn.csv", index=False)
    return merged


def render_gnn_vs_1nn_latex(df: pd.DataFrame) -> str:
    """Aggregate across cases: one row per PRUNE threshold with mean gap."""
    agg = (
        df.groupby("base_mode")
        .agg(
            mean_speedup_1nn=("speedup_1nn", "mean"),
            mean_speedup_gnn=("speedup_gnn", "mean"),
            mean_ratio=("speedup_ratio_gnn_over_1nn", "mean"),
            mean_kept_1nn=("kept_1nn", "mean"),
            mean_kept_gnn=("kept_gnn", "mean"),
        )
        .reset_index()
    )
    agg["base_mode"] = agg["base_mode"].str.replace("WARM+", "", regex=False)

    header = (
        "\\begin{table}[t]\\centering\n"
        "\\caption{Ablation of the GraphSAGE screener against the 1-NN baseline. "
        "Values are means across the six benchmark cases. "
        "``Kept ratio'' is the fraction of post-contingency constraints retained after screening.}\n"
        "\\label{tab:gnn_ablation}\n"
        "\\small\n"
        "\\begin{tabular}{lcccc}\n\\hline\n"
        "\\textbf{Threshold} & "
        "\\textbf{Speedup (1-NN)} & "
        "\\textbf{Speedup (GNN)} & "
        "\\textbf{Ratio} & "
        "\\textbf{Kept ratio (1-NN / GNN)} \\\\\n\\hline\n"
    )
    rows = []
    for _, r in agg.iterrows():
        rows.append(
            f"{r['base_mode']} & "
            f"{r['mean_speedup_1nn']:.2f}x & "
            f"{r['mean_speedup_gnn']:.2f}x & "
            f"{r['mean_ratio']:.3f} & "
            f"{r['mean_kept_1nn']*100:.2f}\\% / {r['mean_kept_gnn']*100:.2f}\\% \\\\"
        )
    body = "\n".join(rows)
    footer = "\n\\hline\n\\end{tabular}\n\\end{table}\n"
    tex = header + body + footer
    (TABLES / "gnn_vs_1nn.tex").write_text(tex, encoding="utf-8")
    return tex


# ---------------------------------------------------------------------------
# 3. Large-case (case1354pegase) evidence (R1-C8)
# ---------------------------------------------------------------------------
def large_case_table() -> pd.DataFrame:
    df = pd.read_csv(DATA / "large_case_summary.csv")
    df = df[df["case_folder"] == "matpower/case1354pegase"].copy()
    df["case_short"] = df["case_folder"].str.replace("matpower/", "", regex=False)
    # baseline = LAZY_ALL (fully exact lazy enforcement over N-1)
    df = df.sort_values("runtime_median_sec")
    keep = [
        "mode_id",
        "n",
        "opt_rate",
        "time_limit_rate",
        "runtime_median_sec",
        "runtime_mean_sec",
        "root_constraints_mean",
        "lazy_added_mean",
        "mip_gap_mean",
    ]
    out = df[keep].copy()
    out.to_csv(TABLES / "large_case_table.csv", index=False)
    return out


def render_large_case_latex(df: pd.DataFrame) -> str:
    header = (
        "\\begin{table}[t]\\centering\n"
        "\\caption{Follow-up large-case evaluation on \\texttt{case1354pegase} "
        "(1354 buses, 1991 lines). "
        "Modes with a compact formulation (\\texttt{SHRINK\\_LAZY}) or an active-set "
        "subset of contingencies (\\texttt{ACTIVESET\\_LAZY}) solve within the "
        "time limit, while the fully explicit \\texttt{WARM\\_LAZY} variant "
        "frequently hits the 1-hour limit.}\n"
        "\\label{tab:large_case}\n"
        "\\small\n"
        "\\begin{tabular}{lrrrrrr}\n\\hline\n"
        "\\textbf{Mode} & "
        "\\textbf{n} & "
        "\\textbf{Opt.\\ rate} & "
        "\\textbf{TL rate} & "
        "\\textbf{Median [s]} & "
        "\\textbf{Root constr.} & "
        "\\textbf{Lazy added} \\\\\n\\hline\n"
    )
    rows = []
    for _, r in df.iterrows():
        median = (
            f"{r['runtime_median_sec']:.0f}" if pd.notna(r["runtime_median_sec"]) else "--"
        )
        root = (
            f"{r['root_constraints_mean']:,.0f}"
            if pd.notna(r["root_constraints_mean"])
            else "--"
        )
        lazy = (
            f"{r['lazy_added_mean']:,.0f}" if pd.notna(r["lazy_added_mean"]) else "--"
        )
        rows.append(
            f"\\texttt{{{r['mode_id']}}} & "
            f"{int(r['n'])} & "
            f"{r['opt_rate']*100:.0f}\\% & "
            f"{r['time_limit_rate']*100:.0f}\\% & "
            f"{median} & "
            f"{root} & "
            f"{lazy} \\\\"
        )
    body = "\n".join(rows)
    footer = "\n\\hline\n\\end{tabular}\n\\end{table}\n"
    tex = header + body + footer
    (TABLES / "large_case.tex").write_text(tex, encoding="utf-8")
    return tex


# ---------------------------------------------------------------------------
# 4. Headline summary (for Response-to-Reviewers + abstract sanity check)
# ---------------------------------------------------------------------------
def headline_summary(spd: pd.DataFrame) -> dict:
    spd_paper = pd.read_csv(DATA / "speedup_summary.csv")
    mean_by_mode = (
        spd_paper[spd_paper["mode"] != "RAW"]
        .groupby("mode")["mean_case_speedup"]
        .first()
        .sort_values(ascending=False)
    )
    per_case_max = (
        spd_paper[spd_paper["mode"] != "RAW"]
        .groupby("case_folder")["speedup_mean"]
        .max()
    )
    d = {
        "best_mode_by_mean_case_speedup": mean_by_mode.index[0],
        "best_mode_value": float(mean_by_mode.iloc[0]),
        "max_case_speedup": float(per_case_max.max()),
        "max_case_name": per_case_max.idxmax(),
        "mean_case_speedup_PRUNE010": float(
            spd_paper.loc[
                spd_paper["mode"] == "WARM+PRUNE-0.10", "mean_case_speedup"
            ].iloc[0]
        ),
        "mean_case_speedup_LAZY_BANDIT": float(
            spd_paper.loc[
                spd_paper["mode"] == "WARM+LAZY+BANDIT", "mean_case_speedup"
            ].iloc[0]
        ),
    }
    (HERE.parent / "data" / "headline.json").write_text(
        json.dumps(d, indent=2), encoding="utf-8"
    )
    return d


# ---------------------------------------------------------------------------
def main() -> None:
    wide = speedup_with_ci()
    render_speedup_ci_latex(wide)
    print(f"wrote {TABLES/'speedup_with_ci.tex'}")

    gnn = gnn_vs_1nn()
    render_gnn_vs_1nn_latex(gnn)
    print(f"wrote {TABLES/'gnn_vs_1nn.tex'}")

    large = large_case_table()
    render_large_case_latex(large)
    print(f"wrote {TABLES/'large_case.tex'}")

    summary = headline_summary(wide)
    print("headline:", summary)


if __name__ == "__main__":
    main()
