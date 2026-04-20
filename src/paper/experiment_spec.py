from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

EXPERIMENT_SEED = 42
DEFAULT_MIP_GAP = 0.05
DEFAULT_LAZY_LODF_TOL = 1e-4
DEFAULT_LAZY_ISF_TOL = 1e-8
DEFAULT_LAZY_VIOL_TOL = 1e-6
DEFAULT_BANDIT_K_LIST = (64, 128, 256)
DEFAULT_BANDIT_EPSILON = 0.10
DEFAULT_NO_REL_HEUR_TIME_RATIO = 0.30

CASES_SMALL = [
    "matpower/case14",
    "matpower/case30",
    "matpower/case57",
    "matpower/case89pegase",
]

CASES_MEDLARGE = [
    "matpower/case118",
    "matpower/case300",
    "matpower/case1354pegase",
    "matpower/case1951rte",
    "matpower/case2383wp",
    "matpower/case2736sp",
    "matpower/case3375wp",
]

CASES_ALL = CASES_SMALL + CASES_MEDLARGE

TRAIN_DATES_24 = [
    "2017-01-05",
    "2017-01-25",
    "2017-02-05",
    "2017-02-25",
    "2017-03-05",
    "2017-03-25",
    "2017-04-05",
    "2017-04-25",
    "2017-05-05",
    "2017-05-25",
    "2017-06-05",
    "2017-06-25",
    "2017-07-05",
    "2017-07-25",
    "2017-08-05",
    "2017-08-25",
    "2017-09-05",
    "2017-09-25",
    "2017-10-05",
    "2017-10-25",
    "2017-11-05",
    "2017-11-25",
    "2017-12-05",
    "2017-12-25",
]

TEST_DATES_6 = [
    "2017-01-15",
    "2017-03-15",
    "2017-05-15",
    "2017-07-15",
    "2017-09-15",
    "2017-11-15",
]

PILOT_N = 2
TIME_LIMIT_MULTIPLIER = 2
DROP_SUCCESS_RATE_MIN = 0.50
DROP_MEDIAN_RUNTIME_FRACTION = 0.90
TRAIN_BASE_MODE_ID = "LAZY_ALL"

TRAIN_TL_SEC_BY_CASE = {
    "matpower/case14": 600,
    "matpower/case30": 600,
    "matpower/case57": 600,
    "matpower/case89pegase": 900,
    "matpower/case118": 1800,
    "matpower/case300": 2400,
    "matpower/case1354pegase": 3600,
    "matpower/case1951rte": 3600,
    "matpower/case2383wp": 3600,
    "matpower/case2736sp": 3600,
    "matpower/case3375wp": 3600,
}

TEST_TL_INIT_SEC_BY_CASE = {
    "matpower/case14": 600,
    "matpower/case30": 600,
    "matpower/case57": 600,
    "matpower/case89pegase": 900,
    "matpower/case118": 1800,
    "matpower/case300": 2400,
    "matpower/case1354pegase": 3600,
    "matpower/case1951rte": 3600,
    "matpower/case2383wp": 3600,
    "matpower/case2736sp": 3600,
    "matpower/case3375wp": 3600,
}

TEST_TL_CAP_SEC_BY_CASE = {
    "matpower/case14": 1800,
    "matpower/case30": 1800,
    "matpower/case57": 1800,
    "matpower/case89pegase": 2400,
    "matpower/case118": 7200,
    "matpower/case300": 7200,
    "matpower/case1354pegase": 14400,
    "matpower/case1951rte": 14400,
    "matpower/case2383wp": 14400,
    "matpower/case2736sp": 14400,
    "matpower/case3375wp": 14400,
}


@dataclass(frozen=True)
class ModeSpec:
    mode_id: str
    mode_family: str
    time_limit_sec: int
    mip_gap: float = DEFAULT_MIP_GAP
    no_rel_heur_time: Optional[float] = None
    tau: Optional[float] = None
    use_warm_start: bool = False
    use_branch_hints: bool = False
    use_commit_hints: bool = False
    commit_mode: str = "hint"
    commit_thr: float = 0.98
    use_gnn_screening: bool = False
    gnn_thr: float = 0.60
    use_gru_warmstart: bool = False
    use_lazy_callback: bool = False
    lazy_top_k: int = 0
    lazy_bandit: bool = False
    bandit_k_list: Tuple[int, ...] = DEFAULT_BANDIT_K_LIST
    bandit_epsilon: float = DEFAULT_BANDIT_EPSILON
    lazy_lodf_tol: float = DEFAULT_LAZY_LODF_TOL
    lazy_isf_tol: float = DEFAULT_LAZY_ISF_TOL
    lazy_viol_tol: float = DEFAULT_LAZY_VIOL_TOL
    sr_sigma: float = 0.3
    sr_l2_thr: float = 400.0
    sr_sigma_thr: float = 5.0
    active_set_batch: int = 2000
    active_set_max_rounds: int = 12
    active_set_cleanup: bool = True
    active_set_cleanup_tol: float = 1e-5
    shrink_window: int = 8
    shrink_overlap: int = 2
    st_commit_fix_thr: float = 0.98
    st_line_keep_thr: float = 0.60
    exact_method: bool = True


TRAIN_BASE_MODE = ModeSpec(
    mode_id="LAZY_ALL",
    mode_family="LAZY",
    time_limit_sec=600,
    use_lazy_callback=True,
)

MODE_CATALOG_SMALL = [
    ModeSpec(mode_id="RAW", mode_family="RAW", time_limit_sec=600),
    ModeSpec(
        mode_id="RAW_GNN",
        mode_family="RAW",
        time_limit_sec=600,
        use_gnn_screening=True,
        gnn_thr=0.60,
    ),
    ModeSpec(
        mode_id="RAW_COMMIT_HINTS",
        mode_family="RAW",
        time_limit_sec=600,
        use_commit_hints=True,
        commit_mode="hint",
        commit_thr=0.98,
    ),
    ModeSpec(
        mode_id="WARM",
        mode_family="WARM",
        time_limit_sec=600,
        use_warm_start=True,
    ),
    ModeSpec(
        mode_id="WARM_BRANCH_HINTS",
        mode_family="WARM",
        time_limit_sec=600,
        use_warm_start=True,
        use_branch_hints=True,
    ),
    ModeSpec(
        mode_id="WARM_COMMIT_HINTS",
        mode_family="WARM",
        time_limit_sec=600,
        use_warm_start=True,
        use_commit_hints=True,
        commit_mode="hint",
        commit_thr=0.98,
    ),
    ModeSpec(
        mode_id="WARM_GRU",
        mode_family="WARM",
        time_limit_sec=600,
        use_warm_start=True,
        use_gru_warmstart=True,
    ),
    ModeSpec(
        mode_id="LAZY_ALL",
        mode_family="LAZY",
        time_limit_sec=600,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="LAZY_TOPK128",
        mode_family="LAZY",
        time_limit_sec=600,
        use_lazy_callback=True,
        lazy_top_k=128,
    ),
    ModeSpec(
        mode_id="LAZY_BANDIT",
        mode_family="LAZY",
        time_limit_sec=600,
        use_lazy_callback=True,
        lazy_bandit=True,
        bandit_k_list=(64, 128, 256),
        bandit_epsilon=0.10,
    ),
    ModeSpec(
        mode_id="WARM_LAZY",
        mode_family="WARM_LAZY",
        time_limit_sec=600,
        use_warm_start=True,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_LAZY_TOPK128",
        mode_family="WARM_LAZY",
        time_limit_sec=600,
        use_warm_start=True,
        use_lazy_callback=True,
        lazy_top_k=128,
    ),
    ModeSpec(
        mode_id="WARM_LAZY_BANDIT",
        mode_family="WARM_LAZY",
        time_limit_sec=600,
        use_warm_start=True,
        use_lazy_callback=True,
        lazy_bandit=True,
        bandit_k_list=(64, 128, 256),
        bandit_epsilon=0.10,
    ),
    ModeSpec(
        mode_id="WARM_LAZY_COMMIT_HINTS",
        mode_family="WARM_LAZY",
        time_limit_sec=600,
        use_warm_start=True,
        use_lazy_callback=True,
        use_commit_hints=True,
        commit_mode="hint",
        commit_thr=0.98,
    ),
    ModeSpec(
        mode_id="WARM_LAZY_GRU",
        mode_family="WARM_LAZY",
        time_limit_sec=600,
        use_warm_start=True,
        use_lazy_callback=True,
        use_gru_warmstart=True,
    ),
    ModeSpec(
        mode_id="WARM_PRUNE_T010",
        mode_family="WARM_PRUNE",
        time_limit_sec=600,
        tau=0.10,
        use_warm_start=True,
    ),
    ModeSpec(
        mode_id="WARM_PRUNE_T020",
        mode_family="WARM_PRUNE",
        time_limit_sec=600,
        tau=0.20,
        use_warm_start=True,
    ),
    ModeSpec(
        mode_id="WARM_PRUNE_T030",
        mode_family="WARM_PRUNE",
        time_limit_sec=600,
        tau=0.30,
        use_warm_start=True,
    ),
    ModeSpec(
        mode_id="WARM_PRUNE_T050",
        mode_family="WARM_PRUNE",
        time_limit_sec=600,
        tau=0.50,
        use_warm_start=True,
    ),
    ModeSpec(
        mode_id="WARM_PRUNE_T080",
        mode_family="WARM_PRUNE",
        time_limit_sec=600,
        tau=0.80,
        use_warm_start=True,
    ),
    ModeSpec(
        mode_id="WARM_LPSCREEN_T010",
        mode_family="WARM_LPSCREEN",
        time_limit_sec=600,
        tau=0.10,
        use_warm_start=True,
    ),
    ModeSpec(
        mode_id="WARM_LPSCREEN_T020",
        mode_family="WARM_LPSCREEN",
        time_limit_sec=600,
        tau=0.20,
        use_warm_start=True,
    ),
    ModeSpec(
        mode_id="WARM_LPSCREEN_T030",
        mode_family="WARM_LPSCREEN",
        time_limit_sec=600,
        tau=0.30,
        use_warm_start=True,
    ),
    ModeSpec(
        mode_id="WARM_LPSCREEN_T050",
        mode_family="WARM_LPSCREEN",
        time_limit_sec=600,
        tau=0.50,
        use_warm_start=True,
    ),
    ModeSpec(
        mode_id="WARM_LPSCREEN_T080",
        mode_family="WARM_LPSCREEN",
        time_limit_sec=600,
        tau=0.80,
        use_warm_start=True,
    ),
    ModeSpec(
        mode_id="WARM_PRUNE_LAZY_T010",
        mode_family="WARM_PRUNE_LAZY",
        time_limit_sec=600,
        tau=0.10,
        use_warm_start=True,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_PRUNE_LAZY_T020",
        mode_family="WARM_PRUNE_LAZY",
        time_limit_sec=600,
        tau=0.20,
        use_warm_start=True,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_PRUNE_LAZY_T030",
        mode_family="WARM_PRUNE_LAZY",
        time_limit_sec=600,
        tau=0.30,
        use_warm_start=True,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_PRUNE_LAZY_T050",
        mode_family="WARM_PRUNE_LAZY",
        time_limit_sec=600,
        tau=0.50,
        use_warm_start=True,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_PRUNE_LAZY_T080",
        mode_family="WARM_PRUNE_LAZY",
        time_limit_sec=600,
        tau=0.80,
        use_warm_start=True,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_LPSCREEN_LAZY_T010",
        mode_family="WARM_LPSCREEN_LAZY",
        time_limit_sec=600,
        tau=0.10,
        use_warm_start=True,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_LPSCREEN_LAZY_T020",
        mode_family="WARM_LPSCREEN_LAZY",
        time_limit_sec=600,
        tau=0.20,
        use_warm_start=True,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_LPSCREEN_LAZY_T030",
        mode_family="WARM_LPSCREEN_LAZY",
        time_limit_sec=600,
        tau=0.30,
        use_warm_start=True,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_LPSCREEN_LAZY_T050",
        mode_family="WARM_LPSCREEN_LAZY",
        time_limit_sec=600,
        tau=0.50,
        use_warm_start=True,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_LPSCREEN_LAZY_T080",
        mode_family="WARM_LPSCREEN_LAZY",
        time_limit_sec=600,
        tau=0.80,
        use_warm_start=True,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_SR_LAZY",
        mode_family="WARM_SR_LAZY",
        time_limit_sec=600,
        use_warm_start=True,
        use_lazy_callback=True,
        sr_sigma=0.30,
        sr_l2_thr=400.0,
        sr_sigma_thr=5.0,
    ),
    ModeSpec(
        mode_id="ACTIVESET",
        mode_family="ACTIVESET",
        time_limit_sec=600,
        exact_method=True,
    ),
    ModeSpec(
        mode_id="ACTIVESET_LAZY",
        mode_family="ACTIVESET_LAZY",
        time_limit_sec=600,
        use_lazy_callback=True,
        exact_method=True,
    ),
    ModeSpec(
        mode_id="SHRINK_LAZY",
        mode_family="SHRINK_LAZY",
        time_limit_sec=600,
        use_lazy_callback=True,
        shrink_window=8,
        shrink_overlap=2,
        exact_method=False,
    ),
    ModeSpec(
        mode_id="STREDUCE",
        mode_family="STREDUCE",
        time_limit_sec=600,
        use_warm_start=True,
        st_commit_fix_thr=0.98,
        st_line_keep_thr=0.60,
    ),
    ModeSpec(
        mode_id="STREDUCE_GRU",
        mode_family="STREDUCE",
        time_limit_sec=600,
        use_warm_start=True,
        use_gru_warmstart=True,
        st_commit_fix_thr=0.98,
        st_line_keep_thr=0.60,
    ),
    ModeSpec(
        mode_id="STREDUCE_LAZY",
        mode_family="STREDUCE_LAZY",
        time_limit_sec=600,
        use_warm_start=True,
        use_lazy_callback=True,
        st_commit_fix_thr=0.98,
        st_line_keep_thr=0.60,
    ),
    ModeSpec(
        mode_id="STREDUCE_LAZY_GRU",
        mode_family="STREDUCE_LAZY",
        time_limit_sec=600,
        use_warm_start=True,
        use_lazy_callback=True,
        use_gru_warmstart=True,
        st_commit_fix_thr=0.98,
        st_line_keep_thr=0.60,
    ),
]

MODE_CATALOG_MEDLARGE_START = [
    ModeSpec(
        mode_id="LAZY_ALL",
        mode_family="LAZY",
        time_limit_sec=1800,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_LAZY",
        mode_family="WARM_LAZY",
        time_limit_sec=1800,
        use_warm_start=True,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_LAZY_BANDIT",
        mode_family="WARM_LAZY",
        time_limit_sec=1800,
        use_warm_start=True,
        use_lazy_callback=True,
        lazy_bandit=True,
        bandit_k_list=(64, 128, 256),
        bandit_epsilon=0.10,
    ),
    ModeSpec(
        mode_id="WARM_PRUNE_LAZY_T030",
        mode_family="WARM_PRUNE_LAZY",
        time_limit_sec=1800,
        tau=0.30,
        use_warm_start=True,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_LPSCREEN_LAZY_T020",
        mode_family="WARM_LPSCREEN_LAZY",
        time_limit_sec=1800,
        tau=0.20,
        use_warm_start=True,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_SR_LAZY",
        mode_family="WARM_SR_LAZY",
        time_limit_sec=1800,
        use_warm_start=True,
        use_lazy_callback=True,
        sr_sigma=0.30,
        sr_l2_thr=400.0,
        sr_sigma_thr=5.0,
    ),
    ModeSpec(
        mode_id="ACTIVESET_LAZY",
        mode_family="ACTIVESET_LAZY",
        time_limit_sec=1800,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="STREDUCE_LAZY",
        mode_family="STREDUCE_LAZY",
        time_limit_sec=1800,
        use_warm_start=True,
        use_lazy_callback=True,
        st_commit_fix_thr=0.98,
        st_line_keep_thr=0.60,
    ),
    ModeSpec(
        mode_id="SHRINK_LAZY",
        mode_family="SHRINK_LAZY",
        time_limit_sec=1800,
        use_lazy_callback=True,
        shrink_window=8,
        shrink_overlap=2,
        exact_method=False,
    ),
]

MODE_CATALOG_MEDLARGE_FULL = [
    ModeSpec(mode_id="RAW", mode_family="RAW", time_limit_sec=1800),
    ModeSpec(
        mode_id="RAW_GNN",
        mode_family="RAW",
        time_limit_sec=1800,
        use_gnn_screening=True,
        gnn_thr=0.60,
    ),
    ModeSpec(
        mode_id="RAW_COMMIT_HINTS",
        mode_family="RAW",
        time_limit_sec=1800,
        use_commit_hints=True,
        commit_mode="hint",
        commit_thr=0.98,
    ),
    ModeSpec(
        mode_id="WARM",
        mode_family="WARM",
        time_limit_sec=1800,
        use_warm_start=True,
    ),
    ModeSpec(
        mode_id="WARM_BRANCH_HINTS",
        mode_family="WARM",
        time_limit_sec=1800,
        use_warm_start=True,
        use_branch_hints=True,
    ),
    ModeSpec(
        mode_id="WARM_COMMIT_HINTS",
        mode_family="WARM",
        time_limit_sec=1800,
        use_warm_start=True,
        use_commit_hints=True,
        commit_mode="hint",
        commit_thr=0.98,
    ),
    ModeSpec(
        mode_id="WARM_GRU",
        mode_family="WARM",
        time_limit_sec=1800,
        use_warm_start=True,
        use_gru_warmstart=True,
    ),
    ModeSpec(
        mode_id="LAZY_ALL",
        mode_family="LAZY",
        time_limit_sec=1800,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="LAZY_TOPK128",
        mode_family="LAZY",
        time_limit_sec=1800,
        use_lazy_callback=True,
        lazy_top_k=128,
    ),
    ModeSpec(
        mode_id="LAZY_BANDIT",
        mode_family="LAZY",
        time_limit_sec=1800,
        use_lazy_callback=True,
        lazy_bandit=True,
        bandit_k_list=(64, 128, 256),
        bandit_epsilon=0.10,
    ),
    ModeSpec(
        mode_id="WARM_LAZY",
        mode_family="WARM_LAZY",
        time_limit_sec=1800,
        use_warm_start=True,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_LAZY_TOPK128",
        mode_family="WARM_LAZY",
        time_limit_sec=1800,
        use_warm_start=True,
        use_lazy_callback=True,
        lazy_top_k=128,
    ),
    ModeSpec(
        mode_id="WARM_LAZY_BANDIT",
        mode_family="WARM_LAZY",
        time_limit_sec=1800,
        use_warm_start=True,
        use_lazy_callback=True,
        lazy_bandit=True,
        bandit_k_list=(64, 128, 256),
        bandit_epsilon=0.10,
    ),
    ModeSpec(
        mode_id="WARM_LAZY_COMMIT_HINTS",
        mode_family="WARM_LAZY",
        time_limit_sec=1800,
        use_warm_start=True,
        use_lazy_callback=True,
        use_commit_hints=True,
        commit_mode="hint",
        commit_thr=0.98,
    ),
    ModeSpec(
        mode_id="WARM_LAZY_GRU",
        mode_family="WARM_LAZY",
        time_limit_sec=1800,
        use_warm_start=True,
        use_lazy_callback=True,
        use_gru_warmstart=True,
    ),
    ModeSpec(
        mode_id="WARM_PRUNE_T030",
        mode_family="WARM_PRUNE",
        time_limit_sec=1800,
        tau=0.30,
        use_warm_start=True,
    ),
    ModeSpec(
        mode_id="WARM_LPSCREEN_T020",
        mode_family="WARM_LPSCREEN",
        time_limit_sec=1800,
        tau=0.20,
        use_warm_start=True,
    ),
    ModeSpec(
        mode_id="WARM_PRUNE_LAZY_T030",
        mode_family="WARM_PRUNE_LAZY",
        time_limit_sec=1800,
        tau=0.30,
        use_warm_start=True,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_LPSCREEN_LAZY_T020",
        mode_family="WARM_LPSCREEN_LAZY",
        time_limit_sec=1800,
        tau=0.20,
        use_warm_start=True,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="WARM_SR_LAZY",
        mode_family="WARM_SR_LAZY",
        time_limit_sec=1800,
        use_warm_start=True,
        use_lazy_callback=True,
        sr_sigma=0.30,
        sr_l2_thr=400.0,
        sr_sigma_thr=5.0,
    ),
    ModeSpec(
        mode_id="ACTIVESET_LAZY",
        mode_family="ACTIVESET_LAZY",
        time_limit_sec=1800,
        use_lazy_callback=True,
    ),
    ModeSpec(
        mode_id="STREDUCE_LAZY",
        mode_family="STREDUCE_LAZY",
        time_limit_sec=1800,
        use_warm_start=True,
        use_lazy_callback=True,
        st_commit_fix_thr=0.98,
        st_line_keep_thr=0.60,
    ),
    ModeSpec(
        mode_id="SHRINK_LAZY",
        mode_family="SHRINK_LAZY",
        time_limit_sec=1800,
        use_lazy_callback=True,
        shrink_window=8,
        shrink_overlap=2,
        exact_method=False,
    ),
]
