from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

from src.data_preparation.data_structure import UnitCommitmentScenario
from src.ml_models.commitment_hints import CommitmentHints
from src.ml_models.gnn_screening import GNNLineScreener


@dataclass
class STReductionProfile:
    scenario: UnitCommitmentScenario
    commit_probs: Dict[Tuple[str, int], float]
    line_scores: Dict[str, float]
    monitored_lines: Set[str]
    keep_line_pairs: Optional[Set[Tuple[str, str]]]
    keep_gen_pairs: Optional[Set[Tuple[str, str]]]
    fixed_commit_vars: int = 0
    fixed_commit_on: int = 0
    fixed_commit_off: int = 0
    kept_line_pairs: int = 0
    kept_gen_pairs: int = 0
    used_commit_model: bool = False
    used_gnn_model: bool = False
    setup_sec: float = 0.0


class STReductionProvider:
    """
    Spatio-temporal reduction inspired by GNN/LSTM reduced-SCUC papers:
    - variable reduction: fix only highly confident commitment decisions
    - constraint reduction: keep only GNN-predicted critical contingency pairs
    """

    def __init__(self, case_folder: str):
        self.case = case_folder.strip().strip("/\\").replace("\\", "/")
        self.commit = CommitmentHints(case_folder=self.case)
        self.gnn = GNNLineScreener(case_folder=self.case)

    def ensure_trained(self) -> Dict[str, bool]:
        return {
            "commit": bool(self.commit.ensure_trained()),
            "gnn": bool(self.gnn.ensure_trained()),
        }

    def _predict_commit_probs(
        self, scenario: UnitCommitmentScenario, instance_name: str
    ) -> Dict[Tuple[str, int], float]:
        if not self.commit.ensure_trained():
            return {}
        X, keys = self.commit._features_for_instance(scenario, instance_name)
        if X.shape[0] == 0:
            return {}
        if self.commit.constant_target is not None:
            prob = [float(self.commit.constant_target)] * X.shape[0]
        elif self.commit.model is not None:
            raw_prob = self.commit.model.predict_proba(X)
            prob_arr = getattr(raw_prob, "tolist", lambda: raw_prob)()
            if len(prob_arr) != len(keys):
                raise ValueError(
                    f"Commitment probability length mismatch for {instance_name}: "
                    f"{len(prob_arr)} predictions vs {len(keys)} keys."
                )
            prob = []
            for row in prob_arr:
                if isinstance(row, (list, tuple)):
                    if len(row) < 2:
                        raise ValueError(
                            f"Unexpected probability row shape for {instance_name}: {row!r}"
                        )
                    prob.append(float(row[1]))
                else:
                    prob.append(float(row))
        else:
            return {}
        return {(gname, int(t)): float(p) for (gname, t), p in zip(keys, prob)}

    def _predict_line_scores(
        self, scenario: UnitCommitmentScenario
    ) -> Dict[str, float]:
        if not self.gnn.ensure_trained():
            return {}
        return {
            str(name): float(score)
            for name, score in self.gnn._predict_scores_for_instance(scenario).items()
        }

    def build_profile(
        self,
        scenario: UnitCommitmentScenario,
        instance_name: str,
        *,
        commit_fix_thr: float = 0.98,
        line_keep_thr: float = 0.60,
    ) -> STReductionProfile:
        t0 = time.time()
        sc_red = copy.deepcopy(scenario)

        commit_probs = self._predict_commit_probs(sc_red, instance_name)
        line_scores = self._predict_line_scores(sc_red)
        critical_lines = {
            ln.name
            for ln in sc_red.lines or []
            if float(line_scores.get(ln.name, 0.0)) >= float(line_keep_thr)
        }

        fixed_on = 0
        fixed_off = 0
        if commit_probs:
            for gen in sc_red.thermal_units:
                if not gen.commitment_status:
                    gen.commitment_status = [None] * int(sc_red.time)
                for t in range(int(sc_red.time)):
                    existing = gen.commitment_status[t]
                    if existing is not None:
                        continue
                    p = commit_probs.get((gen.name, t))
                    if p is None:
                        continue
                    if float(p) >= float(commit_fix_thr):
                        gen.commitment_status[t] = True
                        fixed_on += 1
                    elif float(p) <= 1.0 - float(commit_fix_thr):
                        gen.commitment_status[t] = False
                        fixed_off += 1

        keep_line_pairs: Optional[Set[Tuple[str, str]]] = None
        keep_gen_pairs: Optional[Set[Tuple[str, str]]] = None
        if line_scores:
            keep_line_pairs = set()
            keep_gen_pairs = set()
            for cont in sc_red.contingencies or []:
                if getattr(cont, "lines", None):
                    for out_line in cont.lines:
                        out_critical = out_line.name in critical_lines
                        for line_l in sc_red.lines or []:
                            if line_l.name in critical_lines or out_critical:
                                keep_line_pairs.add((line_l.name, out_line.name))
                if getattr(cont, "units", None):
                    for gen in cont.units:
                        for line_l in sc_red.lines or []:
                            if line_l.name in critical_lines:
                                keep_gen_pairs.add((line_l.name, gen.name))

        return STReductionProfile(
            scenario=sc_red,
            commit_probs=commit_probs,
            line_scores=line_scores,
            monitored_lines=critical_lines,
            keep_line_pairs=keep_line_pairs,
            keep_gen_pairs=keep_gen_pairs,
            fixed_commit_vars=int(fixed_on + fixed_off),
            fixed_commit_on=int(fixed_on),
            fixed_commit_off=int(fixed_off),
            kept_line_pairs=0 if keep_line_pairs is None else int(len(keep_line_pairs)),
            kept_gen_pairs=0 if keep_gen_pairs is None else int(len(keep_gen_pairs)),
            used_commit_model=bool(commit_probs),
            used_gnn_model=bool(line_scores),
            setup_sec=float(time.time() - t0),
        )
