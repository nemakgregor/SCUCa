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
