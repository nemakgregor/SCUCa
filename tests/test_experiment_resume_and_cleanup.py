from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.paper import experiments


class _FakeModel:
    def __init__(self) -> None:
        self.dispose_calls = 0

    def dispose(self) -> None:
        self.dispose_calls += 1


class ResumeAndCleanupTests(unittest.TestCase):
    def _write_attempts(self, attempts: list[tuple[str, str]]) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".csv", delete=False
        )
        with handle:
            writer = csv.DictWriter(handle, fieldnames=experiments.CSV_FIELDS)
            writer.writeheader()
            for result_key, status in attempts:
                writer.writerow(
                    {
                        "result_key": result_key,
                        "stage": "TRAIN",
                        "mode_id": "LAZY_ALL",
                        "instance_name": "matpower/case300/2017-02-05",
                        "status": status,
                    }
                )
        return Path(handle.name)

    def test_latest_error_reopens_exact_and_logical_resume_keys(self) -> None:
        key = "TRAIN::LAZY_ALL::matpower/case300/2017-02-05::1200"
        path = self._write_attempts([(key, "OPTIMAL"), (key, "ERROR")])
        self.addCleanup(path.unlink, missing_ok=True)
        self.assertNotIn(key, experiments._load_completed_keys_from_csv(path))
        logical = experiments._logical_result_key(
            "TRAIN", "LAZY_ALL", "matpower/case300/2017-02-05"
        )
        self.assertNotIn(
            logical, experiments._load_completed_result_lookup_by_logical_key(path)
        )

    def test_latest_success_closes_the_resume_key(self) -> None:
        key = "TRAIN::LAZY_ALL::matpower/case300/2017-02-05::1200"
        path = self._write_attempts([(key, "ERROR"), (key, "OPTIMAL")])
        self.addCleanup(path.unlink, missing_ok=True)
        self.assertIn(key, experiments._load_completed_keys_from_csv(path))

    def test_success_payload_model_is_disposed_once(self) -> None:
        model = _FakeModel()
        payload = SimpleNamespace(model=model)
        mode = SimpleNamespace(
            mode_id="LAZY_ALL", mode_family="LAZY", time_limit_sec=1
        )
        artifacts = SimpleNamespace(case_folder="matpower/case14")
        with (
            patch.object(experiments, "_mark_solve_started"),
            patch.object(experiments, "_solve_payload", return_value=payload),
            patch.object(experiments, "_save_candidate_solution", return_value=None),
            patch.object(
                experiments,
                "_build_success_row",
                return_value={"status": "OPTIMAL"},
            ),
            patch.object(
                experiments, "_is_train_artifact_eligible", return_value=False
            ),
        ):
            row = experiments._run_single_solve(
                run_id="cleanup-test",
                stage="TEST",
                instance_name="matpower/case14/2017-01-15",
                mode=mode,
                case_artifacts=artifacts,
                paths=SimpleNamespace(),
            )
        self.assertEqual(row["status"], "OPTIMAL")
        self.assertEqual(model.dispose_calls, 1)


if __name__ == "__main__":
    unittest.main()
