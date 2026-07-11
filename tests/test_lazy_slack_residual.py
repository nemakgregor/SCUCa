from __future__ import annotations

import unittest

from src.optimization_model.helpers.lazy_contingency_cb import (
    _signed_slack_residuals,
)


class SignedSlackResidualTests(unittest.TestCase):
    def test_positive_row_uses_positive_shared_slack(self) -> None:
        positive, negative = _signed_slack_residuals(
            post_flow=125.0,
            emergency_limit=100.0,
            overflow_pos=20.0,
            overflow_neg=0.0,
        )
        self.assertEqual(positive, 5.0)
        self.assertEqual(negative, -225.0)

    def test_negative_row_uses_negative_shared_slack(self) -> None:
        positive, negative = _signed_slack_residuals(
            post_flow=-130.0,
            emergency_limit=100.0,
            overflow_pos=0.0,
            overflow_neg=25.0,
        )
        self.assertEqual(positive, -230.0)
        self.assertEqual(negative, 5.0)

    def test_existing_slack_covers_the_incumbent(self) -> None:
        positive, negative = _signed_slack_residuals(
            post_flow=125.0,
            emergency_limit=100.0,
            overflow_pos=25.0,
            overflow_neg=0.0,
        )
        self.assertEqual(positive, 0.0)
        self.assertLess(negative, 0.0)


if __name__ == "__main__":
    unittest.main()
