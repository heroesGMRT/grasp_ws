import csv
import tempfile
import unittest
from pathlib import Path

from regrasp_experiment.grip_outcome import GraspAttemptLogger, proximity_outcome


class GripOutcomeTest(unittest.TestCase):
    def test_proximity_true_is_success(self):
        self.assertEqual(proximity_outcome(True), ("success", ""))

    def test_proximity_false_is_failure(self):
        self.assertEqual(proximity_outcome(False), ("fail", "proximity_false"))

    def test_missing_proximity_is_failure(self):
        self.assertEqual(proximity_outcome(None), ("fail", "proximity_timeout"))

    def test_logger_appends_one_attempt_row(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "attempts.csv"
            logger = GraspAttemptLogger(path)
            logger.log(
                attempt=3,
                outcome="success",
                cause="",
                sensor_value=True,
                attempt_duration_s=1.2345,
            )

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attempt"], "3")
        self.assertEqual(rows[0]["outcome"], "success")
        self.assertEqual(rows[0]["sensor_value"], "true")
        self.assertEqual(rows[0]["attempt_duration_s"], "1.234")
