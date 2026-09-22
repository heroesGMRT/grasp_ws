"""Sensor-based grasp outcome and per-attempt CSV logging."""

import csv
from datetime import datetime
from pathlib import Path


ATTEMPT_COLUMNS = (
    "timestamp",
    "attempt",
    "outcome",
    "cause",
    "sensor_value",
    "attempt_duration_s",
)


def proximity_outcome(sensor_value: bool | None) -> tuple[str, str]:
    """Map the post-close proximity result to the recorded grasp outcome."""
    if sensor_value is True:
        return "success", ""
    if sensor_value is False:
        return "fail", "proximity_false"
    return "fail", "proximity_timeout"


class GraspAttemptLogger:
    """Append one sensor-derived outcome row for every gripper-close attempt."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=ATTEMPT_COLUMNS).writeheader()

    def log(
        self,
        attempt: int,
        outcome: str,
        cause: str,
        sensor_value: bool | None,
        attempt_duration_s: float,
    ):
        """Append the outcome and elapsed time since the close command."""
        value = "true" if sensor_value is True else "false" if sensor_value is False else ""
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=ATTEMPT_COLUMNS).writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "attempt": attempt,
                    "outcome": outcome,
                    "cause": cause,
                    "sensor_value": value,
                    "attempt_duration_s": f"{max(0.0, attempt_duration_s):.3f}",
                }
            )
