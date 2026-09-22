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
    "trial_duration_s",
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
        self.columns = ATTEMPT_COLUMNS
        if not self.path.exists() or self.path.stat().st_size == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.columns).writeheader()
        else:
            # Keep appending in the file's own column layout, so logs written
            # before trial_duration_s existed stay valid instead of misaligning.
            with self.path.open(newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle), None)
            if header:
                self.columns = tuple(header)

    def log(
        self,
        attempt: int,
        outcome: str,
        cause: str,
        sensor_value: bool | None,
        attempt_duration_s: float,
        trial_duration_s: float | None = None,
    ):
        """Append the outcome, the time since the close command, and the time
        since the trial started (None when unknown)."""
        value = "true" if sensor_value is True else "false" if sensor_value is False else ""
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(
                handle, fieldnames=self.columns, extrasaction="ignore"
            ).writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "attempt": attempt,
                    "outcome": outcome,
                    "cause": cause,
                    "sensor_value": value,
                    "attempt_duration_s": f"{max(0.0, attempt_duration_s):.3f}",
                    "trial_duration_s": (
                        "" if trial_duration_s is None
                        else f"{max(0.0, trial_duration_s):.3f}"
                    ),
                }
            )