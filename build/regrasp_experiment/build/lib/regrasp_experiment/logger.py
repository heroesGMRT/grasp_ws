"""Logger CSV per-attempt — sesuai Bagian 8 SOP (satu baris = satu attempt)."""
import csv
import os
from datetime import datetime

COLUMNS = [
    "trial_id", "timestamp", "run", "method", "condition", "object_id",
    "attempt", "region", "moved", "outcome", "cause", "latency_ms",
    "scene_success", "notes",
]


class TrialLogger:
    def __init__(self, path="experiment_log.csv"):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(COLUMNS)

    def log_trial(self, trial_id, run, method, condition, object_id, rows, notes=""):
        scene_success = any(r["outcome"] == "success" for r in rows)
        ts = datetime.now().isoformat(timespec="seconds")
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow([
                    trial_id, ts, run, method, condition, object_id,
                    r["attempt"], r["region"], r["moved"], r["outcome"],
                    r["cause"], r["latency_ms"], scene_success, notes,
                ])
        return scene_success
