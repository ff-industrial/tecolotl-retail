# storage.py
import csv
from pathlib import Path

def save_event(path, event):
    file_exists = Path(path).exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp",
            "zone",
            "facing_shelf",
            "seconds",
            "pose_score",
            "shoulder_center_x"
        ])

        if not file_exists:
            writer.writeheader()

        writer.writerow(event)