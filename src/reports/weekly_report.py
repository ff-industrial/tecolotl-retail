# weekly_report.py
import pandas as pd
from pathlib import Path

def create_weekly_report(reports_dir="reports"):
    files = list(Path(reports_dir).glob("heatmap_events_*.csv"))

    df = pd.concat([pd.read_csv(file) for file in files])

    df = df[df["facing_shelf"] == True]

    summary = (
        df.groupby("zone")["seconds"]
        .sum()
        .reset_index()
        .rename(columns={"seconds": "attention_seconds"})
    )

    total = summary["attention_seconds"].sum()
    summary["attention_percent"] = summary["attention_seconds"] / total * 100

    summary = summary.sort_values("attention_seconds", ascending=False)

    summary.to_csv(Path(reports_dir) / "weekly_summary.csv", index=False)

    print(summary)
    print(f"Anaquel con más atención: {summary.iloc[0]['zone']}")