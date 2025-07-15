import os
import sys
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric, ColumnSummaryMetric

def main():
    prev_csv = "evidently/last_week.csv"
    curr_csv = "evidently/current_week.csv"
    if not (os.path.exists(prev_csv) and os.path.exists(curr_csv)):
        print("Required files not found. Please run run_weekly_drift.py first.")
        sys.exit(1)

    prev = pd.read_csv(prev_csv)
    curr = pd.read_csv(curr_csv)

    # Rename "rating" to "target" for compatibility with TargetDriftPreset
    if "rating" in prev.columns:
        prev = prev.rename(columns={"rating": "target"})
        curr = curr.rename(columns={"rating": "target"})

    metrics = [
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetDriftMetric(),
        TargetDriftPreset() if "target" in prev.columns else None,
        # ColumnDriftMetric(column_name="target") if "target" in prev.columns else None,
        # ColumnDriftMetric(column_name="movieId") if "movieId" in prev.columns else None,
        # ColumnDriftMetric(column_name="userId") if "userId" in prev.columns else None,
        # ColumnSummaryMetric(column_name="target") if "target" in prev.columns else None,
        # ColumnSummaryMetric(column_name="movieId") if "movieId" in prev.columns else None,
        # ColumnSummaryMetric(column_name="userId") if "userId" in prev.columns else None,
    ]
    metrics = [m for m in metrics if m is not None]
    report = Report(metrics=metrics)
    report.run(reference_data=prev, current_data=curr)
    os.makedirs("evidently/reports", exist_ok=True)
    report_path = "evidently/reports/weekly_evidently_report.html"
    report.save_html(report_path)
    print(f"Comprehensive report saved to {report_path}")

if __name__ == "__main__":
    main()