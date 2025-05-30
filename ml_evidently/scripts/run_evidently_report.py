import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric

# Load your datasets
reference = pd.read_csv("data/raw/movies.csv")
current = pd.read_csv("data/raw/movies_drift.csv")

# Create a comprehensive report with several metrics and presets
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
    DatasetDriftMetric(),
])

report.run(reference_data=reference, current_data=current)

# Save the report as HTML
report.save_html("reports/drift_report_movies.html")

print("Evidently drift & quality report saved to ml_evidently/reports/drift_report_movies.html")