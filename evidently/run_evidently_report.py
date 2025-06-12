import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric
from evidently.ui.workspace import Workspace
import os

DRIFT_SHARE_THRESHOLD = 0.3  # Global variable for drift threshold

def add_report_to_workspace(workspace, project_name, project_description, report):
    """
    Adds a report to an existing or new project in a workspace.
    """
    # Check if project already exists
    project = None
    for p in workspace.list_projects():
        if p.name == project_name:
            project = p
            break

    # Create a new project if it doesn't exist
    if project is None:
        project = workspace.create_project(project_name)
        project.description = project_description

    # Add report to the project
    workspace.add_report(project.id, report)
    print(f"New report added to project {project_name}")

def main():
    # Load your datasets
    reference = pd.read_csv("data/raw/movies.csv")
    current = pd.read_csv("data/raw/movies_drift.csv")

    # Create a comprehensive report with several metrics and presets
    report = Report(
        metrics=[
            DataDriftPreset(drift_share=DRIFT_SHARE_THRESHOLD),
            DataQualityPreset(),
            DatasetDriftMetric(drift_share=DRIFT_SHARE_THRESHOLD),
        ]
    )

    report.run(reference_data=reference, current_data=current)

    # Save the report as HTML
    output_path = "reports/drift_report_movies.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # <-- FIX
    report.save_html(output_path)
    print(f"Evidently drift & quality report saved to {output_path}")

    # Save to Evidently workspace
    workspace = Workspace.create(os.path.abspath("evidently/workspace"))
    add_report_to_workspace(
        workspace=workspace,
        project_name="movies_drift_monitoring",
        project_description="Evidently dashboard to monitor drift for movies data",
        report=report)

if __name__ == "__main__":
    main()