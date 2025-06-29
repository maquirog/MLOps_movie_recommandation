import json
import os
import subprocess
import yaml
from pathlib import Path
import shutil

def check_and_update_model():
    print("Comparing new model to current production...")
    with open("metrics/scores_test.json") as f:
        current_metrics = json.load(f)

    # Cherche le meilleur run dans mlruns
    #mlruns_dir = "mlruns"
    mlruns_dir = Path("mlruns").resolve()
    best_coverage = current_metrics.get("coverage_10", 0)
    best_model_path = None
    best_metrics_path = None

    for root, dirs, files in os.walk(mlruns_dir):
        for file in files:
            if file == "scores.json" and "artifacts/metrics" in str(root):
                metrics_path = Path(root) / file
                try:
                    with open(metrics_path) as mf:
                        m = json.load(mf)
                        if m.get("coverage_10", 0) > best_coverage:
                            # Revenir au dossier run
                            run_dir = Path(root).parent.parent 
                            model_candidate = run_dir / "artifacts" / "model" / "model.pkl"
                            if model_candidate.exists():
                                best_coverage = m["coverage_10"]
                                best_model_path = model_candidate
                                print(best_model_path)
                                best_metrics_path = metrics_path
                                print(best_coverage)
                except Exception as e:
                    continue

    if best_model_path and best_metrics_path:
        print(f"✅ New model is better. Updating prod model. coverage_10: {best_coverage}")

        # Copier modèle + métriques
        subprocess.run(["cp", best_model_path, "models/model_prod.pkl"], check=True)
        subprocess.run(["cp", best_metrics_path, "metrics/scores_model_prod.json"], check=True)
        #shutil.copy(best_metrics_path, "metrics/scores_model_prod.json")

        # Commit DVC
        subprocess.run(["dvc", "commit", "models/model_prod.pkl"], check=True)

        # Git add + commit + push
        subprocess.run(["git", "add", "metrics/scores_model_prod.json"], check=True)
        subprocess.run(["git", "commit", "-m", "Auto-update model_prod and scores_model_prod"], check=True)
        #try:
        #    subprocess.run(["git", "commit", "-m", "Auto-update metrics"], check=True, capture_output=True, text=True)
        #except subprocess.CalledProcessError as e:
        #    if "nothing to commit" in e.stderr.lower():
        #        print("Aucun changement à committer, tout est à jour.")
        #else:
        #    raise
        subprocess.run(["dvc", "push"], check=True)
        subprocess.run(["git", "push"], check=True)
    else:
        print("⚠️ No better model found. Keeping current production model.")

if __name__ == "__main__":
    check_and_update_model()