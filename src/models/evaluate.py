import pandas as pd
import json
import numpy as np
import os
import mlflow
import argparse
from datetime import datetime

# === Chemins adaptés pour une exécution Docker === #
# --- Env --- #
BASE_DIR = os.environ.get("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
METRICS_DIR= os.environ.get("METRICS_DIR", os.path.join(BASE_DIR, "metrics"))
DATA_DIR= os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))

# --- DEFAULT --- #
DEFAULT_PREDICTIONS_DIR =os.path.join(DATA_DIR, "predictions")
DEFAULT_FAVORITES_PATH = os.path.join(DATA_DIR, "processed/user_favorites.json")
DEFAULT_MOVIES_CSV = os.path.join(DATA_DIR, "processed/movie_matrix.csv")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_total_movie_count(movies_csv=DEFAULT_MOVIES_CSV):
    """
    Charge le dataset des films et retourne le nombre total de films.
    """
    df = pd.read_csv(movies_csv)
    return len(df)

def compute_metrics(favorites, recommendations, total_movies, k=10):
    precision, recall, ndcg_scores, hits = [], [], [], 0
    recommended_set = set()

    for user, liked in favorites.items():
        recs = recommendations.get(user, [])[:k]
        rec_set = set(recs)
        liked_set = set(liked)
        recommended_set.update(rec_set)

        inter = rec_set & liked_set
        precision.append(len(inter) / k)
        recall.append(len(inter) / len(liked_set) if liked_set else 0)
        hits += 1 if inter else 0

        dcg = sum(1 / np.log2(i + 2) if movie in liked_set else 0 for i, movie in enumerate(recs))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(liked_set))))
        ndcg_scores.append(dcg / idcg if idcg else 0)

    return {
        f"precision_{k}": round(np.mean(precision), 4),
        f"recall_{k}": round(np.mean(recall), 4),
        f"hit_rate_{k}": round(hits / len(favorites), 4),
        f"coverage_{k}": round(len(recommended_set) / total_movies, 4),
        f"ndcg_{k}": round(np.mean(ndcg_scores), 4)
    }

def evaluate_and_save_metrics(favorites, recommendations, run_id=None, 
                              k=10, output_dir=None):
    total_movies = load_total_movie_count()
    metrics=compute_metrics(favorites, recommendations, total_movies, k)
    
    # Impression JSON lisible depuis stdout
    print(json.dumps(metrics))
    
    # sauvegarde locale
    # if output_dir:  
    #     os.makedirs(output_dir, exist_ok=True)
    #     file_name = f'scores_{run_id}.json' or 'scores.json'
    #     output_path = os.path.join(output_dir, file_name)
    #     with open(output_path, "w") as f:
    #         json.dump(metrics, f, indent=4)
           
    # sauvegarde sur MLflow 
    if run_id:
        # print(f"📎 Logging to existing run: {run_id}")
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)

    # print("\n📊 Recommandation Evaluation Metrics")
    # for key, val in metrics.items():
    #     print(f"{key}: {val}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None, help="MLflow run ID for logging metrics")
    parser.add_argument("--input_filename", type=str, default="predictions.json", help="Nom du fichier de sortie des prédictions.")
    args = parser.parse_args()
    
    if mlflow.active_run():
        mlflow.end_run()
        
    # Charger les films aimés et recommandés
    favorite_movies = load_json(DEFAULT_FAVORITES_PATH)
    
    recommendations_path = os.path.join(DEFAULT_PREDICTIONS_DIR, args.input_filename)
    recommended_movies = load_json(recommendations_path)
    
    run_id= args.run_id
    evaluate_and_save_metrics(favorite_movies,recommended_movies, run_id=run_id, output_dir=METRICS_DIR)