import pandas as pd
import json
import numpy as np
import os
import mlflow
from mlflow.tracking import MlflowClient
import argparse

# === Chemins adaptés pour une exécution Docker === #
BASE_DIR = "/app"
DEFAULT_FAVORITES_PATH = os.path.join(BASE_DIR, "data/processed/user_favorites.json")
DEFAULT_RECOMMENDATIONS_PATH = os.path.join(BASE_DIR, "data/prediction/predictions.json")
DEFAULT_MOVIES_CSV = os.path.join(BASE_DIR, "data/processed/movie_matrix.csv")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "metrics/scores.json")

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# def load_user_favorites(user_favorites_json=DEFAULT_FAVORITES_PATH):
#     """ 
#     Charge les films préférés de chaque utilisateur à partir du fichier JSON.
#     """
#     with open(user_favorites_json, "r") as f:
#         return json.load(f)

# def load_user_recommendations(user_prediction_json=DEFAULT_RECOMMENDATIONS_PATH):
#     """
#     Charge les films recommandés par le model pour chaque utilisateur à partir du fichier JSON.
#     """
#     with open(user_prediction_json, "r") as f:
#         return json.load(f)

def load_total_movie_count(movies_csv=DEFAULT_MOVIES_CSV):
    """
    Charge le dataset des films et retourne le nombre total de films.
    """
    df = pd.read_csv(movies_csv)
    return len(df)

# def compute_precision_recall_at_k(favorites, recommended_movies, k=10):
#     """
#     Évalue les recommandations en calculant Precision@k et Recall@k.
#     """
#     precision, recall = [], []

#     for user_id, liked_movies in favorites.items():
#         recommended = recommended_movies.get(user_id, [])[:k]
        
#         recommendations = set(recommended)
#         liked_set = set(liked_movies)
        
#         # Precision@k = (Films recommandés pertinents) / k
#         precision_at_k = len(recommendations.intersection(liked_set)) / k
        
#         # Recall@k = (Films aimés retrouvés parmi les recommandés) / (Total des films aimés)
#         recall_at_k = len(recommendations.intersection(liked_set)) / len(liked_set) if liked_set else 0
        
#         precision.append(precision_at_k)
#         recall.append(recall_at_k)
    
#     return np.mean(precision), np.mean(recall)

# def compute_hit_rate_at_k(favorites, recommendations, k=10):
#     """
#     Calcule le Hit Rate pour les recommandations.
#     """
#     hits = 0
#     for user_id, liked_movies in favorites.items():
#         recommended = recommendations.get(user_id, [])
#         recommended_set = set(recommended[:k])
#         liked_set = set(liked_movies)
        
#         # Si au moins un film recommandé est dans les films aimés
#         if recommended_set.intersection(liked_set):
#             hits += 1
    
#     return hits / len(favorites)

# def compute_coverage_at_k(recommendations, total_movies, k=10):
#     """
#     Calcule le Coverage des recommandations.
#     """
#     recommended_set = set()
#     for user_id, recommended in recommendations.items():
#         recommended_set.update(recommended[:k])
    
#     return len(recommended_set) / total_movies


# def compute_ndcg_at_k(favorites, recommendations, k=10):
#     def dcg(recommended, liked):
#         return sum(1 / np.log2(i + 2) if movie in liked else 0
#                    for i, movie in enumerate(recommended[:k]))

#     def idcg(liked):
#         return sum(1 / np.log2(i + 2) for i in range(min(k, len(liked))))

#     scores = []
#     for user_id, liked in favorites.items():
#         rec = recommendations.get(user_id, [])[:k]
#         dcg_val = dcg(rec, liked)
#         idcg_val = idcg(liked)
#         scores.append(dcg_val / idcg_val if idcg_val else 0)

#     return np.mean(scores)

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
                              k=10, output_path=DEFAULT_OUTPUT_PATH,
                              alias=None, model_version=None, model_name = None):
    total_movies = load_total_movie_count()
    metrics=compute_metrics(favorites, recommendations, total_movies, k)
    
    # sauvegarde locale
    if output_path:    
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
            
    if run_id:
        print(f"📎 Logging to existing run: {run_id}")
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"model_version": model_version, 
                             "alias": alias, 
                             "model_name": model_name
                             })

    print("\n📊 Recommandation Evaluation Metrics")
    for key, val in metrics.items():
        print(f"{key}: {val}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None, help="MLflow run ID for logging metrics")
    args = parser.parse_args()
    
    while mlflow.active_run():
        mlflow.end_run()
        
    # Charger les films aimés et recommandés
    favorite_movies = load_json(DEFAULT_FAVORITES_PATH)
    recommended_movies = load_json(DEFAULT_RECOMMENDATIONS_PATH)
    
    run_id= args.run_id
    evaluate_and_save_metrics(favorite_movies,recommended_movies, run_id=run_id)
