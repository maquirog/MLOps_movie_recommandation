import pandas as pd
import json
import numpy as np
import os
import mlflow
from mlflow.tracking import MlflowClient

# === Chemins adaptés pour une exécution Docker === #
# --- Env --- #
BASE_DIR = os.environ.get("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
METRICS_DIR= os.environ.get("METRICS_DIR", os.path.join(BASE_DIR, "metrics"))
DATA_DIR= os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5050")

# --- DEFAULT --- #
DEFAULT_PREDICTIONS_DIR =os.path.join(DATA_DIR, "predictions")
DEFAULT_FAVORITES_PATH = os.path.join(DATA_DIR, "processed/user_favorites.json")
DEFAULT_MOVIES_CSV = os.path.join(DATA_DIR, "processed/movie_matrix.csv")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_user_recommendations(user_prediction_json):
    """
    Charge les films recommandés par le model pour chaque utilisateur à partir du fichier JSON.
    """
    with open(user_prediction_json, "r") as f:
        return json.load(f)

def load_total_movie_count(movies_csv=DEFAULT_MOVIES_CSV):
    """
    Charge le dataset des films et retourne le nombre total de films.
    """
    df = pd.read_csv(movies_csv)
    return len(df)

def get_run_id_from_alias(model_name: str, alias: str) -> str:
    client = MlflowClient()
    # Récupère la version de modèle associée à l'alias
    alias_info = client.get_model_version_by_alias(name=model_name, alias=alias)
    # run_id associé à cette version
    run_id = alias_info.run_id
    return run_id

def compute_precision_recall_at_k(favorites, recommended_movies, k=10):
    """
    Évalue les recommandations en calculant Precision@k et Recall@k.
    """
    precision, recall = [], []

    for user_id, liked_movies in favorites.items():
        recommended = recommended_movies.get(user_id, [])[:k]
        
        recommendations = set(recommended)
        liked_set = set(liked_movies)
        
        # Precision@k = (Films recommandés pertinents) / k
        precision_at_k = len(recommendations.intersection(liked_set)) / k
        
        # Recall@k = (Films aimés retrouvés parmi les recommandés) / (Total des films aimés)
        recall_at_k = len(recommendations.intersection(liked_set)) / len(liked_set) if liked_set else 0
        
        precision.append(precision_at_k)
        recall.append(recall_at_k)
    
    return np.mean(precision), np.mean(recall)

def compute_hit_rate_at_k(favorites, recommendations, k=10):
    """
    Calcule le Hit Rate pour les recommandations.
    """
    hits = 0
    for user_id, liked_movies in favorites.items():
        recommended = recommendations.get(user_id, [])
        recommended_set = set(recommended[:k])
        liked_set = set(liked_movies)
        
        # Si au moins un film recommandé est dans les films aimés
        if recommended_set.intersection(liked_set):
            hits += 1
    
    return hits / len(favorites)

def compute_coverage_at_k(recommendations, total_movies, k=10):
    """
    Calcule le Coverage des recommandations.
    """
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

    scores = []
    for user_id, liked in favorites.items():
        rec = recommendations.get(user_id, [])[:k]
        dcg_val = dcg(rec, liked)
        idcg_val = idcg(liked)
        scores.append(dcg_val / idcg_val if idcg_val else 0)

    return np.mean(scores)

def evaluate_and_log_metrics(favorites,recommendations, run_id_train, movies_csv="data/processed/movie_matrix.csv", k=10, mlflow_alias=None, output_path=None):
    if not recommendations:
        metrics = {
            "precision_10": 0.0,
            "recall_10": 0.0,
            "hit_rate_10": 0.0,
            "coverage_10": 0.0,
            "ndcg_10": 0.0
        }
    else:
        total_movies = load_total_movie_count(movies_csv)
        precision, recall = compute_precision_recall_at_k(favorites, recommendations, k)
        hr = compute_hit_rate_at_k(favorites, recommendations, k)
        cov = compute_coverage_at_k(recommendations, total_movies, k)
        ndcg = compute_ndcg_at_k(favorites, recommendations, k)
        
        metrics = {
            f"precision_{str(k)}": round(precision, 4),
            f"recall_{str(k)}": round(recall, 4),
            f"hit_rate_{str(k)}": round(hr, 4),
            f"coverage_{str(k)}": round(cov, 4),
            f"ndcg_{str(k)}": round(ndcg, 4)
        }
        
    # Log in MLflow
    with mlflow.start_run(run_id=run_id_train):
        mlflow.log_params({"alias": mlflow_alias, "k": k})
        mlflow.log_metrics(metrics)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(metrics, f, indent=4)
            mlflow.log_artifact(output_path)
    
    print(f"\n📊 Evaluation Metrics ({mlflow_alias or 'N/A'}):")
    for key, val in metrics.items():
        print(f"{key}: {val}")

    return metrics
    
if __name__ == "__main__":
    # Chargement des films aimés par utilisateur
    favorite_movies = load_user_favorites()
    
    # Chemins vers les fichiers de recommandations des deux modèles
    path_challenger = "data/prediction/predictions_challenger.json"
    path_champion = "data/prediction/predictions_champion.json"
    
    # Chargement des recommandations
    challenger_recommendations = load_user_recommendations(path_challenger)
    champion_recommendations = load_user_recommendations(path_champion)
    
    # Chargement des run-ids
    challenger_run_id= get_run_id_from_alias(model_name="movie_recommender", alias="challenger")
    champion_run_id = get_run_id_from_alias(model_name="movie_recommender", alias="champion")
    
    # Évaluation
    print("📊 Évaluation du modèle Challenger")
    challenger_metrics = evaluate_and_log_metrics(
        favorite_movies,
        challenger_recommendations,
        run_id_train=challenger_run_id,
        mlflow_alias="challenger",
        output_path="metrics/challenger_metrics.json"
    )
    print(json.dumps(challenger_metrics, indent=4))
    

    print("\n📊 Évaluation du modèle Champion")
    champion_metrics = evaluate_and_log_metrics(
        favorite_movies,
        champion_recommendations,
        run_id_train=champion_run_id,
        mlflow_alias="champion",
        output_path="metrics/champion_metrics.json"
    )
    print(json.dumps(champion_metrics, indent=4))
