import pandas as pd
import json
import numpy as np
import os
import mlflow
import argparse

def load_user_favorites(user_favorites_json="data/processed/user_favorites.json"):
    """
    Charge les films préférés de chaque utilisateur à partir du fichier JSON.
    """
    with open(user_favorites_json, "r") as f:
        return json.load(f)

def load_user_recommendations(user_prediction_json):
    """
    Charge les films recommandés par le model pour chaque utilisateur à partir du fichier JSON.
    """
    with open(user_prediction_json, "r") as f:
        return json.load(f)

def load_total_movie_count(movies_csv="data/processed/movie_matrix.csv"):
    """
    Charge le dataset des films et retourne le nombre total de films.
    """
    df = pd.read_csv(movies_csv)
    return len(df)

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
    for user_id, recommended in recommendations.items():
        recommended_set.update(recommended[:k])
    
    return len(recommended_set) / total_movies


def compute_ndcg_at_k(favorites, recommendations, k=10):
    def dcg(recommended, liked):
        return sum(1 / np.log2(i + 2) if movie in liked else 0
                   for i, movie in enumerate(recommended[:k]))

    def idcg(liked):
        return sum(1 / np.log2(i + 2) for i in range(min(k, len(liked))))

    scores = []
    for user_id, liked in favorites.items():
        rec = recommendations.get(user_id, [])[:k]
        dcg_val = dcg(rec, liked)
        idcg_val = idcg(liked)
        scores.append(dcg_val / idcg_val if idcg_val else 0)

    return np.mean(scores)

def evaluate_and_log_metrics(favorites,recommendations, movies_csv="data/processed/movie_matrix.csv", k=10, mlflow_alias=None, output_path=None):
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
    with mlflow.start_run(run_name=f"evaluation_{mlflow_alias}" if mlflow_alias else "evaluation") as run:
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
    
    # Évaluation
    print("📊 Évaluation du modèle Challenger")
    challenger_metrics = evaluate_and_log_metrics(favorite_movies, challenger_recommendations, mlflow_alias="challenger", output_path="metrics/challenger_metrics.json")
    print(json.dumps(challenger_metrics, indent=4))
    

    print("\n📊 Évaluation du modèle Champion")
    champion_metrics = evaluate_and_log_metrics(favorite_movies, champion_recommendations, mlflow_alias="champion", output_path="metrics/champion_metrics.json")
    print(json.dumps(champion_metrics, indent=4))