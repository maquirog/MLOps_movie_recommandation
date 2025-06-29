import pandas as pd
import json
import numpy as np
import os
import mlflow
from mlflow.tracking import MlflowClient


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_total_movie_count(movies_csv=DEFAULT_MOVIES_CSV):
    """
    Charge le dataset des films et retourne le nombre total de films.
    """
    df = pd.read_csv(movies_csv)
    return len(df)

def get_infos_from_alias(model_name: str, alias: str) -> str:
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name=model_name, alias=alias)
    return mv.run_id, mv.version


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
    
def evaluate_and_log(favorites, recommendations, alias, model_version, run_id=None, model_name="movie_recommender", output_path=None, k=10):
    total_movies = load_total_movie_count()
    metrics = compute_metrics(favorites, recommendations, total_movies, k)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)

    if run_id and alias != "champion":
        print(f"📎 Logging to existing run: {run_id}")
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"model_version": model_version, "alias": alias, "model_name": model_name})
    else:
        print(f"🆕 Creating new run for {alias} v{model_version}")
        with mlflow.start_run(run_name=f"evaluation_{alias}-{model_name}_v{model_version}"):
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"model_version": model_version, "alias": alias, "model_name": model_name, "evaluation_type": "new_dataset"})
            if output_path:
                mlflow.log_artifact(output_path)

    print(f"\n📊 Evaluation ({alias}):")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    while mlflow.active_run():
        mlflow.end_run()

    favs = load_json("data/processed/user_favorites.json")
    recs_challenger = load_json("data/prediction/predictions_challenger.json")
    recs_champion = load_json("data/prediction/predictions_champion.json")

    run_id_chall, ver_chall = get_infos_from_alias("movie_recommender", "challenger")
    run_id_champ, ver_champ = get_infos_from_alias("movie_recommender", "champion")

    evaluate_and_log(favs, recs_challenger, "challenger", ver_chall, run_id=run_id_chall, output_path="metrics/challenger.json")
    evaluate_and_log(favs, recs_champion, "champion", ver_champ, output_path="metrics/champion.json")
    
# def evaluate_metrics(favorites,recommendations, 
#                      movies_csv="data/processed/movie_matrix.csv", k=10,
#                      output_path=None):
#     if not recommendations:
#         metrics = {
#             f"precision_{k}": 0.0,
#             f"recall_{k}": 0.0,
#             f"hit_rate_{k}": 0.0,
#             f"coverage_{k}": 0.0,
#             f"ndcg_{k}": 0.0
#         }
#     else:
#         total_movies = load_total_movie_count(movies_csv)
#         precision, recall = compute_precision_recall_at_k(favorites, recommendations, k)
#         hr = compute_hit_rate_at_k(favorites, recommendations, k)
#         cov = compute_coverage_at_k(recommendations, total_movies, k)
#         ndcg = compute_ndcg_at_k(favorites, recommendations, k)
        
#         metrics = {
#             f"precision_{str(k)}": round(precision, 4),
#             f"recall_{str(k)}": round(recall, 4),
#             f"hit_rate_{str(k)}": round(hr, 4),
#             f"coverage_{str(k)}": round(cov, 4),
#             f"ndcg_{str(k)}": round(ndcg, 4)
#         }
        
#     if output_path:
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         with open(output_path, "w") as f:
#             json.dump(metrics, f, indent=4)
#         mlflow.log_artifact(output_path)
    
#     return metrics
        
# def log_metrics(metrics, run_id_train=None, model_version=None, 
#                 mlflow_alias=None):
#     # Log in MLflow
#     if run_id_train and mlflow_alias != "champion":
#         print(f"Logging metrics to existing run {run_id_train} for {mlflow_alias}")
#         with mlflow.start_run(run_id=run_id_train) as run:
#             mlflow.log_metrics(metrics)
#             mlflow.set_tag('model_version', model_version)
#             mlflow.set_tag("alias", mlflow_alias)
#             mlflow.set_tag("model_name", "movie_recommender")
#     else:
#         print(f"Creating a new run for evaluation of {mlflow_alias} version {model_version}")
#         with mlflow.start_run(run_name=f"evaluation_{mlflow_alias} (v{model_version})", nested=True) as run:
#             mlflow.log_metrics(metrics)
#             mlflow.set_tag('model_version', model_version)
#             mlflow.set_tag("alias", mlflow_alias)
#             mlflow.set_tag("model_name", "movie_recommender")
            
#     print(f"\n📊 Evaluation Metrics ({mlflow_alias}):")
#     for key, val in metrics.items():
#         print(f"{key}: {val}")
    
# if __name__ == "__main__":
#     # 🔐 Assure-toi que TOUTE run est bien fermée
#     while mlflow.active_run():
#         mlflow.end_run()
#     # Chargement des films aimés par utilisateur
#     favorite_movies = load_user_favorites()
    
#     # Chemins vers les fichiers de recommandations des deux modèles
#     path_challenger = "data/prediction/predictions_challenger.json"
#     path_champion = "data/prediction/predictions_champion.json"
    
#     # Chargement des recommandations
#     challenger_recommendations = load_user_recommendations(path_challenger)
#     champion_recommendations = load_user_recommendations(path_champion)
    
#     # Chargement des run-ids
#     challenger_run_id, challenger_model_version= get_infos_from_alias(model_name="movie_recommender", alias="challenger")
#     champion_run_id, champion_model_version= get_infos_from_alias(model_name="movie_recommender", alias="champion")
    
#     # Évaluation
#     challenger_metrics = evaluate_metrics(
#         favorite_movies,
#         challenger_recommendations,
#         output_path="metrics/challenger_metrics.json"
#     )

#     champion_metrics = evaluate_metrics(
#         favorite_movies,
#         champion_recommendations,
#         mlflow_alias="champion",
#         output_path="metrics/champion_metrics.json"
#     )
    
#     # log metric
#     log_metrics(challenger_metrics, run_id_train=challenger_run_id, model_version=challenger_model_version, mlflow_alias="challenger")
#     log_metrics(champion_metrics, run_id_train=champion_run_id, model_version=champion_model_version, mlflow_alias="champion")
