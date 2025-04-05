import pandas as pd
import json
import numpy as np

def load_favorite_movies(user_favorites_json="data/processed/user_favorites.json"):
    """
    Charge les films préférés de chaque utilisateur à partir du fichier JSON.
    """
    with open(user_favorites_json, "r") as f:
        favorite_movies = json.load(f)
    return favorite_movies

def load_recommended_movies(user_prediction_json="data/prediction/predictions.json"):
    """
    Charge les films recommandés par le model pour chaque utilisateur à partir du fichier JSON.
    """
    with open(user_prediction_json, "r") as f:
        recommended_movies = json.load(f)
    return recommended_movies

def load_total_movies(movies_csv="data/processed/movie_matrix.csv"):
    """
    Charge le dataset des films et retourne le nombre total de films.
    """
    df = pd.read_csv(movies_csv)
    return len(df)

def precision_and_recall(favorite_movies, recommended_movies, k=10):
    """
    Évalue les recommandations en calculant Precision@k et Recall@k.
    """
    precision = []
    recall = []

    for user_id, liked_movies in favorite_movies.items():
        # Obtenir les k films recommandés pour cet utilisateur
        recommended = recommended_movies.get(user_id, [])
        
        # Calculer l'intersection entre les films recommandés et les films aimés
        recommended_set = set(recommended[:k])
        liked_set = set(liked_movies)
        
        # Precision@k = (Films recommandés pertinents) / k
        precision_at_k = len(recommended_set.intersection(liked_set)) / k
        
        # Recall@k = (Films aimés retrouvés parmi les recommandés) / (Total des films aimés)
        recall_at_k = len(recommended_set.intersection(liked_set)) / len(liked_set) if liked_set else 0
        
        precision.append(precision_at_k)
        recall.append(recall_at_k)
    
    # Moyenne des précisions et rappels
    avg_precision = sum(precision) / len(precision) if precision else 0
    avg_recall = sum(recall) / len(recall) if recall else 0
    
    return avg_precision, avg_recall

def hit_rate(favorite_movies, recommended_movies, k=10):
    """
    Calcule le Hit Rate pour les recommandations.
    """
    hits = 0
    for user_id, liked_movies in favorite_movies.items():
        recommended = recommended_movies.get(user_id, [])
        recommended_set = set(recommended[:k])
        liked_set = set(liked_movies)
        
        # Si au moins un film recommandé est dans les films aimés
        if recommended_set.intersection(liked_set):
            hits += 1
    
    return hits / len(favorite_movies)

def coverage(recommended_movies, total_movies, k=10):
    """
    Calcule le Coverage des recommandations.
    """
    recommended_set = set()
    for user_id, recommended in recommended_movies.items():
        recommended_set.update(recommended[:k])
    
    return len(recommended_set) / total_movies


def dcg_at_k(recommended, liked_movies, k=10):
    """
    Calcule le DCG@k pour un utilisateur donné.
    """
    recommended_set = set(recommended[:k])
    liked_set = set(liked_movies)
    
    # Calcul du gain pour chaque film recommandé
    dcg = 0
    for i, movie in enumerate(recommended[:k]):
        # Si le film recommandé est aimé par l'utilisateur, on lui donne une pertinence de 1
        relevance = 1 if movie in liked_set else 0
        dcg += relevance / np.log2(i + 2)  # Plus la position est basse, moins il compte
    
    return dcg

def idcg_at_k(liked_movies, k=10):
    """
    Calcule l'IDCG@k pour un utilisateur donné.
    """
    # L'IDCG idéal est 1 pour chaque film aimé dans les positions 1, 2, ..., k
    return np.sum([1 / np.log2(i + 2) for i in range(min(k, len(liked_movies)))])

def ndcg_at_k(favorite_movies, recommended_movies, k=10):
    """
    Calcule le NDCG@k moyen pour tous les utilisateurs.
    """
    ndcg_scores = []
    
    for user_id, liked_movies in favorite_movies.items():
        recommended = recommended_movies.get(user_id, [])
        dcg = dcg_at_k(recommended, liked_movies, k)
        idcg = idcg_at_k(liked_movies, k)
        
        # NDCG = DCG / IDCG
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0)
    
    return np.mean(ndcg_scores)


if __name__ == "__main__":
    # Charger les films aimés
    favorite_movies = load_favorite_movies()
    recommended_movies = load_recommended_movies()
    total_movies = load_total_movies()
    
    # Évaluer les recommandations
    avg_precision, avg_recall = precision_and_recall(favorite_movies, recommended_movies, k=10)
    print(f"Precision@10: {avg_precision:.4f}")
    print(f"Recall@10: {avg_recall:.4f}")
    
    hr = hit_rate(favorite_movies, recommended_movies, k=10)
    print(f"Hit Rate: {hr:.4f}")
    
    coverage_score = coverage(recommended_movies, total_movies, k=10)
    print(f"Coverage: {coverage_score:.4f}")
    
    ndcg_score = ndcg_at_k(favorite_movies, recommended_movies, k=10)
    print(f"NDCG@10: {ndcg_score:.4f}")

