from mlflow import MlflowClient
import mlflow
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.models.evaluate import load_user_favorites, evaluate_and_save_metrics
import argparse
import os


def main():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_neighbors', type=int, 
                       default=30,
                       help='Parameter n_neighbors used for NearestNeighbors training')
    parser.add_argument('--nearest_neighbors_algorithm', type=str, 
                       default='ball_tree',
                       help='Parameter algorithm used for NearestNeighbors training')
    parser.add_argument('--n_movies_metrics', type=int, 
                       default=10,
                       help='Number of movies used for model metrics calculation')
    args = parser.parse_args()

    # Define tracking_uri
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()

    # Define artifact_path name
    artifact_path = "movie_reco"

    # Import Database
    movie_matrix = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'movie_matrix.csv'))
    X = movie_matrix.drop("movieId", axis=1)

    # Train model
    params = {
        'n_neighbors':args.n_neighbors,
        'algorithm':args.nearest_neighbors_algorithm
    }
    model = NearestNeighbors(**params).fit(X)

    # Evaluate model (metrics calculated on 10 reco from 50 users)
    favorite_movies = load_user_favorites(os.path.join(BASE_DIR, 'data', 'processed', 'user_favorites.json'))

    users = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'user_matrix.csv'))
    users_id = list(range(1, 51))
    filtered_users = users[users["userId"].isin(users_id)]
    original_ids = filtered_users["userId"].values
    filtered_users = filtered_users.drop("userId", axis=1)
    _, indices = model.kneighbors(filtered_users)
    selection = indices[:, :10]
    recommended_movies = {
        int(user_id): list(map(int, movie_indices))
        for user_id, movie_indices in zip(original_ids, selection)
    }
    metrics = evaluate_and_save_metrics(
            favorites=favorite_movies,
            recommendations=recommended_movies,
            movies_csv=os.path.join(BASE_DIR, 'data', 'processed', 'movie_matrix.csv'),
            k=args.n_movies_metrics)

    # Store information in tracking server
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        sk_model=model, artifact_path=artifact_path
    )

if __name__ == "__main__":
    main()