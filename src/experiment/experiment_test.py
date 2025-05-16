from mlflow import MlflowClient
import mlflow
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.models.evaluate import load_user_favorites, evaluate_and_save_metrics
import argparse
import os


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_neighbors', type=int, 
                       default=30,
                       help='Parameter n_neighbors used for NearestNeighbors training')
    parser.add_argument('--nearest_neighbours_algorithm', type=str, 
                       default='ball_tree',
                       help='Parameter algorithm used for NearestNeighbors training')
    args = parser.parse_args()

    # Define tracking_uri
    client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

    # Define experiment name, run name and artifact_path name
    movie_experiment = mlflow.set_experiment("Movie_Recommandation_model_Test")
    run_name = "sixth_run"
    artifact_path = "movie_reco"

    # Import Database
    movie_matrix = pd.read_csv("./data/processed/movie_matrix.csv")
    X = movie_matrix.drop("movieId", axis=1)

    # Train model
    params = {
        'n_neighbors':args.n_neighbors,
        'algorithm':args.nearest_neighbours_algorithm
    }
    model = NearestNeighbors(**params).fit(X)

    # Evaluate model (metrics calculated on 10 reco from 50 users)
    favorite_movies = load_user_favorites("./data/processed/user_favorites.json")

    users = pd.read_csv("./data/processed/user_matrix.csv")
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
    metrics = evaluate_and_save_metrics(favorite_movies,recommended_movies)

    # Store information in tracking server
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model, input_example=filtered_users, artifact_path=artifact_path
        )

if __name__ == "__main__":
    main()