from mlflow import MlflowClient
import mlflow
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.models.evaluate import load_user_favorites, evaluate_and_save_metrics
import os
from itertools import product
import argparse

def main():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_movies_metrics', type=int, 
                       default=10,
                       help='Number of movies used for model metrics calculation')
    args = parser.parse_args()

    # Define tracking_uri
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()

    # Set experiment
    artifact_path = "movie_reco"

    # Define parameter grid
    n_neighbors_list = [10, 20, 30]
    algorithm_list = ['ball_tree', 'kd_tree', 'brute']
    param_grid = list(product(n_neighbors_list, algorithm_list))

    # Import Database
    movie_matrix = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'movie_matrix.csv'))
    X = movie_matrix.drop("movieId", axis=1)

    favorite_movies = load_user_favorites(os.path.join(BASE_DIR, 'data', 'processed', 'user_favorites.json'))
    users = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'user_matrix.csv'))
    users_id = list(range(1, 51))
    filtered_users = users[users["userId"].isin(users_id)]
    original_ids = filtered_users["userId"].values
    filtered_users = filtered_users.drop("userId", axis=1)

    # For tracking best run
    best_coverage = -1
    best_run_id = None
    best_params = None

    for n_neighbors, algorithm in param_grid:
        print(f"\nTraining with n_neighbors={n_neighbors}, algorithm={algorithm}")

        params = {
            'n_neighbors': n_neighbors,
            'algorithm': algorithm
        }
        # Train model
        model = NearestNeighbors(**params).fit(X)

        # Get recommendations
        _, indices = model.kneighbors(filtered_users)
        selection = indices[:, :10]
        recommended_movies = {
            int(user_id): list(map(int, movie_indices))
            for user_id, movie_indices in zip(original_ids, selection)
        }

        # Evaluate model (metrics calculated on 10 reco from 50 users)
        metrics = evaluate_and_save_metrics(
            favorites=favorite_movies,
            recommendations=recommended_movies,
            movies_csv=os.path.join(BASE_DIR, 'data', 'processed', 'movie_matrix.csv'),
            k=args.n_movies_metrics)

        # Store information in tracking server
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model=model, artifact_path=artifact_path
            )
            run_id = run.info.run_id
            print(f"Logged run_id: {run_id} with metrics: {metrics}")

            # Track best coverage
            if metrics.get(f'coverage_{str(args.n_movies_metrics)}', 0) > best_coverage:
                best_coverage = metrics[f'coverage_{str(args.n_movies_metrics)}']
                best_run_id = run_id
                best_params = params

    print("\n=== Best run based on 'coverage' ===")
    print(f"Run ID: {best_run_id}")
    print(f"Best Params: {best_params}")
    print(f"Best Coverage: {best_coverage}")

if __name__ == "__main__":
    main()