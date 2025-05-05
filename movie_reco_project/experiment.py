from mlflow import MlflowClient
import mlflow
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.models.evaluate_model import load_user_favorites, evaluate_and_save_metrics
import argparse
import os


def main():
    # Get project root directory (one level up from script location)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(PROJECT_ROOT)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--movie_matrix_path', type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "processed", "movie_matrix.csv"),
                        help='path to the movie matrix file')
    parser.add_argument('--user_matrix_path', type=str, 
                       default=os.path.join(PROJECT_ROOT, "data", "processed", "user_matrix.csv"),
                       help='path to the user matrix file')
    parser.add_argument('--user_favorites_path', type=str, 
                       default=os.path.join(PROJECT_ROOT, "data", "processed", "user_favorites.json"),
                       help='path to users favorites movies file')
    parser.add_argument('--n_neighbors', type=int, 
                       default=30,
                       help='Parameter n_neighbors used for NearestNeighbors training')
    parser.add_argument('--nearest_neighbors_algorithm', type=str, 
                       default='ball_tree',
                       help='Parameter algorithm used for NearestNeighbors training')
    args = parser.parse_args()

    # Define tracking_uri
    client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

    # Define experiment name, run name and artifact_path name
    movie_experiment = mlflow.set_experiment("Movie_Recommandation_model_Project_Test")
    run_name = "first_run"
    artifact_path = "movie_reco"

    # Import Database
    movie_matrix = pd.read_csv(args.movie_matrix_path)
    X = movie_matrix.drop("movieId", axis=1)

    # Train model
    params = {
        'n_neighbors':args.n_neighbors,
        'algorithm':args.nearest_neighbors_algorithm
    }
    model = NearestNeighbors(**params).fit(X)

    # Evaluate model (metrics calculated on 10 reco from 50 users)
    favorite_movies = load_user_favorites(args.user_favorites_path)

    users = pd.read_csv(args.user_matrix_path)
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
    metrics = evaluate_and_save_metrics(favorite_movies,recommended_movies, args.movie_matrix_path)

    # Store information in tracking server
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model, input_example=filtered_users, artifact_path=artifact_path
        )

if __name__ == "__main__":
    main()
