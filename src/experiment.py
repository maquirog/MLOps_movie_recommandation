from mlflow import MlflowClient
import mlflow
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from models.evaluate_model import load_user_favorites, evaluate_and_save_metrics

# Define tracking_uri
client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

# Define experiment name, run name and artifact_path name
apple_experiment = mlflow.set_experiment("Movie_Recommandation_model")
run_name = "first_run"
artifact_path = "movie_reco"

# Import Database
movie_matrix = pd.read_csv("../data/processed/movie_matrix.csv")
X = movie_matrix.drop("movieId", axis=1)

# Train model
params = {
    n_neighbors=20,
    algorithm="ball_tree"
}
model = NearestNeighbors(**params).fit(X)

# Evaluate model
favorite_movies = load_user_favorites()

users = pd.read_csv("../data/processed/user_matrix.csv")
users_id = [1, 2, 3, 4, 5]
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
