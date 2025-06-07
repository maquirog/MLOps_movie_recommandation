from mlflow import MlflowClient
import mlflow
import requests
import os
import json
from itertools import product
import argparse

API_URL = "http://api:8000"
#API_URL = "http://localhost:8000"

def call_train(hyperparams):
    response = requests.post(f"{API_URL}/train", json={"hyperparameters": hyperparams})
    response.raise_for_status()
    print("✅ Model trained successfully.")
    return response.json()

def call_predict(user_ids, n_recommendations):
    json_data = {
        "user_ids": user_ids,
        "n_recommendations": n_recommendations
    }
    response = requests.post(f"{API_URL}/predict", json=json_data)
    response.raise_for_status()
    print("✅ Predictions generated and saved.")
    return response.json()

def call_evaluate(k):
    response = requests.post(f"{API_URL}/evaluate", json={"k": k})
    response.raise_for_status()
    print("✅ Evaluation complete.")
    return response.json()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_movies_metrics', type=int, 
                       default=10,
                       help='Number of movies used for model metrics calculation')
    args = parser.parse_args()

    # Define tracking_uri
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()

    # Define parameter grid
    n_neighbors_list = [10, 20, 30]
    algorithm_list = ['ball_tree', 'kd_tree', 'brute']
    param_grid = list(product(n_neighbors_list, algorithm_list))

    # For tracking best run
    best_coverage = -1
    best_run_id = None
    best_params = None

    for n_neighbors, algorithm in param_grid:
        print(f"\nTraining with n_neighbors={n_neighbors}, algorithm={algorithm}")

        hyperparams = {
            'n_neighbors': n_neighbors,
            'algorithm': algorithm
        }

        call_train(hyperparams)

        call_predict(user_ids=list(range(1, 4)), n_recommendations=10)

        metrics = call_evaluate(k=args.n_movies_metrics)

        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        metrics_path = os.path.join(BASE_DIR, "metrics", "scores.json")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        # Store information in tracking server
        with mlflow.start_run() as run:
            mlflow.log_params(hyperparams)
            mlflow.log_metrics(metrics)
            run_id = run.info.run_id
            print(f"Logged run_id: {run_id} with metrics: {metrics}")

            # Track best coverage
            if metrics.get(f'coverage_{str(args.n_movies_metrics)}', 0) > best_coverage:
                best_coverage = metrics[f'coverage_{str(args.n_movies_metrics)}']
                best_run_id = run_id
                best_params = hyperparams

    print("\n=== Best run based on 'coverage' ===")
    print(f"Run ID: {best_run_id}")
    print(f"Best Params: {best_params}")
    print(f"Best Coverage: {best_coverage}")

if __name__ == "__main__":
    main()