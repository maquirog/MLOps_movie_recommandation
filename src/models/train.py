import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import json
import argparse
import mlflow.sklearn
from mlflow.models.signature import infer_signature


BASE_DIR = os.environ.get("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5050")

default_hyperparameters = {
    "n_neighbors": 20,
    "algorithm": "ball_tree"
}

def train_model(movie_matrix, hyperparams):
    nbrs = NearestNeighbors(**hyperparams).fit(
        movie_matrix.drop("movieId", axis=1)
    )
    return nbrs

def log_model_to_mlflow(model, movie_matrix, hyperparams):
    mlflow.log_params(hyperparams)
        
    example_input = movie_matrix.drop("movieId", axis=1).iloc[[0]]
    signature = infer_signature(example_input, model.kneighbors(example_input)[1])

    mlflow.sklearn.log_model(model, "model", signature=signature)
    print("✅ Model trained and logged.")

def save_model_locally(model, run_id):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"model_{run_id}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Modèle sauvegardé localement sous {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyperparams_dict",
        type=str,
        default=json.dumps(default_hyperparameters),
        help="Hyperparameters as JSON string, e.g., '{\"n_neighbors\": 15, \"algorithm\": \"kd_tree\"}'"
        )
    parser.add_argument("--run_id", type=str, help="MLflow run ID to use for logging")
    parser.add_argument("--save_pkl",type=bool, default=True, help="Save model as .pkl locally")
    args = parser.parse_args()
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        hyperparams = json.loads(args.hyperparams_dict)
    except json.JSONDecodeError:
        print("Error: hyperparams_dict must be a valid JSON string.")
        exit(1)

    # Load the movie matrix
    print("Chargement des données")
    matrix_path = os.path.join(BASE_DIR, "data", "processed", "movie_matrix.csv")
    movie_matrix = pd.read_csv(matrix_path)

    # Train the model
    print("Entrainement du model")
    model = train_model(movie_matrix, hyperparams)
    
    # === MLflow logging ===
    # Safety: stop any active run before manually starting one
    print(f"grid search run ID:{args.run_id}")
    if args.run_id:
        print(f"📌 Logging to existing MLflow run ID: {args.run_id}")
        with mlflow.start_run(run_id=args.run_id):
            log_model_to_mlflow(model, movie_matrix, hyperparams)
            if args.save_pkl:
                save_model_locally(model, args.run_id)
    else:
        print("📌 Aucun run_id fourni, lancement dans expérience 'ManualRuns'")
        experiment_name = "ManualRuns"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        with mlflow.start_run(experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            mlflow.set_tag("manual", "true")
            log_model_to_mlflow(model, movie_matrix, hyperparams)
            if args.save_pkl:
                save_model_locally(model, run_id)
