import pandas as pd
from sklearn.neighbors import NearestNeighbors
import os
import json
import argparse
from datetime import datetime
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

default_hyperparameters = {
    "n_neighbors": 20,
    "algorithm": "ball_tree"
}

def train_model(movie_matrix, hyperparams):
    nbrs = NearestNeighbors(**hyperparams).fit(
        movie_matrix.drop("movieId", axis=1)
    )
    return nbrs

def register_model(run_id, model_name="movie_recommender"):
    client = MlflowClient()
    
    # Enregistre une nouvelle version du modèle dans la registry
    model_version = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id
    )
    print(f"Model version {model_version.version} registered under '{model_name}'")
    return model_version.version

def set_model_alias(model_name, version, alias="Challenger"):
    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        version=version,
        alias=alias)
    print(f"[Registry] Model version {version} now aliased as '{alias}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyperparams_dict",
        type=str,
        default=json.dumps(default_hyperparameters),
        help="Hyperparameters as JSON string, e.g., '{\"n_neighbors\": 15, \"algorithm\": \"kd_tree\"}'"
        )
    args = parser.parse_args()
    try:
        hyperparams = json.loads(args.hyperparams_dict)
    except json.JSONDecodeError:
        print("Error: hyperparams_dict must be a valid JSON string.")
        exit(1)

    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Chargement des données
    movie_matrix = pd.read_csv("data/processed/movie_matrix.csv")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"model__{timestamp}") as run:
        run_id = run.info.run_id
        mlflow.log_params(hyperparams)
        model = train_model(movie_matrix, hyperparams)
        
        # Signature et exemple pour éviter les warnings
        example_input = movie_matrix.drop("movieId", axis=1).iloc[[0]]
        signature = infer_signature(example_input, model.kneighbors(example_input)[1])
        
        
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model",
            input_example=example_input,
            signature=signature
            )
        
 
        print(f"[MLflow] Modèle loggé avec run_id: {run_id}")

    # Register model in MLflow Registry
    model_name = "movie_recommender"
    version = register_model(run_id, model_name)
    
    # Attribution d’un alias
    set_model_alias(model_name, version, alias="Challenger")