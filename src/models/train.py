import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import json
import argparse
import mlflow.sklearn
from mlflow.models.signature import infer_signature


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

default_hyperparameters = {
    "n_neighbors": 20,
    "algorithm": "ball_tree"
}

def train_model(movie_matrix, hyperparams):
    nbrs = NearestNeighbors(**hyperparams).fit(
        movie_matrix.drop("movieId", axis=1)
    )
    return nbrs

def log_model_to_mlflow(model, movie_matrix, hyperparams, run_id):
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(hyperparams)
            
        example_input = movie_matrix.drop("movieId", axis=1).iloc[[0]]
        signature = infer_signature(example_input, model.kneighbors(example_input)[1])

        mlflow.sklearn.log_model(model, "model", 
                            signature=signature)
        print("✅ Model trained and logged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyperparams_dict",
        type=str,
        default=json.dumps(default_hyperparameters),
        help="Hyperparameters as JSON string, e.g., '{\"n_neighbors\": 15, \"algorithm\": \"kd_tree\"}'"
        )
    parser.add_argument("--run_id", type=str, help="MLflow run ID to use for logging")
    args = parser.parse_args()
    
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

    # Save the model to a file
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model sauvegarder sur {model_path}") 
    
    # # Save the model to a file original
    # filehandler = open("models/model.pkl", "wb")
    # pickle.dump(model, filehandler)
    # filehandler.close()
    
    # === MLflow logging ===
    # Safety: stop any active run before manually starting one
    print(f"grid search run ID:{args.run_id}")
    if args.run_id:
        print(f"📌 Logging to existing MLflow run ID: {args.run_id}")
        log_model_to_mlflow(model, movie_matrix, hyperparams, args.run_id)
    else:
        print("❌ Error: run_id not provided. Skipping MLflow logging.")
