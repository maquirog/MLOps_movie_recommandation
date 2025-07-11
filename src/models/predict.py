import pandas as pd
import pickle
import numpy as np
import json
import os
import argparse
from typing import List, Dict, Union
import mlflow

BASE_DIR = os.environ.get("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
METRICS_DIR= os.environ.get("METRICS_DIR", os.path.join(BASE_DIR, "metrics"))
DATA_DIR= os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5050")

DEFAULT_PREDICTIONS_DIR =os.path.join(DATA_DIR, "predictions")

def load_user_data(user_matrix: Union[str, pd.DataFrame], users_id: List[int] = None) -> pd.DataFrame:
    """
    Loads and filters user data from a file or DataFrame.

    Args:
        user_matrix: Path to a CSV file or a DataFrame containing user data.
        users_id: List of user IDs to filter. If None, returns all users.

    Returns:
        Filtered DataFrame containing only the specified users (or all users if users_id is None).
    """
    if isinstance(user_matrix, str):
        user_matrix = pd.read_csv(user_matrix)
    if users_id is not None:
        user_matrix = user_matrix[user_matrix["userId"].isin(users_id)]
    return user_matrix

def load_model_from_source(model_source: str, registry=False, alias=None):
    """
    Load model from a pickle file, MLflow alias, or MLRun URI.

    Args:
        model_source: Path to .pkl file, or mlflow alias (mlflow:model_name@stage),
                      or mlrun URI (mlrun:project/function).

    Returns:
        Loaded model.
    """
    if model_source.endswith(".pkl"):
        with open(model_source, "rb") as f:
            model = pickle.load(f)
        return model
    
    elif model_source.startswith("runs:/"):
        model = mlflow.sklearn.load_model(model_source)
        print(f"✅ Modèle '{model_source} @{alias}' chargé.")
        return model
    
    # elif registry:
    #     model = mlflow.sklearn.load_model(f"models:/{model_source}@{alias}")
    #     print(f"✅ Modèle '{model_source} @{alias}' chargé.")
    #     return model
    
    else:
        print("problemes")


def make_predictions(model, user_data: pd.DataFrame, n_recos: int = 10) -> Dict[int, List[int]]:
    """
    Generates recommendations for users.

    Args:
        model: Trained model for generating recommendations.
        user_data: DataFrame containing user data (excluding the userId column).
        n_recos: Number of recommendations to generate per user.

    Returns:
        Dictionary of recommendations {user_id: [movie_index1, movie_index2, ...]}.
    """
    original_ids = user_data["userId"].values
    features = user_data.drop("userId", axis=1)
    _, indices = model.kneighbors(features)
    selection = indices[:, :n_recos]
    prediction_dict = {
        int(user_id): list(map(int, movie_indices))
        for user_id, movie_indices in zip(original_ids, selection)
    }
    return prediction_dict

def save_predictions_to_file(predictions: Dict[int, List[int]], output_path: str):
    """
    Saves predictions to a JSON file.

    Args:
        predictions: Dictionary of recommendations {user_id: [movie_index1, movie_index2, ...]}.
        output_path: Path to the output file where predictions will be saved.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"✅ Predictions saved to {output_path}")
    
    

if __name__ == "__main__":
    # Parse command-line arguments  
    parser = argparse.ArgumentParser(description="Predict movies for users.")
    parser.add_argument("--user_ids", type=str, help="Comma-separated list of user IDs (e.g., '1,2,3').")
    parser.add_argument("--n_recommendations", type=int, default=10, help="Number of recommendations per user.")
    parser.add_argument("--output_filename", type=str, default="predictions.json", help="Nom du fichier de sortie des prédictions (ex: 'recos_user42.json').")
    parser.add_argument("--no_save_to_file", action="store_true", help="Flag to disable saving predictions to a file.")
    parser.add_argument("--model_source", type=str, required=True, help="Path to .pkl, mlflow:model@stage, or runs:/<run_id>/model")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Load user data
    if args.user_ids:
        users_id = list(map(int, args.user_ids.split(",")))
    else:
        # Default behavior: load all users
        user_matrix_path = os.path.join(BASE_DIR, "data", "processed", "user_matrix.csv")
        print(f"⚙️ Loading all user IDs from {user_matrix_path}...")
        all_users = pd.read_csv(user_matrix_path)
        users_id = all_users["userId"].tolist()

    # Load user data and model
    user_data = load_user_data(user_matrix_path, users_id)
    model = load_model_from_source(args.model_source)

    # Generate predictions
    predictions = make_predictions(model, user_data, n_recos=args.n_recommendations)
    
    # Save predictions to file by default unless --no_save_to_file is provided
    if not args.no_save_to_file:
        predictions_path = os.path.join(DEFAULT_PREDICTIONS_DIR,args.output_filename)
        save_predictions_to_file(predictions, predictions_path)