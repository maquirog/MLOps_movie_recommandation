import pandas as pd
import pickle
import numpy as np
import json
import os
from typing import List, Dict, Union

def load_user_data(user_matrix: Union[str, pd.DataFrame], users_id: List[int]) -> pd.DataFrame:
    """
    Charge et filtre les données utilisateur depuis un fichier ou un DataFrame.

    Args:
        user_matrix: Chemin vers un fichier CSV ou DataFrame contenant les données utilisateur.
        users_id: Liste des IDs utilisateurs à filtrer.

    Returns:
        DataFrame filtré contenant uniquement les utilisateurs spécifiés.
    """
    if isinstance(user_matrix, str):
        user_matrix = pd.read_csv(user_matrix)
    filtered_users = user_matrix[user_matrix["userId"].isin(users_id)]
    return filtered_users

def load_model(model_filename: str):
    """
    Charge un modèle depuis un fichier pickle.

    Args:
        model_filename: Chemin vers le fichier pickle contenant le modèle.

    Returns:
        Modèle chargé.
    """
    with open(model_filename, "rb") as filehandler:
        model = pickle.load(filehandler)
    return model

def make_predictions(model, user_data: pd.DataFrame, n_recos: int = 10) -> Dict[int, List[int]]:
    """
    Effectue des recommandations pour les utilisateurs.

    Args:
        model: Modèle entraîné pour générer les recommandations.
        user_data: DataFrame contenant les données des utilisateurs (sans la colonne userId).
        n_recos: Nombre de recommandations à générer par utilisateur.

    Returns:
        Dictionnaire des recommandations {user_id: [movie_index1, movie_index2, ...]}.
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
    Sauvegarde les prédictions dans un fichier JSON.

    Args:
        predictions: Dictionnaire des recommandations {user_id: [movie_index1, movie_index2, ...]}.
        output_path: Chemin du fichier de sortie pour sauvegarder les prédictions.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"✅ Prédictions sauvegardées dans {output_path}")

if __name__ == "__main__":
    # Exemple d'utilisation
    users_id = [1, 2, 3]

    # Charger les données utilisateur et le modèle
    user_data = load_user_data("data/processed/user_matrix.csv", users_id)
    model = load_model("models/model.pkl")

    # Générer les prédictions
    predictions = make_predictions(model, user_data, n_recos=10)

    # Sauvegarder les prédictions si nécessaire
    save_to_file = True
    if save_to_file:
        save_predictions_to_file(predictions, "data/prediction/predictions.json")

    print(predictions)