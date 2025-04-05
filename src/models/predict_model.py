import pandas as pd
import pickle
import numpy as np
import os
import json


def make_predictions(users_id, model_filename, user_matrix_filename, n_recos=10):
    # Read user_matrix
    users = pd.read_csv(user_matrix_filename)

    # Filter with the list of users_id
    filtered_users = users[users["userId"].isin(users_id)]
    
    # Save original user IDs
    original_ids = filtered_users["userId"].values

    # Delete userId
    filtered_users = filtered_users.drop("userId", axis=1)

    # Open model
    with open(model_filename, "rb") as filehandler:
        model = pickle.load(filehandler)

    # Calculate nearest neighbors
    _, indices = model.kneighbors(filtered_users)

    # Select the n best neighbors
    selection = indices[:, :n_recos]

    # Create a dict: {user_id: [movie_index1, movie_index2, ...]}
    prediction_dict = {
        int(user_id): list(map(int, movie_indices))
        for user_id, movie_indices in zip(original_ids, selection)
    }
    return prediction_dict


if __name__ == "__main__":
    # Take the users Id of the DB
    users = pd.read_csv("data/processed/user_matrix.csv")
    users_id = users["userId"].tolist()

    # Make predictions using `model.pkl`
    predictions = make_predictions(
        users_id, "models/model.pkl", "data/processed/user_matrix.csv"
    )

    # Sauvegarder les prédictions dans un fichier JSON
    os.makedirs("data/prediction", exist_ok=True)
    with open("data/prediction/predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)

    print("✅ Prédictions sauvegardées dans data/predictions.json")

