import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import os


def train_model(movie_matrix):
    nbrs = NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(
        movie_matrix.drop("movieId", axis=1)
    )
    return nbrs


if __name__ == "__main__":
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Load the movie matrix
    movie_matrix = pd.read_csv("data/processed/movie_matrix.csv")

    # Train the model
    model = train_model(movie_matrix)

    # Save the model to a file
    filehandler = open("models/model.pkl", "wb")
    pickle.dump(model, filehandler)
    filehandler.close()