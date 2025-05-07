import pandas as pd
import os
from sklearn.preprocessing import normalize
import json

def is_running_in_docker() -> bool:
    """
    Vérifie si le script s'exécute dans un conteneur Docker.
    """
    return os.path.exists("/.dockerenv")

def get_data_dir(default_local="data/raw", default_docker="/app/data/raw"):
    """
    Retourne le chemin du répertoire de données en fonction de l'environnement d'exécution.
    """
    return default_docker if is_running_in_docker() else default_local

def get_output_dir(default_local="data/processed", default_docker="/app/data/processed"):
    """
    Retourne le chemin du répertoire de sortie en fonction de l'environnement d'exécution.
    """
    return default_docker if is_running_in_docker() else default_local

def read_ratings(ratings_csv, data_dir=None) -> pd.DataFrame:
    """
    Reads a ratings.csv from the specified data directory.

    Parameters
    -------
    ratings_csv : str
        The csv file that will be read. Must be corresponding to a rating file.

    data_dir : str, optional
        The directory where the ratings.csv file is located. Defaults to the appropriate directory
        based on the execution environment.

    Returns
    -------
    pd.DataFrame
        The ratings DataFrame. Its columns are, in order:
        "userId", "movieId", "rating" and "timestamp".
    """
    if data_dir is None:
        data_dir = get_data_dir()
    data = pd.read_csv(os.path.join(data_dir, ratings_csv))
    return data

def read_movies(movies_csv, data_dir=None) -> pd.DataFrame:
    """
    Reads a movies.csv from the specified data directory.

    Parameters
    -------
    movies_csv : str
        The csv file that will be read. Must be corresponding to a movie file.

    data_dir : str, optional
        The directory where the movies.csv file is located. Defaults to the appropriate directory
        based on the execution environment.

    Returns
    -------
    pd.DataFrame
        The movies DataFrame. Its columns are binary and represent the movie genres.
    """
    if data_dir is None:
        data_dir = get_data_dir()
    df = pd.read_csv(os.path.join(data_dir, movies_csv))
    genres = df["genres"].str.get_dummies(sep="|")
    result_df = pd.concat([df[["movieId", "title"]], genres], axis=1)
    return result_df

def keep_only_liked_movies(ratings, min_rating=4.0):
    """
    Keeps only the movies with a rating equal or above the threshold.
    """
    liked_ratings = ratings[ratings["rating"] >= min_rating]
    favorite_movies_lists = liked_ratings.groupby("userId")["movieId"].apply(list)
    return liked_ratings, favorite_movies_lists

def filter_users_with_min_likes(ratings, min_liked_movies=5):
    """
    Filters out users who liked fewer than 'min_liked_movies'.
    """
    liked_counts = ratings.groupby("userId").size()
    active_users = liked_counts[liked_counts >= min_liked_movies].index
    return ratings[ratings["userId"].isin(active_users)]

def create_user_matrix(ratings, movies, aggregation="mean"):
    """
    Builds the user profile matrix by averaging genre vectors of liked movies per user.
    """
    merged = ratings.merge(movies, on="movieId", how="inner")
    merged = merged.drop(columns=["movieId", "title", "rating", "timestamp"])
    user_matrix = merged.groupby("userId").agg(aggregation)
    return user_matrix

def normalize_user_matrix(user_matrix):
    """
    Normalize each user's profile vector to have unit norm (L2).
    """
    normalized_user_matrix = pd.DataFrame(
        normalize(user_matrix, norm='l2', axis=1),
        index=user_matrix.index,
        columns=user_matrix.columns
    )
    return normalized_user_matrix

def save_processed_data(movies_df, user_matrix, favorite_movies_lists, output_dir=None):
    """
    Saves the movie genre matrix and user profile matrix to CSV files.
    """
    if output_dir is None:
        output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    movies_df.drop(columns=["title"]).to_csv(os.path.join(output_dir, "movie_matrix.csv"), index=False)
    user_matrix.to_csv(os.path.join(output_dir, "user_matrix.csv"))

    with open(os.path.join(output_dir, "user_favorites.json"), "w") as f:
        json.dump(favorite_movies_lists.to_dict(), f, indent=4)

if __name__ == "__main__":
    # Créer le dossier de sortie s’il n’existe pas
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    # Load raw data
    ratings = read_ratings("ratings.csv")
    movies = read_movies("movies.csv")

    # Filter and preprocess
    liked_ratings, favorite_movies_lists = keep_only_liked_movies(ratings)
    filtered_ratings = filter_users_with_min_likes(liked_ratings)

    # Create user profile matrix (average genre preference)
    user_matrix = normalize_user_matrix(create_user_matrix(filtered_ratings, movies))

    # Save processed data
    save_processed_data(movies, user_matrix, favorite_movies_lists)