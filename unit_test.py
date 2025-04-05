import pytest
import pandas as pd
import os
from src.features.build_features import (
    read_ratings,
    read_movies,
    keep_only_liked_movies,
    filter_users_with_min_likes,
    create_user_matrix,
    normalize_user_matrix,
    save_processed_data,
)
from src.data.import_raw_data import import_raw_data, main

import pickle
from sklearn.neighbors import NearestNeighbors
from src.models.train_model import train_model
from unittest.mock import patch

from src.models.predict_model import make_predictions

import numpy as np


def test_import_raw_data(tmp_path, requests_mock):
    # Mock the bucket folder URL and filenames
    bucket_folder_url = "https://mock-bucket.s3.amazonaws.com/"
    filenames = ["file1.txt", "file2.txt"]
    raw_data_relative_path = tmp_path / "data" / "raw"
    os.makedirs(raw_data_relative_path)

    # Mock the responses for each file
    for filename in filenames:
        file_url = os.path.join(bucket_folder_url, filename)
        requests_mock.get(file_url, text=f"Content of {filename}")

    # Call the function
    import_raw_data(raw_data_relative_path=str(raw_data_relative_path), 
                    filenames=filenames, 
                    bucket_folder_url=bucket_folder_url)

    # Assert that the files were downloaded correctly
    for filename in filenames:
        file_path = raw_data_relative_path / filename
        assert file_path.exists()
        with open(file_path, "r") as f:
            content = f.read()
            assert content == f"Content of {filename}"


@patch("builtins.input", side_effect=["y"])  # Simulate user input "y" (overwrite)
def test_import_raw_data_file_exists(mock_input, tmp_path, requests_mock):
    # Mock the bucket folder URL and filenames
    bucket_folder_url = "https://mock-bucket.s3.amazonaws.com/"
    filenames = ["file1.txt"]
    raw_data_relative_path = tmp_path / "data" / "raw"
    os.makedirs(raw_data_relative_path)

    # Create a file that already exists
    existing_file = raw_data_relative_path / filenames[0]
    with open(existing_file, "w") as f:
        f.write("Existing content")

    # Mock the response for the file
    file_url = os.path.join(bucket_folder_url, filenames[0])
    requests_mock.get(file_url, text="New content")

    # Call the function
    import_raw_data(raw_data_relative_path=str(raw_data_relative_path), 
                    filenames=filenames, 
                    bucket_folder_url=bucket_folder_url)

    # Assert that the existing file was overwritten
    with open(existing_file, "r") as f:
        content = f.read()
        assert content == "New content"  # The content should now be updated

def test_main(tmp_path, requests_mock):
    # Mock the bucket folder URL and filenames
    bucket_folder_url = "https://mock-bucket.s3.amazonaws.com/"
    filenames = ["file1.txt", "file2.txt"]
    raw_data_relative_path = tmp_path / "data" / "raw"
    os.makedirs(raw_data_relative_path)

    # Mock the responses for each file
    for filename in filenames:
        file_url = os.path.join(bucket_folder_url, filename)
        requests_mock.get(file_url, text=f"Content of {filename}")

    # Call the main function with the mocked parameters
    main(
        raw_data_relative_path=str(raw_data_relative_path),
        filenames=filenames,
        bucket_folder_url=bucket_folder_url,
    )

    # Assert that the files were downloaded correctly
    for filename in filenames:
        file_path = raw_data_relative_path / filename
        assert file_path.exists()
        with open(file_path, "r") as f:
            content = f.read()
            assert content == f"Content of {filename}"


def test_read_ratings(tmp_path):
    test_csv = tmp_path / "ratings.csv"
    test_csv.write_text("userId,movieId,rating,timestamp\n1,101,4.5,964982703\n2,102,3.0,964982703")
    ret = read_ratings(ratings_csv=test_csv.name, data_dir=tmp_path)
    assert isinstance(ret, pd.DataFrame)
    assert list(ret.columns) == ["userId", "movieId", "rating", "timestamp"]
    assert len(ret) == 2

def test_read_movies(tmp_path):
    test_csv = tmp_path / "movies.csv"
    test_csv.write_text("movieId,title,genres\n101,Movie1,Action|Comedy\n102,Movie2,Drama")
    ret = read_movies(movies_csv=test_csv.name, data_dir=tmp_path)
    assert isinstance(ret, pd.DataFrame)
    assert "Action" in ret.columns
    assert "Drama" in ret.columns
    assert len(ret) == 2

def test_keep_only_liked_movies():
    ratings = pd.DataFrame({
        "userId": [1, 1, 2],
        "movieId": [101, 102, 103],
        "rating": [4.5, 3.0, 5.0],
    })
    liked_ratings, favorite_movies_lists = keep_only_liked_movies(ratings, min_rating=4.0)
    assert len(liked_ratings) == 2
    assert favorite_movies_lists[1] == [101]
    assert favorite_movies_lists[2] == [103]

def test_filter_users_with_min_likes():
    ratings = pd.DataFrame({
        "userId": [1, 1, 2, 2, 2, 3],
        "movieId": [101, 102, 103, 104, 105, 106],
        "rating": [4.5, 3.0, 5.0, 4.0, 4.0, 5.0],
    })
    filtered_ratings = filter_users_with_min_likes(ratings, min_liked_movies=2)
    assert len(filtered_ratings["userId"].unique()) == 2
    assert 3 not in filtered_ratings["userId"].values

def test_create_user_matrix():
    ratings = pd.DataFrame({
        "userId": [1, 1, 2],
        "movieId": [101, 102, 103],
        "rating": [4.5, 3.0, 5.0],
        "timestamp": [964982703, 964982704, 964982705],  # Add timestamp column
    })
    movies = pd.DataFrame({
        "movieId": [101, 102, 103],
        "title": ["Movie1", "Movie2", "Movie3"],
        "Action": [1, 0, 0],
        "Comedy": [0, 1, 0],
        "Drama": [0, 0, 1],
    })
    user_matrix = create_user_matrix(ratings, movies, aggregation="mean")
    assert user_matrix.shape == (2, 3)
    assert "Action" in user_matrix.columns

def test_normalize_user_matrix():
    user_matrix = pd.DataFrame({
        "Action": [3, 0],
        "Comedy": [4, 0],
        "Drama": [0, 5],
    }, index=[1, 2])
    normalized = normalize_user_matrix(user_matrix)
    assert pytest.approx(normalized.loc[1, "Action"]) == 0.6
    assert pytest.approx(normalized.loc[1, "Comedy"]) == 0.8
    assert pytest.approx(normalized.loc[2, "Drama"]) == 1.0

def test_save_processed_data(tmp_path):
    movies_df = pd.DataFrame({
        "movieId": [101, 102],
        "title": ["Movie1", "Movie2"],
        "Action": [1, 0],
        "Comedy": [0, 1],
    })
    user_matrix = pd.DataFrame({
        "Action": [0.6, 0.0],
        "Comedy": [0.8, 1.0],
    }, index=[1, 2])
    # Ensure favorite_movies_lists is a pandas.Series
    favorite_movies_lists = pd.Series({1: [101], 2: [102]})
    save_processed_data(movies_df, user_matrix, favorite_movies_lists, output_dir=tmp_path)
    assert os.path.exists(tmp_path / "movie_matrix.csv")
    assert os.path.exists(tmp_path / "user_matrix.csv")
    assert os.path.exists(tmp_path / "user_favorites.json")


def test_train_model():
    # Create a mock movie matrix
    movie_matrix = pd.DataFrame({
        "movieId": [1, 2, 3],
        "Action": [1, 0, 0],
        "Comedy": [0, 1, 0],
        "Drama": [0, 0, 1],
    })

    # Train the model
    model = train_model(movie_matrix)

    # Assert that the model is an instance of NearestNeighbors
    assert isinstance(model, NearestNeighbors)

    # Assert that the model has been fitted
    assert hasattr(model, "n_samples_fit_")  # This attribute exists after fitting


def test_model_saving(tmp_path):
    # Create a mock movie matrix
    movie_matrix = pd.DataFrame({
        "movieId": [1, 2, 3],
        "Action": [1, 0, 0],
        "Comedy": [0, 1, 0],
        "Drama": [0, 0, 1],
    })

    # Train the model
    model = train_model(movie_matrix)

    # Save the model to a temporary file
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as filehandler:
        pickle.dump(model, filehandler)

    # Assert that the file exists
    assert model_path.exists()

    # Load the model back and verify it
    with open(model_path, "rb") as filehandler:
        loaded_model = pickle.load(filehandler)

    # Assert that the loaded model is the same as the original model
    assert isinstance(loaded_model, NearestNeighbors)
    assert loaded_model.n_neighbors == model.n_neighbors


def test_make_predictions(tmp_path):
    # Create a larger mock user matrix with more samples
    user_matrix = pd.DataFrame({
        "userId": range(1, 21),  # 20 users
        "Action": np.random.rand(20),
        "Comedy": np.random.rand(20),
        "Drama": np.random.rand(20),
    })

    # Save the mock user matrix to a temporary file
    user_matrix_path = tmp_path / "user_matrix.csv"
    user_matrix.to_csv(user_matrix_path, index=False)

    # Create a mock model with 10 neighbors
    model = NearestNeighbors(n_neighbors=10, algorithm="ball_tree").fit(
        user_matrix.drop("userId", axis=1)
    )

    # Save the mock model to a temporary file
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as filehandler:
        pickle.dump(model, filehandler)

    # Call the make_predictions function
    users_id = [1, 2, 3]
    predictions = make_predictions(users_id, model_path, user_matrix_path)

    # Assert the predictions are a numpy array
    assert isinstance(predictions, np.ndarray)

    # Assert the shape of the predictions (3 users, 10 recommendations each)
    assert predictions.shape == (3, 10)

    # Assert that the predictions contain valid indices
    for row in predictions:
        assert all(idx in range(len(user_matrix)) for idx in row)

    # Assert that the predictions contain valid indices
    for row in predictions:
        assert all(idx in range(len(user_matrix)) for idx in row)

