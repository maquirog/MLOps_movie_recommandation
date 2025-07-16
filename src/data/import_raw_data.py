import requests
import os
import logging
import pandas as pd 

def import_raw_data(raw_data_relative_path, filenames, bucket_folder_url):
    '''Import filenames from bucket_folder_url into raw_data_relative_path.'''
    os.makedirs(raw_data_relative_path, exist_ok=True)
    for filename in filenames:
        input_file = os.path.join(bucket_folder_url, filename)
        output_file = os.path.join(raw_data_relative_path, filename)
        print(f"[INFO] Downloading {input_file} as {os.path.basename(output_file)}")
        try:
            response = requests.get(input_file)
            if response.status_code == 200:
                with open(output_file, "wb") as f:
                    f.write(response.content)
                print(f"[INFO] File {output_file} created successfully.")
            else:
                print(f"[ERROR] Failed to download {input_file}. Status code: {response.status_code}")
        except Exception as e:
            print(f"[ERROR] An error occurred while downloading {input_file}: {e}")

def split_ratings_into_buckets(ratings_path, output_dir, n_buckets=10):
    """Split ratings.csv into n_buckets saved as bucket_0.csv ... bucket_n.csv"""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(ratings_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle for randomness
    bucket_size = len(df) // n_buckets
    for i in range(n_buckets):
        start = i * bucket_size
        end = (i + 1) * bucket_size if i < n_buckets - 1 else len(df)
        bucket_file = os.path.join(output_dir, f"bucket_{i}.csv")
        df.iloc[start:end].to_csv(bucket_file, index=False)
        print(f"[INFO] Wrote {bucket_file} with {end - start} rows")
    print(f"[INFO] All buckets created in {output_dir}")

def main(raw_data_relative_path="./data/raw",
         filenames=["genome-scores.csv", "genome-tags.csv", "links.csv",
                   "movies.csv", "ratings.csv", "README.txt", "tags.csv"],
         bucket_folder_url="https://mlops-project-db.s3.eu-west-1.amazonaws.com/movie_recommandation/",
         n_buckets=30,
         buckets_dir="./data/buckets"):
    """Download data from AWS S3 to ./data/raw and split ratings.csv into buckets."""
    import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)
    ratings_path = os.path.join(raw_data_relative_path, "ratings.csv")
    if os.path.exists(ratings_path):
        split_ratings_into_buckets(ratings_path, buckets_dir, n_buckets=n_buckets)
    else:
        print(f"[ERROR] ratings.csv not found at {ratings_path}; cannot split into buckets.")
    logger = logging.getLogger(__name__)
    logger.info('Raw data set created and buckets generated successfully.')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()