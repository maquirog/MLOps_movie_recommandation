import os
import sys
import pandas as pd

def prepare_weekly_dataset(current_week, buckets_dir="data/buckets", output_path="data/weekly/current_ratings.csv"):
    dfs = []
    # Concatène les buckets 0 à current_week inclus
    for i in range(current_week + 1):
        bucket_path = os.path.join(buckets_dir, f"bucket_{i}.csv")
        if os.path.exists(bucket_path):
            dfs.append(pd.read_csv(bucket_path))
        else:
            print(f"Warning: {bucket_path} does not exist, skipping.")
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.to_csv(output_path, index=False)
        print(f"Written {output_path} with {len(combined)} rows (buckets 0 to {current_week})")
    else:
        print("No data buckets found, nothing to write.")

if __name__ == "__main__":
    # Récupère le numéro de la semaine courante passé en argument
    if len(sys.argv) < 2:
        print("Usage: python prepare_weekly_dataset.py <current_week>")
        sys.exit(1)
    current_week = int(sys.argv[1])
    prepare_weekly_dataset(current_week)