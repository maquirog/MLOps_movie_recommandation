import os
import sys
import pandas as pd

def main():
    # Accept week as a CLI argument or environment variable
    if len(sys.argv) >= 2:
        week = int(sys.argv[1])
    else:
        week = int(os.environ.get('CURRENT_WEEK', '0'))

    if week < 1:
        print("Week must be >= 1 to compare with previous week")
        sys.exit(1)

    bucket_dir = "data/buckets"
    prev_path = os.path.join(bucket_dir, f"bucket_{week-1}.csv")
    curr_path = os.path.join(bucket_dir, f"bucket_{week}.csv")
    if not os.path.exists(prev_path):
        print(f"Previous week bucket not found: {prev_path}")
        sys.exit(1)
    if not os.path.exists(curr_path):
        print(f"Current week bucket not found: {curr_path}")
        sys.exit(1)

    prev_df = pd.read_csv(prev_path)
    curr_df = pd.read_csv(curr_path)

    # Save to evidently/ for the next stage
    os.makedirs("evidently", exist_ok=True)
    prev_df.to_csv("evidently/last_week.csv", index=False)
    curr_df.to_csv("evidently/current_week.csv", index=False)
    print("Saved evidently/current_week.csv and evidently/last_week.csv for reporting.")

if __name__ == "__main__":
    main()