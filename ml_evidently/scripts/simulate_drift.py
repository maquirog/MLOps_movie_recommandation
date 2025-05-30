import pandas as pd
import os

os.makedirs("../reports", exist_ok=True)

movies = pd.read_csv("../../data/raw/movies.csv")
# Simuler un drift : remplacer "Comedy" par "Horror"
drifted = movies.copy()
mask = drifted['genres'].str.contains("Comedy")
drifted.loc[mask, 'genres'] = drifted.loc[mask, 'genres'].str.replace("Comedy", "Horror")
drifted.to_csv("../../data/raw/movies_drift.csv", index=False)
print("movies_drift.csv généré.")