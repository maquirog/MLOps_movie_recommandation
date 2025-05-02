from fastapi import FastAPI, HTTPException
from src.api.models import PredictionRequest  # Import du modèle PredictionRequest depuis models.py
from src.models.predict_model import load_user_data, load_model, make_predictions
from typing import Dict, List

# Initialisation de l'application FastAPI
app = FastAPI()

# Paramètres de configuration
MODEL_PATH = "models/model.pkl"
USER_MATRIX_PATH = "data/processed/user_matrix.csv"

# Chargement du modèle au démarrage de l'application
model = load_model(MODEL_PATH)

# Endpoint pour effectuer des prédictions
@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        # Charger les données utilisateur
        user_data = load_user_data(USER_MATRIX_PATH, request.user_ids)
        
        # Vérification si des utilisateurs sont trouvés
        if user_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Aucun utilisateur correspondant aux IDs {request.user_ids} trouvé dans la matrice utilisateur."
            )
        
        # Générer les prédictions
        predictions = make_predictions(model, user_data, request.n_recommendations)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """
    Vérifie l'état de l'API.
    """
    try:
        # Vérifie si le modèle et les données sont chargés
        return {
            "status": "ok",
            "model_loaded": model is not None,
            "user_matrix_available": True  # Vous pouvez ajouter un vérificateur pour le fichier
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du health check: {str(e)}")