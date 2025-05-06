from pydantic import BaseModel
from typing import List, Dict

# Modèles Pydantic pour l'API

class PredictionRequest(BaseModel):
    """
    Modèle pour valider les données d'une requête de prédiction.
    """
    user_ids: List[int] = [1, 2, 3]
    n_recommendations: int = 10

class PredictionResponse(BaseModel):
    """
    Modèle pour structurer la réponse d'une prédiction.
    """
    predictions: Dict[int, List[int]]