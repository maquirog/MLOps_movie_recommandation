
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class TrainRequest(BaseModel):
    hyperparams: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dictionnaire d'hyperparamètres, ex: {\"n_neighbors\": 10, \"algorithm\": \"kd_tree\"}"
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Identifiant optionnel de run pour tracer l'entraînement"
    )

class PredictionRequest(BaseModel):
    user_ids: Optional[List[int]] = None  # No default value
    n_recommendations: int = 10

class PredictionResponse(BaseModel):
    predictions: Dict[int, List[int]]
