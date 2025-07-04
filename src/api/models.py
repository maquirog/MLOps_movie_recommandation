
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
    model_source: Optional[str] = None
    output_filename: Optional[str] = None
    
class EvaluateRequest(BaseModel):
    run_id: Optional[str] = None
    input_filename: Optional[str] = None
    output_filename: Optional[str] = None

class PredictionResponse(BaseModel):
    predictions: Dict[int, List[int]]
    
class TrainerExperimentRequest(BaseModel):
    experiment_name: Optional[str] = None  
    hyperparams: Optional[Dict] = None    
