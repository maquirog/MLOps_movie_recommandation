from pydantic import BaseModel
from typing import List, Dict, Optional

class PredictionRequest(BaseModel):
    user_ids: Optional[List[int]] = None  # No default value
    n_recommendations: int = 10

class PredictionResponse(BaseModel):
    predictions: Dict[int, List[int]]