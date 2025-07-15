import os
from fastapi import Header, HTTPException

def get_api_key(x_api_key: str = Header(...)):
    expected_key = os.environ.get("API_KEY")
    if not expected_key or x_api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key"
        )