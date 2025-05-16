import os
import json  # For formatting JSON responses
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import JSONResponse  # For custom JSON responses
from src.api.models import PredictionRequest
from typing import Dict
import docker
import requests

# Initialize FastAPI app
app = FastAPI()
router = APIRouter()

# Initialize Docker client
docker_client = docker.from_env()

# Utility function to trigger a microservice using Docker REST API
def trigger_microservice(service_name: str, command: str = None):
    try:
        # Define shared named volumes for production or container-to-container communication
        shared_volumes = {
            "shared_src": {"bind": "/app/src", "mode": "rw"},
            "shared_data": {"bind": "/app/data", "mode": "rw"},
            "shared_models": {"bind": "/app/models", "mode": "rw"},
            "shared_metrics": {"bind": "/app/metrics", "mode": "rw"},
        }

        # Run the container with detach set to False to capture logs
        logs = docker_client.containers.run(
            image=f"maquirog/{service_name}:latest",
            command=command,  # Override CMD with the provided command
            detach=False,  # Run in the foreground to capture logs
            volumes=shared_volumes,  # Dynamically overwrite volumes
            working_dir="/app",
            stdout=True,
            stderr=True,
        )

        # Logs are returned directly as a bytes object when detach=False
        decoded_logs = logs.decode("utf-8")
        print(f"🚀 Logs for {service_name} microservice:\n{decoded_logs}")

        # Beautify the response JSON
        response = {
            "status": "success",
            "message": f"{service_name} microservice completed successfully",
            "logs": decoded_logs
        }
        return JSONResponse(content=json.loads(json.dumps(response, indent=4)))  # Beautify JSON output
    except docker.errors.DockerException as e:
        print(f"Error while running container: {str(e)}")  # Print error for debugging
        raise HTTPException(status_code=500, detail=f"Failed to trigger {service_name}: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    """
    Minimal API health check.
    """
    return {"status": "ok"}

# Microservice endpoints
@router.post("/import_raw_data")
def import_raw_data():
    # Override the CMD for the import_raw_data service
    return trigger_microservice("import_raw_data", command="python src/data/import_raw_data.py")

@router.post("/build_features")
def build_features():
    # Override the CMD for the build_features service
    return trigger_microservice("build_features", command="python src/data/build_features.py")

@router.post("/train")
def train_model():
    # Override the CMD for the train service
    return trigger_microservice("train", command="python src/models/train.py")

# Prediction endpoint
@router.post("/predict")
def predict(request: PredictionRequest):
    # Check if specific user IDs are provided
    if request.user_ids:
        user_ids = ",".join(map(str, request.user_ids))
        user_ids_arg = f"--user_ids {user_ids}"
    else:
        user_ids_arg = ""  # Default behavior: all users

    # Use the trigger_microservice function to start the predict container
    return trigger_microservice(
        service_name="predict",
        command=f"python src/models/predict.py {user_ids_arg} --n_recommendations {request.n_recommendations} --save_to_file"
    )

@router.post("/evaluate")
def evaluate():
    # Override the CMD for the evaluate service
    return trigger_microservice("evaluate", command="python src/models/evaluate.py")


# Include router in FastAPI app
app.include_router(router)