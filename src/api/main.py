import json  # For formatting JSON responses
from fastapi import FastAPI, APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse  # For custom JSON responses
from src.api.models import PredictionRequest
import docker
import os
from fastapi import Body


# Initialize FastAPI app
app = FastAPI()
router = APIRouter()

# Initialize Docker client
docker_client = docker.from_env(timeout=300)

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

        # Create and run the container
        container = docker_client.containers.run(
            image=f"maquirog/{service_name}:latest",
            command=command,
            detach=True,  # Run in the background to get a container object
            volumes=shared_volumes,
            working_dir="/app",
            stdout=True,
            stderr=True,
        )

        # Wait for the container to finish
        logs = container.logs(follow=True)
        decoded_logs = logs.decode("utf-8")
        print(f"🚀 Logs for {service_name} microservice:\n{decoded_logs}")

        # Remove the container after it stops
        container.remove()

        # Beautify the response JSON
        response = {
            "status": "success",
            "message": f"{service_name} microservice completed successfully",
            "logs": decoded_logs,
        }
        return JSONResponse(content=json.loads(json.dumps(response, indent=4)))
    except docker.errors.DockerException as e:
        print(f"Error while running container: {str(e)}")
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
def predict(request: PredictionRequest = Body(default=None)):
    # Set default behavior if no body is provided
    if not request:
        user_ids_arg = ""  # Default behavior: process all users
        n_recommendations = 10
    else:
        # Check if specific user IDs are provided
        if request.user_ids:
            user_ids = ",".join(map(str, request.user_ids))
            user_ids_arg = f"--user_ids {user_ids}"
        else:
            user_ids_arg = ""  # Default behavior: all users
        n_recommendations = request.n_recommendations

    # Use the trigger_microservice function to start the predict container
    return trigger_microservice(
        service_name="predict",
        command=f"python src/models/predict.py {user_ids_arg} --n_recommendations {n_recommendations}"
    )

@router.post("/evaluate")
def evaluate():
    # Override the CMD for the evaluate service
    return trigger_microservice("evaluate", command="python src/models/evaluate.py")

@app.get("/prometheus_metrics", response_class=Response)
def get_prometheus_metrics():
    metrics_path = "metrics/scores.json"
    if not os.path.isfile(metrics_path):
        return Response(content="# Metrics file not found\n", media_type="text/plain", status_code=404)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    # Convert metrics dict to Prometheus format
    prom_lines = []
    for key, value in metrics.items():
        metric_name = key.lower().replace("@", "_at_").replace(".", "_").replace("-", "_")
        prom_lines.append(f'{metric_name} {value}')
    prom_str = "\n".join(prom_lines) + "\n"
    return Response(content=prom_str, media_type="text/plain")

# Include router in FastAPI app
app.include_router(router)