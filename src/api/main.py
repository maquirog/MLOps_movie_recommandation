import json  # For formatting JSON responses
from fastapi import FastAPI, APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse  # For custom JSON responses
from src.api.models import TrainRequest, PredictionRequest, EvaluateRequest, TrainerExperimentRequest
from dotenv import dotenv_values
import docker
import os
from fastapi import Body

# Initialize FastAPI app
app = FastAPI()
router = APIRouter()

BASE_DIR = os.environ.get("HOST_PROJECT_PATH")

# Initialize Docker client
docker_client = docker.from_env(timeout=300)

def trigger_microservice(service_name: str, command: str = None):
    volume_mode = os.environ.get("VOLUME_MODE", "bind")
    # Decide how to mount volumes based on environment variable
    if volume_mode == "bind":
        # Use host paths for bind mounts
        volumes = {
            os.path.join(BASE_DIR, "src"): {"bind": "/app/src", "mode": "rw"},
            os.path.join(BASE_DIR, "data"): {"bind": "/app/data", "mode": "rw"},
            os.path.join(BASE_DIR, "models"): {"bind": "/app/models", "mode": "rw"},
            os.path.join(BASE_DIR, "metrics"): {"bind": "/app/metrics", "mode": "rw"},
            os.path.join(BASE_DIR, "predictions"): {"bind": "/app/predictions", "mode": "rw"},
            os.path.join(BASE_DIR, "reports"): {"bind": "/app/reports", "mode": "rw"},
            os.path.join(BASE_DIR, "mlruns"): {"bind": "/app/mlruns", "mode": "rw"}
        }
        # Debug printout
        print("Using bind mounts from host for volumes:")
        for host_path, mount_info in volumes.items():
            print(f"  Host: {host_path} -> Container: {mount_info['bind']} (mode: {mount_info['mode']})")
    else:
        # Use named Docker volumes (for CI/production)
        volumes = {
            "shared_src": {"bind": "/app/src", "mode": "rw"},
            "shared_data": {"bind": "/app/data", "mode": "rw"},
            "shared_models": {"bind": "/app/models", "mode": "rw"},
            "shared_metrics": {"bind": "/app/metrics", "mode": "rw"},
            "shared_predictions": {"bind": "/app/predictions", "mode": "rw"},
            "shared_reports": {"bind": "/app/reports", "mode": "rw"},
            "shared_mlruns": {"bind": "/app/mlruns", "mode": "rw"},
            "shared_dvc": {"bind": "/app/.dvc", "mode": "rw"}
        }
        print("Using named Docker volumes for volumes:")
        for vol_name, mount_info in volumes.items():
            print(f"  Volume: {vol_name} -> Container: {mount_info['bind']} (mode: {mount_info['mode']})")

    env_vars = [
        f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}",
        f"MLFLOW_TRACKING_URI={os.environ.get('MLFLOW_TRACKING_URI', '')}",
        f"MLFLOW_EXPERIMENT_NAME={os.environ.get('MLFLOW_EXPERIMENT_NAME', '')}",
        f"API_URL={os.environ.get('API_URL', '')}",
        f"BASE_DIR={os.environ.get('BASE_DIR', '')}",
        f"DATA_DIR={os.environ.get('DATA_DIR', '')}",
        f"MODELS_DIR={os.environ.get('MODELS_DIR', '')}",
        f"METRICS_DIR={os.environ.get('METRICS_DIR', '')}",
        f"MODEL_NAME={os.environ.get('MODEL_NAME', '')}",
        f"METRIC_KEY={os.environ.get('METRIC_KEY', '')}",
    ]
    if service_name == "champion_selector":
        volumes.update({
            os.path.join(BASE_DIR, ".dvc"): {"bind": "/app/.dvc", "mode": "rw"},
            os.path.join(BASE_DIR, ".git"): {"bind": "/app/.git", "mode": "rw"}
        })
        env_vars += [
            f"GITHUB_USERNAME={os.environ.get('GITHUB_USERNAME', '')}",
            f"GITHUB_EMAIL={os.environ.get('GITHUB_EMAIL', '')}",
            f"GITHUB_TOKEN={os.environ.get('GITHUB_TOKEN', '')}",
            f"AWS_ACCESS_KEY_ID={os.environ.get('AWS_ACCESS_KEY_ID', '')}",
            f"AWS_SECRET_ACCESS_KEY={os.environ.get('AWS_SECRET_ACCESS_KEY', '')}"
        ]
        
    # Create and run the container
    container = docker_client.containers.run(
        image=f"maquirog/{service_name}:latest",
        command=command,
        detach=True,
        volumes=volumes,
        working_dir="/app",
        stdout=True,
        stderr=True,
        network="mlops-net", # ou mlops_movie_recommandation ?
        environment=env_vars
    )

    # Wait for the container to finish
    logs = container.logs(follow=True)
    decoded_logs = logs.decode("utf-8")
    print(f"🚀 Logs for {service_name} microservice:\n{decoded_logs}")
    
    result = container.wait()  # Attend la fin et récupère le code de sortie
    exit_code = result.get("StatusCode", 1)

    # Remove the container after it stops
    container.remove()

    
    if exit_code != 0:
        # Ici tu peux lever une exception HTTP pour que FastAPI renvoie 500
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"{service_name} microservice failed with exit code {exit_code}",
                "logs": decoded_logs,
            }
        )
        
    # Beautify the response JSON
    response = {
        "status": "success",
        "message": f"{service_name} microservice completed successfully",
        "logs": decoded_logs,
    }
    return JSONResponse(content=json.loads(json.dumps(response, indent=4)))


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Microservice endpoints
@router.post("/import_raw_data")
def import_raw_data():
    return trigger_microservice("import_raw_data", command="python src/data/import_raw_data.py")

@router.post("/build_features")
def build_features():
    return trigger_microservice("build_features", command="python src/data/build_features.py")

#@router.post("/train")
#def train_model():
#    return trigger_microservice("train", command="python src/models/train.py")


# Training endpoint
@router.post("/train")
def train(request: TrainRequest = Body(default=None)):
    print(f"🎯 Received training request: {request}")
    if not request or request.hyperparams is None:
        hyperparams_arg = ""
    else:
        json_params = json.dumps(request.hyperparams)
        hyperparams_arg = f"--hyperparams_dict '{json_params}'"

    run_id_arg = ""
    if request and request.run_id:
        run_id_arg = f"--run_id {request.run_id}"
        
    command = f"python src/models/train.py {hyperparams_arg} {run_id_arg}"

    return trigger_microservice(
        service_name="train",
        command=command
    )

# Prediction endpoint
@router.post("/predict")
def predict(request: PredictionRequest = Body(default=None)):
    args = []
    if request:
        if request.user_ids:
            user_ids = ",".join(map(str, request.user_ids))
            args.append(f"--user_ids {user_ids}")
        
        if request.n_recommendations:
            args.append(f"--n_recommendations {request.n_recommendations}")
        
        if request.model_source:
            args.append(f"--model_source {request.model_source}")
        
        if request.output_filename:
            args.append(f"--output_filename {request.output_filename}")

    command = f"python src/models/predict.py {' '.join(args)}"
    return trigger_microservice("predict", command=command)
    

@router.post("/evaluate")
def evaluate(request: EvaluateRequest = Body(default=None)):
    args = []

    if request:
        if request.run_id:
            args.append(f"--run_id {request.run_id}")
        if request.input_filename:
            args.append(f"--input_filename {request.input_filename}")
        if request.output_filename:
            args.append(f"--output_filename {request.output_filename}")

    command = f"python src/models/evaluate.py {' '.join(args)}"
    return trigger_microservice("evaluate", command=command)

@router.post("/trainer_experiment")
def run_trainer_experiment(request: TrainerExperimentRequest = Body(...)):
    # Génère un nom d'expérience si non fourni
    experiment_name = request.experiment_name
    if experiment_name is None:
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = f"weekly_experiment_{now}"

    # Prépare l’argument hyperparams_dict si fourni
    if request.hyperparams:
        json_params = json.dumps(request.hyperparams)
    else:
        json_params = ""

    command = f"bash src/experiment_trainer/entrypoint.sh {experiment_name} '{json_params}'"
    
    return trigger_microservice(service_name="trainer_experiment", command=command)

@router.post("/run_champion_selector")
def run_champion_selector():
    return trigger_microservice(
        service_name="champion_selector",
        command="python src/champion_selector/compare_and_promote.py"
    )

@app.get("/prometheus_metrics", response_class=Response)
def get_prometheus_metrics():
    metrics_path = "metrics/champion_scores.json"
    if not os.path.isfile(metrics_path):
        return Response(content="# Metrics file not found\n", media_type="text/plain", status_code=404)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    prom_lines = []
    for key, value in metrics.items():
        metric_name = key.lower().replace("@", "_at_").replace(".", "_").replace("-", "_")
        prom_lines.append(f'{metric_name} {value}')
    prom_str = "\n".join(prom_lines) + "\n"
    return Response(content=prom_str, media_type="text/plain")

app.include_router(router)