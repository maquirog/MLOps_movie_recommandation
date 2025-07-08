from mlflow import MlflowClient
import mlflow
import os
import json
from itertools import product
import yaml
import argparse
from mlflow.exceptions import MlflowException
from src.utils.calls import call_train_api, call_predict_api, call_evaluate_api, call_train, call_predict, call_evaluate

# === 🌍 Variables d'environnement === #
BASE_DIR = os.environ.get(
    "BASE_DIR", 
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
METRICS_DIR = os.environ.get("METRICS_DIR", os.path.join(BASE_DIR, "metrics"))
PREDICT_DIR = os.environ.get("PREDICT_DIR", os.path.join(DATA_DIR, "predictions"))

MODEL_NAME = os.environ.get("MODEL_NAME", "movie_recommender")
METRIC_KEY = os.environ.get("METRIC_KEY", "ndcg_10")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5050")

USE_API = True

if USE_API:
    train_func = call_train_api
    predict_func = call_predict_api
    evaluate_func = call_evaluate_api
else:
    train_func = call_train
    predict_func = call_predict
    evaluate_func = call_evaluate

client = MlflowClient()


def load_hyperparams_grid():
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hyperparameters.yaml')
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    hyperparam_grid = config.get("hyperparameters", {})
    keys = list(hyperparam_grid.keys())
    return list(product(*hyperparam_grid.values())), keys

    
# === MLFlow Run === #

def train_predict_evaluate_log_run(hyperparams, experiment_id,
                                   call_train_func=train_func, call_predict_func=predict_func, call_evaluate_func=evaluate_func):
    with mlflow.start_run() as run:
        active_run = mlflow.active_run().info
        run_id = active_run.run_id
        print(f"🔮🔮 grid search expID: {experiment_id} & run ID:{run_id}🔮🔮", flush=True)
    
        # Paths
        model_source = os.path.join(MODELS_DIR, f"model_{run_id}.pkl")
        predictions_filename = f"predictions_{run_id}.json"
        metrics_filename = os.path.join(METRICS_DIR, f"scores_{run_id}.json")
        
        # === Pipeline calls === #
        call_train_func(hyperparams, run_id)
        call_predict_func(model_source, output_filename = predictions_filename)
        call_evaluate_func(run_id=run_id, input_filename = predictions_filename, output_filename = metrics_filename)
        
        with open(metrics_filename, "r") as f:
            metrics = json.load(f)

        print(f"📈 Run {run_id} terminée avec succès. Metrics : {metrics}", flush=True)
        return run_id, metrics

# === Post-processing === #

def clean_dirs(best_run_id):
    # === Nettoyage prédictions ===
    for f in os.listdir(PREDICT_DIR):
        full_path = os.path.join(PREDICT_DIR, f)
        if f != f"predictions_{best_run_id}.json":
            os.remove(full_path)

    pred_old = os.path.join(PREDICT_DIR, f"predictions_{best_run_id}.json")
    pred_new = os.path.join(PREDICT_DIR, "predictions_challenger.json")
    if os.path.exists(pred_old):
        os.rename(pred_old, pred_new)
    else:
        print(f"⚠️ Fichier {pred_old} introuvable pour renommage.", flush=True)

    # === Nettoyage metrics ===
    if METRICS_DIR:
        for f in os.listdir(METRICS_DIR):
            full_path = os.path.join(METRICS_DIR, f)
            if f != f"scores_{best_run_id}.json":
                os.remove(full_path)

        metric_old = os.path.join(METRICS_DIR, f"scores_{best_run_id}.json")
        metric_new = os.path.join(METRICS_DIR, "challenger_scores.json")
        if os.path.exists(metric_old):
            os.rename(metric_old, metric_new)
        else:
            print(f"⚠️ Fichier {metric_old} introuvable pour renommage.", flush=True)

    # === Nettoyage models ===
    for f in os.listdir(MODELS_DIR):
        full_path = os.path.join(MODELS_DIR, f)
        if f not in [f"model_{best_run_id}.pkl", "model_champion.pkl"]:
            os.remove(full_path)

    model_old = os.path.join(MODELS_DIR, f"model_{best_run_id}.pkl")
    model_new = os.path.join(MODELS_DIR, "model_challenger.pkl")
    if os.path.exists(model_old):
        os.rename(model_old, model_new)
    else:
        print(f"⚠️ Fichier {model_old} introuvable pour renommage.", flush=True)

def register_challenger(run_id, alias="challenger", model_name=MODEL_NAME, metrics_dir = METRICS_DIR):
    print("💾 New Challenger defined, time to register ", flush=True)
    client = MlflowClient()
    
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(model_name)
    
    model_version = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id
    )
    
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags({"model_version": model_version.version, "alias": alias, "model_name": model_name})
    
    print(f"✅ Model {model_name} version {model_version.version} enregistré et aliasé comme '{alias}'", flush=True)
    
    return model_version.version

def set_model_alias(version, alias="challenger", model_name=MODEL_NAME):
    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        version=version,
        alias=alias)
    print(f"[Registry] Model version {version} now aliased as '{alias}'", flush=True)

# === Entrypoint === #

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--hyperparams_dict",
        type=str,
        help="Hyperparameters as JSON string, e.g., '{\"n_neighbors\": 15, \"algorithm\": \"kd_tree\"}'"
        )
    return parser.parse_args()

def main():
    args = parse_args()
    experiment_name = args.experiment_name
    
    # Set l'expérience AVANT de récupérer l'ID ou de commencer les runs
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_name)
    
    #experiment_id = get_experiment_id_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    
    print(f"Experiment id créée: {experiment_name} ({experiment_id})", flush=True)  

    if args.hyperparams_dict is None:
        param_combinations, keys = load_hyperparams_grid()
    else:
        # 🔍 Parse le JSON string en dict
        try:
            hyperparams_dict = json.loads(args.hyperparams_dict)
            if not isinstance(hyperparams_dict, dict):
                raise ValueError("Le JSON ne représente pas un dictionnaire.")
        except Exception as e:
            raise ValueError(f"Erreur de parsing JSON dans --hyperparams_dict: {e}")
        
        # 🧱 Structure pour compatibilité avec le for-loop
        keys = list(hyperparams_dict.keys())
        values = list(hyperparams_dict.values())
        param_combinations = list(product(*values))
        
    best = {METRIC_KEY: -1, "run_id": None, "params": None}


    for values in param_combinations:
        hyperparams = dict(zip(keys, values))
        print(f"🏋️‍♂️ Training with hyperparameters: {hyperparams}", flush=True)
        run_id, metrics = train_predict_evaluate_log_run(hyperparams, experiment_id)
        if metrics.get(METRIC_KEY, 0) > best[METRIC_KEY]:
                best.update({
                    METRIC_KEY: metrics[METRIC_KEY],
                    "run_id": run_id,
                    "params": hyperparams
                })
                


    print("\n=== 🏆 Best Run Summary ===", flush=True)
    print(f"Run ID       : {best['run_id']}", flush=True)
    print(f"Hyperparams  : {best['params']}", flush=True)
    print(f"{METRIC_KEY}  : {best[METRIC_KEY]}", flush=True)
    
    
    clean_dirs(best["run_id"])
    version = register_challenger(best["run_id"])
    set_model_alias(version, "challenger")

if __name__ == "__main__":
    main()