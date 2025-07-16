import os
import mlflow
import pickle
import json
import subprocess
import logging
from fastapi import HTTPException
from mlflow.tracking import MlflowClient
from src.utils.calls import  call_predict_api, call_evaluate_api, call_predict, call_evaluate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# === 🌍 Variables d'environnement === #
API_URL = os.environ.get("API_URL", "http://api:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "movie_recommender")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5050")
BASE_DIR = os.environ.get("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
METRIC_KEY = os.environ.get("METRIC_KEY", "ndcg_10")


METRICS_DIR = os.environ.get("METRICS_DIR", os.path.join(BASE_DIR, "metrics"))
USE_API = True

# === API or Local Mode === #
predict_func = call_predict_api if USE_API else call_predict
evaluate_func = call_evaluate_api if USE_API else call_evaluate
    
# === Setup MLflow === #
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# === 🔁 MLflow: Registry & Metrics === #

def get_version_from_alias(model_name: str, alias: str) -> str:
    try:
        mv = client.get_model_version_by_alias(name=model_name, alias=alias)
        logger.info(f"Alias '{alias}' correspond à la version {mv.version} du modèle '{model_name}'")
        return mv.version
    except Exception as e:
        logger.warning(f"Alias '{alias}' introuvable pour le modèle '{model_name}'.")
        return None

    
# localiser la dernière run_id
def get_first_run_id_by_model_version(model_version, model_name=MODEL_NAME) -> str:
    # Recherche toutes les runs liées à ce modèle
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.model_name = '{model_name}' and tags.model_version = '{model_version}'",
        order_by=["start_time ASC"],  # tri décroissant sur le temps
        max_results=1
    )  
    if runs.empty:
        raise HTTPException(status_code=404, detail=f"Aucune run trouvée pour model_name={model_name} model_version={model_version}")
    return runs.iloc[0].run_id

def get_metrics_from_run(run_id):
    try:
        metrics = client.get_run(run_id).data.metrics
        logger.info(f"Métriques récupérées pour run_id={run_id}: {metrics}")
        return metrics
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Métriques introuvables pour run_id={run_id}") from e

def compare_metrics(challenger_run_id: str, champion_run_id: str, metric_key: str) -> bool:
    challenger = get_metrics_from_run(challenger_run_id)
    champion = get_metrics_from_run(champion_run_id)

    if not champion:
        logger.warning("Aucune métrique pour le champion, promotion automatique du challenger.")
        return True

    c_val, champ_val = challenger.get(metric_key, 0), champion.get(metric_key, 0)
    logger.info(f"Comparaison métrique {metric_key} : challenger={c_val} vs champion={champ_val}")
    return c_val > champ_val

# === 📈 Prédiction & Évaluation === #

def predict_evaluate_log_champion(model_version, dataset_hash):
    # Paths
    model="model_champion.pkl"
    model_source = os.path.join(MODELS_DIR,model)
    predictions_filename="predictions_champion.json"
    metrics_filename = os.path.join(METRICS_DIR, f"champion_scores.json")
    
    if not os.path.exists(model_source):
        raise HTTPException(status_code=404, detail=f"Fichier modèle champion introuvable : {model_source}")

    
    # Prédictions
    predict_func(model, output_filename = predictions_filename)
    
    # Nouvelle run pour logguer les performances du champion sur le nouveau dataset
    with mlflow.start_run():
        new_run_id = mlflow.active_run().info.run_id
        evaluate_func(run_id=new_run_id, input_filename=predictions_filename, output_filename = metrics_filename)

        if not os.path.exists(metrics_filename):
            raise HTTPException(status_code=404, detail=f"Fichier métriques introuvable : {metrics_filename}")

        with open(metrics_filename, "r") as f:
            metrics = json.load(f)
            
        logger.info(f"Evaluation metrics du champion : {metrics}")

        mlflow.set_tags({
            "model_version": model_version, 
            "alias": "champion", 
            "model_name": MODEL_NAME,
            "dataset_hash_evaluate":dataset_hash,
            "note": "Re-evaluation on new dataset"})

    return new_run_id
    

def export_champion_model_as_pkl(model_name, champion_version, output_dir=MODELS_DIR):
    model_uri = f"models:/{model_name}/{champion_version}"
    model = mlflow.sklearn.load_model(model_uri)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    output_path = os.path.join(MODELS_DIR, f"model_champion.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modèle exporté en pickle à : {output_path}")
    
# === 🛠 Git & DVC === #
def initialise_git():
    username = os.getenv("GITHUB_USERNAME")
    token = os.getenv("GITHUB_TOKEN")
    email = os.getenv("GITHUB_EMAIL")
    
    if not all([username, token, email]):
        raise HTTPException(status_code=500, detail="Informations GitHub manquantes dans les variables d'environnement")
    
    print("🔐 Configuring Git safe.directory...")
    subprocess.run(["git", "config", "--global", "--add", "safe.directory", "/app"], check=True)
    subprocess.run(["git", "config", "--global", "user.name", username], check=True)
    subprocess.run(["git", "config", "--global", "user.email", email], check=True)
    remote_url = f"https://{username}:{token}@github.com/maquirog/MLOps_movie_recommandation.git"
    subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True)
    
def version_dataset_and_get_hash(dataset_path=os.path.join(DATA_DIR, "processed")):
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail=f"Dossier {dataset_path} introuvable.")
    subprocess.run(["dvc", "add", dataset_path], check=True)
    subprocess.run(["git", "add", f"{dataset_path}.dvc"], check=True)
    subprocess.run(["git", "commit", "-m", "Ajout dataset processed"], check=True)
    
    # Récupère le hash DVC du dataset ajouté
    dvc_file = f"{dataset_path}.dvc"
    with open(dvc_file, "r") as f:
        for line in f:
            if "md5:" in line:
                return line.strip().split(":")[1].strip()
    return None


def log_dataset_hash(version: str, run_id: str, dataset_hash: str):
    """Ajoute le hash du dataset aux tags MLflow pour traçabilité"""
    client.set_model_version_tag(MODEL_NAME, version, "dataset_hash", dataset_hash)
    client.set_tag(run_id, "dataset_hash_train", dataset_hash)
    client.set_tag(run_id, "dataset_hash_evaluate", dataset_hash)

# === 🚀 Promotion === #

def promote_challenger(new_champ_version):
    # Update Alias MLFLow
    client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=new_champ_version)
    client.delete_registered_model_alias(name=MODEL_NAME, alias="challenger")
    
    # Update fichier locaux
    print("Save champion_model.pkl localy from registry MLflow")
    export_champion_model_as_pkl(MODEL_NAME, new_champ_version)
    
    challenger_metrics = os.path.join(METRICS_DIR, "challenger_scores.json")
    champion_metrics = os.path.join(METRICS_DIR, "champion_scores.json")
    if os.path.exists(challenger_metrics):
        os.rename(challenger_metrics, champion_metrics)
        print(f"✅ Renommé {challenger_metrics} → {champion_metrics}")
    else:
        raise HTTPException(status_code=404, detail=f"Fichier {challenger_metrics} introuvable, rien à renommer.")
    
    # DVC add + Git add
    print("Ajout des fichiers au suivi DVC et Git...")

    model_dvc_file = "models/model_champion.pkl.dvc"
    subprocess.run(["dvc", "add", "models/model_champion.pkl"], check=True)
    subprocess.run(["git", "add", model_dvc_file, champion_metrics], check=True)
    subprocess.run(["git", "commit", "-m", "Promote new champion model and metrics"], check=True)
    
    print("Push DVC et Git...")
    subprocess.run(["dvc", "push"], check=True)
    subprocess.run(["git", "push", "origin", "HEAD"], check=True)

    print("Promotion effectuée avec versionnement DVC/Git.")

def main():
        # === Récupération des versions === #
    challenger_version = get_version_from_alias(MODEL_NAME, "challenger")
    champion_version = get_version_from_alias(MODEL_NAME, "champion")
    
    if challenger_version is None:
        raise HTTPException(status_code=404, detail="Aucun challenger à promouvoir.")
    
    # === Initialisation Git/DVC ===
    initialise_git()
    dataset_hash = version_dataset_and_get_hash()
    
    # === Run ID challenger ===
    challenger_run_id = get_first_run_id_by_model_version(str(challenger_version))
    log_dataset_hash(challenger_version, challenger_run_id, dataset_hash)
    
    if champion_version is None:
        logger.info("Aucun champion existant, promotion directe du challenger.")
        promote_challenger(challenger_version)
        return True
    
    # === Évaluation du champion sur le nouveau dataset === #
    logger.info("Champion détecté, évaluation en cours...")
    champion_run_id = predict_evaluate_log_champion(champion_version, dataset_hash)
    
    # === Comparaison et décision ===
    is_better = compare_metrics(challenger_run_id, champion_run_id, METRIC_KEY)
    if is_better:
        logger.info("Challenger meilleur, promotion en cours...")
        promote_challenger(challenger_version)
        return True
    else:
        logger.info("Champion conservé, suppression de l'alias challenger.")
        client.delete_registered_model_alias(MODEL_NAME, "challenger")
        return False


if __name__ == "__main__":
    try:
        promoted = main()
        logger.info(f"Processus terminé. Promotion effectuée : {promoted}")
    except HTTPException as e:
        logger.error(f"Erreur HTTP: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Erreur critique : {str(e)}")
        raise