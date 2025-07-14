import os
import mlflow
import pickle
import json
import subprocess
from fastapi import HTTPException
from mlflow.tracking import MlflowClient
from src.utils.calls import  call_predict_api, call_evaluate_api, call_predict, call_evaluate

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
if USE_API:
    predict_func = call_predict_api
    evaluate_func = call_evaluate_api
else:
    predict_func = call_predict
    evaluate_func = call_evaluate

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

def get_version_from_alias(model_name: str, alias: str) -> str:
    try:
        mv = client.get_model_version_by_alias(name=model_name, alias=alias)
        print(f"Le {alias} est {model_name} v{mv.version}")
        return mv.version
    except Exception as e:
        print(f"Alias '{alias}' introuvable pour le modèle '{model_name}'.")
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
        raise ValueError(f"Aucune run trouvée pour model_name={model_name} model_version={model_version}")
    
    return runs.iloc[0].run_id

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


def predict_evaluate_log_run_champion(model_version, dataset_hash,
                                      call_predict_func=predict_func, 
                                      call_evaluate_func=evaluate_func):
    # Paths
    model_source = os.path.join(MODELS_DIR,"model_champion.pkl")
    predictions_filename="predictions_champion.json"
    metrics_filename = os.path.join(METRICS_DIR, f"champion_scores.json")
    
    if not os.path.exists(model_source):
        raise HTTPException(status_code=404, detail=f"Fichier modèle champion introuvable : {model_source}")

    
    # Prédictions
    call_predict_func(model_source, output_filename = predictions_filename)
    
    # Nouvelle run pour logguer les performances du champion sur le nouveau dataset
    with mlflow.start_run():
        new_run_id = mlflow.active_run().info.run_id
        call_evaluate_func(run_id=new_run_id, input_filename=predictions_filename, output_filename = metrics_filename)

        if not os.path.exists(metrics_filename):
            raise HTTPException(status_code=404, detail=f"Fichier métriques introuvable : {metrics_filename}")

        with open(metrics_filename, "r") as f:
            metrics = json.load(f)
            
        print("\n📊 Recommandation Evaluation Metrics")
        for key, val in metrics.items():
            print(f"{key}: {val}")


        mlflow.set_tags({
            "model_version": model_version, 
            "alias": "champion", 
            "model_name": MODEL_NAME,
            "note": "Re-evaluation on new dataset",
            "dataset_hash_evaluate": dataset_hash})

    return new_run_id
    
def get_metrics_from_run(run_id):
    try:
        metrics = client.get_run(run_id).data.metrics
        print(metrics)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Métriques introuvables pour run_id={run_id}") from e


def compare_metrics_mlflow(challenger_run_id, champion_run_id, metric_key=METRIC_KEY):
    challenger_metrics = get_metrics_from_run(challenger_run_id)
    champion_metrics = get_metrics_from_run(champion_run_id)

    if challenger_metrics is None:
        raise HTTPException(status_code=404, detail="Pas de métriques challenger trouvées")
    if champion_metrics is None:
        print("Pas de métriques champion trouvées")
        return True  # Promote challenger by default if no champion

    print(f"Challenger {metric_key}: {challenger_metrics.get(metric_key, 0)}")
    print(f"Champion {metric_key}: {champion_metrics.get(metric_key, 0)}")

    return challenger_metrics.get(metric_key, 0) > champion_metrics.get(metric_key, 0)

def export_champion_model_as_pkl(model_name, champion_version, output_dir=MODELS_DIR):
    model_uri = f"models:/{model_name}/{champion_version}"
    print(f"Chargement du modèle depuis : {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    
    output_path = os.path.join(MODELS_DIR, f"model_champion.pkl")
    
    # Sauvegarde en .pkl
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Modèle exporté en pickle à : {output_path}")
    
def get_hyperparams_from_run(run_id):
    try:
        run = client.get_run(run_id)
        return run.data.params
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Hyperparamètres introuvables pour run_id={run_id}") from e
    
def version_champion_model(champion_version, dataset_hash, metrics, hyperparams=None):
    """
    Crée un dossier versionné (model_vX/) avec modèle + métadonnées, DVC tracked.
    """
    version_str = f"model_v{champion_version}"
    version_dir = os.path.join(MODELS_DIR, "models_versions", version_str)
    os.makedirs(version_dir, exist_ok=True)

    # 1. Copie du modèle
    src_model_path = os.path.join(MODELS_DIR, "model_champion.pkl")
    dst_model_path = os.path.join(version_dir, "model.pkl")
    print(f"DEBUG: Copying {src_model_path} → {dst_model_path}")
    subprocess.run(["cp", src_model_path, dst_model_path], check=True)

    # 2. Métadonnées
    metadata = {
        "model_version": champion_version,
        "dataset_hash": dataset_hash,
        "metric_key": METRIC_KEY,
        "metric_value": metrics.get(METRIC_KEY),
        # "week": datetime.utcnow().isoformat(),
        "hyperparameters": hyperparams or "unknown"
    }
    metadata_path = os.path.join(version_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # 3. DVC tracking
    subprocess.run(["dvc", "add", version_dir], check=True)
    subprocess.run(["git", "add", f"{version_dir}.dvc"], check=True)
    subprocess.run(["dvc", "add", src_model_path], check=True)
    subprocess.run(["git", "add", f"{src_model_path}.dvc"], check=True)
    subprocess.run(["git", "commit", "-m", f"Ajout modèle promu {version_str}"], check=True)

    

def promote_challenger_to_champ(new_champ_version, dataset_hash):
    # Update Alias MLFLow
    client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=new_champ_version)
    client.delete_registered_model_alias(name=MODEL_NAME, alias="challenger")
    
    # Update fichier locaux
    print("Save champion_model.pkl localy from registry MLflow")
    export_champion_model_as_pkl(MODEL_NAME, new_champ_version)
    
    # Renomer les scores
    challenger_metrics = os.path.join(METRICS_DIR, "challenger_scores.json")
    champion_metrics = os.path.join(METRICS_DIR, "champion_scores.json")
    if os.path.exists(challenger_metrics):
        os.rename(challenger_metrics, champion_metrics)
        print(f"✅ Renommé {challenger_metrics} → {champion_metrics}")
    else:
        raise HTTPException(status_code=404, detail=f"Fichier {challenger_metrics} introuvable, rien à renommer.")
    
    # Versionning models sur dvc
    challenger_run_id = get_first_run_id_by_model_version(str(new_champ_version))
    hyperparams = get_hyperparams_from_run(challenger_run_id)
    with open(os.path.join(METRICS_DIR, "champion_scores.json")) as f:
        metrics = json.load(f)
    version_champion_model(
        champion_version=new_champ_version,
        dataset_hash=dataset_hash,
        metrics=metrics,
        hyperparams=hyperparams  # ou passe ton dict de params ici
    )

    # Supprimer l'ancien modèle challenger local s’il existe
    challenger_model_path = os.path.join(MODELS_DIR, "model_challenger.pkl")
    if os.path.exists(challenger_model_path):
        os.remove(challenger_model_path)
        print(f"🧹 Supprimé : {challenger_model_path}")
    else:
        print(f"ℹ️ Aucun fichier modèle challenger à supprimer : {challenger_model_path}")
        
    print("Promotion effectuée avec versionnement DVC/Git.")

if __name__ == "__main__":
    try:
        client = MlflowClient()

        # Récupère les versions actuelles selon alias
        challenger_version = get_version_from_alias(MODEL_NAME, "challenger")
        champion_version = get_version_from_alias(MODEL_NAME, "champion")
        
        if challenger_version is None:
            raise Exception("Challenger est introuvable, arrêt du script.")
        else:
            # Récupère la run du challenger
            print("Versionning du dataset sur DVC")
            challenger_run_id = get_first_run_id_by_model_version(str(challenger_version))
            initialise_git()
            dataset_hash = version_dataset_and_get_hash()
            # Tag sur la run du challenger
            client.set_tag(challenger_run_id, "dataset_hash_train", dataset_hash)
            # Tag sur la version du modèle (registry)
            client.set_model_version_tag(
                name=MODEL_NAME,
                version=challenger_version,
                key="dataset_hash",
                value=dataset_hash
            )
            
            if champion_version is None:
                print("Pas de champion : promouvoir challenger en champion")
                promote_challenger_to_champ(new_champ_version=challenger_version, dataset_hash=dataset_hash)
            else:
                print("Model challenger et champion trouvés, début de la comparaison...")
                # Lancer une run du champion sur le nouveau dataset
                first_champion_run_id = get_first_run_id_by_model_version(str(champion_version))
                
                new_run_id_champ = predict_evaluate_log_run_champion(champion_version, dataset_hash=dataset_hash)

                # Compare les métriques (exemple sur ndcg_10)
                if compare_metrics_mlflow(challenger_run_id, new_run_id_champ, metric_key=METRIC_KEY):
                    print("Promouvoir challenger en champion, champion déprécié")
                    promote_challenger_to_champ(new_champ_version=challenger_version, dataset_hash=dataset_hash)
                else:
                    print("Champion reste, challenger déprécié")
                    # Supprime l'alias challenger (ou fait ce que tu veux)
                    client.delete_registered_model_alias(name=MODEL_NAME, alias="challenger")
                    #### update possible: suppression du fichier model_challenger.pkl
            
            try:
                print("Push DVC et Git...")
                subprocess.run(["dvc", "push"], check=True)
                subprocess.run(["git", "push", "origin", "HEAD"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Erreur lors du push DVC/Git : {e}")

        
    except HTTPException as http_exc:
        # Dans un contexte API, tu gérerais cette exception en la renvoyant au client
        print(f"HTTPException levée : {http_exc.detail}")
        exit(1)
    except Exception as e:
        print(f"Erreur critique : {str(e)}")
        exit(1)