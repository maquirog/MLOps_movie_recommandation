import os
import mlflow
import pickle
import json
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
PREDICT_DIR = os.path.join(DATA_DIR, "prediction")

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


def predict_evaluate_log_run_champion(latest_champion_run_id, model_version,
                                      call_predict_func=predict_func, call_evaluate_func=evaluate_func):
    predictions_filename="predictions_champion.json"
    model_source = os.path.join(MODELS_DIR,"model_champion.pkl")
    
    # Paths
    model_source = os.path.join(MODELS_DIR,"model_champion.pkl")
    predictions_filename="predictions_champion.json"
    metrics_filename = os.path.join(METRICS_DIR, f"champion_scores.json")
    
    # 1. Prédictions
    call_predict_func(model_source, output_filename = predictions_filename)
    
    # 2. Nouvelle run pour logguer les performances du champion sur le nouveau dataset
    with mlflow.start_run():
        new_run_id = mlflow.active_run().info.run_id
    
        # 3. Évaluation
        call_evaluate_func(run_id=new_run_id, input_filename=predictions_filename, output_filename = metrics_filename)
    
        with open(metrics_filename, "r") as f:
            metrics = json.load(f)
            
        print("\n📊 Recommandation Evaluation Metrics")
        for key, val in metrics.items():
            print(f"{key}: {val}")


        mlflow.set_tags({
            "model_version": model_version, 
            "alias": "champion", 
            "model_name": MODEL_NAME,
            "note": "Re-evaluation on new dataset"})

    return new_run_id
    
def get_metrics_from_run(run_id):
    print(client.get_run(run_id).data.metrics)
    return client.get_run(run_id).data.metrics


def compare_metrics_mlflow(challenger_run_id, champion_run_id, metric_key=METRIC_KEY):
    challenger_metrics = get_metrics_from_run(challenger_run_id)
    champion_metrics = get_metrics_from_run(champion_run_id)

    if challenger_metrics is None:
        print("Pas de métriques challenger trouvées")
        return False
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

def promote_challenger_to_champ(version):
        # Le challenger devient champion
        client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=version)
        # Supprimer alias challenger
        client.delete_registered_model_alias(name=MODEL_NAME, alias="challenger")
        #Sauvegarder le nouveau champion
        print("Print save .pkl localy")
        export_champion_model_as_pkl(MODEL_NAME, challenger_version)

if __name__ == "__main__":
    client = MlflowClient()

    # Récupère les versions actuelles selon alias
    challenger_version = get_version_from_alias(MODEL_NAME, "challenger")
    champion_version = get_version_from_alias(MODEL_NAME, "champion")
    
    if challenger_version is None:
        print("Challenger est introuvable, arrêt du script.")
        exit(1)
    elif champion_version is None:
        print("Pas de champion : promouvoir challenger en champion")
        promote_challenger_to_champ(challenger_version)
    else:
        print("Model challenger et champion trouvés, début de la comparaison...")
        # Lancer une run du champion sur le nouveau dataset
        first_champion_run_id = get_first_run_id_by_model_version(str(champion_version))
        
        new_run_id_champ = predict_evaluate_log_run_champion(first_champion_run_id, champion_version)
        
        # Récupère la run du challenger
        challenger_run_id = get_first_run_id_by_model_version(str(challenger_version))

        # Compare les métriques (exemple sur ndcg_10)
        if compare_metrics_mlflow(challenger_run_id, new_run_id_champ, metric_key=METRIC_KEY):
            print("Promouvoir challenger en champion, champion déprécié")
            promote_challenger_to_champ(challenger_version)
        else:
            print("Champion reste, challenger déprécié")
            # Supprime l'alias challenger (ou fait ce que tu veux)
            client.delete_registered_model_alias(name=MODEL_NAME, alias="challenger")
