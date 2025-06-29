import os
import mlflow
import pickle
from mlflow.tracking import MlflowClient

# récupérer les versions des modèles challenger et champion
def get_version_from_alias(model_name: str, alias: str) -> str:
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(name=model_name, alias=alias)
        print(f"Le {alias} est {model_name} v{mv.version}")
        return mv.version
    except Exception as e:
        print(f"Alias '{alias}' introuvable pour le modèle '{model_name}'.")
        return None

# localiser la dernière run_id
def get_latest_run_id_by_model_version(model_version: str, model_name: str = "movie_recommender") -> str:
    client = MlflowClient()
    
    # Recherche toutes les runs liées à ce modèle
    runs = client.search_runs(
        experiment_ids=["0"],  # ou le bon experiment ID si autre que "0"
        filter_string=f"tags.model_name = '{model_name}' and tags.model_version = '{model_version}'",
        order_by=["start_time DESC"],  # tri décroissant sur le temps
        max_results=1
    )
    
    if not runs:
        raise ValueError(f"Aucune run trouvée pour model_version={model_version}")

    return runs[0].info.run_id

def get_metrics_from_run(run_id):
    client = MlflowClient()
    print(client.get_run(run_id).data.metrics)
    return client.get_run(run_id).data.metrics


def compare_metrics_mlflow(challenger_run_id, champion_run_id, metric_key="ndcg_10"):
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

def export_champion_model_as_pkl(model_name, champion_version, output_path="models/champion_model.pkl"):
    model_uri = f"models:/{model_name}/{champion_version}"
    print(f"Chargement du modèle depuis : {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sauvegarde en .pkl
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Modèle exporté en pickle à : {output_path}")

if __name__ == "__main__":
    model_name = "movie_recommender"
    client = MlflowClient()

    # Récupère les versions actuelles selon alias
    challenger_version = get_version_from_alias(model_name, "challenger")
    champion_version = get_version_from_alias(model_name, "champion")
    
    if challenger_version is None or champion_version is None:
        print("Un des alias est introuvable, arrêt du script.")
        exit(1)

    # Récupère la dernière run_id associée à chaque version
    challenger_run_id = get_latest_run_id_by_model_version(str(challenger_version), model_name=model_name)
    champion_run_id = get_latest_run_id_by_model_version(str(champion_version), model_name=model_name)

    # Compare les métriques (exemple sur ndcg_10)
    if compare_metrics_mlflow(challenger_run_id, champion_run_id, metric_key="ndcg_10"):
        print("Promouvoir challenger en champion, champion déprécié")
        # Le challenger devient champion
        client.set_registered_model_alias(name=model_name, alias="champion", version=challenger_version)
        # Supprimer alias challenger
        client.delete_registered_model_alias(name=model_name, alias="challenger")
        #Sauvegarder le nouveau champion
        export_champion_model_as_pkl(model_name, challenger_version)
    else:
        print("Champion reste, challenger déprécié")
        # Supprime l'alias challenger (ou fait ce que tu veux)
        client.delete_registered_model_alias(name=model_name, alias="challenger")
