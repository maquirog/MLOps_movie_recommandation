import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd

class NullModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return [[] for _ in range(len(model_input))]

def register_null_model(model_name: str = "movie_recommender"):
    with mlflow.start_run(run_name="null_model_baseline") as run:
        run_id = run.info.run_id

        # Logger le modèle nul
        mlflow.pyfunc.log_model(
            artifact_path="null_model",
            python_model=NullModel()
        )
        print(f"[MLflow] Modèle nul loggé avec run_id: {run_id}")

        # Log de métriques nulles
        null_metrics = {
            "precision_10": 0.0,
            "recall_10": 0.0,
            "hit_rate_10": 0.0,
            "coverage_10": 0.0,
            "ndcg_10": 0.0
        }
        mlflow.log_metrics(null_metrics)
        
        # ▶️ 4. Ajouter un tag (optionnel mais recommandé)
        mlflow.set_tag("model_type", "null")
        mlflow.set_tag("stage", "initialisation")    
        
        
        # Registry
        client = MlflowClient()

        try:
            client.get_registered_model(name=model_name)
        except:
            client.create_registered_model(name=model_name)
            print(f"[Registry] Registered Model '{model_name}' créé.")

        # Crée une nouvelle version à partir du modèle loggué
        model_version = client.create_model_version(
            name=model_name,
            source=mlflow.get_artifact_uri("null_model"),
            run_id=run_id
        )
        print(f"[Registry] Modèle enregistré: version {model_version.version}")

        # 👉 Promotion vers 'Production'
        client.set_registered_model_alias(
            name=model_name,
            version=model_version.version,
            alias="Champion"
        )
        print(f"[Registry] Version {model_version.version} est maintenant promu '@champion'.")

if __name__ == "__main__":
    register_null_model()
