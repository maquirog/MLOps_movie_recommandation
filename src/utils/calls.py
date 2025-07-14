import requests
import subprocess
import json
import sys
import os

API_URL = os.environ.get("API_URL", "http://api:8000")
STRICT_MODE = os.environ.get("STRICT_MODE", "false").lower() == "true"

## === Helpers === ##
def _raise_error(message, details=None):
    print(f"❌ {message}", flush=True)
    if details:
        print(details, flush=True)
    raise RuntimeError(message)

## === Call API === ###
def call_train_api(hyperparams, run_id=None):
    payload = {"hyperparams": hyperparams}
    if run_id:
        payload["run_id"] = run_id
    response = requests.post(f"{API_URL}/train", 
                             json=payload,
                             headers={"Content-Type": "application/json"},
                             timeout=300
                             )
    
    try:
        response.raise_for_status()
        print("✅ Train API called successfully.", flush=True)
        return response.json()  # Retourne la réponse JSON du serveur
    except requests.HTTPError as e:
        content = response.text if 'response' in locals() else "Pas de réponse."
        _raise_error("Erreur lors de l'appel à l'API /train", f"{e}\n{content}")


def call_predict_api(model_source=None, output_filename=None):
    payload = {
        "model_source":model_source,
        "output_filename": output_filename
    }

    try:
        print(f"📡 Envoi requête à {API_URL}/predict ...", flush=True)
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300
        )
        response.raise_for_status()  # Raise HTTPError for bad status codes
        data = response.json()
        print("✅ Prédiction lancée avec succès via l'API.", flush=True)
        return data
    except requests.RequestException as e:
        content = response.text if 'response' in locals() else "Pas de réponse."
        _raise_error("Erreur lors de l'appel à l'API /predict", f"{e}\n{content}")



def call_evaluate_api(run_id=None, input_filename=None, output_filename = None):
    payload = {"run_id": run_id}
    if input_filename:
        payload["input_filename"] = input_filename
    if output_filename:
        payload["output_filename"] = output_filename

    try:
        print(f"📡 Envoi requête à {API_URL}/evaluate ...", flush=True)
        response = requests.post(
            f"{API_URL}/evaluate",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300
        )
        response.raise_for_status()
        data = response.json()
        print("✅ Évaluation réussie via l’API.", flush=True)
        return data
    except requests.RequestException as e:
        content = response.text if 'response' in locals() else "Pas de réponse."
        _raise_error("Erreur lors de l'appel à l'API /evaluate", f"{e}\n{content}")



### === Call en local === ###
def call_train(hyperparams, run_id):
    json_params = json.dumps(hyperparams)
    command = f"python ../models/train.py --hyperparams_dict '{json_params}' --run_id {run_id}"
    try:
        subprocess.run(command, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
    except subprocess.CalledProcessError as e:
        _raise_error("Échec du script local de training", f"Commande: {command}")
        
def call_predict(run_id=None, output_filename = None):
    print("🧠 Predicting locally...", flush=True)
    command = "python ../models/predict.py"
    if run_id:
        command += f" --model_source runs:/{run_id}/model"
    if output_filename:
        command += f" --output_filename {output_filename}"
    try:
        subprocess.run(command, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
    except subprocess.CalledProcessError as e:
        _raise_error("Échec du script local de prédiction", f"Commande: {command}")
        
def call_evaluate(run_id=None, input_filename = None):
    print("📊 Evaluating locally...", flush=True)
    command = "python ../models/evaluate.py"
    if run_id:
        command += f" --run_id {run_id}"
        
    if input_filename:
        command += f" --input_filename {input_filename}"

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("⚠️ Evaluate script crashed.")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        return None
    
    first_line = result.stdout.strip().splitlines()[0]

    try:
        metrics = json.loads(first_line)
        print("✅ Metrics récupérées :", metrics)
        return metrics
    except subprocess.CalledProcessError as e:
        _raise_error("Evaluate script crashed", f"stdout: {e.stdout}\nstderr: {e.stderr}")
    except (json.JSONDecodeError, IndexError):
        _raise_error("Impossible de parser les métriques retournées", f"Sortie brute :\n{result.stdout}")