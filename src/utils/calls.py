import requests
import subprocess
import json
import sys
import os

API_URL = os.environ.get("API_URL", "http://api:8000")

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
        print(f"❌ Error calling Train API: {e}", flush=True)
        print("Response content:", response.text, flush=True)
        return None


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
        print(f"❌ Erreur lors de l'appel à l'API /predict : {e}", flush=True)
        return None


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
        print(f"❌ Erreur lors de l'appel à l'API /evaluate : {e}", flush=True)
        return None


### === Call en local === ###
def call_train(hyperparams, run_id):
    json_params = json.dumps(hyperparams)
    command = f"python ../models/train.py --hyperparams_dict '{json_params}' --run_id {run_id}"
    subprocess.run(command, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)

def call_predict(run_id=None, output_filename = None):
    print("🧠 Predicting locally...", flush=True)
    command = "python ../models/predict.py"
    if run_id:
        command += f" --model_source runs:/{run_id}/model"
    if output_filename:
        command += f" --output_filename {output_filename}"
    subprocess.run(command, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)

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
    except json.JSONDecodeError:
        print("⚠️ Impossible de parser les métriques retournées.")
        print("Sortie brute :", result.stdout)
        return None
