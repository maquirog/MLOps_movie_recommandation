import pytest
import requests
import os

API_URL = os.environ.get("API_URL","http://localhost:8000")
API_KEY = os.environ.get("API_KEY","my_secret_api_key")
HEADERS = {"x-api-key": API_KEY}
 
def print_api_logs(resp, step_name=""):
    print(f"\n=== API STEP: {step_name} ===")
    print(f"Status code: {resp.status_code}")
    try:
        data = resp.json()
        # Print logs if available
        if isinstance(data, dict):
            logs = data.get("logs") or data.get("detail") or data
            print("Logs:")
            print(logs)
        else:
            print("Response data:")
            print(data)
    except Exception:
        print("Raw response (non-JSON):")
        print(resp.text)
    print("=== END STEP ===\n")

def test_api_pipeline():

    resp = requests.get(f"{API_URL}/health")
    print_api_logs(resp, "health")
    assert resp.status_code == 200


    resp = requests.post(f"{API_URL}/import_raw_data", headers=HEADERS)
    print_api_logs(resp, "import_raw_data")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


    resp = requests.post(f"{API_URL}/prepare_weekly_dataset", json={"current_week": 1}, headers=HEADERS)
    print_api_logs(resp, "prepare_weekly_dataset")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


    resp = requests.post(f"{API_URL}/build_features", headers=HEADERS)
    print_api_logs(resp, "build_features")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


    resp = requests.post(f"{API_URL}/trainer_experiment", json={"experiment_name": "test_exp"}, headers=HEADERS)
    print_api_logs(resp, "trainer_experiment")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


    resp = requests.post(f"{API_URL}/run_evidently_report", json={"current_week": 1}, headers=HEADERS)
    print_api_logs(resp, "run_evidently_report")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"