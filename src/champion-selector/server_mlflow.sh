mlflow server \
  --host 0.0.0.0 \
  --port 5050 \
  --backend-store-uri file://$(pwd)/mlruns \
  --default-artifact-root file://$(pwd)/mlruns \
  --serve-artifacts          