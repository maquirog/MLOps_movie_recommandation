mlflow server \
  --host 0.0.0.0 \
  --port 8082 \
  --backend-store-uri file://$(pwd)/mlruns \
  --default-artifact-root file://$(pwd)/mlruns \
  --serve-artifacts          