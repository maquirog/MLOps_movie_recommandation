from mlflow.tracking import MlflowClient

def get_metrics_by_alias(client, experiment_id, alias):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"params.alias = '{alias}'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    if not runs:
        return None
    run = runs[0]
    return run.data.metrics

def compare_metrics_mlflow(experiment_id, model_name="movie_recommender", metric_key="ndcg_10"):
    client = MlflowClient()

    challenger_metrics = get_metrics_by_alias(client, experiment_id, "challenger")
    champion_metrics = get_metrics_by_alias(client, experiment_id, "champion")

    if challenger_metrics is None:
        print("Pas de métriques challenger trouvées")
        return False
    if champion_metrics is None:
        print("Pas de métriques champion trouvées")
        return True  # Promote challenger by default if no champion

    print(f"Challenger {metric_key}: {challenger_metrics.get(metric_key, 0)}")
    print(f"Champion {metric_key}: {champion_metrics.get(metric_key, 0)}")

    return challenger_metrics.get(metric_key, 0) > champion_metrics.get(metric_key, 0)

# Usage
experiment_id = "0"  # Mets ton experiment id
if compare_metrics_mlflow(experiment_id):
    print("Promouvoir challenger")
else:
    print("Champion reste")
