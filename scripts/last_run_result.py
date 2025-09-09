import mlflow

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("GaussGAN-manual")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attribute.start_time desc"]
)

last_run = runs[0]
print("Metrics for run", last_run.info.run_id)
for metric_name, metric_value in last_run.data.metrics.items():
    print(f"{metric_name}: {metric_value}")