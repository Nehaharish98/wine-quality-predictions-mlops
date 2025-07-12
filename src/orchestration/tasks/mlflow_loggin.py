from prefect import task
import mlflow
import mlflow.sklearn

@task
def log_to_mlflow(model, metrics, params, model_name):
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_experiment("quality-predictions")

    with mlflow.start_run():
        mlflow.set_tag("developer", "Neha")
        mlflow.set_tag("model", model_name)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, artifact_path=model_name)

        print("Logged model to MLflow")