from prefect import flow
from orchestration.tasks.data_loading import load_data
from orchestration.tasks.preprocessing import preprocess_data
from orchestration.tasks.training import train_logistic_regression
from orchestration.tasks.evaluation import evaluate_model
from orchestration.tasks.mlflow_loggin import log_to_mlflow


@flow(name="wine-quality-pipeline")
def wine_quality_pipeline(
    red_path="src/data/winequality-red.csv",
    white_path="src/data/winequality-white.csv"
):

    df = load_data(red_path, white_path)

    
    X_train, X_val, y_train, y_val, dv = preprocess_data(df)

    
    model = train_logistic_regression(X_train, y_train, dv)

    
    acc, auc = evaluate_model(model, X_val, y_val)

    
    metrics = {"accuracy": acc, "auc": auc}
    params = {"max_iter": 1000}
    log_to_mlflow(model, metrics, params, model_name="LogisticRegression")

    print(f"✅ Pipeline completed. Accuracy={acc:.4f}, AUC={auc:.4f}")

if __name__ == "__main__":
    wine_quality_pipeline.serve(
        name="monthly_quality_check",
        cron="0 0 1 * *",
        tags=["monthly", "wine-quality"],
        #work_pool_name="my-local-pool",
    )