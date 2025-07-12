# src/orchestration/flows/wine_quality_pipeline.py

from prefect import flow
from orchestration.tasks.data_loading import load_data
from orchestration.tasks.preprocessing import preprocess_data
from orchestration.tasks.training import train_logistic_regression
from orchestration.tasks.evaluation import evaluate_model
from orchestration.tasks.mlflow_loggin import log_to_mlflow

@flow
def wine_quality_pipeline(
    red_path="src/data/winequality-red.csv",
    white_path="src/data/winequality-white.csv"
):
    # 1. Load Data
    df = load_data(red_path, white_path)

    # 2. Preprocess
    X_train, X_val, y_train, y_val, dv = preprocess_data(df)

    # 3. Train
    model = train_logistic_regression(X_train, y_train, dv)

    # 4. Evaluate
    acc, auc = evaluate_model(model, X_val, y_val)

    # 5. Log to MLflow
    metrics = {"accuracy": acc, "auc": auc}
    params = {"max_iter": 1000}
    log_to_mlflow(model, metrics, params, model_name="LogisticRegression")

    print(f"Pipeline completed. Accuracy={acc:.4f}, AUC={auc:.4f}") 