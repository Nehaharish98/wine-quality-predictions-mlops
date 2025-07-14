import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
from mlflow.models.signature import infer_signature
import os
import json 

PROCESSED_DATA_PATH = "data/processed/wine_data_processed.csv"

def train_wine_model(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    processed_data_path=PROCESSED_DATA_PATH
):
    # --- MLflow Setup ---
    # Set MLflow tracking URI to a local SQLite database for model registry
    # This creates mlflow.db in your project root if it doesn't exist
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Wine Quality Prediction Experiment")

    if not os.path.exists(processed_data_path):
        print(f"Processed data not found at {processed_data_path}. Running preprocessing...")
        from src.data.preprocess import preprocess_wine_data
        processed_data_path = preprocess_wine_data()


    # --- Load Data ---
    try:
        data = pd.read_csv(processed_data_path)
        print(f"Loaded processed data from: {processed_data_path}")
    except FileNotFoundError:
        print(f"Error: Processed data not found at {processed_data_path}. Please run src/data/preprocess.py first.")
        return

    X = data.drop("quality", axis=1)
    y = data["quality"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    # Stratify ensures proportional representation of quality classes in train/test sets

    # --- MLflow Run Context ---
    with mlflow.start_run(run_name="RandomForest_Wine_Quality") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # --- Log Parameters ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", 0.2)

        # --- Train Model ---
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)

        # --- Make Predictions and Evaluate ---
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        # Use average='weighted' for multi-class classification for comprehensive metrics
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # --- Log Metrics ---
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # --- Log Classification Report as an Artifact ---
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_path = "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=4)
        mlflow.log_artifact(report_path)
        print(f"Classification report saved as artifact: {report_path}")

        # --- Infer Model Signature (Input/Output Schema) ---
        # This helps MLflow understand the model's expected input and output
        # Use X_train as a sample input. Make sure it matches the features your model expects.
        example_input = X_train.sample(1) # Take one sample for signature inference
        signature = infer_signature(example_input, model.predict(example_input))

        # --- Log and Register Model ---
        # 'wine_quality_model' is the artifact path within the MLflow run
        # 'WineQualityClassifier' is the name it will appear under in the Model Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="wine_quality_model",
            signature=signature,
            registered_model_name="WineQualityClassifier" # The name for the Model Registry!
        )
        print(f"Model logged as artifact 'wine_quality_model' in run {run_id}")
        print(f"Model registered in MLflow Model Registry as 'WineQualityClassifier'")

        return model, run_id

if __name__ == "__main__":
    # You can call the function with default parameters or specific ones
    trained_model, run_id = train_wine_model(n_estimators=150, max_depth=15)