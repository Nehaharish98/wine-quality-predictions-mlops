from prefect import flow, task, get_run_logger
from src.data.load_data import load_data
from src.features.preprocessing import preprocess_data
from src.models.training import train_model
from src.models.evaluation import evaluate_model

@task
def load(path):
    logger = get_run_logger()
    logger.info(f"Loading data from {path}")
    return load_data(path)

@task
def preprocess(df):
    logger = get_run_logger()
    logger.info("Preprocessing data")
    return preprocess_data(df)

@task
def train(X, y):
    logger = get_run_logger()
    logger.info("Training model")
    return train_model(X, y)

@task
def evaluate(model, X_val, y_val):
    logger = get_run_logger()
    logger.info("Evaluating model")
    return evaluate_model(model, X_val, y_val)

@flow(name="Wine Quality ML Pipeline")
def wine_quality_pipeline(train_path: str, val_path: str):
    train_df = load.submit(train_path)
    train_df_processed = preprocess.submit(train_df)

    X_train = train_df_processed.result().drop("target", axis=1)
    y_train = train_df_processed.result()["target"]

    model = train.submit(X_train, y_train)

    val_df = load.submit(val_path)
    val_df_processed = preprocess.submit(val_df)

    X_val = val_df_processed.result().drop("target", axis=1)
    y_val = val_df_processed.result()["target"]

    evaluate.submit(model, X_val, y_val)

if __name__ == "__main__":
    wine_quality_pipeline(
        train_path="data/train.csv",
        val_path="data/val.csv"
    )
