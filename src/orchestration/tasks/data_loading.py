from prefect import task
import pandas as pd

@task
def load_data(red_path: str, white_path: str):
    df_red = pd.read_csv(red_path, sep=";")
    df_white = pd.read_csv(white_path, sep=";")
    df = pd.concat([df_red, df_white])
    return df
