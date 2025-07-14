# src/data/preprocess.py
import pandas as pd
import os

def preprocess_wine_data(output_dir="data/processed"):
    """
    Loads raw wine quality data, combines it, performs preprocessing,
    including one-hot encoding for categorical features, and saves the processed data.
    """
    red_wine_path = "data/winequality-red.csv"
    white_wine_path = "data/winequality-white.csv"

    if not os.path.exists(red_wine_path) or not os.path.exists(white_wine_path):
        raise FileNotFoundError(f"Raw data not found. Expected: {red_wine_path} and {white_wine_path}")

    red_wine = pd.read_csv(red_wine_path, sep=';')
    white_wine = pd.read_csv(white_wine_path, sep=';')

    # Add a 'type' column to distinguish red/white
    red_wine['type'] = 'red'
    white_wine['type'] = 'white'

    combined_wine = pd.concat([red_wine, white_wine], ignore_index=True)

    # Basic preprocessing: rename columns to be Python-friendly (e.g., replace spaces)
    combined_wine.columns = [col.replace(' ', '_') for col in combined_wine.columns]

    # --- MODIFIED: Apply One-Hot Encoding to 'type' column and convert to int ---
    # Get dummy variables for 'type'
    dummy_cols = pd.get_dummies(combined_wine['type'], prefix='wine_type', drop_first=False)

    # Convert True/False columns to 1/0 integers
    dummy_cols = dummy_cols.astype(int)

    # Drop the original 'type' column and concatenate the new dummy columns
    combined_wine = pd.concat([combined_wine.drop('type', axis=1), dummy_cols], axis=1)
    # --- END MODIFIED ---

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    processed_file_path = os.path.join(output_dir, "wine_data_processed.csv")

    combined_wine.to_csv(processed_file_path, index=False)
    print(f"Processed data saved to: {processed_file_path}")
    return processed_file_path

if __name__ == "__main__":
    preprocess_wine_data()