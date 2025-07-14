import mlflow.pyfunc
from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

model_name = "WineQualityClassifier"
model_alias = "Production"

model = None 
try:
    os.environ["MLFLOW_TRACKING_URI"] = "http://host.docker.internal:5002"
    model = mlflow.pyfunc.load_model(f"models:/{model_name}@{model_alias}")
    print(f"Model '{model_name}' (alias: {model_alias}) loaded successfully.")
except Exception as e:
    print(f"Error loading model from MLflow Registry: {e}")
    print("Please ensure:")
    print(f"1. A model named '{model_name}' is registered.")
    print(f"2. A version of '{model_name}' has the alias '{model_alias}' assigned in MLflow UI.")
    print(f"3. MLflow tracking server is running at {os.getenv('MLFLOW_TRACKING_URI')}.")
    model = None


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    try:
        raw_data = request.get_json(force=True)

        expected_features = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
            'pH', 'sulphates', 'alcohol', 'wine_type_red', 'wine_type_white'
        ]

        if not isinstance(raw_data, list):
            raw_data = [raw_data]

        input_df = pd.DataFrame(raw_data)

    
        if 'type' in input_df.columns:
            type_dummies = pd.get_dummies(input_df['type'], prefix='wine_type')
            if 'wine_type_red' not in type_dummies.columns:
                type_dummies['wine_type_red'] = 0
            if 'wine_type_white' not in type_dummies.columns:
                type_dummies['wine_type_white'] = 0
            type_dummies = type_dummies[['wine_type_red', 'wine_type_white']].astype(int)
            input_df = pd.concat([input_df.drop('type', axis=1), type_dummies], axis=1)
        else:
            input_df['wine_type_red'] = 0
            input_df['wine_type_white'] = 0

        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0.0
        input_df = input_df[expected_features]

        predictions = model.predict(input_df).tolist()
        return jsonify({"predictions": predictions})

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}. Check input format."}), 400