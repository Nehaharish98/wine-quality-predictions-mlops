from fastapi import FastAPI
from pydantic import BaseModel

from api.schemas import WineFeatures
from api.utils import load_model, predict

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model("src/models/log_reg.bin")
    yield

app = FastAPI(title="Wine Quality Prediction API", lifespan=lifespan)


# Load model at startup
@app.on_event("startup")
def startup_event():
    app.state.model = load_model("src/models/log_reg.bin")

# Define request schema
class WineFeatures(BaseModel):
    features: dict

@app.post("/predict")
def predict_quality(data: WineFeatures):
    model_bundle = app.state.model
    prediction = predict(model_bundle, data.features)
    return {"prediction": int(prediction)}