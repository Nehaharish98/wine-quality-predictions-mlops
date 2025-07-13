from fastapi import FastAPI, Depends
from pydantic import BaseModel

from api.schemas import WineFeatures
from api.utils import load_model, predict

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model("models/log_reg.bin")
    yield

app = FastAPI(title="Wine Quality Prediction API", lifespan=lifespan)


# Load model at startup
@app.on_event("startup")
def startup_event():
    app.state.model = load_model("models/log_reg.bin")

# Define request schema
class WineFeatures(BaseModel):
    features: dict

def get_app():
    from fastapi import Request
    def _get_app(request: Request):
        return request.app
    return _get_app

@app.post("/predict")
async def predict_quality(data: WineFeatures, app: FastAPI = Depends(get_app())):
    dv, model = app.state.model
    X = dv.transform([data.features])
    prediction = model.predict(X)[0]
    return {"prediction": int(prediction)}