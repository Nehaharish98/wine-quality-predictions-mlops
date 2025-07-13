from pydantic import BaseModel

class WineFeatures(BaseModel):
    features: dict