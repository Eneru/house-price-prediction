from fastapi import FastAPI
import joblib
import pandas as pd

from house_price_prediction.config import BEST_MODEL_FILEPATH, PREPROCESSED_MODEL_FILEPATH
from house_price_prediction.modeling.schemas import HouseFeatures

app = FastAPI()

model = joblib.load(BEST_MODEL_FILEPATH)
preprocessor = joblib.load(PREPROCESSED_MODEL_FILEPATH)


@app.post("/predict")
def predict(features: HouseFeatures):
    df = pd.DataFrame([features.__dict__])
    X = preprocessor.transform(df) if preprocessor else df
    prediction = model.predict(X)
    return {"predicted_price": float(prediction[0])}
