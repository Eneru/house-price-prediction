import json

import joblib
import pandas as pd

from house_price_prediction.config import BEST_MODEL_FILEPATH, PREPROCESSED_MODEL_FILEPATH
from house_price_prediction.modeling.schemas import HouseFeatures


def load_preprocessed_data():
    return joblib.load(PREPROCESSED_MODEL_FILEPATH)


def load_best_model():
    return joblib.load(BEST_MODEL_FILEPATH)


def ask_features():
    result = HouseFeatures()
    is_a_valid_json = False
    while not is_a_valid_json:
        features_as_json = input("Give a json representing the features you want to filter:")
        try:
            features_json_obj = json.loads(features_as_json)
            is_a_valid_json = True
            result = HouseFeatures(**features_json_obj)
        except ValueError:
            print("The JSON is not a valid JSON file!")
    return result


def predict_price(preprocessor, model, features: HouseFeatures):
    df = pd.DataFrame([features.model_dump()])
    X_processed = preprocessor.transform(df) if preprocessor else df
    prediction = model.predict(X_processed)
    return prediction[0]


def main():
    print("Loading preprocessed data and best model...")
    preprocessor = load_preprocessed_data()
    model = load_best_model()
    print("Asking features as json...")
    features = ask_features()
    print("Predicting...")
    price = predict_price(preprocessor, model, features)
    print(f"🏷️ Predicted price: ${price:,.2f}")
    print("Done.")


if __name__ == "__main__":
    main()
