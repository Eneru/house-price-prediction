import joblib
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from house_price_prediction.config import (
    BEST_MODEL_FILEPATH,
    PROCESSED_X_TRAIN_FILEPATH,
    PROCESSED_X_VALID_FILEPATH,
    PROCESSED_Y_TRAIN_FILEPATH,
    PROCESSED_Y_VALID_FILEPATH,
)


def load_preprocessed_data():
    X_train = np.load(PROCESSED_X_TRAIN_FILEPATH)
    X_valid = np.load(PROCESSED_X_VALID_FILEPATH)
    y_train = np.load(PROCESSED_Y_TRAIN_FILEPATH)
    y_valid = np.load(PROCESSED_Y_VALID_FILEPATH)

    return X_train, X_valid, y_train, y_valid


def clean_previous_best_model():
    if BEST_MODEL_FILEPATH.exists():
        BEST_MODEL_FILEPATH.unlink()


def train_and_evaluate_models(X_train, X_valid, y_train, y_valid):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            max_features="sqrt",
            min_samples_split=5,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        rmse = root_mean_squared_error(y_valid, y_pred)
        r2 = r2_score(y_valid, y_pred)

        results[name] = {"model": model, "RMSE": rmse, "R2": r2}

        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²:   {r2:.4f}")

    return results


def get_best_model(model_training_results):
    best_model_name = min(model_training_results, key=lambda k: model_training_results[k]["RMSE"])
    best_model = model_training_results[best_model_name]["model"]

    print(
        f"\n✅ Best Model: {best_model_name} (RMSE: {model_training_results[best_model_name]['RMSE']:.2f})"
    )

    return best_model


def save_best_model(best_model):
    joblib.dump(best_model, BEST_MODEL_FILEPATH)


def main():
    print("Loading preprocessed data...")
    X_train, X_valid, y_train, y_valid = load_preprocessed_data()
    print("Cleaning previous best model...")
    clean_previous_best_model()
    print("Training and evaluating models...")
    model_training_results = train_and_evaluate_models(X_train, X_valid, y_train, y_valid)
    print("Getting best model...")
    best_model = get_best_model(model_training_results)
    print("Saving best model...")
    save_best_model(best_model)
    print("Done.")


if __name__ == "__main__":
    main()
