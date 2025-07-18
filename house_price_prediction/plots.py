import joblib
import matplotlib.pyplot as plt
import numpy as np

from house_price_prediction.config import (
    BEST_MODEL_FILEPATH,
    BEST_PREDICTION_PLOT_FILEPATH,
    PLOTS_FILEPATHES,
    PROCESSED_X_VALID_FILEPATH,
    PROCESSED_Y_VALID_FILEPATH,
)


def load_preprocessed_data():
    X_valid = np.load(PROCESSED_X_VALID_FILEPATH)
    y_valid = np.load(PROCESSED_Y_VALID_FILEPATH)

    return X_valid, y_valid


def clean_previous_plots():
    for plotFile in PLOTS_FILEPATHES:
        if plotFile.exists():
            plotFile.unlink()


def get_best_model():
    return joblib.load(BEST_MODEL_FILEPATH)


def save_plot_prediction(best_model, X_valid, y_valid):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_valid, best_model.predict(X_valid), alpha=0.3)
    plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], "--r")
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("Predicted vs Actual")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BEST_PREDICTION_PLOT_FILEPATH)
    plt.close()


def main():
    print("Loading preprocessed data...")
    X_valid, y_valid = load_preprocessed_data()
    print("Cleaning previous plots...")
    clean_previous_plots()
    print("Getting best model...")
    best_model = get_best_model()
    print("Saving best model...")
    save_plot_prediction(best_model, X_valid, y_valid)
    print("Done.")


if __name__ == "__main__":
    main()
