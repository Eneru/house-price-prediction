import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from house_price_prediction.config import (
    DATASET_FILEPATH,
    KAGGLE_CONFIG_DIR,
    PREPROCESSED_MODEL_FILEPATH,
    PROCESSED_X_TRAIN_FILEPATH,
    PROCESSED_X_VALID_FILEPATH,
    PROCESSED_Y_TRAIN_FILEPATH,
    PROCESSED_Y_VALID_FILEPATH,
    RAW_DATA_DIR,
)


def remove_data() -> None:
    try:
        PROCESSED_X_TRAIN_FILEPATH.unlink()
        PROCESSED_X_VALID_FILEPATH.unlink()
        PROCESSED_Y_TRAIN_FILEPATH.unlink()
        PROCESSED_Y_VALID_FILEPATH.unlink()
        PREPROCESSED_MODEL_FILEPATH.unlink()
        print("\tRemoved.")
    except BaseException:
        print("\tAlready removed.")


def get_data(is_removed_if_exists: bool = False) -> None:
    if is_removed_if_exists and DATASET_FILEPATH.exists():
        print(f"\t{DATASET_FILEPATH} will be removed to download it again.")
        DATASET_FILEPATH.unlink()
    # File had already been downloaded
    if DATASET_FILEPATH.exists():
        print(f"\t{DATASET_FILEPATH} already exists, it will not be downloaded.")
        return
    os.environ["KAGGLE_CONFIG_DIR"] = str(KAGGLE_CONFIG_DIR)
    # Import does the environment variable check so it must be set first
    from kaggle.api.kaggle_api_extended import KaggleApi

    print(f"\tDownloading {DATASET_FILEPATH}...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        dataset="prevek18/ames-housing-dataset", path=str(RAW_DATA_DIR), unzip=True
    )
    print("\tDone.")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_FILEPATH)
    return df


def init_preprocessing_pipeline(numerical_cols, categorical_cols):
    # Numerical: impute missing + scale
    num_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # Categorical: impute missing + one-hot encode
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Combine both
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ]
    )

    return preprocessor


def fit_and_transform(X_train, X_valid, preprocessor: ColumnTransformer):
    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)
    return X_train_processed, X_valid_processed


def clean_data(df: pd.DataFrame):
    missing_threshold = 0.95
    missing_ratios = df.isnull().mean()
    high_missing_cols = missing_ratios[missing_ratios > missing_threshold].index.tolist()

    if high_missing_cols:
        print(
            f"\tDropping columns with >{missing_threshold * 100}% missing values: ",
            high_missing_cols,
        )
        df.drop(columns=high_missing_cols, inplace=True)

    useless_columns = ["PID", "Order"]
    print("\tDropping useless columns: ", useless_columns)
    df.drop(columns=useless_columns, inplace=True)

    numerical_cols = (
        df.select_dtypes(include=["int64", "float64"]).drop(columns=["SalePrice"]).columns.tolist()
    )
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    target_col = "SalePrice"

    print(f"\tDropping target column: {target_col}")
    X = df.drop(columns=target_col)
    y = df[target_col]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = init_preprocessing_pipeline(numerical_cols, categorical_cols)

    X_train_processed, X_valid_processed = fit_and_transform(X_train, X_valid, preprocessor)

    return preprocessor, X_train_processed, X_valid_processed, y_train, y_valid


def save_data(preprocessor, X_train_processed, X_valid_processed, y_train, y_valid):
    # Save preprocessed data
    np.save(PROCESSED_X_TRAIN_FILEPATH, X_train_processed)
    np.save(PROCESSED_X_VALID_FILEPATH, X_valid_processed)
    np.save(PROCESSED_Y_TRAIN_FILEPATH, y_train.to_numpy())
    np.save(PROCESSED_Y_VALID_FILEPATH, y_valid.to_numpy())

    # Save the preprocessing pipeline
    joblib.dump(preprocessor, PREPROCESSED_MODEL_FILEPATH)


def main():
    print("Removing existing data first...")
    remove_data()
    print("Getting the data from Kaggle...")
    get_data()
    print("Loading data...")
    df = load_data()
    print("Cleaning data...")
    preprocessor, X_train_processed, X_valid_processed, y_train, y_valid = clean_data(df)
    print("Saving cleaned data...")
    save_data(preprocessor, X_train_processed, X_valid_processed, y_train, y_valid)
    print("Done.")


if __name__ == "__main__":
    main()
