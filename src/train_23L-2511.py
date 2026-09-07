"""
train_23L-2511.py
MLOps Assignment 1 - House Price Prediction
Student ID: 23L-2511

Loads a housing dataset from data/, trains a RandomForestRegressor,
and saves the trained model to model/.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

STUDENT_ID = "23L-2511"

# Hyperparameters
LEARNING_RATE = 0.05  # placeholder for Part 3 (RandomForest has no learning_rate, kept for demo purposes)
N_ESTIMATORS = 300
RANDOM_STATE = 42

DATA_PATH = os.path.join("data", "house_prices.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, f"house_price_model_{STUDENT_ID}.pkl")


def load_data(path: str) -> pd.DataFrame:
    """Load the raw dataset from the data/ directory."""
    print(f"[Student: {STUDENT_ID}] Loading dataset from {path} ...")
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def preprocess(df: pd.DataFrame):
    """Basic preprocessing: split features/target, handle missing values.
    NOTE: No scaling here — scaling happens AFTER the train/test split
    to avoid data leakage."""
    df = df.dropna()

    if "price" not in df.columns:
        raise ValueError("Expected a 'price' column as the prediction target.")

    X = df.drop(columns=["price"])
    y = df["price"]

    # keep only numeric columns for this baseline model
    X = X.select_dtypes(include=["number"])

    return X, y


def train_model(X_train, y_train):
    """Train a RandomForestRegressor model."""
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.4f}")


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def main():
    df = load_data(DATA_PATH)
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, MODEL_PATH)


if __name__ == "__main__":
    main()