import pandas as pd
import joblib
import numpy as np
import logging
import argparse
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO)

def test_model(model_path):
    train_df = pd.read_csv("src/data/processed/train.csv")
    test_df = pd.read_csv("src/data/processed/test.csv")

    X_train = train_df[["total_meters", "rooms_count", "floor", "floors_count"]]
    y_train = train_df["price"]
    X_test = test_df[["total_meters", "rooms_count", "floor", "floors_count"]]
    y_test = test_df["price"]

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)

    logging.info(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    logging.info(f"R2 train: {r2_train:.2f}, R2 test: {r2_test:.2f}")

    with open("metrics.json", "w") as f:
        import json
        json.dump({
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_train": r2_train,
            "r2_test": r2_test
        }, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    test_model(args.model_path)
