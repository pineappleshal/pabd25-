import pandas as pd
import joblib
import logging
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

logging.basicConfig(level=logging.INFO)

def train_model(model_path, model_type):
    df = pd.read_csv("src/data/processed/train.csv")
    X = df[["total_meters", "rooms_count", "floor", "floors_count"]]
    y = df["price"]

    if model_type == "gb":
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        logging.info("Using GradientBoostingRegressor")
    else:
        model = LinearRegression()
        logging.info("Using LinearRegression")

    model.fit(X, y)
    joblib.dump(model, model_path)

    logging.info(f"Trained model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", choices=["linear", "gb"], default="linear")
    args = parser.parse_args()
    train_model(args.model_path, args.model_type)