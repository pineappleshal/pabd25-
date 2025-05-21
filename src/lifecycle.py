"""This is full life cycle for ml model"""

import argparse
import datetime
import glob
import os

import cianparser
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor


TEST_SIZE = 0.2
N_ROOMS = 1
MODEL_NAME = "gradient_boosting_v1.pkl"

logging.basicConfig(
    filename="train.log",
    filemode="a",
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


def parse_cian(n_rooms=1):
    """
    Parse data to data/raw
    :param int n_rooms: The number of flats rooms
    :return None
    """
    moscow_parser = cianparser.CianParser(location="Москва")

    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = f"data/raw/{n_rooms}_{t}.csv"
    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=(n_rooms,),
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 10,
            "object_type": "secondary",
        },
    )
    df = pd.DataFrame(data)

    df.to_csv(csv_path, encoding="utf-8", index=False)


def preprocess_data(test_size):
    """
    Filter, sort and remove duplicates.
    Добавлены признаки: rooms_count, floor, floors_count
    """
    raw_data_path = "./data/raw"
    file_list = glob.glob(raw_data_path + "/*.csv")
    logging.info(f"Preprocess_data. Use files to train: {file_list}")
    main_df = pd.read_csv(file_list[0])
    for i in range(1, len(file_list)):
        data = pd.read_csv(file_list[i])
        df = pd.DataFrame(data)
        main_df = pd.concat([main_df, df], axis=0)
    main_df["url_id"] = main_df["url"].map(lambda x: x.split("/")[-2])
    required_columns = ["url_id", "total_meters", "price", "rooms_count", "floor", "floors_count"]
    main_df = main_df[required_columns].set_index("url_id")
    main_df = main_df.dropna()
    main_df = main_df.astype({
        "total_meters": float,
        "price": float,
        "rooms_count": int,
        "floor": int,
        "floors_count": int
    })

    main_df = main_df[
        (main_df["price"] < 100_000_000) &
        (main_df["total_meters"] < 1000) &
        (main_df["rooms_count"] <= 10) &
        (main_df["floor"] <= main_df["floors_count"])
    ]
    main_df = main_df.sort_index()
    main_df = main_df.drop_duplicates()

    train_df, test_df = train_test_split(main_df, test_size=test_size, shuffle=False)

    logging.info(f"Preprocess_data. train_df: {len(train_df)} samples")
    logging.info("\n" + str(train_df.head()))
    logging.info(f"Preprocess_data. test_df: {len(test_df)} samples")
    logging.info("\n" + str(test_df.head()))

    train_df.to_csv("data/processed/train.csv")
    test_df.to_csv("data/processed/test.csv")


def train_model(model_path, model_type):
    train_df = pd.read_csv("data/processed/train.csv")
    X = train_df[["total_meters", "rooms_count", "floor", "floors_count"]]
    y = train_df["price"]

    if model_type == "gb":
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        logging.info("Using GradientBoostingRegressor")
    else:
        model = LinearRegression()
        logging.info("Using LinearRegression")

    model.fit(X, y)
    joblib.dump(model, model_path)

    logging.info(f"Train model. Coefficients or feature importances: {getattr(model, 'coef_', getattr(model, 'feature_importances_', None))}")
    logging.info(f"Train model. Bias or intercept: {getattr(model, 'intercept_', 'n/a')}")


def test_model(model_path):
    """Test model with new data"""
    test_df = pd.read_csv("data/processed/test.csv")
    train_df = pd.read_csv("data/processed/train.csv")
    X_test = test_df[["total_meters", "rooms_count", "floor", "floors_count"]]
    y_test = test_df["price"]
    X_train = train_df[["total_meters", "rooms_count", "floor", "floors_count"]]
    y_train = train_df["price"]
    model = joblib.load(model_path)
    # Предсказание на тестовой выборке
    y_pred = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)

    logging.info(f"Test model. MSE: {mse:.2f}")
    logging.info(f"Test model. RMSE: {rmse:.2f}")
    logging.info(f"Test model. MAE: {mae:.2f}")
    logging.info(f"Test model. R2 train: {r2_train:.2f}")
    logging.info(f"Test model. R2 test: {r2_test:.2f}")


if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--split",
        type=float,
        help="Split data, test size, from 0 to 0.5",
        default=TEST_SIZE,
    )
    parser.add_argument(
        "-n", "--n_rooms", help="Number of rooms to parse", type=int, default=N_ROOMS
    )
    parser.add_argument(
    "--model_type", choices=["linear", "gb"], default="linear", help="Type of model to train"
)
    parser.add_argument("-m", "--model", help="Model name", default=MODEL_NAME)
    parser.add_argument(
        "-p", "--parse_data", help="Flag to parse new data", action="store_true"
    )
    args = parser.parse_args()

    test_size = float(args.split)
    assert 0.0 <= test_size <= 0.5
    model_path = os.path.join("..", "notebooks", "models", args.model)

    if args.parse_data:
        parse_cian(args.n_rooms)
    preprocess_data(test_size)
    train_model(model_path, args.model_type)
    test_model(model_path)
