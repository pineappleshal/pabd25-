import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import argparse

logging.basicConfig(level=logging.INFO)

def preprocess_data(test_size: float):
    os.makedirs("src/data/processed", exist_ok=True)
    
    file_list = glob.glob("src/data/raw/*.csv")
    logging.info(f"Preprocess_data. Use files to train: {file_list}")

    main_df = pd.concat([pd.read_csv(f) for f in file_list])
    main_df["url_id"] = main_df["url"].map(lambda x: x.split("/")[-2])

    cols = ["url_id", "total_meters", "price", "rooms_count", "floor", "floors_count"]
    main_df = main_df[cols].dropna().set_index("url_id")
    main_df = main_df.astype({
        "total_meters": float,
        "price": float,
        "rooms_count": int,
        "floor": int,
        "floors_count": int
    })

    main_df = main_df[
        (main_df["price"] < 1e8) &
        (main_df["total_meters"] < 1000) &
        (main_df["rooms_count"] <= 10) &
        (main_df["floor"] <= main_df["floors_count"])
    ].drop_duplicates().sort_index()

    train_df, test_df = train_test_split(main_df, test_size=test_size, shuffle=False)
    train_df.to_csv("src/data/processed/train.csv")
    test_df.to_csv("src/data/processed/test.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    preprocess_data(args.test_size)