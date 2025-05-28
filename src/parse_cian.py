import datetime
import os
import pandas as pd
import cianparser
import argparse


def parse_cian(n_rooms: int):
    os.makedirs("src/data/raw", exist_ok=True)
    parser = cianparser.CianParser(location="Москва")
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = f"src/data/raw/{n_rooms}_{t}.csv"

    data = parser.get_flats(
        deal_type="sale",
        rooms=(n_rooms,),
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 20,
            "object_type": "secondary",
        },
    )
    df = pd.DataFrame(data)
    df.to_csv(csv_path, encoding="utf-8", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rooms", type=int, default=1)
    args = parser.parse_args()
    parse_cian(args.n_rooms)
