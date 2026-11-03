from pathlib import Path

import pandas as pd
from utils.preprocessing import clean_data

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "used_cars_data1.csv"


def load_data():
    df_raw = pd.read_csv(DATA_PATH)
    df_clean = clean_data(df_raw.copy())
    return df_raw, df_clean
