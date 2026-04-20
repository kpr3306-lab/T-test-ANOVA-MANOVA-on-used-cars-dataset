import pandas as pd
import numpy as np
from utils.preprocessing import clean_data

def load_data():
    df_raw = pd.read_csv("data/used_cars_data1.csv")
    df_clean = clean_data(df_raw.copy())
    return df_raw, df_clean