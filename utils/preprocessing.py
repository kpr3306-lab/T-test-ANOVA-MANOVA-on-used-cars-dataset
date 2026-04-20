import matplotlib.pyplot as plt
import numpy as np

def clean_data(df):

    def strip_units(series):
        return series.astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)

    df["Mileage"] = strip_units(df["Mileage"])
    df["Engine"] = strip_units(df["Engine"])
    df["Power"] = strip_units(df["Power"])

    for col in ["Mileage", "Engine", "Power"]:
        df[col] = df[col].fillna(df[col].median())

    df["Seats"] = df["Seats"].fillna(df["Seats"].mode()[0])

    # KEEP YEAR
    df["Car_Age"] = 2026 - df["Year"]

    df["Log_Price"] = np.log(df["Price"])

    return df