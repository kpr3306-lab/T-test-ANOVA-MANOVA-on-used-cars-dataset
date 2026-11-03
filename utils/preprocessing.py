import re

import numpy as np
import pandas as pd

REFERENCE_YEAR = 2026

# The dataset has no body-type column, so it is inferred from the model name.
# shortcut: keyword lists cover the common Indian-market models; unknown models fall back to Hatchback (or MUV if 7+ seats).
BODY_KEYWORDS = {
    "Hatchback": ["liva", "1 series", "etios cross", "sail hatchback", "cross polo", "crosspolo"],
    "SUV": [
        "xuv500", "xuv300", "fortuner", "ecosport", "creta", "duster", "scorpio", "vitara", "brezza", "endeavour",
        "q3", "q5", "q7", "x1", "x3", "x5", "x6", "terrano", "cr-v", "m-class", "benz gla", "benz gle", "benz glc",
        "benz gl-class", "benz gls", "pajero", "santa fe", "compass", "kuv", "tuv", "safari", "thar", "cayenne",
        "yeti", "xc60", "xc90", "tucson", "x-trail", "nuvosport", "xenon", "captur", "s-cross", "br-v",
        "brv", "wr-v", "wrv", "land rover", "ssangyong", "rexton", "harrier", "bolero", "nexon",
        "venue", "kicks", "avventura", "kodiaq", "tiguan", "hexa", "macan", "outlander", "captiva", "mux", "s cross",
    ],
    "MUV": [
        "innova", "ertiga", "xylo", "mobilio", "omni", "eeco", "quanto", "qualis", "lodgy", "marazzo",
        "enjoy", "evalia", "carnival", "r-class", "xl6", "tavera", "sumo", "versa", "aria",
    ],
    "Sedan": [
        "city", "verna", "vento", "amaze", "series", "c-class", "e-class", "s-class", "s class", "cla", "a3", "a4",
        "a6", "a8", "ciaz", "corolla", "etios", "xcent", "superb", "rapid", "sx4", "civic", "indigo",
        "octavia", "xf", "jaguar xe", "xj", "laura", "accord", "sunny", "jetta", "dzire", "elantra", "zest",
        "accent", "cruze", "manza", "optra", "linea", "camry", "sail", "passat", "panamera", "s60",
        "s80", "sonata", "esteem", "tigor", "scala", "fluence", "verito", "aspire", "fiesta", "ikon",
        "logan", "teana", "fusion", "mondeo",
    ],
}


def infer_body_type(name, seats):
    name = f" {str(name).lower()} "
    for body, words in BODY_KEYWORDS.items():
        if any(re.search(rf"[\s-]{re.escape(w)}[\s-]", name) for w in words):
            return body
    return "MUV" if seats >= 7 else "Hatchback"


def clean_data(df):

    # Brand: first word, normalised (ISUZU/Isuzu, Land Rover)
    df["Brand"] = df["Name"].astype(str).str.split().str[0].str.title()
    df["Brand"] = df["Brand"].replace({"Land": "Land Rover", "Bmw": "BMW"})
    df["Model"] = df["Name"].astype(str).str.split().str[1].str.title()

    def strip_units(series):
        return series.astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)

    df["Mileage"] = strip_units(df["Mileage"])
    df["Engine"] = strip_units(df["Engine"])
    df["Power"] = strip_units(df["Power"])

    # Zero mileage / zero seats are data-entry errors, not real values
    df.loc[df["Mileage"] == 0, "Mileage"] = np.nan
    df.loc[df["Seats"] == 0, "Seats"] = np.nan

    for col in ["Mileage", "Engine", "Power"]:
        df[col] = df[col].fillna(df[col].median())

    df["Seats"] = df["Seats"].fillna(df["Seats"].mode()[0])

    # A single car with 6.5M km skews every km statistic
    df = df[df["Kilometers_Driven"] < 1_000_000].copy()

    df["Body_Type"] = [infer_body_type(n, s) for n, s in zip(df["Name"], df["Seats"])]
    df["Seats"] = df["Seats"].astype(int).astype(str)

    df["Car_Age"] = REFERENCE_YEAR - df["Year"]
    df["Log_Price"] = np.log(df["Price"])

    return df.drop(columns=["New_Price"]).reset_index(drop=True)
