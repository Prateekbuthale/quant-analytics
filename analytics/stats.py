# analytics/stats.py

import pandas as pd
import numpy as np

def compute_returns(df, price_col="close"):
    df = df.copy()
    df["returns"] = np.log(df[price_col]).diff()
    return df

def rolling_volatility(df, window=30):
    df = df.copy()
    df["volatility"] = df["returns"].rolling(window).std()
    return df
