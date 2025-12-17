# analytics/hedge.py

import numpy as np
import pandas as pd

def ols_hedge_ratio(y, x):
    """
    y = dependent (e.g. BTC)
    x = independent (e.g. ETH)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]

    return beta[1]  # hedge ratio

def compute_spread(y, x, hedge_ratio):
    return y - hedge_ratio * x

def compute_zscore(series, window=30):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std
