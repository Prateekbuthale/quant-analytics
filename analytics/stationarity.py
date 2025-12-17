# analytics/stationarity.py

from statsmodels.tsa.stattools import adfuller

def adf_test(series):
    result = adfuller(series.dropna())
    return {
        "adf_stat": result[0],
        "p_value": result[1],
        "lags": result[2],
        "n_obs": result[3]
    }
