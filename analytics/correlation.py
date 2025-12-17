# analytics/correlation.py

def rolling_correlation(series1, series2, window=30):
    return series1.rolling(window).corr(series2)
