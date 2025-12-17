# analytics/correlation_matrix.py

import pandas as pd
import numpy as np


def compute_correlation_matrix(df, columns=None, window=None):
    """
    Compute correlation matrix for specified columns
    
    Args:
        df: DataFrame with time series data
        columns: List of column names to correlate (if None, uses all numeric columns)
        window: Rolling window for correlation (if None, computes full period correlation)
    
    Returns:
        Correlation matrix as DataFrame
    """
    if df.empty:
        return pd.DataFrame()
    
    # Select columns
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude timestamp columns
        numeric_cols = [col for col in numeric_cols if col not in ['ts', 'timestamp']]
        columns = numeric_cols
    
    # Filter to available columns
    available_cols = [col for col in columns if col in df.columns]
    
    if len(available_cols) < 2:
        return pd.DataFrame()
    
    # Get numeric data
    data = df[available_cols].copy()
    
    # Remove rows with all NaN
    data = data.dropna(how='all')
    
    if data.empty or len(data) < 2:
        return pd.DataFrame()
    
    # Compute correlation
    if window is None:
        # Full period correlation
        corr_matrix = data.corr()
    else:
        # Rolling correlation - compute for last window
        if len(data) >= window:
            corr_matrix = data.tail(window).corr()
        else:
            corr_matrix = data.corr()
    
    return corr_matrix


def compute_cross_correlation_lags(df, col1, col2, max_lag=10):
    """
    Compute cross-correlation at different lags
    
    Args:
        df: DataFrame with time series
        col1: First column name
        col2: Second column name
        max_lag: Maximum lag to compute
    
    Returns:
        Dictionary with lag values and correlations
    """
    if df.empty or col1 not in df.columns or col2 not in df.columns:
        return {}
    
    series1 = df[col1].dropna()
    series2 = df[col2].dropna()
    
    if len(series1) < max_lag * 2 or len(series2) < max_lag * 2:
        return {}
    
    # Align series
    min_len = min(len(series1), len(series2))
    series1 = series1.iloc[:min_len]
    series2 = series2.iloc[:min_len]
    
    correlations = {}
    
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            corr = series1.corr(series2)
        elif lag > 0:
            # col1 leads col2
            if len(series1) > lag:
                corr = series1.iloc[:-lag].corr(series2.iloc[lag:])
            else:
                corr = np.nan
        else:
            # col2 leads col1
            lag_abs = abs(lag)
            if len(series2) > lag_abs:
                corr = series1.iloc[lag_abs:].corr(series2.iloc[:-lag_abs])
            else:
                corr = np.nan
        
        correlations[lag] = corr
    
    return correlations


def compute_multi_timeframe_correlation(datastore, symbols, timeframes):
    """
    Compute correlation matrix across multiple symbols and timeframes
    
    Args:
        datastore: MarketDataStore instance
        symbols: List of symbols
        timeframes: List of timeframes
    
    Returns:
        DataFrame with multi-index (symbol_timeframe) columns
    """
    all_data = {}
    
    for symbol in symbols:
        for tf in timeframes:
            try:
                df = datastore.execute(
                    """
                    SELECT ts, close
                    FROM ohlc
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY ts
                    """,
                    [symbol, tf]
                ).fetchdf()
                
                if not df.empty:
                    key = f"{symbol}_{tf}"
                    all_data[key] = df.set_index('ts')['close']
            except Exception:
                continue
    
    if not all_data:
        return pd.DataFrame()
    
    # Combine into single dataframe
    combined_df = pd.DataFrame(all_data)
    
    # Compute correlation
    corr_matrix = combined_df.corr()
    
    return corr_matrix

