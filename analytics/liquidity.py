# analytics/liquidity.py

import pandas as pd
import numpy as np


def compute_volume_profile(df, price_col="close", volume_col="volume", bins=20):
    """
    Compute volume profile (volume at different price levels)
    
    Args:
        df: DataFrame with price and volume columns
        price_col: Column name for price
        volume_col: Column name for volume
        bins: Number of price bins
    
    Returns:
        DataFrame with price_bin and total_volume
    """
    if df.empty or volume_col not in df.columns:
        return pd.DataFrame()
    
    # Create price bins
    price_min = df[price_col].min()
    price_max = df[price_col].max()
    
    if price_min == price_max:
        return pd.DataFrame()
    
    df_copy = df.copy()
    df_copy['price_bin'] = pd.cut(df_copy[price_col], bins=bins, 
                                   labels=False, include_lowest=True)
    
    volume_profile = df_copy.groupby('price_bin').agg({
        volume_col: 'sum',
        price_col: ['min', 'max', 'mean']
    }).reset_index()
    
    volume_profile.columns = ['bin', 'total_volume', 'price_min', 'price_max', 'price_mean']
    volume_profile = volume_profile.sort_values('price_mean')
    
    return volume_profile


def compute_bid_ask_spread_proxy(df, window=20):
    """
    Estimate bid-ask spread using high-low range as proxy
    
    Args:
        df: DataFrame with high, low, close columns
        window: Rolling window for smoothing
    
    Returns:
        Series with spread estimates
    """
    if df.empty or 'high' not in df.columns or 'low' not in df.columns:
        return pd.Series()
    
    # Spread proxy: (high - low) / close
    spread = (df['high'] - df['low']) / df['close']
    
    # Rolling average
    spread_smooth = spread.rolling(window=window).mean()
    
    return spread_smooth


def filter_by_liquidity(df, volume_col="volume", min_volume_percentile=10):
    """
    Filter dataframe to keep only liquid periods
    
    Args:
        df: DataFrame to filter
        volume_col: Column name for volume
        min_volume_percentile: Minimum volume percentile to keep (0-100)
    
    Returns:
        Filtered DataFrame
    """
    if df.empty or volume_col not in df.columns:
        return df
    
    volume_threshold = df[volume_col].quantile(min_volume_percentile / 100)
    return df[df[volume_col] >= volume_threshold].copy()


def compute_liquidity_metrics(df, volume_col="volume", price_col="close"):
    """
    Compute various liquidity metrics
    
    Returns:
        Dictionary with liquidity metrics
    """
    if df.empty:
        return {}
    
    metrics = {}
    
    if volume_col in df.columns:
        metrics['avg_volume'] = df[volume_col].mean()
        metrics['median_volume'] = df[volume_col].median()
        metrics['volume_std'] = df[volume_col].std()
        metrics['total_volume'] = df[volume_col].sum()
    
    if 'high' in df.columns and 'low' in df.columns and price_col in df.columns:
        # Average spread
        spread = (df['high'] - df['low']) / df[price_col]
        metrics['avg_spread_pct'] = spread.mean() * 100
        metrics['median_spread_pct'] = spread.median() * 100
    
    return metrics


def compute_volume_weighted_price(df, price_col="close", volume_col="volume"):
    """
    Compute Volume Weighted Average Price (VWAP)
    
    Args:
        df: DataFrame with price and volume
        price_col: Column name for price
        volume_col: Column name for volume
    
    Returns:
        Series with VWAP values
    """
    if df.empty or volume_col not in df.columns or price_col not in df.columns:
        return pd.Series()
    
    # Cumulative VWAP
    df_copy = df.copy()
    df_copy['price_volume'] = df_copy[price_col] * df_copy[volume_col]
    df_copy['cum_price_volume'] = df_copy['price_volume'].cumsum()
    df_copy['cum_volume'] = df_copy[volume_col].cumsum()
    
    vwap = df_copy['cum_price_volume'] / df_copy['cum_volume']
    
    return vwap

