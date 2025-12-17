import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
import json
import io
from datetime import datetime

from storage.datastore import MarketDataStore
from ingestion.binance_ws import BinanceWebSocketIngestor
from resampling.sampler import Resampler

from analytics.stats import compute_returns, rolling_volatility
from analytics.hedge import ols_hedge_ratio, compute_spread, compute_zscore
from analytics.correlation import rolling_correlation
from analytics.stationarity import adf_test
from analytics.liquidity import (
    compute_volume_profile, compute_bid_ask_spread_proxy,
    filter_by_liquidity, compute_liquidity_metrics, compute_volume_weighted_price
)
from analytics.correlation_matrix import (
    compute_correlation_matrix, compute_cross_correlation_lags,
    compute_multi_timeframe_correlation
)
from alerts.rules import AlertManager, AlertRule
import plotly.graph_objects as go
import numpy as np


st.set_page_config(page_title="Quant Dashboard", layout="wide")


@st.cache_resource
def get_datastore():
    """Get or create the shared datastore instance"""
    return MarketDataStore()

@st.cache_resource
def get_ingestor():
    """Get or create the WebSocket ingestor - only once"""
    datastore = get_datastore()
    ingestor = BinanceWebSocketIngestor(
        symbols=["btcusdt", "ethusdt"],
        datastore=datastore
    )
    ingestor.start()
    return ingestor

# Get shared datastore instance
datastore = get_datastore()

# Start ingestion (cached, so only runs once)
try:
    ingestor = get_ingestor()
    if 'ingestion_started' not in st.session_state:
        st.session_state.ingestion_started = True
        st.info("✅ WebSocket ingestion started")
except Exception as e:
    st.error(f"❌ Failed to start ingestion: {e}")
    import traceback
    with st.expander("Error Details"):
        st.code(traceback.format_exc())

resampler = Resampler(datastore)


@st.cache_data(ttl=5)
def run_resampling():
    try:
        for sym in ["btcusdt", "ethusdt"]:
            for tf in ["1s", "1m", "5m"]:
                resampler.resample(sym, tf)
        return True
    except Exception as e:
        st.error(f"Resampling error: {e}")
        return False


run_resampling()


st.title("Quant Analytics Dashboard")

# Ingestion status
if 'ingestion_started' in st.session_state:
    st.success("🟢 WebSocket ingestion is running")
else:
    st.warning("🟡 Waiting for ingestion to start...")

# Database status
with st.expander("📊 Database Status", expanded=False):
    try:
        stats = datastore.get_stats()
        if "error" in stats:
            st.error(f"Database error: {stats['error']}")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Ticks", stats.get("ticks_count", 0))
            with col2:
                st.metric("Total OHLC Bars", stats.get("ohlc_count", 0))
            with col3:
                latest_tick = stats.get("latest_tick")
                st.metric("Latest Tick", latest_tick.strftime("%H:%M:%S") if latest_tick else "N/A")
            with col4:
                latest_ohlc = stats.get("latest_ohlc")
                st.metric("Latest OHLC", latest_ohlc.strftime("%H:%M:%S") if latest_ohlc else "N/A")
    except Exception as e:
        st.error(f"Error checking database: {e}")

st.write("Data ingestion running in background...")


st.sidebar.header("Analytics Controls")
timeframe = st.sidebar.selectbox("Timeframe", ["1s", "1m", "5m"])
window = st.sidebar.slider("Rolling Window", 10, 100, 30)

# Liquidity Filters
st.sidebar.header("💧 Liquidity Filters")
enable_liquidity_filter = st.sidebar.checkbox("Enable Liquidity Filter", value=False, key="liquidity_filter")
if enable_liquidity_filter:
    min_volume_percentile = st.sidebar.slider(
        "Minimum Volume Percentile",
        min_value=0,
        max_value=50,
        value=10,
        help="Filter out periods with volume below this percentile"
    )
else:
    min_volume_percentile = None

# Live Analytics Controls
st.sidebar.header("🔄 Live Updates")
auto_refresh = st.sidebar.checkbox("Enable Live Mode", value=False, key="live_mode")

if auto_refresh:
    # Calculate refresh interval based on timeframe
    refresh_intervals = {
        "1s": 0.5,  # 500ms for tick-based
        "1m": 5,    # 5 seconds for 1m bars
        "5m": 300   # 5 minutes for 5m bars
    }
    refresh_interval = refresh_intervals.get(timeframe, 5)
    refresh_interval_sec = st.sidebar.selectbox(
        "Refresh Interval",
        ["500ms", "1s", "5s", "30s", "1m", "5m"],
        index=1 if timeframe == "1s" else (2 if timeframe == "1m" else 5),
        key="refresh_interval"
    )
    
    # Convert to seconds
    if refresh_interval_sec.endswith('ms'):
        sleep_time = float(refresh_interval_sec.replace('ms', '')) / 1000
    elif refresh_interval_sec.endswith('s'):
        sleep_time = float(refresh_interval_sec.replace('s', ''))
    elif refresh_interval_sec.endswith('m'):
        sleep_time = float(refresh_interval_sec.replace('m', '')) * 60
    else:
        sleep_time = 5
    
    st.sidebar.info(f"🔄 Auto-refreshing every {refresh_interval_sec}")
else:
    sleep_time = None


if st.button("Show 1m BTC bars"):
    df_bars = datastore.execute(
        """
        SELECT *
        FROM ohlc
        WHERE symbol='btcusdt' AND timeframe='1m'
        ORDER BY ts DESC
        LIMIT 5
        """
    ).fetchdf()
    st.dataframe(df_bars)


if st.button("Check latest BTC ticks"):
    df_ticks = datastore.get_ticks("btcusdt", 50)
    st.dataframe(df_ticks)


btc = datastore.execute(
    """
    SELECT ts, open, high, low, close, volume
    FROM ohlc
    WHERE symbol='btcusdt' AND timeframe=?
    ORDER BY ts
    """,
    [timeframe]
).fetchdf()

eth = datastore.execute(
    """
    SELECT ts, open, high, low, close, volume
    FROM ohlc
    WHERE symbol='ethusdt' AND timeframe=?
    ORDER BY ts
    """,
    [timeframe]
).fetchdf()

# Check if we have data
if btc.empty or eth.empty:
    st.warning("⚠️ Waiting for data... Please wait a few moments for data ingestion to populate the database.")
    st.stop()

df = btc.merge(eth, on="ts", suffixes=("_btc", "_eth"))

# Check if merge resulted in empty dataframe
if df.empty:
    st.warning("⚠️ No overlapping timestamps found between BTC and ETH data. Waiting for more data...")
    st.stop()

# Apply liquidity filter if enabled
if enable_liquidity_filter and 'volume_btc' in df.columns and 'volume_eth' in df.columns:
    df_before = len(df)
    # Use combined volume for filtering
    df['combined_volume'] = df['volume_btc'] + df['volume_eth']
    df = filter_by_liquidity(df, volume_col="combined_volume", 
                            min_volume_percentile=min_volume_percentile)
    df_filtered = len(df)
    if df_before > df_filtered:
        st.sidebar.info(f"💧 Filtered: {df_before} → {df_filtered} rows ({df_before - df_filtered} removed)")

# Apply liquidity filter if enabled
if enable_liquidity_filter and 'volume_btc' in df.columns:
    df_before = len(df)
    # Use combined volume for filtering
    df['combined_volume'] = df['volume_btc'] + df['volume_eth']
    df = filter_by_liquidity(df, volume_col="combined_volume", 
                            min_volume_percentile=min_volume_percentile)
    df_filtered = len(df)
    if df_before > df_filtered:
        st.sidebar.info(f"💧 Filtered: {df_before} → {df_filtered} rows ({df_before - df_filtered} removed)")

# Check if we have enough data points for calculations
if len(df) < window:
    st.info(f"📊 Collecting data... Need at least {window} data points (currently have {len(df)}).")

hedge = ols_hedge_ratio(df["close_btc"], df["close_eth"])
df["spread"] = compute_spread(df["close_btc"], df["close_eth"], hedge)
df["zscore"] = compute_zscore(df["spread"], window)
df["corr"] = rolling_correlation(df["close_btc"], df["close_eth"], window)

st.subheader("Stationarity Test (ADF)")

if st.button("Run ADF Test on Spread"):
    if df["spread"].notna().sum() < 10:  # Need at least 10 non-NaN values
        st.error("Not enough data points for ADF test. Please wait for more data.")
    else:
        res = adf_test(df["spread"])
        c1, c2 = st.columns(2)

        with c1:
            st.metric("ADF Statistic", f"{res['adf_stat']:.4f}")
            st.metric("p-value", f"{res['p_value']:.4f}")

        with c2:
            st.write(f"Lags used: {res['lags']}")
            st.write(f"Observations: {res['n_obs']}")

        if res["p_value"] < 0.05:
            st.success("Likely stationary (mean-reverting)")
        else:
            st.warning("Not stationary at 5% significance")

st.subheader("Prices")
if not df.empty:
    st.plotly_chart(
        px.line(df, x="ts", y=["close_btc", "close_eth"]),
        width='stretch',
        key="prices_chart"
    )
else:
    st.info("Waiting for price data...")

st.subheader("Spread & Z-Score")
if not df.empty and df["spread"].notna().any():
    st.plotly_chart(
        px.line(df, x="ts", y=["spread", "zscore"]),
        width='stretch',
        key="spread_zscore_chart"
    )
else:
    st.info("Waiting for spread data...")

st.subheader("Rolling Correlation")
if not df.empty and df["corr"].notna().any():
    st.plotly_chart(
        px.line(df, x="ts", y="corr"),
        width='stretch',
        key="correlation_chart"
    )
else:
    st.info("Waiting for correlation data...")

# Cross-Correlation Heatmap Section
st.subheader("📊 Cross-Correlation Heatmap")

# Create tabs for different correlation views
corr_tab1, corr_tab2, corr_tab3 = st.tabs([
    "Current Pair Correlation",
    "Multi-Timeframe Correlation",
    "Cross-Correlation Lags"
])

with corr_tab1:
    # Correlation matrix for current pair
    if not df.empty:
        corr_cols = ["close_btc", "close_eth", "spread", "zscore", "corr"]
        available_cols = [col for col in corr_cols if col in df.columns]
        
        if len(available_cols) >= 2:
            # Full period correlation
            corr_matrix_full = compute_correlation_matrix(df, columns=available_cols)
            
            # Rolling correlation (last window period)
            if len(df) >= window:
                corr_matrix_rolling = compute_correlation_matrix(
                    df.tail(window), columns=available_cols
                )
            else:
                corr_matrix_rolling = corr_matrix_full
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Full Period Correlation**")
                if not corr_matrix_full.empty:
                    fig_full = go.Figure(data=go.Heatmap(
                        z=corr_matrix_full.values,
                        x=corr_matrix_full.columns,
                        y=corr_matrix_full.index,
                        colorscale='RdBu',
                        zmid=0,
                        text=corr_matrix_full.values.round(3),
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        colorbar=dict(title="Correlation")
                    ))
                    fig_full.update_layout(
                        title="Full Period",
                        width=400,
                        height=400
                    )
                    st.plotly_chart(fig_full, use_container_width=True)
            
            with col2:
                st.write(f"**Rolling Correlation (Last {window} periods)**")
                if not corr_matrix_rolling.empty:
                    fig_rolling = go.Figure(data=go.Heatmap(
                        z=corr_matrix_rolling.values,
                        x=corr_matrix_rolling.columns,
                        y=corr_matrix_rolling.index,
                        colorscale='RdBu',
                        zmid=0,
                        text=corr_matrix_rolling.values.round(3),
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        colorbar=dict(title="Correlation")
                    ))
                    fig_rolling.update_layout(
                        title=f"Rolling ({window} periods)",
                        width=400,
                        height=400
                    )
                    st.plotly_chart(fig_rolling, use_container_width=True)

with corr_tab2:
    # Multi-timeframe correlation
    st.write("**Correlation Across Timeframes**")
    
    selected_symbols = st.multiselect(
        "Select Symbols",
        ["btcusdt", "ethusdt"],
        default=["btcusdt", "ethusdt"],
        key="corr_symbols"
    )
    
    selected_timeframes = st.multiselect(
        "Select Timeframes",
        ["1s", "1m", "5m"],
        default=["1m", "5m"],
        key="corr_timeframes"
    )
    
    if selected_symbols and selected_timeframes:
        try:
            multi_corr = compute_multi_timeframe_correlation(
                datastore, selected_symbols, selected_timeframes
            )
            
            if not multi_corr.empty:
                fig_multi = go.Figure(data=go.Heatmap(
                    z=multi_corr.values,
                    x=multi_corr.columns,
                    y=multi_corr.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=multi_corr.values.round(3),
                    texttemplate='%{text}',
                    textfont={"size": 9},
                    colorbar=dict(title="Correlation")
                ))
                fig_multi.update_layout(
                    title="Multi-Timeframe Correlation Matrix",
                    width=600,
                    height=600,
                    xaxis_title="",
                    yaxis_title=""
                )
                st.plotly_chart(fig_multi, use_container_width=True)
            else:
                st.info("Not enough data for multi-timeframe correlation")
        except Exception as e:
            st.error(f"Error computing multi-timeframe correlation: {e}")

with corr_tab3:
    # Cross-correlation at different lags
    st.write("**Cross-Correlation Analysis (Lead-Lag)**")
    
    if not df.empty and "close_btc" in df.columns and "close_eth" in df.columns:
        max_lag = st.slider("Maximum Lag", 1, 20, 10, key="max_lag")
        
        lag_correlations = compute_cross_correlation_lags(
            df, "close_btc", "close_eth", max_lag=max_lag
        )
        
        if lag_correlations:
            lags = list(lag_correlations.keys())
            correlations = list(lag_correlations.values())
            
            # Create bar chart
            fig_lag = go.Figure(data=go.Bar(
                x=lags,
                y=correlations,
                marker=dict(
                    color=correlations,
                    colorscale='RdBu',
                    showscale=True,
                    colorbar=dict(title="Correlation")
                ),
                text=[f"{c:.3f}" if not pd.isna(c) else "N/A" for c in correlations],
                textposition='outside'
            ))
            fig_lag.update_layout(
                title="Cross-Correlation at Different Lags",
                xaxis_title="Lag (positive = BTC leads ETH)",
                yaxis_title="Correlation",
                height=400
            )
            fig_lag.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_lag, use_container_width=True)
            
            # Find maximum correlation lag
            valid_corrs = {k: v for k, v in lag_correlations.items() if not pd.isna(v)}
            if valid_corrs:
                max_corr_lag = max(valid_corrs.items(), key=lambda x: abs(x[1]))
                st.info(
                    f"**Maximum correlation** ({max_corr_lag[1]:.3f}) at lag {max_corr_lag[0]}. "
                    f"{'BTC leads ETH' if max_corr_lag[0] > 0 else 'ETH leads BTC' if max_corr_lag[0] < 0 else 'No lead-lag'}"
                )
        else:
            st.info("Not enough data for lag correlation analysis")
    else:
        st.info("Waiting for data...")

# Liquidity Analysis Section
st.subheader("💧 Liquidity Analysis")

if not df.empty and 'volume_btc' in df.columns and 'volume_eth' in df.columns:
    # Compute liquidity metrics
    liquidity_metrics_btc = compute_liquidity_metrics(
        df[['volume_btc', 'high_btc', 'low_btc', 'close_btc']].rename(columns={
            'volume_btc': 'volume',
            'high_btc': 'high',
            'low_btc': 'low',
            'close_btc': 'close'
        }),
        volume_col='volume',
        price_col='close'
    )
    
    liquidity_metrics_eth = compute_liquidity_metrics(
        df[['volume_eth', 'high_eth', 'low_eth', 'close_eth']].rename(columns={
            'volume_eth': 'volume',
            'high_eth': 'high',
            'low_eth': 'low',
            'close_eth': 'close'
        }),
        volume_col='volume',
        price_col='close'
    )
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**BTC Liquidity**")
        if liquidity_metrics_btc:
            st.metric("Avg Volume", f"{liquidity_metrics_btc.get('avg_volume', 0):,.0f}")
            if 'avg_spread_pct' in liquidity_metrics_btc:
                st.metric("Avg Spread %", f"{liquidity_metrics_btc['avg_spread_pct']:.4f}%")
    
    with col2:
        st.write("**ETH Liquidity**")
        if liquidity_metrics_eth:
            st.metric("Avg Volume", f"{liquidity_metrics_eth.get('avg_volume', 0):,.0f}")
            if 'avg_spread_pct' in liquidity_metrics_eth:
                st.metric("Avg Spread %", f"{liquidity_metrics_eth['avg_spread_pct']:.4f}%")
    
    with col3:
        st.write("**Combined**")
        if liquidity_metrics_btc and liquidity_metrics_eth:
            total_volume = (
                liquidity_metrics_btc.get('total_volume', 0) +
                liquidity_metrics_eth.get('total_volume', 0)
            )
            st.metric("Total Volume", f"{total_volume:,.0f}")
    
    # Volume Profile Chart
    if 'volume_btc' in df.columns:
        st.write("**Volume Profile**")
        volume_profile_btc = compute_volume_profile(
            df[['close_btc', 'volume_btc']].rename(columns={
                'close_btc': 'close',
                'volume_btc': 'volume'
            }),
            price_col='close',
            volume_col='volume',
            bins=20
        )
        
        if not volume_profile_btc.empty:
            fig_vol_profile = px.bar(
                volume_profile_btc,
                x='price_mean',
                y='total_volume',
                title="BTC Volume Profile (Price Distribution)",
                labels={'price_mean': 'Price', 'total_volume': 'Total Volume'}
            )
            st.plotly_chart(fig_vol_profile, use_container_width=True, key="volume_profile")
    
    # VWAP Chart
    if 'volume_btc' in df.columns and 'close_btc' in df.columns:
        st.write("**Volume Weighted Average Price (VWAP)**")
        vwap_btc = compute_volume_weighted_price(
            df[['close_btc', 'volume_btc']].rename(columns={
                'close_btc': 'close',
                'volume_btc': 'volume'
            }),
            price_col='close',
            volume_col='volume'
        )
        
        if not vwap_btc.empty:
            vwap_df = pd.DataFrame({
                'ts': df['ts'],
                'close': df['close_btc'],
                'vwap': vwap_btc.values
            })
            
            fig_vwap = px.line(
                vwap_df,
                x='ts',
                y=['close', 'vwap'],
                title="BTC Price vs VWAP",
                labels={'value': 'Price', 'variable': 'Type'}
            )
            st.plotly_chart(fig_vwap, use_container_width=True, key="vwap_chart")
else:
    st.info("Volume data not available for liquidity analysis")

# Initialize Alert Manager
if 'alert_manager' not in st.session_state:
    st.session_state.alert_manager = AlertManager()
    # Add default z-score alert
    default_alert = AlertRule(
        name="Z-Score > 2",
        condition=">",
        threshold=2.0,
        metric="zscore",
        enabled=True
    )
    st.session_state.alert_manager.add_rule(default_alert)

# Alert Management Section
st.sidebar.header("🚨 Alert Management")
with st.sidebar.expander("Manage Alerts", expanded=False):
    # Display existing alerts
    st.write("**Active Alerts:**")
    for rule in st.session_state.alert_manager.get_rules():
        col1, col2 = st.columns([3, 1])
        with col1:
            status = "🟢" if rule.enabled else "🔴"
            st.write(f"{status} {rule.name}")
            st.caption(f"{rule.metric} {rule.condition} {rule.threshold} (Triggered: {rule.trigger_count}x)")
        with col2:
            if st.button("Delete", key=f"del_{rule.name}"):
                st.session_state.alert_manager.remove_rule(rule.name)
                st.rerun()
    
    # Add new alert
    st.write("**Create New Alert:**")
    new_alert_name = st.text_input("Alert Name", key="new_alert_name")
    new_alert_metric = st.selectbox(
        "Metric",
        ["zscore", "spread", "corr", "price_btc", "price_eth"],
        key="new_alert_metric"
    )
    new_alert_condition = st.selectbox(
        "Condition",
        [">", "<", ">=", "<=", "=="],
        key="new_alert_condition"
    )
    new_alert_threshold = st.number_input("Threshold", value=2.0, key="new_alert_threshold")
    
    if st.button("Add Alert", key="add_alert"):
        if new_alert_name:
            new_rule = AlertRule(
                name=new_alert_name,
                condition=new_alert_condition,
                threshold=new_alert_threshold,
                metric=new_alert_metric,
                enabled=True
            )
            st.session_state.alert_manager.add_rule(new_rule)
            st.success(f"Alert '{new_alert_name}' added!")
            st.rerun()

# Check alerts
if not df.empty and df["zscore"].notna().any():
    # Get the last non-NaN values for alert checking
    valid_zscore = df["zscore"].dropna()
    if not valid_zscore.empty:
        latest_z = valid_zscore.iloc[-1]
        latest_spread = df["spread"].dropna().iloc[-1] if df["spread"].notna().any() else None
        latest_corr = df["corr"].dropna().iloc[-1] if df["corr"].notna().any() else None
        latest_price_btc = df["close_btc"].iloc[-1] if not df["close_btc"].empty else None
        latest_price_eth = df["close_eth"].iloc[-1] if not df["close_eth"].empty else None
        
        # Prepare data for alert checking
        alert_data = {
            "zscore": latest_z,
            "spread": latest_spread,
            "corr": latest_corr,
            "price_btc": latest_price_btc,
            "price_eth": latest_price_eth
        }
        
        # Check all alerts
        triggered_alerts = st.session_state.alert_manager.check_all(alert_data)
        
        # Display triggered alerts
        if triggered_alerts:
            st.subheader("🚨 Active Alerts")
            for alert in triggered_alerts:
                value = alert_data.get(alert.metric, "N/A")
                st.error(
                    f"**{alert.name}** triggered! "
                    f"{alert.metric} = {value:.4f} {alert.condition} {alert.threshold}"
                )
                if alert.last_triggered:
                    st.caption(f"Last triggered: {alert.last_triggered.strftime('%H:%M:%S')}")

# Data Export Section
st.sidebar.header("💾 Data Export")
with st.sidebar.expander("Export Data", expanded=False):
    if not df.empty:
        # Export options
        export_format = st.radio("Export Format", ["CSV", "JSON"], key="export_format")
        
        # Prepare data for export
        export_df = df[["ts", "close_btc", "close_eth", "spread", "zscore", "corr"]].copy()
        
        if export_format == "CSV":
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"analytics_data_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_csv"
            )
        else:  # JSON
            json_str = export_df.to_json(orient='records', date_format='iso')
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"analytics_data_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_json"
            )
        
        # Export analytics summary
        st.write("**Export Analytics Summary:**")
        # Prepare summary data
        summary = {
            "timeframe": timeframe,
            "window": window,
            "hedge_ratio": float(hedge) if 'hedge' in locals() else None,
            "data_points": len(df),
            "export_timestamp": datetime.now().isoformat()
        }
        # Add latest values if available
        if not df.empty:
            if df["zscore"].notna().any():
                summary["latest_zscore"] = float(df["zscore"].dropna().iloc[-1])
            if df["spread"].notna().any():
                summary["latest_spread"] = float(df["spread"].dropna().iloc[-1])
            if df["corr"].notna().any():
                summary["latest_correlation"] = float(df["corr"].dropna().iloc[-1])
        
        summary_json = json.dumps(summary, indent=2)
        st.download_button(
            label="📥 Download Summary JSON",
            data=summary_json,
            file_name=f"analytics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="download_summary"
        )
    else:
        st.info("No data available for export")

# Auto-refresh logic
if auto_refresh and sleep_time:
    time.sleep(sleep_time)
    st.rerun()
