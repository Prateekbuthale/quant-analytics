# Quant Analytics Dashboard

## Overview

This project is a real-time quantitative analytics dashboard designed as a research-oriented prototype for statistical arbitrage and market microstructure analysis. The system ingests live tick data from Binance via WebSockets, stores and processes the data using DuckDB, computes key quantitative analytics, and visualizes results through an interactive Streamlit dashboard.

While the application runs locally, the architecture is intentionally designed to reflect modularity, extensibility, and foresight, allowing the system to scale conceptually into a larger real-time analytics stack without requiring fundamental rewrites.

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- Internet connection (for Binance WebSocket)

### Installation

```bash
git clone <repository-url>
cd quant-analytics

```
### Single Command
```bash
./run.sh
```

OR
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.\.venv\Scripts\Activate.ps1  # Windows

python -m pip install -r requirements.txt
```
### Run the Application
```bash
streamlit run app.py
```

## Database Initialization

The DuckDB database is created automatically on first run.

---

## Dependencies

Core dependencies used in this project:

- **Streamlit** – Dashboard UI and orchestration
- **DuckDB** – Embedded analytical database
- **websocket-client** – Real-time WebSocket ingestion
- **pandas, numpy** – Data manipulation and numerical computation
- **statsmodels** – Statistical testing (ADF)
- **plotly** – Interactive visualizations

All dependencies are listed in `requirements.txt`.

---

## Methodology & Data Flow

### Data Ingestion
- Live trade data is streamed from Binance using WebSockets.
- One WebSocket thread per symbol enables parallel ingestion.
- Incoming ticks are buffered in memory and flushed to storage at fixed intervals (1 second).

### Storage
- Raw tick data is stored in a DuckDB `ticks` table.
- Resampled OHLC bars are stored in a DuckDB `ohlc` table.
- DuckDB acts as the single source of truth for all analytics.

### Resampling
- Tick data is incrementally resampled into OHLC bars for multiple timeframes (1s, 1m, 5m).
- Only new data since the last processed timestamp is aggregated.
- Aggregation is performed using DuckDB SQL primitives for efficiency and clarity.

### Analytics Computation
- Analytics operate strictly on OHLC data.
- BTC and ETH time series are merged by timestamp.
- Optional liquidity filters can be applied before analysis.
- Computed metrics are visualized and optionally exported.

---

## Analytics Implemented

### Pairs Trading Analytics
- **Hedge Ratio (OLS)**  
  Ordinary Least Squares regression is used to estimate the hedge ratio between two assets.
- **Spread**  
  Measures deviation from the equilibrium relationship between the pair.
- **Z-Score**  
  Normalized spread used to identify mean-reversion opportunities.

### Correlation Analysis
- Rolling Pearson correlation between assets.
- Cross-correlation and multi-timeframe correlation matrices.

### Stationarity Testing
- Augmented Dickey-Fuller (ADF) test applied to the spread.
- Used to assess statistical stationarity and mean-reversion suitability.

### Liquidity Analysis
- Volume profile analysis.
- Bid-ask spread proxy estimation.
- Volume-weighted average price (VWAP).
- Liquidity-based filtering of low-quality data.

### Statistical Metrics
- Log returns.
- Rolling volatility.

---

## System Architecture
https://app.eraser.io/workspace/FEm79GYZSPXmDNHFpTgq?origin=share

---

## Design Philosophy & Architectural Decisions

This system is designed as a prototype with deliberate architectural foresight.

### Loose Coupling & Clear Interfaces
- Data ingestion, storage, resampling, analytics, and visualization are implemented as independent modules.
- Components interact through well-defined interfaces, reducing tight coupling.

### Scalability Without Rewriting
- The ingestion layer is abstracted such that alternative data sources (e.g., CME futures, REST APIs, historical CSV files) can be introduced with minimal changes.
- Analytics operate on standardized OHLC data, making them agnostic to the underlying data feed.

### Deliberate Extensibility
- New analytics can be added as independent modules without modifying existing logic.
- Additional symbols, timeframes, or alert rules can be introduced incrementally.
- Visualization components remain separate from computation logic.

### Clarity Over Complexity
- The system prioritizes readability and correctness over premature optimization.
- Incremental processing is used where appropriate while keeping logic simple and transparent.

### Awareness of Scaling Constraints
- DuckDB is used as an embedded analytical database suitable for single-process workloads.
- Concurrent write access is intentionally controlled and isolated.
- Future scaling would involve decoupling ingestion, storage, and UI into separate services.

---

## Threading & Concurrency Model

- WebSocket threads: One per symbol (daemon threads)
- Flush thread: Periodic database writes
- Streamlit main thread: UI rendering and user interaction
- Database access is protected via thread-safe mechanisms

This design ensures data integrity during concurrent ingestion and resampling.

---

## ChatGPT Usage Transparency

ChatGPT was used to assist with:
- Structuring the project architecture
- Debugging environment and dependency issues
- Refactoring code for modularity and clarity
- Drafting documentation and architecture explanations

All prompts were focused on engineering guidance and code organization.  
All implementation decisions, validation, and final code were authored and reviewed by the developer.

---

## Limitations & Future Extensions

### Current Limitations
- No historical backfill beyond live data
- No order execution or trading logic
- Single-machine, single-process execution

### Potential Extensions
- Kalman filter-based dynamic hedge ratios
- Mean-reversion backtesting framework
- Support for additional exchanges and asset classes
- Distributed ingestion and storage services

---

## Conclusion

This project demonstrates an end-to-end real-time quantitative analytics pipeline, combining data ingestion, storage, statistical analysis, and visualization. The modular design and deliberate architectural choices make it suitable as a foundation for more advanced quantitative research systems while remaining simple, readable, and aligned with evaluation requirements.
