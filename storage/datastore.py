# storage/datastore.py

import duckdb
import pandas as pd
from pathlib import Path
import threading

class MarketDataStore:
    def __init__(self, db_path="data/market.duckdb"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        # Connect with read_only=False for read-write access
        self.con = duckdb.connect(db_path, read_only=False)
        self._lock = threading.Lock()
        self._create_tables()

    # ---------- EXECUTION HELPERS ----------

    def execute(self, query: str, params=None):
        """Thread-safe execute wrapper"""
        with self._lock:
            return self.con.execute(query, params or [])

    def fetchdf(self, query: str, params=None):
        """Thread-safe fetchdf"""
        with self._lock:
            return self.con.execute(query, params or []).fetchdf()

    def fetchone(self, query: str, params=None):
        """Thread-safe fetchone"""
        with self._lock:
            return self.con.execute(query, params or []).fetchone()

    def _create_tables(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                ts TIMESTAMP,
                symbol VARCHAR,
                price DOUBLE,
                size DOUBLE
            )
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS ohlc (
                ts TIMESTAMP,
                symbol VARCHAR,
                timeframe VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE
            )
        """)

    # ---------- INSERT METHODS ----------

    def insert_ticks(self, df: pd.DataFrame):
        """
        Expects columns: ts, symbol, price, size
        Thread-safe insert operation
        """
        if df.empty:
            return
        with self._lock:
            try:
                self.con.register("ticks_df", df)
                self.con.execute("INSERT INTO ticks SELECT * FROM ticks_df")
            finally:
                # Unregister to avoid pending result conflicts
                try:
                    self.con.unregister("ticks_df")
                except Exception:
                    pass

    def insert_ohlc(self, df: pd.DataFrame):
        """
        Expects columns:
        ts, symbol, timeframe, open, high, low, close, volume
        Thread-safe insert operation
        """
        if df.empty:
            return
        with self._lock:
            try:
                self.con.register("ohlc_df", df)
                self.con.execute("INSERT INTO ohlc SELECT * FROM ohlc_df")
            finally:
                try:
                    self.con.unregister("ohlc_df")
                except Exception:
                    pass

    # ---------- QUERY METHODS ----------

    
    
    def get_ticks(self, symbol, limit=100):
        query = """
            SELECT *
            FROM ticks
            WHERE symbol = ?
            ORDER BY ts DESC
            LIMIT ?
        """
        return self.fetchdf(query, [symbol, limit])


    def get_ohlc(self, symbol, timeframe, lookback_minutes=60):
        # Use parameterized query to prevent SQL injection
        query = """
            SELECT *
            FROM ohlc
            WHERE symbol = ?
            AND timeframe = ?
            AND ts >= NOW() - INTERVAL ? MINUTE
            ORDER BY ts
        """
        return self.fetchdf(query, [symbol, timeframe, lookback_minutes])
    
    def get_stats(self):
        """Get database statistics"""
        try:
            ticks_count = self.con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
            ohlc_count = self.con.execute("SELECT COUNT(*) FROM ohlc").fetchone()[0]
            latest_tick = self.con.execute("SELECT MAX(ts) FROM ticks").fetchone()[0]
            latest_ohlc = self.con.execute("SELECT MAX(ts) FROM ohlc").fetchone()[0]
            return {
                "ticks_count": ticks_count,
                "ohlc_count": ohlc_count,
                "latest_tick": latest_tick,
                "latest_ohlc": latest_ohlc
            }
        except Exception as e:
            return {"error": str(e)}
