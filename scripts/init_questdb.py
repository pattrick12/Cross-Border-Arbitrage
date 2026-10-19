"""
QuestDB Schema Initialization
Creates all tables required for Phase 2 via the REST API.
Run once after QuestDB starts: python scripts/init_questdb.py
"""
import os
import sys
import requests
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("QuestDB-Init")

QUESTDB_HTTP = os.getenv("QUESTDB_HTTP", "http://localhost:9000")

SCHEMAS = [
    # Raw tick storage (high-frequency, partitioned by hour)
    """
    CREATE TABLE IF NOT EXISTS market_ticks (
        timestamp TIMESTAMP,
        symbol SYMBOL,
        source SYMBOL,
        bid DOUBLE,
        ask DOUBLE,
        mid DOUBLE,
        spread DOUBLE,
        bid_volume DOUBLE,
        ask_volume DOUBLE
    ) timestamp(timestamp) PARTITION BY HOUR
      WAL
      DEDUP UPSERT KEYS(timestamp, symbol);
    """,

    # 1-minute OHLCV bars
    """
    CREATE TABLE IF NOT EXISTS ohlcv_1m (
        timestamp TIMESTAMP,
        symbol SYMBOL,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume DOUBLE,
        tick_count LONG,
        vwap DOUBLE
    ) timestamp(timestamp) PARTITION BY DAY
      WAL;
    """,

    # Rolling correlation matrix snapshots
    """
    CREATE TABLE IF NOT EXISTS correlation_matrix (
        timestamp TIMESTAMP,
        symbol_a SYMBOL,
        symbol_b SYMBOL,
        correlation DOUBLE,
        window_size INT
    ) timestamp(timestamp) PARTITION BY DAY
      WAL;
    """,

    # Computed model features for ML pipeline
    """
    CREATE TABLE IF NOT EXISTS model_features (
        timestamp TIMESTAMP,
        symbol SYMBOL,
        log_return DOUBLE,
        rolling_mean_5 DOUBLE,
        rolling_std_5 DOUBLE,
        rolling_mean_20 DOUBLE,
        rolling_std_20 DOUBLE,
        arima_mean DOUBLE,
        garch_vol DOUBLE,
        ou_zscore DOUBLE,
        graph_cluster INT
    ) timestamp(timestamp) PARTITION BY DAY
      WAL;
    """,

    # Arbitrage signals (Phase 1 + Phase 2)
    """
    CREATE TABLE IF NOT EXISTS arbitrage_signals (
        timestamp TIMESTAMP,
        signal_id SYMBOL,
        source SYMBOL,
        path STRING,
        currencies INT,
        profit_pct DOUBLE,
        signal_type SYMBOL,
        confidence DOUBLE
    ) timestamp(timestamp) PARTITION BY DAY
      WAL;
    """,

    # Risk metrics and position sizing
    """
    CREATE TABLE IF NOT EXISTS risk_metrics (
        timestamp TIMESTAMP,
        symbol SYMBOL,
        var_95 DOUBLE,
        cvar_95 DOUBLE,
        kelly_fraction DOUBLE,
        position_size DOUBLE,
        volatility DOUBLE,
        correlation_avg DOUBLE
    ) timestamp(timestamp) PARTITION BY DAY
      WAL;
    """,

    # Backtest results
    """
    CREATE TABLE IF NOT EXISTS backtest_results (
        timestamp TIMESTAMP,
        run_id SYMBOL,
        strategy SYMBOL,
        equity DOUBLE,
        pnl DOUBLE,
        sharpe DOUBLE,
        max_drawdown DOUBLE,
        win_rate DOUBLE,
        trades INT
    ) timestamp(timestamp) PARTITION BY DAY
      WAL;
    """,
]


def execute_sql(sql: str) -> bool:
    """Execute a SQL statement against QuestDB REST API."""
    try:
        resp = requests.get(
            f"{QUESTDB_HTTP}/exec",
            params={"query": sql.strip()},
            timeout=10,
        )
        if resp.status_code == 200:
            return True
        else:
            logger.error(f"SQL failed ({resp.status_code}): {resp.text}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to QuestDB. Is it running?")
        return False


def wait_for_questdb(max_retries: int = 30, delay: float = 2.0):
    """Wait for QuestDB to become available."""
    logger.info(f"Waiting for QuestDB at {QUESTDB_HTTP}...")
    for i in range(max_retries):
        try:
            resp = requests.get(f"{QUESTDB_HTTP}/exec", params={"query": "SELECT 1"}, timeout=5)
            if resp.status_code == 200:
                logger.info("QuestDB is ready.")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(delay)
    logger.error("QuestDB did not become available.")
    return False


def main():
    if not wait_for_questdb():
        sys.exit(1)

    success = 0
    total = len(SCHEMAS)

    for i, sql in enumerate(SCHEMAS):
        table_name = sql.split("CREATE TABLE IF NOT EXISTS")[1].split("(")[0].strip()
        logger.info(f"[{i+1}/{total}] Creating table: {table_name}")
        if execute_sql(sql):
            logger.info(f"  ✓ {table_name}")
            success += 1
        else:
            logger.error(f"  ✗ {table_name}")

    logger.info(f"\nSchema init complete: {success}/{total} tables created.")


if __name__ == "__main__":
    main()
