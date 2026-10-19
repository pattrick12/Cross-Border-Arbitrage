"""
Backtest Replay Engine
=======================
Replays historical ticks from QuestDB through the full pipeline
by re-publishing them to Kafka at configurable speed.

The key insight: the SAME pipeline code runs in both live and
backtest mode — the only difference is where the ticks come from.

Usage:
    python -m backtest.replay --speed 100  # 100x real-time
    python -m backtest.replay --start "2024-01-01" --end "2024-02-01"
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime

from confluent_kafka import Producer
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from backtest.metrics import compute_metrics, generate_equity_curve, BacktestMetrics

load_dotenv()
logger = get_logger("BacktestReplay")

BROKER = os.getenv("REDPANDA_BROKER", "localhost:19092")
QUESTDB_PG = os.getenv("QUESTDB_PG", "localhost:8812")


class BacktestReplay:
    """
    Replays historical data through the Kafka pipeline.

    Modes:
        1. From QuestDB: reads market_ticks table
        2. From CSV: reads a tick-level CSV file
        3. Synthetic: generates synthetic data for testing

    Parameters
    ----------
    speed_multiplier : float
        Replay speed (1.0 = real-time, 100.0 = 100x faster).
    topic : str
        Kafka topic to publish ticks to (default: 'ticks').
    """

    def __init__(
        self,
        speed_multiplier: float = 100.0,
        topic: str = "ticks",
    ):
        self.speed_multiplier = speed_multiplier
        self.topic = topic
        self.producer = Producer({"bootstrap.servers": BROKER})
        self.total_published = 0

    def replay_from_questdb(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        symbols: Optional[list] = None,
    ):
        """Replay ticks from QuestDB market_ticks table."""
        try:
            import psycopg2
        except ImportError:
            logger.error("psycopg2 not installed")
            return

        dsn = f"host={QUESTDB_PG.split(':')[0]} port={QUESTDB_PG.split(':')[1]} dbname=qdb user=admin password=quest"

        conditions = []
        if start_time:
            conditions.append(f"timestamp >= '{start_time}'")
        if end_time:
            conditions.append(f"timestamp <= '{end_time}'")
        if symbols:
            sym_list = ",".join(f"'{s}'" for s in symbols)
            conditions.append(f"symbol IN ({sym_list})")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT timestamp, symbol, bid, ask, bid_volume
            FROM market_ticks
            WHERE {where_clause}
            ORDER BY timestamp ASC
        """

        logger.info(f"Querying QuestDB: {query[:100]}...")

        try:
            conn = psycopg2.connect(dsn)
            df = pd.read_sql(query, conn, parse_dates=["timestamp"])
            conn.close()
        except Exception as e:
            logger.error(f"QuestDB query failed: {e}")
            return

        if df.empty:
            logger.warning("No data found in QuestDB for the given range.")
            return

        logger.info(f"Loaded {len(df)} ticks for replay ({df['symbol'].nunique()} symbols)")
        self._replay_dataframe(df)

    def replay_synthetic(
        self,
        n_ticks: int = 10_000,
        n_symbols: int = 5,
    ):
        """Generate and replay synthetic ticks for pipeline testing."""
        logger.info(f"Generating {n_ticks} synthetic ticks across {n_symbols} symbols...")

        symbols = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CHF"][:n_symbols]
        base_prices = {"EUR_USD": 1.0550, "GBP_USD": 1.2650, "USD_JPY": 149.5,
                       "AUD_USD": 0.655, "USD_CHF": 0.875}

        rows = []
        prices = {s: base_prices.get(s, 1.0) for s in symbols}

        for i in range(n_ticks):
            sym = symbols[i % len(symbols)]
            prices[sym] *= 1 + np.random.normal(0, 0.0003)
            spread = abs(prices[sym]) * 0.0002
            rows.append({
                "timestamp": pd.Timestamp.now() + pd.Timedelta(milliseconds=i * 50),
                "symbol": sym,
                "bid": prices[sym] - spread / 2,
                "ask": prices[sym] + spread / 2,
                "bid_volume": np.random.randint(100_000, 5_000_000),
            })

        df = pd.DataFrame(rows)
        logger.info(f"Replaying {len(df)} synthetic ticks...")
        self._replay_dataframe(df)

    def _replay_dataframe(self, df: pd.DataFrame):
        """Core replay loop — publishes ticks at controlled speed."""
        n = len(df)
        t_start = time.time()

        # Compute inter-tick delays from timestamps
        if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df = df.sort_values("timestamp")
            df["delay_s"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
        else:
            df["delay_s"] = 0.05  # Default 50ms

        published = 0

        for _, row in df.iterrows():
            tick = {
                "type": "PRICE",
                "time": str(row.get("timestamp", datetime.utcnow().isoformat())),
                "instrument": row["symbol"],
                "bids": [{"price": str(round(row["bid"], 6)),
                          "liquidity": int(row.get("bid_volume", 1_000_000))}],
                "asks": [{"price": str(round(row["ask"], 6)),
                          "liquidity": int(row.get("bid_volume", 1_000_000))}],
                "closeoutBid": str(round(row["bid"], 6)),
                "closeoutAsk": str(round(row["ask"], 6)),
                "status": "tradeable",
                "tradeable": True,
            }

            self.producer.produce(
                self.topic,
                key=row["symbol"],
                value=json.dumps(tick),
            )
            published += 1

            # Controlled delay
            delay = row["delay_s"] / self.speed_multiplier
            if delay > 0 and delay < 1.0:
                time.sleep(delay)

            # Periodic flush + stats
            if published % 1000 == 0:
                self.producer.flush()
                elapsed = time.time() - t_start
                rate = published / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Replayed {published}/{n} ticks "
                    f"({published / n * 100:.1f}%) "
                    f"@ {rate:.0f} ticks/sec"
                )

            self.producer.poll(0)

        self.producer.flush()
        self.total_published = published
        elapsed = time.time() - t_start
        logger.info(
            f"Replay complete: {published} ticks in {elapsed:.1f}s "
            f"({published / elapsed:.0f} ticks/sec)"
        )


def main():
    parser = argparse.ArgumentParser(description="Backtest Replay Engine")
    parser.add_argument("--speed", type=float, default=100.0,
                        help="Replay speed multiplier (default: 100x)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start time (e.g., '2024-01-01')")
    parser.add_argument("--end", type=str, default=None,
                        help="End time (e.g., '2024-02-01')")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of QuestDB")
    parser.add_argument("--n-ticks", type=int, default=10_000,
                        help="Number of synthetic ticks")

    args = parser.parse_args()

    replayer = BacktestReplay(speed_multiplier=args.speed)

    if args.synthetic:
        replayer.replay_synthetic(n_ticks=args.n_ticks)
    else:
        replayer.replay_from_questdb(start_time=args.start, end_time=args.end)


if __name__ == "__main__":
    main()
