"""
Stream Processor — PyFlink + Standalone Mode
=============================================
Consumes raw ticks from Kafka, computes:
  1. OHLCV 1-minute bars
  2. Rolling log-returns
  3. Rolling Pearson correlation matrix
  4. Feature vectors for ML pipeline

Publishes computed features to Kafka 'features' topic
and persists state to QuestDB.

Supports two execution modes:
  --pyflink : Runs as a PyFlink DataStream job (requires Java 11 + apache-flink)
  default   : Runs as a standalone Python Kafka consumer (always works)
"""

import os
import sys
import json
import time
import argparse
import threading
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, Optional

from confluent_kafka import Consumer, Producer, KafkaError
from questdb.ingress import Sender, TimestampNanos
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from streaming.correlation import RollingCorrelation

load_dotenv()
logger = get_logger("StreamProcessor")

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
BROKER = os.getenv("REDPANDA_BROKER", "localhost:19092")
QUESTDB_INGEST = os.getenv("QUESTDB_INGEST", "localhost:9009")

CURRENCIES = [
    "EUR", "USD", "JPY", "GBP", "AUD", "CAD", "CHF",
    "INR", "CNY", "RUB", "KWD", "SGD", "HKD", "BRL", "MXN",
]

# All symbols we track (both pairs and their derived mid-price series)
# We track by instrument name (e.g., "EUR_USD")
OHLCV_WINDOW_SEC = int(os.getenv("OHLCV_WINDOW_SEC", "60"))
CORRELATION_WINDOW = int(os.getenv("CORRELATION_WINDOW_TICKS", "300"))


# ─────────────────────────────────────────────────────────────
# OHLCV Aggregator
# ─────────────────────────────────────────────────────────────
class OHLCVBar:
    """Accumulates ticks into a single OHLCV bar."""

    __slots__ = ["open", "high", "low", "close", "volume", "tick_count",
                 "vwap_num", "vwap_den", "start_ts"]

    def __init__(self, price: float, volume: float, timestamp: float):
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = volume
        self.tick_count = 1
        self.vwap_num = price * volume
        self.vwap_den = volume
        self.start_ts = timestamp

    def update(self, price: float, volume: float):
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume
        self.tick_count += 1
        self.vwap_num += price * volume
        self.vwap_den += volume

    @property
    def vwap(self) -> float:
        return self.vwap_num / self.vwap_den if self.vwap_den > 0 else self.close

    def to_dict(self, symbol: str) -> dict:
        return {
            "symbol": symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "tick_count": self.tick_count,
            "vwap": self.vwap,
            "start_ts": self.start_ts,
        }


class OHLCVAggregator:
    """Time-based OHLCV bar aggregator with configurable window."""

    def __init__(self, window_sec: int = 60):
        self.window_sec = window_sec
        self.current_bars: Dict[str, OHLCVBar] = {}
        self.bar_start_time: Dict[str, float] = {}

    def ingest(self, symbol: str, mid: float, volume: float, ts: float) -> Optional[dict]:
        """
        Feed a tick. Returns a completed bar dict if the window has elapsed,
        otherwise None.
        """
        if symbol not in self.current_bars:
            self.current_bars[symbol] = OHLCVBar(mid, volume, ts)
            self.bar_start_time[symbol] = ts
            return None

        bar_start = self.bar_start_time[symbol]

        # Check if window elapsed
        if ts - bar_start >= self.window_sec:
            completed = self.current_bars[symbol].to_dict(symbol)
            # Start new bar
            self.current_bars[symbol] = OHLCVBar(mid, volume, ts)
            self.bar_start_time[symbol] = ts
            return completed
        else:
            self.current_bars[symbol].update(mid, volume)
            return None


# ─────────────────────────────────────────────────────────────
# Rolling Return Calculator
# ─────────────────────────────────────────────────────────────
class RollingReturns:
    """Computes rolling statistics on log-returns."""

    def __init__(self, short_window: int = 5, long_window: int = 20):
        self.short_window = short_window
        self.long_window = long_window
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=long_window + 1))
        self.returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=long_window))

    def update(self, symbol: str, mid: float) -> Optional[dict]:
        """
        Feed a mid price. Returns feature dict if enough data,
        otherwise None.
        """
        self.prices[symbol].append(mid)
        if len(self.prices[symbol]) < 2:
            return None

        log_ret = float(np.log(mid / self.prices[symbol][-2]))
        self.returns[symbol].append(log_ret)

        rets = list(self.returns[symbol])
        if len(rets) < self.short_window:
            return None

        short_rets = rets[-self.short_window:]
        features = {
            "log_return": log_ret,
            "rolling_mean_5": float(np.mean(short_rets)),
            "rolling_std_5": float(np.std(short_rets)) if len(short_rets) > 1 else 0.0,
        }

        if len(rets) >= self.long_window:
            long_rets = rets[-self.long_window:]
            features["rolling_mean_20"] = float(np.mean(long_rets))
            features["rolling_std_20"] = float(np.std(long_rets))
        else:
            features["rolling_mean_20"] = features["rolling_mean_5"]
            features["rolling_std_20"] = features["rolling_std_5"]

        return features


# ─────────────────────────────────────────────────────────────
# Standalone Stream Processor (Pure Python)
# ─────────────────────────────────────────────────────────────
class StandaloneStreamProcessor:
    """
    Pure Python stream processor using confluent-kafka.
    Functionally identical to the PyFlink version but without JVM dependency.
    """

    def __init__(self):
        # Kafka consumer
        self.consumer = Consumer({
            "bootstrap.servers": BROKER,
            "group.id": "stream-processor-group",
            "auto.offset.reset": "latest",
        })
        self.consumer.subscribe(["ticks"])

        # Kafka producer (for features topic)
        self.producer = Producer({"bootstrap.servers": BROKER})

        # Processing components
        all_instruments = self._get_instrument_list()
        self.ohlcv = OHLCVAggregator(window_sec=OHLCV_WINDOW_SEC)
        self.returns = RollingReturns(short_window=5, long_window=20)
        self.correlation = RollingCorrelation(
            symbols=all_instruments,
            window_size=CORRELATION_WINDOW,
            min_observations=30,
        )

        # QuestDB sender
        self.qdb_conf = f"tcp::addr={QUESTDB_INGEST};"

        # Stats
        self.tick_count = 0
        self.bar_count = 0
        self.feature_count = 0
        self.corr_count = 0

    def _get_instrument_list(self):
        """Get list of all instruments we might see."""
        from ingestion.main import PAIR_CONFIG
        return list(PAIR_CONFIG.keys())

    def run(self):
        """Main processing loop."""
        logger.info("Starting Standalone Stream Processor...")
        logger.info(f"OHLCV window: {OHLCV_WINDOW_SEC}s | "
                     f"Correlation window: {CORRELATION_WINDOW} ticks")

        try:
            with Sender.from_conf(self.qdb_conf) as sender:
                while True:
                    msg = self.consumer.poll(0.1)
                    if msg is None:
                        continue
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            continue
                        logger.error(f"Consumer error: {msg.error()}")
                        break

                    try:
                        self._process_tick(msg, sender)
                    except Exception as e:
                        logger.error(f"Error processing tick: {e}")

        except KeyboardInterrupt:
            logger.info("Stopping Stream Processor...")
        finally:
            self.consumer.close()
            self.producer.flush()
            logger.info(f"Stats: ticks={self.tick_count} bars={self.bar_count} "
                         f"features={self.feature_count} correlations={self.corr_count}")

    @staticmethod
    def _native(val):
        """Convert numpy scalars to Python native types for QuestDB."""
        if isinstance(val, (np.floating, np.float64, np.float32)):
            return float(val)
        if isinstance(val, (np.integer, np.int64, np.int32)):
            return int(val)
        return val

    def _native_dict(self, d: dict) -> dict:
        """Convert all values in a dict to native Python types."""
        return {k: self._native(v) for k, v in d.items()}

    def _process_tick(self, msg, sender: Sender):
        """Process a single tick message."""
        data = json.loads(msg.value().decode("utf-8"))

        if data.get("type") != "PRICE":
            return

        symbol = data.get("instrument")
        if not symbol:
            return

        bid = float(data["closeoutBid"])
        ask = float(data["closeoutAsk"])
        mid = float((bid + ask) / 2.0)
        spread = float(ask - bid)
        volume = float(data.get("bids", [{}])[0].get("liquidity", 0))
        ts = float(time.time())

        self.tick_count += 1

        # 1. Persist raw tick to QuestDB
        sender.row(
            "market_ticks",
            symbols={"symbol": symbol, "source": "MOCK"},
            columns={
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread": spread,
                "bid_volume": volume,
                "ask_volume": volume,
            },
            at=TimestampNanos.now(),
        )

        # 2. OHLCV aggregation
        completed_bar = self.ohlcv.ingest(symbol, mid, volume, ts)
        if completed_bar:
            self.bar_count += 1
            sender.row(
                "ohlcv_1m",
                symbols={"symbol": completed_bar["symbol"]},
                columns={
                    "open": float(completed_bar["open"]),
                    "high": float(completed_bar["high"]),
                    "low": float(completed_bar["low"]),
                    "close": float(completed_bar["close"]),
                    "volume": float(completed_bar["volume"]),
                    "tick_count": int(completed_bar["tick_count"]),
                    "vwap": float(completed_bar["vwap"]),
                },
                at=TimestampNanos.now(),
            )

        # 3. Rolling returns
        ret_features = self.returns.update(symbol, mid)

        # 4. Correlation matrix
        has_enough_data = self.correlation.update(symbol, mid)
        if has_enough_data and self.correlation.needs_recompute:
            corr_matrix = self.correlation.compute()
            if corr_matrix is not None:
                self.corr_count += 1
                # Persist upper triangle to QuestDB
                for sym_a, sym_b, corr_val in self.correlation.get_upper_triangle():
                    sender.row(
                        "correlation_matrix",
                        symbols={"symbol_a": sym_a, "symbol_b": sym_b},
                        columns={
                            "correlation": float(corr_val),
                            "window_size": int(CORRELATION_WINDOW),
                        },
                        at=TimestampNanos.now(),
                    )

        # 5. Publish feature vector to Kafka
        if ret_features:
            self.feature_count += 1
            feature_msg = {
                "symbol": symbol,
                "timestamp": ts,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread": spread,
                **ret_features,
            }
            self.producer.produce(
                "features",
                key=symbol,
                value=json.dumps(feature_msg),
            )
            self.producer.poll(0)

            # Persist to QuestDB
            sender.row(
                "model_features",
                symbols={"symbol": symbol},
                columns={k: float(v) for k, v in ret_features.items() if isinstance(v, (int, float, np.floating))},
                at=TimestampNanos.now(),
            )

        # Flush QuestDB periodically
        if self.tick_count % 100 == 0:
            sender.flush()

        # Log stats periodically
        if self.tick_count % 5000 == 0:
            logger.info(
                f"Processed {self.tick_count} ticks | "
                f"{self.bar_count} bars | "
                f"{self.feature_count} features | "
                f"{self.corr_count} correlation updates"
            )


# ─────────────────────────────────────────────────────────────
# PyFlink Stream Processor
# ─────────────────────────────────────────────────────────────
def run_pyflink():
    """
    PyFlink DataStream implementation.

    Connects to Redpanda via Flink's Kafka connector,
    processes ticks using Flink's windowing primitives,
    and sinks results to QuestDB.
    """
    try:
        from pyflink.datastream import StreamExecutionEnvironment
        from pyflink.datastream.connectors.kafka import (
            KafkaSource, KafkaOffsetsInitializer, KafkaSink,
            KafkaRecordSerializationSchema,
        )
        from pyflink.common.serialization import SimpleStringSchema
        from pyflink.common import WatermarkStrategy, Duration
        from pyflink.datastream.functions import (
            MapFunction, ProcessFunction, KeyedProcessFunction,
        )
        from pyflink.datastream.state import ValueStateDescriptor
        from pyflink.common.typeinfo import Types
        from pyflink.datastream.window import TumblingProcessingTimeWindows, Time
    except ImportError:
        logger.error(
            "PyFlink not installed. Install with: pip install apache-flink\n"
            "Also requires Java 11: brew install openjdk@11\n"
            "Falling back to standalone mode."
        )
        processor = StandaloneStreamProcessor()
        processor.run()
        return

    logger.info("Starting PyFlink Stream Processor...")

    # Create execution environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(2)

    # Add Kafka connector JARs
    # These need to be downloaded to a local jars directory
    jar_dir = os.path.join(os.path.dirname(__file__), "..", "jars")
    if os.path.exists(jar_dir):
        for jar in os.listdir(jar_dir):
            if jar.endswith(".jar"):
                env.add_jars(f"file://{os.path.abspath(os.path.join(jar_dir, jar))}")

    # Kafka Source
    kafka_source = (
        KafkaSource.builder()
        .set_bootstrap_servers(BROKER)
        .set_topics("ticks")
        .set_group_id("flink-stream-processor")
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )

    # Create DataStream
    watermark_strategy = (
        WatermarkStrategy
        .for_bounded_out_of_orderness(Duration.of_seconds(5))
    )

    tick_stream = env.from_source(
        kafka_source, watermark_strategy, "Kafka Ticks Source"
    )

    # Parse JSON ticks
    class ParseTick(MapFunction):
        def map(self, value):
            data = json.loads(value)
            if data.get("type") != "PRICE":
                return None
            symbol = data.get("instrument", "")
            bid = float(data.get("closeoutBid", 0))
            ask = float(data.get("closeoutAsk", 0))
            mid = (bid + ask) / 2.0
            volume = float(data.get("bids", [{}])[0].get("liquidity", 0))
            return json.dumps({
                "symbol": symbol, "bid": bid, "ask": ask,
                "mid": mid, "volume": volume,
                "timestamp": time.time(),
            })

    parsed = tick_stream.map(ParseTick()).filter(lambda x: x is not None)

    # Key by symbol for per-instrument processing
    keyed = parsed.key_by(lambda x: json.loads(x)["symbol"])

    # Process with stateful OHLCV aggregation
    class OHLCVProcess(KeyedProcessFunction):
        def open(self, runtime_context):
            self.bar_state = runtime_context.get_state(
                ValueStateDescriptor("bar", Types.STRING())
            )

        def process_element(self, value, ctx):
            data = json.loads(value)
            symbol = data["symbol"]
            mid = data["mid"]
            volume = data["volume"]
            ts = data["timestamp"]

            # Load or create bar state
            state_json = self.bar_state.value()
            if state_json:
                bar_data = json.loads(state_json)
                # Check window elapsed
                if ts - bar_data["start_ts"] >= OHLCV_WINDOW_SEC:
                    # Emit completed bar
                    yield json.dumps({"type": "ohlcv", **bar_data})
                    # New bar
                    bar_data = {
                        "symbol": symbol, "open": mid, "high": mid,
                        "low": mid, "close": mid, "volume": volume,
                        "tick_count": 1, "start_ts": ts,
                    }
                else:
                    bar_data["high"] = max(bar_data["high"], mid)
                    bar_data["low"] = min(bar_data["low"], mid)
                    bar_data["close"] = mid
                    bar_data["volume"] += volume
                    bar_data["tick_count"] += 1
            else:
                bar_data = {
                    "symbol": symbol, "open": mid, "high": mid,
                    "low": mid, "close": mid, "volume": volume,
                    "tick_count": 1, "start_ts": ts,
                }

            self.bar_state.update(json.dumps(bar_data))

            # Also emit the raw feature for downstream
            yield json.dumps({"type": "feature", **data})

    processed = keyed.process(OHLCVProcess())

    # Kafka Sink for features
    kafka_sink = (
        KafkaSink.builder()
        .set_bootstrap_servers(BROKER)
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
            .set_topic("features")
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
        )
        .build()
    )

    processed.sink_to(kafka_sink)

    # Execute
    logger.info("Submitting PyFlink job...")
    env.execute("CrossBorder-StreamProcessor")


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stream Processor")
    parser.add_argument(
        "--pyflink", action="store_true",
        help="Run using PyFlink (requires Java 11 + apache-flink)",
    )
    args = parser.parse_args()

    if args.pyflink:
        run_pyflink()
    else:
        processor = StandaloneStreamProcessor()
        processor.run()


if __name__ == "__main__":
    main()
