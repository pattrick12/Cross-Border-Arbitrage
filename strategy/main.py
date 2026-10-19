"""
Strategy Execution Engine
==========================
The final orchestrator that ties all Phase 2 components together.

Flow per tick:
    1. Consume features from Kafka 'features' topic
    2. Get OU z-score signal (stat-arb)
    3. Get ML signal (XGBoost P(up))
    4. Run Monte Carlo → CVaR
    5. If safe → compute position size (Kelly)
    6. Publish execution order to Kafka 'execution_orders'

Also runs periodic background tasks:
    - Graph embedding refresh (every 5 minutes)
    - Cointegration re-testing (every 10 minutes)
    - ML model re-training (every hour)
"""

import os
import sys
import json
import time
import threading
import numpy as np
from collections import defaultdict
from typing import Dict, Optional
from confluent_kafka import Consumer, Producer, KafkaError
from questdb.ingress import Sender, TimestampNanos
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from strategy.stat_arb import StatArbManager, Signal
from strategy.cointegration import CointegrationTester
from strategy.graph_embeddings import CurrencyGraphEmbedder
from strategy.ml.pipeline import MLPipeline
from risk.monte_carlo import MonteCarloEngine
from risk.position_sizing import KellyPositionSizer
from streaming.correlation import RollingCorrelation

load_dotenv()
logger = get_logger("StrategyEngine")

BROKER = os.getenv("REDPANDA_BROKER", "localhost:19092")
QUESTDB_INGEST = os.getenv("QUESTDB_INGEST", "localhost:9009")

# All tracked instruments
INSTRUMENTS = list({
    "EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CAD", "USD_CHF",
    "EUR_JPY", "EUR_GBP", "GBP_JPY", "EUR_CHF", "AUD_JPY",
    "USD_INR", "USD_CNY", "USD_RUB", "USD_KWD",
    "USD_SGD", "USD_HKD", "USD_BRL", "USD_MXN",
    "EUR_INR", "EUR_CNY", "GBP_INR", "JPY_INR",
    "EUR_SGD", "GBP_HKD", "AUD_SGD",
})


class StrategyEngine:
    """
    Main strategy engine that orchestrates all Phase 2 components.
    """

    def __init__(self, capital: float = 100_000):
        self.capital = capital

        # Kafka
        self.consumer = Consumer({
            "bootstrap.servers": BROKER,
            "group.id": "strategy-engine-group",
            "auto.offset.reset": "latest",
        })
        self.consumer.subscribe(["features"])
        self.producer = Producer({"bootstrap.servers": BROKER})

        # Components
        self.correlation = RollingCorrelation(
            symbols=INSTRUMENTS,
            window_size=300,
            min_observations=50,
        )
        self.graph_embedder = CurrencyGraphEmbedder(
            symbols=INSTRUMENTS,
            embedding_dim=16,
            n_clusters=4,
        )
        self.coint_tester = CointegrationTester()
        self.stat_arb = StatArbManager()
        self.ml_pipeline = MLPipeline(model_dir="./models")
        self.mc_engine = MonteCarloEngine(n_paths=10_000)
        self.position_sizer = KellyPositionSizer(
            max_kelly_fraction=float(os.getenv("MAX_KELLY_FRACTION", "0.5")),
            max_position_pct=float(os.getenv("MAX_POSITION_PCT", "0.10")),
        )

        # State
        self.current_prices: Dict[str, float] = {}
        self.current_features: Dict[str, Dict] = {}
        self.price_history: Dict[str, list] = defaultdict(list)
        self.tick_count = 0
        self.signal_count = 0
        self.initialized = False

        # Timing for periodic tasks
        self._last_graph_update = 0
        self._last_coint_test = 0
        self._graph_update_interval = 300   # 5 minutes
        self._coint_test_interval = 600     # 10 minutes

        # QuestDB
        self.qdb_conf = f"tcp::addr={QUESTDB_INGEST};"

    def run(self):
        """Main event loop."""
        logger.info("=" * 60)
        logger.info("  STRATEGY ENGINE - Phase 2")
        logger.info(f"  Capital: ${self.capital:,.0f}")
        logger.info(f"  Instruments: {len(INSTRUMENTS)}")
        logger.info("=" * 60)

        # Try to load pre-trained ML models
        if self.ml_pipeline.load():
            logger.info("Loaded pre-trained ML models.")
        else:
            logger.info("No pre-trained models found. Will train when enough data accumulates.")

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
                        self._process_feature(msg, sender)
                    except Exception as e:
                        logger.error(f"Error processing feature: {e}", exc_info=True)

        except KeyboardInterrupt:
            logger.info("Stopping Strategy Engine...")
        finally:
            self.consumer.close()
            self.producer.flush()
            logger.info(f"Strategy Engine stopped. "
                         f"Processed {self.tick_count} features, "
                         f"generated {self.signal_count} signals.")

    def _process_feature(self, msg, sender: Sender):
        """Process a single feature message from the streaming processor."""
        data = json.loads(msg.value().decode("utf-8"))
        symbol = data.get("symbol")
        if not symbol:
            return

        mid = data.get("mid", 0)
        self.current_prices[symbol] = mid
        self.current_features[symbol] = data
        self.price_history[symbol].append(mid)

        # Keep price history bounded
        if len(self.price_history[symbol]) > 5000:
            self.price_history[symbol] = self.price_history[symbol][-5000:]

        self.tick_count += 1

        # Update correlation
        self.correlation.update(symbol, mid)

        # Run periodic tasks
        now = time.time()
        if now - self._last_graph_update > self._graph_update_interval:
            self._update_graph_embeddings()
            self._last_graph_update = now

        if now - self._last_coint_test > self._coint_test_interval:
            self._update_cointegration()
            self._last_coint_test = now

        # === SIGNAL GENERATION ===

        # 1. Stat-arb signals (OU z-score)
        ou_signals = self.stat_arb.update(self.current_prices)

        for sig in ou_signals:
            self._handle_signal(sig, sender)

        # 2. ML prediction (if trained)
        if self.ml_pipeline.is_trained and self.tick_count % 10 == 0:
            features = {
                "log_return": data.get("log_return", 0),
                "rolling_mean_5": data.get("rolling_mean_5", 0),
                "rolling_std_5": data.get("rolling_std_5", 0),
                "rolling_mean_20": data.get("rolling_mean_20", 0),
                "rolling_std_20": data.get("rolling_std_20", 0),
                "arima_mean": 0,
                "garch_vol": 0,
                "ou_zscore": 0,
            }

            # Get OU z-score for this symbol if available
            zscores = self.stat_arb.get_all_zscores(self.current_prices)
            for key, z in zscores.items():
                if symbol in key:
                    features["ou_zscore"] = z
                    break

            graph_emb = self.graph_embedder.get_embedding_vector(symbol)
            prob_up = self.ml_pipeline.predict(symbol, features, graph_emb)

            # Generate signal if strong prediction
            if prob_up > 0.65 or prob_up < 0.35:
                self._handle_ml_signal(symbol, prob_up, sender)

        # Log periodically
        if self.tick_count % 5000 == 0:
            n_pairs = len(self.stat_arb.models)
            logger.info(
                f"Tick {self.tick_count}: {len(self.current_prices)} prices | "
                f"{n_pairs} stat-arb pairs | {self.signal_count} signals"
            )

    def _handle_signal(self, sig, sender: Sender):
        """Process a stat-arb signal through risk engine."""
        # Get current vol estimate
        returns = np.diff(self.price_history.get(sig.symbol_a, [0, 0]))
        current_vol = float(np.std(returns[-20:])) if len(returns) > 20 else 0.01

        # Monte Carlo risk check
        cov = self.correlation.get_covariance_matrix()
        if cov is not None and cov.shape[0] > 1:
            weights = np.ones(cov.shape[0]) / cov.shape[0]
            risk = self.mc_engine.run(cov, weights)
        else:
            risk = self.mc_engine.run_single_asset(current_vol)

        # Position sizing
        win_prob = sig.confidence
        position = self.position_sizer.size_position(
            capital=self.capital,
            win_prob=win_prob,
            risk_metrics=risk,
            current_vol=current_vol,
        )

        if position.should_trade:
            self.signal_count += 1
            order = {
                "type": "stat_arb",
                "signal": sig.signal.value,
                "symbol_a": sig.symbol_a,
                "symbol_b": sig.symbol_b,
                "z_score": sig.z_score,
                "confidence": sig.confidence,
                "position_size": position.position_size,
                "position_pct": position.position_pct,
                "kelly": position.kelly_fraction,
                "var_95": risk.var_95,
                "cvar_95": risk.cvar_95,
                "timestamp": time.time(),
            }

            self.producer.produce(
                "execution_orders",
                key=f"{sig.symbol_a}_{sig.symbol_b}",
                value=json.dumps(order),
            )
            self.producer.poll(0)

            # Persist to QuestDB
            sender.row(
                "risk_metrics",
                symbols={"symbol": f"{sig.symbol_a}/{sig.symbol_b}"},
                columns={
                    "var_95": risk.var_95,
                    "cvar_95": risk.cvar_95,
                    "kelly_fraction": position.kelly_fraction,
                    "position_size": position.position_size,
                    "volatility": current_vol,
                    "correlation_avg": risk.correlation_avg,
                },
                at=TimestampNanos.now(),
            )
            sender.flush()

            logger.info(
                f"SIGNAL: {sig.signal.value} {sig.symbol_a}/{sig.symbol_b} "
                f"z={sig.z_score:.2f} size={position.position_size:.0f} "
                f"CVaR={risk.cvar_95:.4f}"
            )

    def _handle_ml_signal(self, symbol: str, prob_up: float, sender: Sender):
        """Process an ML-generated signal."""
        direction = "BUY" if prob_up > 0.5 else "SELL"

        returns = np.diff(self.price_history.get(symbol, [0, 0]))
        current_vol = float(np.std(returns[-20:])) if len(returns) > 20 else 0.01

        risk = self.mc_engine.run_single_asset(current_vol)

        position = self.position_sizer.size_position(
            capital=self.capital,
            win_prob=prob_up if prob_up > 0.5 else 1 - prob_up,
            risk_metrics=risk,
            current_vol=current_vol,
        )

        if position.should_trade:
            self.signal_count += 1
            order = {
                "type": "ml_directional",
                "signal": direction,
                "symbol": symbol,
                "probability": prob_up,
                "position_size": position.position_size,
                "position_pct": position.position_pct,
                "kelly": position.kelly_fraction,
                "cvar_95": risk.cvar_95,
                "timestamp": time.time(),
            }

            self.producer.produce(
                "execution_orders",
                key=symbol,
                value=json.dumps(order),
            )
            self.producer.poll(0)

    def _update_graph_embeddings(self):
        """Periodic graph embedding refresh."""
        if not self.correlation.is_ready:
            return

        corr = self.correlation.compute()
        if corr is None:
            return

        logger.info("Refreshing graph embeddings...")
        try:
            pairs = self.graph_embedder.run_pipeline(corr)
            logger.info(f"Graph pipeline: {len(pairs)} candidate pairs")
        except Exception as e:
            logger.error(f"Graph embedding failed: {e}")

    def _update_cointegration(self):
        """Periodic cointegration re-testing."""
        if not self.graph_embedder.cluster_pairs:
            return

        logger.info("Re-testing cointegration...")

        # Build price data dict for in-memory testing
        price_data = {
            sym: np.array(hist)
            for sym, hist in self.price_history.items()
            if len(hist) > 100
        }

        try:
            passed = self.coint_tester.filter_candidates(
                self.graph_embedder.cluster_pairs,
                price_data=price_data,
            )

            # Update stat-arb models
            for pair_info in passed:
                self.stat_arb.add_pair(
                    pair_info["symbol_a"],
                    pair_info["symbol_b"],
                    pair_info["beta"],
                )

                # Calibrate with available data
                a_data = price_data.get(pair_info["symbol_a"])
                b_data = price_data.get(pair_info["symbol_b"])
                if a_data is not None and b_data is not None:
                    key = f"{pair_info['symbol_a']}/{pair_info['symbol_b']}"
                    if key in self.stat_arb.models:
                        n = min(len(a_data), len(b_data))
                        spread = a_data[-n:] - pair_info["beta"] * b_data[-n:]
                        self.stat_arb.models[key].calibrate(spread)

        except Exception as e:
            logger.error(f"Cointegration update failed: {e}")


def main():
    capital = float(os.getenv("INITIAL_CAPITAL", "100000"))
    engine = StrategyEngine(capital=capital)
    engine.run()


if __name__ == "__main__":
    main()