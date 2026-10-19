"""
Realistic Mock FX Stream Generator
===================================
Generates correlated multi-asset tick data with:
- Correlated Geometric Brownian Motion (GBM) via Cholesky decomposition
- Volatility clustering (GARCH-like regime switches)
- Mean-reverting spread dynamics
- Session-aware volume & tick frequency
- Occasional micro-inefficiencies (synthetic arbitrage windows)
"""

import os
import sys
import time
import random
import json
import math
import numpy as np
from datetime import datetime, timezone
from confluent_kafka import Producer
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

load_dotenv()
logger = get_logger("Ingestion")


# ─────────────────────────────────────────────────────────────
# Currency & Pair Configuration
# ─────────────────────────────────────────────────────────────

CURRENCIES = [
    "EUR", "USD", "JPY", "GBP", "AUD", "CAD", "CHF",
    "INR", "CNY", "RUB", "KWD", "SGD", "HKD", "BRL", "MXN",
]

# Pairs as (base, quote) with realistic mid prices, pip sizes, typical spread
PAIR_CONFIG = {
    # Majors
    "EUR_USD": {"mid": 1.0550, "pip": 0.0001, "spread_pips": 1.2,  "daily_vol": 0.006},
    "USD_JPY": {"mid": 149.50, "pip": 0.01,   "spread_pips": 1.5,  "daily_vol": 0.007},
    "GBP_USD": {"mid": 1.2650, "pip": 0.0001, "spread_pips": 1.8,  "daily_vol": 0.007},
    "AUD_USD": {"mid": 0.6550, "pip": 0.0001, "spread_pips": 1.5,  "daily_vol": 0.008},
    "USD_CAD": {"mid": 1.3550, "pip": 0.0001, "spread_pips": 2.0,  "daily_vol": 0.006},
    "USD_CHF": {"mid": 0.8750, "pip": 0.0001, "spread_pips": 2.0,  "daily_vol": 0.006},
    # Crosses
    "EUR_JPY": {"mid": 157.70, "pip": 0.01,   "spread_pips": 2.5,  "daily_vol": 0.008},
    "EUR_GBP": {"mid": 0.8340, "pip": 0.0001, "spread_pips": 2.0,  "daily_vol": 0.005},
    "GBP_JPY": {"mid": 189.10, "pip": 0.01,   "spread_pips": 3.5,  "daily_vol": 0.010},
    "EUR_CHF": {"mid": 0.9230, "pip": 0.0001, "spread_pips": 2.0,  "daily_vol": 0.004},
    "AUD_JPY": {"mid": 97.90,  "pip": 0.01,   "spread_pips": 3.0,  "daily_vol": 0.009},
    # Emerging
    "USD_INR": {"mid": 83.25,  "pip": 0.01,   "spread_pips": 5.0,  "daily_vol": 0.003},
    "USD_CNY": {"mid": 7.24,   "pip": 0.0001, "spread_pips": 8.0,  "daily_vol": 0.002},
    "USD_RUB": {"mid": 92.50,  "pip": 0.01,   "spread_pips": 30.0, "daily_vol": 0.015},
    "USD_KWD": {"mid": 0.3075, "pip": 0.0001, "spread_pips": 3.0,  "daily_vol": 0.001},
    "USD_SGD": {"mid": 1.3450, "pip": 0.0001, "spread_pips": 3.0,  "daily_vol": 0.004},
    "USD_HKD": {"mid": 7.8150, "pip": 0.0001, "spread_pips": 2.0,  "daily_vol": 0.001},
    "USD_BRL": {"mid": 4.9200, "pip": 0.0001, "spread_pips": 15.0, "daily_vol": 0.010},
    "USD_MXN": {"mid": 17.25,  "pip": 0.0001, "spread_pips": 10.0, "daily_vol": 0.008},
    # Cross-emerging
    "EUR_INR": {"mid": 87.83,  "pip": 0.01,   "spread_pips": 8.0,  "daily_vol": 0.005},
    "EUR_CNY": {"mid": 7.64,   "pip": 0.0001, "spread_pips": 10.0, "daily_vol": 0.004},
    "GBP_INR": {"mid": 105.30, "pip": 0.01,   "spread_pips": 10.0, "daily_vol": 0.006},
    "JPY_INR": {"mid": 0.5570, "pip": 0.0001, "spread_pips": 5.0,  "daily_vol": 0.004},
    "EUR_SGD": {"mid": 1.4190, "pip": 0.0001, "spread_pips": 5.0,  "daily_vol": 0.005},
    "GBP_HKD": {"mid": 9.8850, "pip": 0.0001, "spread_pips": 8.0,  "daily_vol": 0.005},
    "AUD_SGD": {"mid": 0.8810, "pip": 0.0001, "spread_pips": 5.0,  "daily_vol": 0.006},
}

# Cross-currency price relationships for triangular consistency
# E.g., EUR_JPY ≈ EUR_USD × USD_JPY
TRIANGULAR_RELATIONSHIPS = [
    ("EUR_JPY", "EUR_USD", "USD_JPY", lambda a, b: a * b),
    ("EUR_GBP", "EUR_USD", "GBP_USD", lambda a, b: a / b),
    ("GBP_JPY", "GBP_USD", "USD_JPY", lambda a, b: a * b),
    ("EUR_CHF", "EUR_USD", "USD_CHF", lambda a, b: a * b),
    ("AUD_JPY", "AUD_USD", "USD_JPY", lambda a, b: a * b),
    ("EUR_INR", "EUR_USD", "USD_INR", lambda a, b: a * b),
    ("EUR_CNY", "EUR_USD", "USD_CNY", lambda a, b: a * b),
    ("GBP_INR", "GBP_USD", "USD_INR", lambda a, b: a * b),
    ("EUR_SGD", "EUR_USD", "USD_SGD", lambda a, b: a * b),
    ("GBP_HKD", "GBP_USD", "USD_HKD", lambda a, b: a * b),
    ("AUD_SGD", "AUD_USD", "USD_SGD", lambda a, b: a * b),
]


class VolatilityRegime:
    """GARCH-like volatility regime switching."""

    def __init__(self, base_vol: float):
        self.base_vol = base_vol
        self.current_vol = base_vol
        self.regime = "normal"  # normal | elevated | spike
        self.regime_ticks_remaining = 0

    def step(self) -> float:
        """Returns current tick volatility and potentially switches regime."""
        if self.regime_ticks_remaining <= 0:
            # Regime transition probabilities
            r = random.random()
            if self.regime == "normal":
                if r < 0.002:    # 0.2% chance of spike
                    self.regime = "spike"
                    self.regime_ticks_remaining = random.randint(5, 30)
                    self.current_vol = self.base_vol * random.uniform(3.0, 6.0)
                elif r < 0.02:   # 2% chance of elevated
                    self.regime = "elevated"
                    self.regime_ticks_remaining = random.randint(50, 200)
                    self.current_vol = self.base_vol * random.uniform(1.5, 2.5)
                else:
                    self.regime_ticks_remaining = random.randint(100, 500)
                    self.current_vol = self.base_vol
            elif self.regime == "elevated":
                if r < 0.05:
                    self.regime = "spike"
                    self.regime_ticks_remaining = random.randint(5, 20)
                    self.current_vol = self.base_vol * random.uniform(4.0, 8.0)
                else:
                    self.regime = "normal"
                    self.regime_ticks_remaining = random.randint(200, 1000)
                    self.current_vol = self.base_vol
            else:  # spike
                self.regime = "elevated"
                self.regime_ticks_remaining = random.randint(30, 100)
                self.current_vol = self.base_vol * random.uniform(1.5, 2.5)

        self.regime_ticks_remaining -= 1
        return self.current_vol


class RealisticFXGenerator:
    """
    Multi-asset correlated FX tick generator.

    Uses per-tick GBM steps on "base" USD pairs, then derives cross rates
    via triangular relationships. Adds realistic spread dynamics and
    occasional micro-inefficiencies to create synthetic arbitrage windows.
    """

    # Base pairs whose prices are generated independently (USD-denominated)
    BASE_PAIRS = [
        "EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD",
        "USD_CAD", "USD_CHF", "USD_INR", "USD_CNY",
        "USD_RUB", "USD_KWD", "USD_SGD", "USD_HKD",
        "USD_BRL", "USD_MXN",
    ]

    def __init__(self, topic: str = "ticks", broker: str = "localhost:19092"):
        self.topic = topic
        self.producer = Producer({"bootstrap.servers": broker})

        # Current mid prices
        self.prices = {pair: cfg["mid"] for pair, cfg in PAIR_CONFIG.items()}

        # Volatility regimes per base pair
        # Convert daily vol to per-tick vol (assuming ~50,000 ticks/day)
        ticks_per_day = 50_000
        self.vol_regimes = {}
        for pair in self.BASE_PAIRS:
            daily_vol = PAIR_CONFIG[pair]["daily_vol"]
            tick_vol = daily_vol / math.sqrt(ticks_per_day)
            self.vol_regimes[pair] = VolatilityRegime(tick_vol)

        # Correlation matrix for base pairs (simplified)
        # EUR, JPY, GBP, AUD, CAD, CHF, INR, CNY, RUB, KWD, SGD, HKD, BRL, MXN
        n = len(self.BASE_PAIRS)
        self.correlation = np.eye(n)
        # Set some realistic correlations
        pair_idx = {p: i for i, p in enumerate(self.BASE_PAIRS)}

        corr_pairs = [
            ("EUR_USD", "GBP_USD", 0.7),
            ("EUR_USD", "USD_CHF", -0.85),
            ("EUR_USD", "AUD_USD", 0.5),
            ("USD_JPY", "AUD_USD", 0.3),
            ("GBP_USD", "AUD_USD", 0.45),
            ("USD_CAD", "AUD_USD", -0.4),
            ("USD_INR", "USD_CNY", 0.3),
            ("USD_BRL", "USD_MXN", 0.6),
            ("USD_SGD", "USD_HKD", 0.4),
        ]
        for p1, p2, rho in corr_pairs:
            if p1 in pair_idx and p2 in pair_idx:
                i, j = pair_idx[p1], pair_idx[p2]
                self.correlation[i, j] = rho
                self.correlation[j, i] = rho

        # Cholesky decomposition for correlated random numbers
        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(self.correlation)
        if eigvals.min() < 0:
            self.correlation += (-eigvals.min() + 0.01) * np.eye(n)
        self.cholesky = np.linalg.cholesky(self.correlation)

        # Tick counter for micro-inefficiency injection
        self.tick_count = 0
        self.arb_window_active = False
        self.arb_window_remaining = 0

        # Mean-reversion anchor (prices tend to drift back slowly)
        self.anchor_prices = {pair: cfg["mid"] for pair, cfg in PAIR_CONFIG.items()}
        self.mean_reversion_strength = 0.0001  # Very weak pull

        logger.info(f"Initialized RealisticFXGenerator with {len(PAIR_CONFIG)} pairs")
        logger.info(f"Base pairs: {len(self.BASE_PAIRS)}, "
                     f"Derived pairs: {len(PAIR_CONFIG) - len(self.BASE_PAIRS)}")

    def _step_base_prices(self):
        """Generate correlated price moves for base USD pairs."""
        n = len(self.BASE_PAIRS)

        # Correlated standard normals
        z = np.random.standard_normal(n)
        correlated_z = self.cholesky @ z

        for i, pair in enumerate(self.BASE_PAIRS):
            vol = self.vol_regimes[pair].step()
            price = self.prices[pair]

            # GBM step: dS = S * (mu*dt + sigma*dW)
            # mu includes weak mean-reversion toward anchor
            anchor = self.anchor_prices[pair]
            mean_rev = self.mean_reversion_strength * (anchor - price) / anchor
            drift = mean_rev
            diffusion = vol * correlated_z[i]

            new_price = price * (1.0 + drift + diffusion)

            # Floor at a small positive value
            self.prices[pair] = max(new_price, PAIR_CONFIG[pair]["mid"] * 0.5)

    def _derive_cross_prices(self):
        """
        Derive cross-pair prices from base pairs via triangular relationships.
        Adds small noise to simulate real-world imperfect arbitrage.
        """
        for cross, leg1, leg2, formula in TRIANGULAR_RELATIONSHIPS:
            if leg1 in self.prices and leg2 in self.prices:
                fair_value = formula(self.prices[leg1], self.prices[leg2])
                # Add tiny noise (0.01% - simulates market microstructure)
                noise = random.gauss(0, fair_value * 0.0001)
                self.prices[cross] = fair_value + noise

        # JPY_INR is special: JPY/USD * USD/INR = JPY_INR (but JPY/USD = 1/USD_JPY)
        if "USD_JPY" in self.prices and "USD_INR" in self.prices:
            jpy_usd = 1.0 / self.prices["USD_JPY"]
            self.prices["JPY_INR"] = jpy_usd * self.prices["USD_INR"]

    def _inject_micro_inefficiency(self):
        """
        Occasionally skew a cross-pair price to create a brief arbitrage window.
        This simulates real-world latency-driven opportunities.
        """
        self.tick_count += 1

        if self.arb_window_active:
            self.arb_window_remaining -= 1
            if self.arb_window_remaining <= 0:
                self.arb_window_active = False
                # Snap prices back to fair value
                self._derive_cross_prices()
            return

        # ~0.1% chance per tick of opening an arb window
        if random.random() < 0.001:
            self.arb_window_active = True
            self.arb_window_remaining = random.randint(3, 15)  # 3-15 ticks

            # Pick a random cross pair and skew it
            cross_pairs = [t[0] for t in TRIANGULAR_RELATIONSHIPS]
            target = random.choice(cross_pairs)

            # Skew by 0.05% - 0.2% (after fees, might not be profitable)
            skew = random.choice([-1, 1]) * random.uniform(0.0005, 0.002)
            self.prices[target] *= (1.0 + skew)

    def _compute_spread(self, pair: str) -> float:
        """
        Dynamic spread based on:
        - Base spread for the pair
        - Current volatility regime
        - Random microstructure noise
        """
        cfg = PAIR_CONFIG[pair]
        base_spread = cfg["spread_pips"] * cfg["pip"]

        # Widen spread during high vol
        vol_regime = self.vol_regimes.get(pair)
        if vol_regime and vol_regime.regime == "spike":
            base_spread *= random.uniform(2.0, 4.0)
        elif vol_regime and vol_regime.regime == "elevated":
            base_spread *= random.uniform(1.2, 1.8)

        # Random microstructure noise (±20%)
        base_spread *= random.uniform(0.8, 1.2)

        return max(base_spread, cfg["pip"])

    def _compute_volume(self, pair: str) -> float:
        """Simulate realistic liquidity (notional in base currency units)."""
        if "KWD" in pair or "RUB" in pair:
            base_volume = random.uniform(50_000, 500_000)
        elif "INR" in pair or "CNY" in pair or "BRL" in pair or "MXN" in pair:
            base_volume = random.uniform(100_000, 2_000_000)
        else:
            base_volume = random.uniform(500_000, 10_000_000)

        # Round to nearest 1000
        return round(base_volume / 1000) * 1000

    def generate_tick(self, pair: str) -> dict:
        """Generate a single tick for the given pair."""
        mid = self.prices[pair]
        spread = self._compute_spread(pair)
        half_spread = spread / 2.0

        bid = mid - half_spread
        ask = mid + half_spread
        volume = self._compute_volume(pair)

        # Round to appropriate precision
        pip = PAIR_CONFIG[pair]["pip"]
        decimals = max(0, -int(math.log10(pip)) + 1) if pip < 1 else 2
        bid = round(bid, decimals)
        ask = round(ask, decimals)

        tick = {
            "type": "PRICE",
            "time": datetime.now(timezone.utc).isoformat(),
            "instrument": pair,
            "bids": [{"price": str(bid), "liquidity": int(volume)}],
            "asks": [{"price": str(ask), "liquidity": int(volume)}],
            "closeoutBid": str(bid),
            "closeoutAsk": str(ask),
            "status": "tradeable",
            "tradeable": True,
        }
        return tick

    def delivery_report(self, err, msg):
        if err is not None:
            logger.error(f"Message delivery failed: {err}")

    def run(self):
        """Main generator loop."""
        logger.info(f"Starting Realistic FX Stream to topic '{self.topic}'...")
        logger.info(f"Generating ticks for {len(PAIR_CONFIG)} currency pairs")
        logger.info("Volatility regimes: normal / elevated / spike")
        logger.info("Triangular consistency with micro-inefficiency injection")

        pairs = list(PAIR_CONFIG.keys())
        batch_count = 0

        try:
            while True:
                # Step 1: Move base prices (correlated GBM)
                self._step_base_prices()

                # Step 2: Derive cross rates (triangular consistency)
                self._derive_cross_prices()

                # Step 3: Occasionally inject micro-inefficiency
                self._inject_micro_inefficiency()

                # Step 4: Emit ticks for a random subset of pairs
                # (not all pairs tick simultaneously — realistic)
                num_pairs_this_tick = random.randint(1, min(5, len(pairs)))
                active_pairs = random.sample(pairs, num_pairs_this_tick)

                for pair in active_pairs:
                    tick = self.generate_tick(pair)
                    self.producer.produce(
                        self.topic,
                        key=tick["instrument"],
                        value=json.dumps(tick),
                        callback=self.delivery_report,
                    )

                self.producer.poll(0)
                batch_count += 1

                # Log stats periodically
                if batch_count % 1000 == 0:
                    sample_pair = "EUR_USD"
                    regime = self.vol_regimes[sample_pair].regime
                    arb_status = "ARB_WINDOW" if self.arb_window_active else "normal"
                    logger.info(
                        f"Batch {batch_count}: EUR_USD={self.prices[sample_pair]:.5f} "
                        f"regime={regime} market={arb_status}"
                    )

                # Realistic inter-tick delay: 10ms - 80ms
                time.sleep(random.uniform(0.01, 0.08))

        except KeyboardInterrupt:
            logger.info("Stopping FX Stream...")
        finally:
            self.producer.flush()
            logger.info("Producer flushed and stopped.")


def main():
    broker = os.getenv("REDPANDA_BROKER", "localhost:19092")
    logger.info("Starting OANDA Ingestion Service (REALISTIC MOCK MODE)...")
    generator = RealisticFXGenerator(broker=broker)
    generator.run()


if __name__ == "__main__":
    main()