"""
Cointegration Testing Module
==============================
Implements Engle-Granger cointegration tests for candidate FX pairs.

Pipeline:
  1. Retrieve historical close prices from QuestDB
  2. Run ADF stationarity test on individual series
  3. Estimate hedge ratio (β) via OLS regression
  4. Test spread stationarity via ADF on residuals
  5. Filter pairs that pass cointegration at significance level
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("Cointegration")

try:
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning("statsmodels not installed. Cointegration tests disabled.")

try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


class CointegrationTester:
    """
    Tests candidate pairs for cointegration using the Engle-Granger method.

    Parameters
    ----------
    significance_level : float
        P-value threshold for ADF test (default 0.05).
    min_observations : int
        Minimum number of price observations required (default 100).
    questdb_dsn : str
        QuestDB PostgreSQL connection string.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_observations: int = 100,
        questdb_dsn: str = "host=localhost port=8812 dbname=qdb user=admin password=quest",
    ):
        self.significance_level = significance_level
        self.min_observations = min_observations
        self.questdb_dsn = questdb_dsn

    def fetch_prices(
        self, symbol: str, limit: int = 500
    ) -> Optional[pd.Series]:
        """
        Fetch close prices from QuestDB ohlcv_1m table.

        Falls back to in-memory data if QuestDB is unavailable.
        """
        if not HAS_PSYCOPG2:
            return None

        try:
            conn = psycopg2.connect(self.questdb_dsn)
            query = f"""
                SELECT timestamp, close
                FROM ohlcv_1m
                WHERE symbol = '{symbol}'
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            df = pd.read_sql(query, conn, parse_dates=["timestamp"])
            conn.close()

            if len(df) < self.min_observations:
                return None

            return df.set_index("timestamp")["close"].sort_index()
        except Exception as e:
            logger.debug(f"QuestDB fetch failed for {symbol}: {e}")
            return None

    def adf_test(self, series: np.ndarray) -> Tuple[float, float, bool]:
        """
        Augmented Dickey-Fuller test for stationarity.

        Returns
        -------
        (test_statistic, p_value, is_stationary)
        """
        if not HAS_STATSMODELS:
            return 0.0, 1.0, False

        try:
            result = adfuller(series, autolag="AIC")
            stat = result[0]
            pval = result[1]
            return stat, pval, pval < self.significance_level
        except Exception as e:
            logger.debug(f"ADF test failed: {e}")
            return 0.0, 1.0, False

    def estimate_hedge_ratio(
        self, y: np.ndarray, x: np.ndarray
    ) -> Tuple[float, float]:
        """
        OLS regression: y = α + β·x + ε

        Returns
        -------
        (beta, intercept)
        """
        if not HAS_STATSMODELS:
            # Fallback: simple ratio
            beta = np.std(y) / np.std(x) * np.sign(np.corrcoef(y, x)[0, 1])
            return float(beta), 0.0

        X = add_constant(x)
        model = OLS(y, X).fit()
        intercept = model.params[0]
        beta = model.params[1]
        return float(beta), float(intercept)

    def compute_spread(
        self, price_a: np.ndarray, price_b: np.ndarray, beta: float
    ) -> np.ndarray:
        """
        Compute the spread: Z_t = price_A - β × price_B
        """
        return price_a - beta * price_b

    def test_pair(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
    ) -> Dict:
        """
        Full Engle-Granger cointegration test on a pair.

        Returns
        -------
        dict with keys: cointegrated, beta, intercept, spread_adf_stat,
        spread_adf_pval, half_life
        """
        result = {
            "cointegrated": False,
            "beta": 0.0,
            "intercept": 0.0,
            "spread_adf_stat": 0.0,
            "spread_adf_pval": 1.0,
            "half_life": float("inf"),
        }

        if len(prices_a) < self.min_observations or len(prices_b) < self.min_observations:
            return result

        # Ensure same length
        n = min(len(prices_a), len(prices_b))
        a = prices_a[-n:]
        b = prices_b[-n:]

        # Step 1: Estimate hedge ratio via OLS
        beta, intercept = self.estimate_hedge_ratio(a, b)
        result["beta"] = beta
        result["intercept"] = intercept

        # Step 2: Compute spread
        spread = self.compute_spread(a, b, beta)

        # Step 3: ADF test on spread
        stat, pval, is_stationary = self.adf_test(spread)
        result["spread_adf_stat"] = stat
        result["spread_adf_pval"] = pval
        result["cointegrated"] = is_stationary

        # Step 4: Estimate half-life of mean reversion
        if is_stationary and len(spread) > 2:
            result["half_life"] = self._estimate_half_life(spread)

        return result

    def test_pair_from_db(self, symbol_a: str, symbol_b: str) -> Dict:
        """
        Fetch prices from QuestDB and test cointegration.
        """
        prices_a = self.fetch_prices(symbol_a)
        prices_b = self.fetch_prices(symbol_b)

        if prices_a is None or prices_b is None:
            logger.debug(f"Insufficient data for {symbol_a}/{symbol_b}")
            return {"cointegrated": False}

        # Align on timestamp
        combined = pd.concat([prices_a, prices_b], axis=1, join="inner")
        if len(combined) < self.min_observations:
            return {"cointegrated": False}

        return self.test_pair(combined.iloc[:, 0].values, combined.iloc[:, 1].values)

    def filter_candidates(
        self,
        candidate_pairs: List[Tuple[str, str, int]],
        price_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict]:
        """
        Filter candidate pairs by cointegration.

        Parameters
        ----------
        candidate_pairs : list of (symbol_a, symbol_b, cluster_id)
        price_data : optional dict of {symbol: price_array} for in-memory testing

        Returns
        -------
        List of dicts with cointegration results for pairs that pass.
        """
        passed = []

        for sym_a, sym_b, cluster_id in candidate_pairs:
            if price_data and sym_a in price_data and sym_b in price_data:
                result = self.test_pair(price_data[sym_a], price_data[sym_b])
            else:
                result = self.test_pair_from_db(sym_a, sym_b)

            result["symbol_a"] = sym_a
            result["symbol_b"] = sym_b
            result["cluster_id"] = cluster_id

            if result["cointegrated"]:
                logger.info(
                    f"  ✓ {sym_a}/{sym_b}: β={result['beta']:.4f} "
                    f"ADF p={result['spread_adf_pval']:.4f} "
                    f"half_life={result['half_life']:.1f}"
                )
                passed.append(result)
            else:
                logger.debug(
                    f"  ✗ {sym_a}/{sym_b}: ADF p={result['spread_adf_pval']:.4f}"
                )

        logger.info(f"Cointegration: {len(passed)}/{len(candidate_pairs)} pairs passed")
        return passed

    def _estimate_half_life(self, spread: np.ndarray) -> float:
        """
        Estimate mean-reversion half-life via AR(1) regression.

        spread_t = α + φ·spread_{t-1} + ε
        half_life = -ln(2) / ln(φ)
        """
        if not HAS_STATSMODELS:
            return float("inf")

        try:
            y = np.diff(spread)
            x = spread[:-1]
            X = add_constant(x)
            model = OLS(y, X).fit()
            phi = model.params[1]

            if phi >= 0:
                return float("inf")  # Not mean-reverting

            half_life = -np.log(2) / phi
            return max(half_life, 0.1)  # Floor at 0.1 bars
        except Exception:
            return float("inf")
