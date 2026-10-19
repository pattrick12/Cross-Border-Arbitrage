"""
Econometric Feature Extraction — ARIMA + GARCH
================================================
Extracts time-series features from OHLCV data:

1. ARIMA(p,d,q) → conditional mean forecast
2. GARCH(1,1) → conditional volatility forecast

These features feed into the LSTM as additional input channels.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import get_logger

logger = get_logger("Econometrics")
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False
    logger.warning("statsmodels not installed. ARIMA disabled.")

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    logger.warning("arch library not installed. GARCH disabled.")


class ARIMAExtractor:
    """
    Fits ARIMA(p,d,q) to a return series and extracts
    the conditional mean forecast.

    Parameters
    ----------
    order : tuple (p, d, q)
        ARIMA order (default: (2, 0, 1) for log-returns).
    """

    def __init__(self, order: Tuple[int, int, int] = (2, 0, 1)):
        self.order = order
        self._model = None
        self._fitted = None

    def fit(self, returns: np.ndarray) -> bool:
        """
        Fit ARIMA model to the return series.

        Returns True if fitting succeeded.
        """
        if not HAS_ARIMA:
            return False

        if len(returns) < 30:
            return False

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = ARIMA(returns, order=self.order)
                self._fitted = self._model.fit(
                    method_kwargs={"maxiter": 200, "disp": False}
                )
            return True
        except Exception as e:
            logger.debug(f"ARIMA fit failed: {e}")
            return False

    def forecast(self, steps: int = 1) -> float:
        """
        Forecast the conditional mean for the next `steps` periods.

        Returns the 1-step-ahead forecast.
        """
        if self._fitted is None:
            return 0.0

        try:
            fc = self._fitted.forecast(steps=steps)
            return float(fc.iloc[0]) if hasattr(fc, "iloc") else float(fc[0])
        except Exception:
            return 0.0

    def get_residuals(self) -> Optional[np.ndarray]:
        """Get model residuals (needed for GARCH)."""
        if self._fitted is None:
            return None
        try:
            return self._fitted.resid.values
        except Exception:
            return None


class GARCHExtractor:
    """
    Fits GARCH(1,1) to return residuals and extracts
    the conditional volatility forecast.

    Parameters
    ----------
    p : int
        GARCH lag order (default: 1).
    q : int
        ARCH lag order (default: 1).
    dist : str
        Error distribution: 'normal', 'studentst', 'skewt' (default: 'normal').
    """

    def __init__(self, p: int = 1, q: int = 1, dist: str = "normal"):
        self.p = p
        self.q = q
        self.dist = dist
        self._model = None
        self._fitted = None

    def fit(self, returns: np.ndarray) -> bool:
        """
        Fit GARCH(p,q) to the return series.

        Returns True if fitting succeeded.
        """
        if not HAS_ARCH:
            return False

        if len(returns) < 50:
            return False

        try:
            # Scale returns (arch library works better with ×100 returns)
            scaled = returns * 100

            self._model = arch_model(
                scaled,
                vol="Garch",
                p=self.p,
                q=self.q,
                dist=self.dist,
                rescale=False,
            )
            self._fitted = self._model.fit(
                disp="off",
                show_warning=False,
            )
            return True
        except Exception as e:
            logger.debug(f"GARCH fit failed: {e}")
            return False

    def forecast_volatility(self, horizon: int = 1) -> float:
        """
        Forecast conditional volatility for the next period.

        Returns annualized volatility estimate.
        """
        if self._fitted is None:
            return 0.0

        try:
            fc = self._fitted.forecast(horizon=horizon)
            # fc.variance is in scaled (×100) units, convert back
            var_forecast = fc.variance.iloc[-1, 0]
            vol = np.sqrt(var_forecast) / 100  # Un-scale
            return float(vol)
        except Exception:
            return 0.0

    def get_conditional_volatility(self) -> Optional[np.ndarray]:
        """Get the full conditional volatility series."""
        if self._fitted is None:
            return None
        try:
            return self._fitted.conditional_volatility.values / 100
        except Exception:
            return None


class EconometricFeatureExtractor:
    """
    Combined ARIMA + GARCH feature extractor.

    Usage:
        extractor = EconometricFeatureExtractor()
        features = extractor.extract(returns_array)
        # features = {'arima_mean': 0.001, 'garch_vol': 0.015}
    """

    def __init__(
        self,
        arima_order: Tuple[int, int, int] = (2, 0, 1),
        garch_p: int = 1,
        garch_q: int = 1,
        refit_interval: int = 100,
    ):
        self.arima = ARIMAExtractor(order=arima_order)
        self.garch = GARCHExtractor(p=garch_p, q=garch_q)
        self.refit_interval = refit_interval
        self._steps_since_fit = 0
        self._is_fitted = False

    def fit(self, returns: np.ndarray) -> bool:
        """Fit both ARIMA and GARCH models."""
        arima_ok = self.arima.fit(returns)
        garch_ok = self.garch.fit(returns)
        self._is_fitted = arima_ok or garch_ok
        self._steps_since_fit = 0
        return self._is_fitted

    def extract(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Extract ARIMA mean and GARCH volatility features.

        Auto-refits the models periodically.
        """
        self._steps_since_fit += 1

        # Refit periodically
        if not self._is_fitted or self._steps_since_fit >= self.refit_interval:
            self.fit(returns)

        return {
            "arima_mean": self.arima.forecast(steps=1),
            "garch_vol": self.garch.forecast_volatility(horizon=1),
        }

    def extract_batch(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for multiple assets.

        Parameters
        ----------
        returns_df : DataFrame with columns = asset symbols, rows = time steps.

        Returns
        -------
        DataFrame with columns: symbol, arima_mean, garch_vol
        """
        results = []
        for col in returns_df.columns:
            rets = returns_df[col].dropna().values
            if len(rets) < 50:
                results.append({"symbol": col, "arima_mean": 0.0, "garch_vol": 0.0})
                continue

            features = self.extract(rets)
            features["symbol"] = col
            results.append(features)

        return pd.DataFrame(results)
