"""
Statistical Arbitrage — Ornstein-Uhlenbeck Process
====================================================
Implements the core stat-arb engine:

1. Calibrate OU process parameters (θ, μ, σ) via Maximum Likelihood Estimation
2. Compute real-time z-scores for trade signaling
3. Generate entry/exit signals based on z-score thresholds

The OU process models a mean-reverting spread:
    dX_t = θ(μ - X_t)dt + σ dW_t

Where:
    θ = mean-reversion speed (higher = faster reversion)
    μ = long-term mean (equilibrium level)
    σ = volatility of the process
"""

import os
import sys
import time
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("StatArb")

try:
    from scipy.optimize import minimize
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not installed. OU calibration disabled.")


class Signal(Enum):
    """Trading signal from OU model."""
    LONG_SPREAD = "LONG"      # z < -entry_threshold → BUY spread
    SHORT_SPREAD = "SHORT"    # z > +entry_threshold → SELL spread
    EXIT = "EXIT"             # z crosses zero → close position
    HOLD = "HOLD"             # no action


@dataclass
class OUParameters:
    """Calibrated OU process parameters."""
    theta: float = 0.0      # Mean-reversion speed
    mu: float = 0.0         # Long-term mean
    sigma: float = 0.0      # Volatility
    sigma_eq: float = 0.0   # Equilibrium std dev = σ / √(2θ)
    half_life: float = 0.0  # Half-life = ln(2) / θ
    log_likelihood: float = float("-inf")
    calibrated: bool = False


@dataclass
class OUSignal:
    """Output signal from the OU model."""
    symbol_a: str
    symbol_b: str
    signal: Signal
    z_score: float
    spread: float
    params: OUParameters
    confidence: float = 0.0
    timestamp: float = 0.0


class OrnsteinUhlenbeck:
    """
    OU process calibration and signal generation for a single pair.

    Parameters
    ----------
    symbol_a : str
        First instrument of the spread.
    symbol_b : str
        Second instrument of the spread.
    beta : float
        Hedge ratio from cointegration (spread = A - β·B).
    entry_threshold : float
        Z-score threshold for entry (default 1.5).
    exit_threshold : float
        Z-score threshold for exit (default 0.5).
    stop_loss_threshold : float
        Z-score threshold for stop-loss (default 3.0).
    min_calibration_points : int
        Minimum spread observations for calibration (default 60).
    """

    def __init__(
        self,
        symbol_a: str,
        symbol_b: str,
        beta: float,
        entry_threshold: float = 1.5,
        exit_threshold: float = 0.5,
        stop_loss_threshold: float = 3.0,
        min_calibration_points: int = 60,
    ):
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.beta = beta
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.min_calibration_points = min_calibration_points

        self.params = OUParameters()
        self.spread_history: List[float] = []
        self.current_position: Signal = Signal.HOLD

        # Recalibrate every N new observations
        self._recalibrate_interval = 50
        self._obs_since_calibration = 0

    def compute_spread(self, price_a: float, price_b: float) -> float:
        """Compute the spread: Z_t = price_A - β × price_B."""
        return price_a - self.beta * price_b

    def calibrate(self, spread_data: Optional[np.ndarray] = None) -> OUParameters:
        """
        Calibrate OU parameters via Maximum Likelihood Estimation.

        The discrete-time log-likelihood for OU is based on:
            X_{t+1} | X_t ~ N(X_t·e^{-θΔt} + μ(1-e^{-θΔt}), σ²(1-e^{-2θΔt})/(2θ))

        Parameters
        ----------
        spread_data : optional array of spread observations.
                      Uses internal history if not provided.
        """
        if not HAS_SCIPY:
            logger.error("scipy not installed, cannot calibrate OU.")
            return self.params

        data = spread_data if spread_data is not None else np.array(self.spread_history)

        if len(data) < self.min_calibration_points:
            logger.debug(
                f"{self.symbol_a}/{self.symbol_b}: Only {len(data)} points, "
                f"need {self.min_calibration_points}"
            )
            return self.params

        dt = 1.0  # Assume unit time steps (1 bar)
        n = len(data)

        def neg_log_likelihood(params):
            theta, mu, sigma = params
            if theta <= 0 or sigma <= 0:
                return 1e10

            exp_neg_theta = np.exp(-theta * dt)
            variance = (sigma ** 2) * (1 - np.exp(-2 * theta * dt)) / (2 * theta)

            if variance <= 0:
                return 1e10

            # Predicted means
            predicted = data[:-1] * exp_neg_theta + mu * (1 - exp_neg_theta)
            residuals = data[1:] - predicted

            # Log-likelihood (Gaussian)
            ll = -0.5 * (n - 1) * np.log(2 * np.pi * variance)
            ll -= 0.5 * np.sum(residuals ** 2) / variance

            return -ll  # Minimize negative log-likelihood

        # Initial guesses from data moments
        mu0 = np.mean(data)
        sigma0 = np.std(np.diff(data))
        theta0 = 1.0  # Moderate mean-reversion

        # Optimize
        try:
            result = minimize(
                neg_log_likelihood,
                x0=[theta0, mu0, sigma0],
                method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
            )

            if result.success or result.fun < 1e9:
                theta, mu, sigma = result.x
                theta = max(theta, 1e-6)  # Ensure positive
                sigma = max(sigma, 1e-10)

                self.params = OUParameters(
                    theta=theta,
                    mu=mu,
                    sigma=sigma,
                    sigma_eq=sigma / np.sqrt(2 * theta),
                    half_life=np.log(2) / theta,
                    log_likelihood=-result.fun,
                    calibrated=True,
                )
                self._obs_since_calibration = 0

                logger.info(
                    f"OU calibrated {self.symbol_a}/{self.symbol_b}: "
                    f"θ={theta:.4f} μ={mu:.6f} σ={sigma:.6f} "
                    f"σ_eq={self.params.sigma_eq:.6f} "
                    f"half_life={self.params.half_life:.1f} bars"
                )
            else:
                logger.warning(f"OU calibration did not converge for {self.symbol_a}/{self.symbol_b}")

        except Exception as e:
            logger.error(f"OU calibration failed: {e}")

        return self.params

    def compute_zscore(self, current_spread: float) -> float:
        """
        Compute the normalized z-score.

        z = (X_t - μ) / σ_eq
        """
        if not self.params.calibrated or self.params.sigma_eq <= 0:
            return 0.0

        return (current_spread - self.params.mu) / self.params.sigma_eq

    def generate_signal(self, price_a: float, price_b: float) -> OUSignal:
        """
        Generate a trading signal based on current prices.

        Strategy:
            z < -entry_threshold → LONG spread (buy A, sell B)
            z > +entry_threshold → SHORT spread (sell A, buy B)
            |z| < exit_threshold → EXIT (close position)
            |z| > stop_loss → EXIT (emergency close)
        """
        spread = self.compute_spread(price_a, price_b)
        self.spread_history.append(spread)
        self._obs_since_calibration += 1

        # Auto-recalibrate
        if self._obs_since_calibration >= self._recalibrate_interval:
            self.calibrate()

        z = self.compute_zscore(spread)

        # Compute signal confidence (how extreme the z-score is)
        # P(|Z| > |z|) using standard normal CDF
        confidence = 2 * (1 - norm.cdf(abs(z))) if HAS_SCIPY else 0.0

        signal = Signal.HOLD

        # Stop-loss check first
        if abs(z) > self.stop_loss_threshold:
            if self.current_position in (Signal.LONG_SPREAD, Signal.SHORT_SPREAD):
                signal = Signal.EXIT

        # Entry signals
        elif z < -self.entry_threshold and self.current_position == Signal.HOLD:
            signal = Signal.LONG_SPREAD

        elif z > self.entry_threshold and self.current_position == Signal.HOLD:
            signal = Signal.SHORT_SPREAD

        # Exit signals
        elif abs(z) < self.exit_threshold:
            if self.current_position in (Signal.LONG_SPREAD, Signal.SHORT_SPREAD):
                signal = Signal.EXIT

        # Update position tracking
        if signal == Signal.LONG_SPREAD:
            self.current_position = Signal.LONG_SPREAD
        elif signal == Signal.SHORT_SPREAD:
            self.current_position = Signal.SHORT_SPREAD
        elif signal == Signal.EXIT:
            self.current_position = Signal.HOLD

        return OUSignal(
            symbol_a=self.symbol_a,
            symbol_b=self.symbol_b,
            signal=signal,
            z_score=z,
            spread=spread,
            params=self.params,
            confidence=1.0 - confidence,  # Higher = more confident
            timestamp=time.time(),
        )


class StatArbManager:
    """
    Manages multiple OU models for a portfolio of cointegrated pairs.
    """

    def __init__(self):
        self.models: Dict[str, OrnsteinUhlenbeck] = {}

    def add_pair(
        self,
        symbol_a: str,
        symbol_b: str,
        beta: float,
        **kwargs,
    ):
        """Register a cointegrated pair for monitoring."""
        key = f"{symbol_a}/{symbol_b}"
        self.models[key] = OrnsteinUhlenbeck(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            beta=beta,
            **kwargs,
        )
        logger.info(f"Added stat-arb pair: {key} (β={beta:.4f})")

    def calibrate_all(self, price_data: Optional[Dict[str, np.ndarray]] = None):
        """Calibrate all models."""
        for key, model in self.models.items():
            if price_data:
                a_data = price_data.get(model.symbol_a)
                b_data = price_data.get(model.symbol_b)
                if a_data is not None and b_data is not None:
                    n = min(len(a_data), len(b_data))
                    spread = a_data[-n:] - model.beta * b_data[-n:]
                    model.calibrate(spread)
            elif model.spread_history:
                model.calibrate()

    def update(
        self, prices: Dict[str, float]
    ) -> List[OUSignal]:
        """
        Update all models with new prices and collect signals.

        Parameters
        ----------
        prices : dict mapping symbol → current mid price

        Returns
        -------
        List of OUSignal objects (only non-HOLD signals).
        """
        signals = []
        for key, model in self.models.items():
            if model.symbol_a in prices and model.symbol_b in prices:
                sig = model.generate_signal(
                    prices[model.symbol_a],
                    prices[model.symbol_b],
                )
                if sig.signal != Signal.HOLD:
                    signals.append(sig)
        return signals

    def get_all_zscores(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Get current z-scores for all pairs."""
        zscores = {}
        for key, model in self.models.items():
            if model.symbol_a in prices and model.symbol_b in prices:
                spread = model.compute_spread(
                    prices[model.symbol_a], prices[model.symbol_b]
                )
                zscores[key] = model.compute_zscore(spread)
        return zscores
