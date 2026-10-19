"""
Monte Carlo Risk Engine
========================
Generates correlated future price paths via Cholesky decomposition
to compute Value at Risk (VaR) and Conditional Value at Risk (CVaR).

Pipeline:
    Covariance Matrix → Cholesky → Correlated Normal Draws
    → Simulate 10,000 portfolio return paths
    → Compute VaR (5th percentile) and CVaR (mean of tail losses)
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("MonteCarlo")


@dataclass
class RiskMetrics:
    """Output of Monte Carlo simulation."""
    var_95: float = 0.0         # 95% VaR (loss that is exceeded 5% of the time)
    cvar_95: float = 0.0        # 95% CVaR (expected loss given loss > VaR)
    var_99: float = 0.0         # 99% VaR
    cvar_99: float = 0.0        # 99% CVaR
    expected_return: float = 0.0
    volatility: float = 0.0     # Portfolio volatility
    max_simulated_loss: float = 0.0
    correlation_avg: float = 0.0
    n_paths: int = 0
    compute_time_ms: float = 0.0


class MonteCarloEngine:
    """
    Vectorized Monte Carlo simulation for portfolio risk estimation.

    Uses Cholesky decomposition of the asset covariance matrix to
    generate correlated return scenarios, then computes VaR and CVaR
    from the simulated P&L distribution.

    Parameters
    ----------
    n_paths : int
        Number of simulated price paths (default: 10,000).
    horizon : int
        Simulation horizon in time steps (default: 1).
    confidence : float
        VaR/CVaR confidence level (default: 0.95).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_paths: int = 10_000,
        horizon: int = 1,
        confidence: float = 0.95,
        seed: Optional[int] = None,
    ):
        self.n_paths = n_paths
        self.horizon = horizon
        self.confidence = confidence
        self.rng = np.random.default_rng(seed)

        # Cached Cholesky decomposition
        self._cholesky: Optional[np.ndarray] = None
        self._cov_matrix: Optional[np.ndarray] = None

    def set_covariance_matrix(self, cov_matrix: np.ndarray) -> bool:
        """
        Set and decompose the asset covariance matrix.

        Returns True if Cholesky decomposition succeeded.
        """
        n = cov_matrix.shape[0]

        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(cov_matrix)
        if eigvals.min() < 0:
            # Add small diagonal to make PSD
            cov_matrix = cov_matrix + (-eigvals.min() + 1e-8) * np.eye(n)
            logger.debug("Adjusted covariance matrix to be PSD")

        try:
            self._cholesky = np.linalg.cholesky(cov_matrix)
            self._cov_matrix = cov_matrix
            return True
        except np.linalg.LinAlgError:
            logger.error("Cholesky decomposition failed. Matrix not PSD.")
            return False

    def simulate_returns(
        self,
        mean_returns: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate correlated return scenarios.

        Parameters
        ----------
        mean_returns : expected returns per asset (optional, defaults to 0).

        Returns
        -------
        np.ndarray of shape (n_paths, n_assets) — simulated returns.
        """
        if self._cholesky is None:
            raise ValueError("Covariance matrix not set. Call set_covariance_matrix() first.")

        n_assets = self._cholesky.shape[0]

        if mean_returns is None:
            mean_returns = np.zeros(n_assets)

        # Generate independent standard normals
        Z = self.rng.standard_normal((self.n_paths, n_assets))

        # Correlate via Cholesky: R = mu + L @ Z^T
        correlated = Z @ self._cholesky.T

        # Scale by horizon
        if self.horizon > 1:
            correlated *= np.sqrt(self.horizon)
            mean_returns = mean_returns * self.horizon

        return correlated + mean_returns

    def compute_portfolio_pnl(
        self,
        simulated_returns: np.ndarray,
        weights: np.ndarray,
        portfolio_value: float = 1.0,
    ) -> np.ndarray:
        """
        Compute portfolio P&L from simulated asset returns.

        Parameters
        ----------
        simulated_returns : (n_paths, n_assets)
        weights : asset weights (sum to 1)
        portfolio_value : total portfolio value

        Returns
        -------
        np.ndarray of shape (n_paths,) — portfolio P&L per path.
        """
        portfolio_returns = simulated_returns @ weights
        return portfolio_returns * portfolio_value

    def compute_risk_metrics(
        self,
        pnl: np.ndarray,
    ) -> RiskMetrics:
        """
        Compute VaR and CVaR from P&L distribution.
        """
        metrics = RiskMetrics()
        metrics.n_paths = len(pnl)

        # Sort losses (negative P&L)
        losses = -pnl  # Positive = loss

        # VaR: percentile of the loss distribution
        metrics.var_95 = float(np.percentile(losses, 95))
        metrics.var_99 = float(np.percentile(losses, 99))

        # CVaR: expected loss given loss > VaR
        tail_95 = losses[losses >= metrics.var_95]
        metrics.cvar_95 = float(np.mean(tail_95)) if len(tail_95) > 0 else metrics.var_95

        tail_99 = losses[losses >= metrics.var_99]
        metrics.cvar_99 = float(np.mean(tail_99)) if len(tail_99) > 0 else metrics.var_99

        # Other metrics
        metrics.expected_return = float(np.mean(pnl))
        metrics.volatility = float(np.std(pnl))
        metrics.max_simulated_loss = float(np.max(losses))

        return metrics

    def run(
        self,
        cov_matrix: np.ndarray,
        weights: np.ndarray,
        mean_returns: Optional[np.ndarray] = None,
        portfolio_value: float = 1.0,
    ) -> RiskMetrics:
        """
        Full Monte Carlo simulation: simulate → compute risk.

        Parameters
        ----------
        cov_matrix : (n_assets, n_assets) covariance matrix
        weights : (n_assets,) portfolio weights
        mean_returns : (n_assets,) expected returns (optional)
        portfolio_value : total portfolio value

        Returns
        -------
        RiskMetrics with VaR, CVaR, etc.
        """
        t0 = time.time()

        # Normalize weights
        weights = np.array(weights, dtype=float)
        weights = weights / np.sum(np.abs(weights)) if np.sum(np.abs(weights)) > 0 else weights

        # Set covariance and decompose
        if not self.set_covariance_matrix(cov_matrix):
            return RiskMetrics()

        # Simulate
        sim_returns = self.simulate_returns(mean_returns)

        # Portfolio P&L
        pnl = self.compute_portfolio_pnl(sim_returns, weights, portfolio_value)

        # Compute metrics
        metrics = self.compute_risk_metrics(pnl)

        # Add correlation info
        n = cov_matrix.shape[0]
        if n > 1:
            # Convert covariance to correlation
            std_devs = np.sqrt(np.diag(cov_matrix))
            std_devs[std_devs == 0] = 1e-10
            corr = cov_matrix / np.outer(std_devs, std_devs)
            np.fill_diagonal(corr, 0)
            metrics.correlation_avg = float(np.mean(np.abs(corr)))

        metrics.compute_time_ms = (time.time() - t0) * 1000

        logger.info(
            f"MC simulation: {self.n_paths} paths | "
            f"VaR95={metrics.var_95:.4f} CVaR95={metrics.cvar_95:.4f} | "
            f"E[R]={metrics.expected_return:.6f} σ={metrics.volatility:.4f} | "
            f"{metrics.compute_time_ms:.1f}ms"
        )

        return metrics

    def run_single_asset(
        self,
        volatility: float,
        expected_return: float = 0.0,
        portfolio_value: float = 1.0,
    ) -> RiskMetrics:
        """
        Simplified MC for a single asset / trade.
        """
        cov = np.array([[volatility ** 2]])
        weights = np.array([1.0])
        mean = np.array([expected_return])
        return self.run(cov, weights, mean, portfolio_value)
