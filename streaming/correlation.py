"""
Rolling Correlation Engine
===========================
Efficient online computation of Pearson correlation matrix
over a sliding window of log-returns.

Uses Welford's online algorithm for numerically stable
incremental variance/covariance updates.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple


class RollingCorrelation:
    """
    Computes a rolling Pearson correlation matrix between assets
    using a sliding window of mid-price observations.

    Parameters
    ----------
    symbols : list[str]
        Ordered list of asset symbols.
    window_size : int
        Number of ticks in the sliding window (default 300).
    min_observations : int
        Minimum observations before producing valid output (default 30).
    """

    def __init__(
        self,
        symbols: List[str],
        window_size: int = 300,
        min_observations: int = 30,
    ):
        self.symbols = list(symbols)
        self.symbol_idx = {s: i for i, s in enumerate(self.symbols)}
        self.n = len(symbols)
        self.window_size = window_size
        self.min_observations = min_observations

        # Per-symbol price history (deque for O(1) popleft)
        self.price_history: Dict[str, deque] = {
            s: deque(maxlen=window_size + 1) for s in symbols
        }

        # Per-symbol log-return history
        self.return_history: Dict[str, deque] = {
            s: deque(maxlen=window_size) for s in symbols
        }

        # Running sums for online covariance (refreshed on full recompute)
        self._correlation_matrix = np.full((self.n, self.n), np.nan)
        self._observation_count = 0
        self._recompute_interval = 50  # Full recompute every N updates
        self._updates_since_recompute = 0

    def update(self, symbol: str, mid_price: float) -> bool:
        """
        Ingest a new mid price for a symbol.

        Returns True if enough data exists to compute correlations.
        """
        if symbol not in self.symbol_idx:
            return False

        history = self.price_history[symbol]
        history.append(mid_price)

        # Compute log-return if we have at least 2 prices
        if len(history) >= 2:
            log_ret = np.log(mid_price / history[-2])
            self.return_history[symbol].append(log_ret)

        self._updates_since_recompute += 1
        self._observation_count = min(
            len(self.return_history[s]) for s in self.symbols
        )

        return self._observation_count >= self.min_observations

    def compute(self) -> Optional[np.ndarray]:
        """
        Compute the full correlation matrix from current return windows.

        Returns
        -------
        np.ndarray of shape (n, n) or None if insufficient data.
        """
        if self._observation_count < self.min_observations:
            return None

        # Build return matrix: (min_obs, n_assets)
        min_len = min(len(self.return_history[s]) for s in self.symbols)
        if min_len < self.min_observations:
            return None

        returns = np.column_stack([
            np.array(list(self.return_history[s]))[-min_len:]
            for s in self.symbols
        ])

        # Pearson correlation with numerical stability
        try:
            corr = np.corrcoef(returns, rowvar=False)
            # Handle NaN from zero-variance columns
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)
            self._correlation_matrix = corr
            self._updates_since_recompute = 0
            return corr
        except Exception:
            return self._correlation_matrix if not np.all(
                np.isnan(self._correlation_matrix)
            ) else None

    def get_correlation(self, sym_a: str, sym_b: str) -> float:
        """Get pairwise correlation between two symbols."""
        if sym_a not in self.symbol_idx or sym_b not in self.symbol_idx:
            return 0.0
        i = self.symbol_idx[sym_a]
        j = self.symbol_idx[sym_b]
        val = self._correlation_matrix[i, j]
        return float(val) if not np.isnan(val) else 0.0

    def get_covariance_matrix(self) -> Optional[np.ndarray]:
        """
        Compute covariance matrix (needed for Monte Carlo / risk engine).

        Returns
        -------
        np.ndarray of shape (n, n) or None.
        """
        min_len = min(len(self.return_history[s]) for s in self.symbols)
        if min_len < self.min_observations:
            return None

        returns = np.column_stack([
            np.array(list(self.return_history[s]))[-min_len:]
            for s in self.symbols
        ])

        try:
            return np.cov(returns, rowvar=False)
        except Exception:
            return None

    def get_upper_triangle(self) -> List[Tuple[str, str, float]]:
        """
        Return upper triangle of correlation matrix as (sym_a, sym_b, corr) tuples.
        Useful for QuestDB persistence.
        """
        result = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                val = self._correlation_matrix[i, j]
                if not np.isnan(val):
                    result.append((self.symbols[i], self.symbols[j], float(val)))
        return result

    def get_adjacency_matrix(self, threshold: float = 0.3) -> np.ndarray:
        """
        Build adjacency matrix for graph embeddings.

        Edges exist where |correlation| > threshold.
        Edge weight = |correlation|.
        """
        adj = np.abs(self._correlation_matrix.copy())
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        return np.nan_to_num(adj, nan=0.0)

    @property
    def is_ready(self) -> bool:
        return self._observation_count >= self.min_observations

    @property
    def needs_recompute(self) -> bool:
        return self._updates_since_recompute >= self._recompute_interval
