"""
Position Sizing — Fractional Kelly Criterion
==============================================
Computes optimal position sizes based on:
  1. Kelly criterion (from win probability + win/loss ratio)
  2. CVaR constraint (max risk per trade)
  3. Dynamic volatility scaling
  4. Hard portfolio limits

Formula:
    f* = W - (1-W)/R     (Full Kelly)
    f  = f* × fraction   (Fractional Kelly, typically 0.25-0.5)

Where:
    W = win probability (from XGBoost P(up))
    R = average win / average loss ratio
    fraction = Kelly scaling factor (reduced when vol or CVaR is high)
"""

import os
import sys
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from risk.monte_carlo import RiskMetrics

logger = get_logger("PositionSizing")


@dataclass
class PositionSizeResult:
    """Output of position sizing calculation."""
    kelly_fraction: float = 0.0     # Raw Kelly fraction
    scaled_fraction: float = 0.0    # After CVaR/vol adjustment
    position_size: float = 0.0      # Dollar amount to risk
    position_pct: float = 0.0       # As percentage of capital
    scaling_factor: float = 1.0     # CVaR/vol scaling applied
    reason: str = ""                # Why this size was chosen

    @property
    def should_trade(self) -> bool:
        """Whether the position is large enough to execute."""
        return self.position_size > 0 and self.position_pct > 0.001


class KellyPositionSizer:
    """
    Fractional Kelly position sizer with CVaR constraints.

    Parameters
    ----------
    max_kelly_fraction : float
        Maximum Kelly multiplier (default: 0.5 = half Kelly).
    min_kelly_fraction : float
        Minimum Kelly to trade (default: 0.02).
    max_position_pct : float
        Maximum position as % of capital (default: 0.10 = 10%).
    base_vol : float
        "Normal" volatility level for scaling (default: 0.01 = 1%).
    max_cvar : float
        Maximum acceptable CVaR (default: 0.05 = 5%).
    """

    def __init__(
        self,
        max_kelly_fraction: float = 0.5,
        min_kelly_fraction: float = 0.02,
        max_position_pct: float = 0.10,
        base_vol: float = 0.01,
        max_cvar: float = 0.05,
    ):
        self.max_kelly_fraction = max_kelly_fraction
        self.min_kelly_fraction = min_kelly_fraction
        self.max_position_pct = max_position_pct
        self.base_vol = base_vol
        self.max_cvar = max_cvar

    def compute_kelly(
        self,
        win_prob: float,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
    ) -> float:
        """
        Compute the raw Kelly fraction.

        f* = W - (1-W)/R
        Where R = avg_win / avg_loss

        Parameters
        ----------
        win_prob : P(win) from ML model (0 to 1)
        avg_win : average winning trade return (positive)
        avg_loss : average losing trade return (positive)
        """
        if avg_loss <= 0 or win_prob <= 0 or win_prob >= 1:
            return 0.0

        R = avg_win / avg_loss
        kelly = win_prob - (1 - win_prob) / R

        return max(kelly, 0.0)  # Don't go negative (= don't trade)

    def compute_scaling_factor(
        self,
        risk_metrics: Optional[RiskMetrics] = None,
        current_vol: float = 0.01,
    ) -> float:
        """
        Compute dynamic scaling factor based on market conditions.

        Scale DOWN when:
          - CVaR is high (more tail risk)
          - Volatility is high (more uncertainty)

        Returns a factor between 0 and 1.
        """
        factor = 1.0

        # Volatility scaling: linear reduction when vol exceeds base
        if current_vol > self.base_vol:
            vol_ratio = self.base_vol / current_vol
            factor *= max(vol_ratio, 0.1)  # Floor at 10%

        # CVaR scaling
        if risk_metrics and risk_metrics.cvar_95 > 0:
            if risk_metrics.cvar_95 > self.max_cvar:
                cvar_ratio = self.max_cvar / risk_metrics.cvar_95
                factor *= max(cvar_ratio, 0.1)

        return min(factor, 1.0)

    def size_position(
        self,
        capital: float,
        win_prob: float,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
        risk_metrics: Optional[RiskMetrics] = None,
        current_vol: float = 0.01,
    ) -> PositionSizeResult:
        """
        Full position sizing calculation.

        Parameters
        ----------
        capital : total available capital
        win_prob : P(win) from classifier
        avg_win : average winning trade size (absolute)
        avg_loss : average losing trade size (absolute)
        risk_metrics : CVaR metrics from Monte Carlo
        current_vol : current volatility estimate

        Returns
        -------
        PositionSizeResult with the recommended position size.
        """
        result = PositionSizeResult()

        # Step 1: Raw Kelly
        raw_kelly = self.compute_kelly(win_prob, avg_win, avg_loss)
        result.kelly_fraction = raw_kelly

        if raw_kelly < self.min_kelly_fraction:
            result.reason = f"Kelly too low ({raw_kelly:.4f} < {self.min_kelly_fraction})"
            return result

        # Step 2: Dynamic scaling
        scaling = self.compute_scaling_factor(risk_metrics, current_vol)
        result.scaling_factor = scaling

        # Step 3: Apply fractional Kelly (never exceed max)
        scaled = raw_kelly * scaling
        capped = min(scaled, self.max_kelly_fraction)
        result.scaled_fraction = capped

        # Step 4: Convert to position size
        position_pct = min(capped, self.max_position_pct)
        result.position_pct = position_pct
        result.position_size = capital * position_pct

        # Step 5: Reason
        if scaling < 1.0:
            result.reason = f"Scaled down by {scaling:.2f}x (vol/CVaR)"
        elif capped < raw_kelly:
            result.reason = f"Capped at max Kelly {self.max_kelly_fraction}"
        else:
            result.reason = "Full fractional Kelly"

        logger.info(
            f"Position: kelly={raw_kelly:.4f} × scale={scaling:.2f} "
            f"→ {position_pct:.4f} ({result.position_size:.2f} of {capital:.0f})"
        )

        return result

    def size_portfolio(
        self,
        capital: float,
        signals: Dict[str, float],
        risk_metrics: Optional[RiskMetrics] = None,
        current_vols: Optional[Dict[str, float]] = None,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
    ) -> Dict[str, PositionSizeResult]:
        """
        Size positions for a portfolio of signals.

        Ensures total exposure doesn't exceed capital constraints.

        Parameters
        ----------
        capital : total capital
        signals : dict of {symbol: win_probability}
        risk_metrics : portfolio-level CVaR
        current_vols : dict of {symbol: current_volatility}
        """
        results = {}
        total_allocated = 0.0
        remaining_capital = capital

        # Sort by signal strength (strongest first)
        sorted_signals = sorted(signals.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)

        for symbol, win_prob in sorted_signals:
            vol = current_vols.get(symbol, self.base_vol) if current_vols else self.base_vol

            position = self.size_position(
                capital=remaining_capital,
                win_prob=win_prob,
                avg_win=avg_win,
                avg_loss=avg_loss,
                risk_metrics=risk_metrics,
                current_vol=vol,
            )

            results[symbol] = position

            if position.should_trade:
                total_allocated += position.position_size
                remaining_capital = max(0, capital - total_allocated)

        logger.info(f"Portfolio sizing: {len(results)} signals, "
                     f"allocated {total_allocated:.2f} / {capital:.0f}")

        return results
