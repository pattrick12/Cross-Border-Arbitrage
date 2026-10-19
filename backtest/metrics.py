"""
Backtest Performance Metrics
=============================
Computes standard trading performance metrics from
a sequence of trades/returns.

Metrics:
    - Sharpe Ratio (annualized)
    - Sortino Ratio (downside risk only)
    - Maximum Drawdown
    - Win Rate / Loss Rate
    - Profit Factor
    - Equity Curve
    - Calmar Ratio
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class BacktestMetrics:
    """Complete backtest results."""
    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # Periods

    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

    # Equity
    final_equity: float = 0.0
    peak_equity: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def compute_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0,
                   periods_per_year: int = 252 * 24 * 60) -> float:
    """
    Annualized Sharpe ratio.

    Sharpe = (E[R] - Rf) / σ(R) × √(periods_per_year)
    """
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    return float(np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year))


def compute_sortino(returns: np.ndarray, risk_free_rate: float = 0.0,
                    periods_per_year: int = 252 * 24 * 60) -> float:
    """
    Sortino ratio — uses downside deviation only.

    Sortino = (E[R] - Rf) / σ_down(R) × √(periods_per_year)
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    downside = returns[returns < 0]
    if len(downside) < 2 or np.std(downside) == 0:
        return 0.0 if np.mean(excess) <= 0 else float("inf")
    return float(np.mean(excess) / np.std(downside) * np.sqrt(periods_per_year))


def compute_max_drawdown(equity_curve: np.ndarray) -> tuple:
    """
    Maximum drawdown and duration.

    Returns (max_dd, max_dd_duration_periods)
    """
    if len(equity_curve) == 0:
        return 0.0, 0

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = float(np.min(drawdown))

    # Duration of max drawdown
    in_drawdown = drawdown < 0
    if not np.any(in_drawdown):
        return 0.0, 0

    # Count consecutive drawdown periods
    dd_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0]
    dd_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0]

    if len(dd_starts) == 0:
        return float(max_dd), len(equity_curve)

    if len(dd_ends) == 0 or dd_ends[-1] < dd_starts[-1]:
        dd_ends = np.append(dd_ends, len(equity_curve) - 1)

    max_duration = 0
    for s, e in zip(dd_starts, dd_ends):
        max_duration = max(max_duration, e - s)

    return float(max_dd), int(max_duration)


def compute_metrics(
    trade_returns: np.ndarray,
    initial_capital: float = 100_000,
    periods_per_year: int = 252 * 24 * 60,
) -> BacktestMetrics:
    """
    Compute all metrics from a series of trade returns (not prices).

    Parameters
    ----------
    trade_returns : array of per-period returns (e.g., log-returns)
    initial_capital : starting capital for equity curve
    periods_per_year : for Sharpe annualization
    """
    m = BacktestMetrics()

    if len(trade_returns) == 0:
        return m

    returns = np.array(trade_returns, dtype=float)

    # Equity curve
    equity = initial_capital * np.cumprod(1 + returns)
    equity = np.insert(equity, 0, initial_capital)

    m.final_equity = float(equity[-1])
    m.peak_equity = float(np.max(equity))
    m.total_return = float(equity[-1] / initial_capital - 1)

    # Annualized
    n_periods = len(returns)
    if n_periods > 0:
        m.annualized_return = float((1 + m.total_return) ** (periods_per_year / n_periods) - 1)
    m.volatility = float(np.std(returns) * np.sqrt(periods_per_year))

    # Risk-adjusted ratios
    m.sharpe_ratio = compute_sharpe(returns, periods_per_year=periods_per_year)
    m.sortino_ratio = compute_sortino(returns, periods_per_year=periods_per_year)

    # Drawdown
    m.max_drawdown, m.max_drawdown_duration = compute_max_drawdown(equity)

    # Calmar
    if m.max_drawdown != 0:
        m.calmar_ratio = float(m.annualized_return / abs(m.max_drawdown))

    # Trade-level stats
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    m.total_trades = len(returns)
    m.winning_trades = len(wins)
    m.losing_trades = len(losses)
    m.win_rate = len(wins) / len(returns) if len(returns) > 0 else 0

    m.avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
    m.avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0
    m.best_trade = float(np.max(returns))
    m.worst_trade = float(np.min(returns))

    # Profit factor = gross profits / gross losses
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 1e-10
    m.profit_factor = gross_profit / gross_loss

    return m


def generate_equity_curve(
    trade_returns: np.ndarray,
    initial_capital: float = 100_000,
) -> pd.DataFrame:
    """
    Generate equity curve DataFrame.
    """
    equity = initial_capital * np.cumprod(1 + trade_returns)
    equity = np.insert(equity, 0, initial_capital)

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak

    return pd.DataFrame({
        "equity": equity,
        "peak": peak,
        "drawdown": drawdown,
    })
