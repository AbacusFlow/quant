"""绩效指标计算"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_metrics(equity: pd.Series) -> dict:
    """仅基于净值序列的核心指标(适用于组合回测)"""
    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_periods = len(returns)
    annual_return = (1 + total_return) ** (TRADING_DAYS / n_periods) - 1 if n_periods > 0 else 0.0

    sharpe = 0.0
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)

    cummax = equity.cummax()
    max_drawdown = (equity / cummax - 1).min()

    return {
        "总收益率": total_return,
        "年化收益率": annual_return,
        "夏普比率": sharpe,
        "最大回撤": max_drawdown,
    }


def compute_metrics(equity: pd.Series, trades_df: pd.DataFrame) -> dict:
    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_periods = len(returns)
    annual_return = (1 + total_return) ** (TRADING_DAYS / n_periods) - 1 if n_periods > 0 else 0.0

    sharpe = 0.0
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)

    # 最大回撤
    cummax = equity.cummax()
    drawdown = equity / cummax - 1
    max_drawdown = drawdown.min()

    # 胜率:按完整买卖回合计算
    wins, rounds = 0, 0
    buy_cost = None
    for _, row in trades_df.iterrows():
        if row["side"] == "buy":
            buy_cost = row["amount"] + row["fee"]
        elif row["side"] == "sell" and buy_cost is not None:
            pnl = (row["amount"] - row["fee"]) - buy_cost
            rounds += 1
            if pnl > 0:
                wins += 1
            buy_cost = None
    win_rate = wins / rounds if rounds > 0 else float("nan")

    return {
        "总收益率": total_return,
        "年化收益率": annual_return,
        "夏普比率": sharpe,
        "最大回撤": max_drawdown,
        "交易回合数": rounds,
        "胜率": win_rate,
        "总交易次数": len(trades_df),
    }
