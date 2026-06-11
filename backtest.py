"""回测引擎:T+1 次日开盘成交,满仓/空仓切换,A股费用建模"""
from dataclasses import dataclass, field

import pandas as pd

import config


@dataclass
class Trade:
    date: pd.Timestamp
    side: str        # buy / sell
    price: float
    shares: int
    amount: float    # 成交金额
    fee: float       # 总费用
    cash_after: float


@dataclass
class BacktestResult:
    equity: pd.Series          # 每日净值(总资产)
    trades: list[Trade] = field(default_factory=list)

    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": t.date.date(),
                    "side": t.side,
                    "price": round(t.price, 4),
                    "shares": t.shares,
                    "amount": round(t.amount, 2),
                    "fee": round(t.fee, 2),
                    "cash_after": round(t.cash_after, 2),
                }
                for t in self.trades
            ]
        )


def _buy_fee(amount: float) -> float:
    return max(amount * config.STOCK_COMMISSION_RATE, config.COMMISSION_MIN)


def _sell_fee(amount: float) -> float:
    commission = max(amount * config.STOCK_COMMISSION_RATE, config.COMMISSION_MIN)
    return commission + amount * config.STAMP_TAX_RATE


def run_backtest(df: pd.DataFrame, position: pd.Series,
                 initial_capital: float = config.INITIAL_CAPITAL) -> BacktestResult:
    """逐日模拟。position 为当日收盘后产生的目标仓位,次日开盘价执行(T+1)。"""
    cash = initial_capital
    shares = 0
    trades: list[Trade] = []
    equity = pd.Series(index=df.index, dtype=float)

    # 当日执行的目标仓位 = 前一日信号
    target = position.shift(1).fillna(0).astype(int)

    for date in df.index:
        open_price = df.at[date, "open"]
        close_price = df.at[date, "close"]
        tgt = target.at[date]

        if tgt == 1 and shares == 0:
            # 按开盘价买入,A股 100 股一手
            est_shares = int(cash / (open_price * (1 + config.STOCK_COMMISSION_RATE)) // 100) * 100
            while est_shares > 0:
                amount = est_shares * open_price
                fee = _buy_fee(amount)
                if amount + fee <= cash:
                    break
                est_shares -= 100
            if est_shares > 0:
                amount = est_shares * open_price
                fee = _buy_fee(amount)
                cash -= amount + fee
                shares = est_shares
                trades.append(Trade(date, "buy", open_price, shares, amount, fee, cash))
        elif tgt == 0 and shares > 0:
            amount = shares * open_price
            fee = _sell_fee(amount)
            cash += amount - fee
            trades.append(Trade(date, "sell", open_price, shares, amount, fee, cash))
            shares = 0

        equity.at[date] = cash + shares * close_price

    equity.name = "equity"
    return BacktestResult(equity=equity, trades=trades)
