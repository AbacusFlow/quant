"""多标的组合回测引擎。

- 信号:目标权重在 T 日收盘后产生,T+1 日开盘执行
- 成交:开盘价 + 滑点(买加卖减),100 股整手
- 费用:佣金(万2.5,最低5元);印花税仅对股票卖出收取(ETF 免)
- 先卖后买,买入按目标权重比例分配可用资金
"""
from dataclasses import dataclass, field

import pandas as pd

import config


@dataclass
class PortfolioTrade:
    date: pd.Timestamp
    symbol: str
    side: str          # buy / sell
    price: float       # 含滑点的实际成交价
    shares: int
    amount: float
    fee: float
    cash_after: float


@dataclass
class PortfolioResult:
    equity: pd.Series                    # 每日总资产(收盘计)
    holdings: pd.DataFrame               # 每日各标的持仓股数
    trades: list[PortfolioTrade] = field(default_factory=list)

    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": t.date.date(),
                    "symbol": t.symbol,
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


def _commission(amount: float) -> float:
    return max(amount * config.COMMISSION_RATE, config.COMMISSION_MIN)


def _sell_fee(amount: float, stamp_tax: bool) -> float:
    fee = _commission(amount)
    if stamp_tax:
        fee += amount * config.STAMP_TAX_RATE
    return fee


def align_prices(prices: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """所有标的共同交易日(交集),保证每个交易日所有标的都有行情"""
    common = None
    for df in prices.values():
        common = df.index if common is None else common.intersection(df.index)
    return common.sort_values()


def run_portfolio_backtest(
    prices: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    initial_capital: float = config.INITIAL_CAPITAL,
    stamp_tax: bool = False,
    slippage: float = config.SLIPPAGE_RATE,
    rebalance_band: float = config.REBALANCE_BAND,
) -> PortfolioResult:
    """逐日模拟组合调仓。

    prices: {symbol: DataFrame(open/close, 日期索引)}
    weights: 目标权重表(日期 x symbol),T 日收盘后的目标,T+1 开盘执行;行和应 <= 1
    """
    symbols = list(weights.columns)
    index = align_prices(prices).intersection(weights.index).sort_values()

    # 当日执行的目标权重 = 前一日信号
    target = weights.reindex(index).shift(1).fillna(0.0)

    cash = initial_capital
    shares: dict[str, int] = {s: 0 for s in symbols}
    trades: list[PortfolioTrade] = []
    equity = pd.Series(index=index, dtype=float)
    holdings = pd.DataFrame(0, index=index, columns=symbols, dtype=int)

    for date in index:
        opens = {s: prices[s].at[date, "open"] for s in symbols}
        closes = {s: prices[s].at[date, "close"] for s in symbols}

        # 以开盘价计算当前总资产,得出各标的目标股数
        equity_open = cash + sum(shares[s] * opens[s] for s in symbols)
        desired = {}
        for s in symbols:
            tgt_w = target.at[date, s]
            tgt_value = equity_open * tgt_w
            desired[s] = int(tgt_value / opens[s] // 100) * 100 if tgt_value > 0 else 0
            # 再平衡带宽:偏离金额过小不交易,避免每日漂移再平衡的费用拖累
            # (按原始目标权重判断清仓:仅权重为 0 的真清仓不受带宽限制,
            #  避免正权重因整手取整为 0 而被误清仓)
            if tgt_w > 0 and abs(desired[s] - shares[s]) * opens[s] < rebalance_band * equity_open:
                desired[s] = shares[s]

        # 先卖
        for s in symbols:
            diff = desired[s] - shares[s]
            if diff < 0:
                sell_shares = -diff
                price = opens[s] * (1 - slippage)
                amount = sell_shares * price
                fee = _sell_fee(amount, stamp_tax)
                cash += amount - fee
                shares[s] -= sell_shares
                trades.append(PortfolioTrade(date, s, "sell", price, sell_shares, amount, fee, cash))

        # 后买:按各标的目标买入金额比例分配现金预算,避免先买的标的挤占后买的资金
        buys = {s: desired[s] - shares[s] for s in symbols if desired[s] - shares[s] > 0}
        total_buy_value = sum(n * opens[s] * (1 + slippage) for s, n in buys.items())
        cash_snapshot = cash  # 卖出后的现金快照,预算分配与成交顺序无关
        for s, diff in buys.items():
            price = opens[s] * (1 + slippage)
            budget = cash_snapshot * (diff * price / total_buy_value)
            # 佣金计入各自预算内,保证分配结果与成交顺序无关
            buy_shares = min(diff, int(budget / (price * (1 + config.COMMISSION_RATE)) // 100) * 100)
            while buy_shares > 0:
                amount = buy_shares * price
                fee = _commission(amount)
                if amount + fee <= budget and amount + fee <= cash:
                    break
                buy_shares -= 100
            if buy_shares > 0:
                amount = buy_shares * price
                fee = _commission(amount)
                cash -= amount + fee
                shares[s] += buy_shares
                trades.append(PortfolioTrade(date, s, "buy", price, buy_shares, amount, fee, cash))

        equity.at[date] = cash + sum(shares[s] * closes[s] for s in symbols)
        holdings.loc[date] = [shares[s] for s in symbols]

    equity.name = "equity"
    return PortfolioResult(equity=equity, holdings=holdings, trades=trades)
