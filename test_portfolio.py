"""组合回测引擎单元测试:手工构造数据验证 T+1、费用、调仓逻辑。

运行: python -m pytest test_portfolio.py -v  (或 python test_portfolio.py)
"""
import pandas as pd

import config
from portfolio import run_portfolio_backtest


def make_prices(days: int, price_a: float = 10.0, price_b: float = 20.0) -> dict:
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    a = pd.DataFrame({"open": price_a, "close": price_a}, index=idx)
    b = pd.DataFrame({"open": price_b, "close": price_b}, index=idx)
    return {"A": a, "B": b}


def test_t_plus_1_execution():
    """T 日信号 T+1 日才成交"""
    prices = make_prices(3)
    idx = prices["A"].index
    weights = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    weights.loc[idx[0], "A"] = 1.0  # 第1日收盘信号

    result = run_portfolio_backtest(prices, weights, initial_capital=100_000, slippage=0.0)
    # 第1日信号 -> 第2日买入;第2日信号归0 -> 第3日卖出
    assert len(result.trades) == 2
    assert result.trades[0].date == idx[1] and result.trades[0].side == "buy"
    assert result.trades[1].date == idx[2] and result.trades[1].side == "sell"


def test_lot_size_and_fees():
    """100股整手 + 佣金计算"""
    prices = make_prices(3, price_a=10.0)
    idx = prices["A"].index
    weights = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    weights["A"] = 1.0

    result = run_portfolio_backtest(prices, weights, initial_capital=100_000, slippage=0.0)
    t = result.trades[0]
    assert t.shares % 100 == 0
    # 10万本金,10元/股 -> 最多9900股(留出佣金)
    assert t.shares in (9900, 10000)
    amount = t.shares * 10.0
    assert t.fee == max(amount * config.COMMISSION_RATE, config.COMMISSION_MIN)
    assert t.amount + t.fee <= 100_000


def test_rotation_sell_then_buy():
    """从 A 切换到 B:先卖 A 后买 B,同日完成"""
    prices = make_prices(5)
    idx = prices["A"].index
    weights = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    weights.iloc[0:2, weights.columns.get_loc("A")] = 1.0
    weights.iloc[2:, weights.columns.get_loc("B")] = 1.0

    result = run_portfolio_backtest(prices, weights, initial_capital=100_000, slippage=0.0)
    sides = [(t.symbol, t.side) for t in result.trades]
    assert ("A", "buy") in sides and ("A", "sell") in sides and ("B", "buy") in sides
    switch_day = idx[3]  # 第3日收盘信号,第4日执行
    day_trades = [t for t in result.trades if t.date == switch_day]
    assert [t.side for t in day_trades] == ["sell", "buy"]


def test_slippage_direction():
    """买入价上浮、卖出价下调"""
    prices = make_prices(4)
    idx = prices["A"].index
    weights = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    weights.loc[idx[0], "A"] = 1.0  # 买入后,后续信号为0 -> 卖出

    result = run_portfolio_backtest(prices, weights, initial_capital=100_000, slippage=0.001)
    buy = next(t for t in result.trades if t.side == "buy")
    sell = next(t for t in result.trades if t.side == "sell")
    assert buy.price == 10.0 * 1.001
    assert sell.price == 10.0 * 0.999


def test_no_stamp_tax_for_etf():
    """ETF 卖出不收印花税,股票收"""
    prices = make_prices(4)
    idx = prices["A"].index
    weights = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    weights.loc[idx[0], "A"] = 1.0

    etf = run_portfolio_backtest(prices, weights, initial_capital=100_000, stamp_tax=False, slippage=0.0)
    stock = run_portfolio_backtest(prices, weights, initial_capital=100_000, stamp_tax=True, slippage=0.0)
    etf_sell = next(t for t in etf.trades if t.side == "sell")
    stock_sell = next(t for t in stock.trades if t.side == "sell")
    assert stock_sell.fee > etf_sell.fee
    assert abs((stock_sell.fee - etf_sell.fee) - etf_sell.amount * config.STAMP_TAX_RATE) < 1e-6


def test_equity_conservation():
    """价格不变时,净值只因费用减少"""
    prices = make_prices(5)
    idx = prices["A"].index
    weights = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    weights["A"] = 1.0

    result = run_portfolio_backtest(prices, weights, initial_capital=100_000, slippage=0.0)
    total_fees = sum(t.fee for t in result.trades)
    assert abs(result.equity.iloc[-1] - (100_000 - total_fees)) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} 个测试全部通过")
