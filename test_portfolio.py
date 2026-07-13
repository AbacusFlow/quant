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
    assert t.fee == max(amount * config.ETF_COMMISSION_RATE, config.COMMISSION_MIN)
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


def test_rebalance_band_skips_small_drift_but_zero_target_sells_all():
    """再平衡带宽的不对称行为(锚定,防重构时被'顺手统一'):
    - 目标权重 >0 且偏离金额 < band → 不交易(避免每日小额漂移调仓的费用拖累)
    - 目标权重 =0 → 清仓,不受带宽限制(真离场不能被带宽吞掉)
    """
    prices = make_prices(5, price_a=10.0)
    idx = prices["A"].index
    weights = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    weights.loc[idx[0], "A"] = 1.00   # T+1:第2日建仓 A
    weights.loc[idx[1], "A"] = 0.99   # T+1:第3日目标 0.99,偏离 ~1% < band 2% → 应不交易
    # idx[2] 起信号归 0 → T+1:第4日应清仓(权重归零不受带宽限制)

    result = run_portfolio_backtest(prices, weights, initial_capital=100_000,
                                    slippage=0.0, rebalance_band=0.02)
    by_day = {}
    for t in result.trades:
        by_day.setdefault(t.date, []).append(t)
    assert idx[1] in by_day and by_day[idx[1]][0].side == "buy"   # 第2日建仓
    assert idx[2] not in by_day                                    # 第3日带宽内,不动
    day4 = by_day.get(idx[3], [])
    assert len(day4) == 1 and day4[0].side == "sell"               # 第4日清仓
    assert day4[0].shares == result.trades[0].shares               # 卖光全部持仓


def test_multi_buy_budget_order_independent():
    """同日多标的买入按现金快照比例分配预算,结果与列顺序/字典插入顺序无关"""
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    a = pd.DataFrame({"open": 10.0, "close": 10.0}, index=idx)
    b = pd.DataFrame({"open": 20.0, "close": 20.0}, index=idx)

    def run(order):
        prices = {s: (a if s == "A" else b) for s in order}
        weights = pd.DataFrame(0.0, index=idx, columns=order)
        weights.loc[idx[0], "A"] = 0.5
        weights.loc[idx[0], "B"] = 0.5
        return run_portfolio_backtest(prices, weights, initial_capital=100_000, slippage=0.0)

    r1, r2 = run(["A", "B"]), run(["B", "A"])
    h1 = {t.symbol: t.shares for t in r1.trades if t.side == "buy"}
    h2 = {t.symbol: t.shares for t in r2.trades if t.side == "buy"}
    assert h1 == h2, f"买入结果依赖顺序: {h1} vs {h2}"
    # 各按 ~50% 预算成交(10元/20元 → 约 4900-5000 / 2400-2500 股),现金不透支
    assert 4800 <= h1["A"] <= 5000 and 2400 <= h1["B"] <= 2500
    total_cost = sum(t.amount + t.fee for t in r1.trades if t.side == "buy")
    assert total_cost <= 100_000 + 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} 个测试全部通过")
