"""策略函数单元测试:集成轮动与回撤控制。

运行: python test_strategy.py
"""
import numpy as np
import pandas as pd

from strategy import (
    apply_drawdown_control,
    apply_vol_targeting,
    etf_momentum_ensemble,
    etf_momentum_rotation,
)


def make_closes(days: int = 120) -> pd.DataFrame:
    """A 持续上涨,B 持续下跌"""
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    a = pd.Series(np.linspace(10, 20, days), index=idx)
    b = pd.Series(np.linspace(20, 10, days), index=idx)
    return pd.DataFrame({"A": a, "B": b})


def test_rotation_picks_uptrend():
    """轮动应持有上涨的 A,不碰下跌的 B"""
    closes = make_closes()
    w = etf_momentum_rotation(closes, lookback=20)
    tail = w.iloc[30:]
    assert (tail["A"] == 1.0).all()
    assert (tail["B"] == 0.0).all()


def test_rotation_cash_when_all_negative():
    """全部下跌时空仓"""
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    closes = pd.DataFrame({
        "A": np.linspace(20, 10, 60),
        "B": np.linspace(30, 15, 60),
    }, index=idx)
    w = etf_momentum_rotation(closes, lookback=20)
    assert (w.iloc[25:].sum(axis=1) == 0).all()


def test_ensemble_is_average():
    """集成权重 = 各子策略平均,行和不超过 1"""
    closes = make_closes()
    lookbacks = (15, 20, 25)
    ens = etf_momentum_ensemble(closes, lookbacks=lookbacks)
    manual = sum(etf_momentum_rotation(closes, lookback=lb) for lb in lookbacks) / len(lookbacks)
    assert np.allclose(ens.values, manual.values)
    assert (ens.sum(axis=1) <= 1.0 + 1e-9).all()


def test_drawdown_control_scales_down():
    """虚拟净值跌破均线时权重被缩放"""
    days = 150
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    # A 先涨后崩
    a = np.concatenate([np.linspace(10, 20, 75), np.linspace(20, 8, 75)])
    closes = pd.DataFrame({"A": a, "B": np.full(days, 10.0)}, index=idx)
    weights = pd.DataFrame({"A": 1.0, "B": 0.0}, index=idx)

    controlled = apply_drawdown_control(weights, closes, ma_window=20, scale=0.5)
    # 上涨阶段不缩放
    assert (controlled["A"].iloc[30:70] == 1.0).all()
    # 崩盘后期应被缩放
    assert (controlled["A"].iloc[-20:] == 0.5).all()


def test_drawdown_control_no_lookahead():
    """T 日控制系数只依赖 T 日及之前的数据:截断未来数据不改变历史系数"""
    closes = make_closes(120)
    weights = etf_momentum_rotation(closes, lookback=20)
    full = apply_drawdown_control(weights, closes, ma_window=20)
    part = apply_drawdown_control(weights.iloc[:80], closes.iloc[:80], ma_window=20)
    assert np.allclose(full.iloc[:80].values, part.values)


def test_vol_targeting_no_lookahead():
    """波动率目标 T 日系数只依赖 T 日及之前的数据:截断未来不改变历史系数"""
    closes = make_closes(120)
    weights = etf_momentum_rotation(closes, lookback=20)
    full = apply_vol_targeting(weights, closes, lookback=20)
    part = apply_vol_targeting(weights.iloc[:80], closes.iloc[:80], lookback=20)
    assert np.allclose(full.iloc[:80].values, part.values)


def test_vol_targeting_scales_down():
    """波动放大的末段应降仓(factor<1),且任何时候不加杠杆(factor<=1)"""
    days = 300
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    t = np.arange(days)
    # 前半段低波动、后半段高波动(均值约 0 的确定性振荡)
    amp = np.where(t < 150, 0.004, 0.04)
    rets = amp * np.sin(t)
    price = 10 * np.cumprod(1 + rets)
    closes = pd.DataFrame({"A": price, "B": np.full(days, 10.0)}, index=idx)
    weights = pd.DataFrame({"A": 1.0, "B": 0.0}, index=idx)

    out = apply_vol_targeting(weights, closes, lookback=20)
    factor = out["A"]
    assert (factor <= 1.0 + 1e-9).all()        # 无杠杆,封顶 1.0
    assert (factor.iloc[-20:] < 1.0).all()     # 高波动末段已实现波动超历史中位 → 降仓


def test_vol_targeting_warmup_full():
    """暖机期与波动不超历史中位时 factor==1(权重不变):波动单调下降则始终满仓"""
    days = 120
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    t = np.arange(days)
    # 波动单调下降:每个 T 的已实现波动都是历史最小,扩张中位数 >= 当前 → scale 恒为 1
    amp = np.linspace(0.05, 0.005, days)
    rets = amp * np.sin(t)
    price = 10 * np.cumprod(1 + rets)
    closes = pd.DataFrame({"A": price, "B": np.full(days, 10.0)}, index=idx)
    weights = pd.DataFrame({"A": 1.0, "B": 0.0}, index=idx)

    out = apply_vol_targeting(weights, closes, lookback=20)
    assert np.allclose(out["A"].values, 1.0)   # 含暖机期(realized=NaN→1)与低波动段


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} 个测试全部通过")
