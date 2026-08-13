"""研究脚本 experiment_buffer 的区间边界逻辑单元测试。

这些辅助函数只服务于研究报告,但报告结论(尤其近两年切片)对边界处理很敏感:
漏掉区间首日收益会低估年化,窗口前的持仓状态丢失会虚增抖动次数。

运行: python scripts/test_experiment_buffer.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from experiment_buffer import (
    fingerprint,
    seg_slice,
    turnover,
    whipsaw_count,
    year_return,
)

IDX = pd.to_datetime(["2023-12-28", "2023-12-29", "2024-01-02", "2024-01-03"])


def test_seg_slice_includes_prior_day_as_base():
    """区间起点前一个交易日必须作为基准纳入,否则漏掉起点当日收益"""
    eq = pd.Series([1.0, 1.1, 1.2, 1.3], index=IDX)
    s = seg_slice(eq, start="2024-01-01")
    assert list(s.index) == list(IDX[1:])          # 含 2023-12-29 基准点
    assert abs(s.iloc[-1] / s.iloc[0] - 1 - (1.3 / 1.1 - 1)) < 1e-12


def test_seg_slice_start_before_series_keeps_all():
    """起点早于序列时不应越界,全量返回"""
    eq = pd.Series([1.0, 1.1, 1.2, 1.3], index=IDX)
    assert list(seg_slice(eq, start="2000-01-01").index) == list(IDX)


def test_seg_slice_end_is_exclusive():
    eq = pd.Series([1.0, 1.1, 1.2, 1.3], index=IDX)
    assert list(seg_slice(eq, end="2024-01-01").index) == list(IDX[:2])


def test_year_return_uses_prior_year_last_day():
    """年度收益以上一年最后一个交易日为基准"""
    eq = pd.Series([1.0, 1.1, 1.2, 1.3], index=IDX)
    assert abs(year_return(eq, 2024) - (1.3 / 1.1 - 1)) < 1e-12
    assert abs(year_return(eq, 2023) - (1.1 / 1.0 - 1)) < 1e-12   # 首年退化


def test_whipsaw_carries_state_across_window_start():
    """窗口开始前已持有的标的,不能被当作窗口内新建仓"""
    idx = pd.date_range("2024-01-01", periods=7, freq="B")
    w = pd.DataFrame({"A": [0, 1, 1, 1, 1, 0, 0]}, index=idx, dtype=float)
    assert whipsaw_count(w, 3) == 0                          # 持有 4 天,非抖动
    # 若直接切片后再计数,窗口起点的存量持仓会被误判为「新建仓 2 天即离场」
    assert whipsaw_count(w.iloc[3:], 3) == 1                 # 错误做法的表现
    assert whipsaw_count(w, 3, start=idx[3]) == 0            # 正确:状态延续


def test_whipsaw_counts_exit_inside_window():
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    w = pd.DataFrame({"A": [0, 0, 0, 1, 0, 0]}, index=idx, dtype=float)
    assert whipsaw_count(w, 3, start=idx[3]) == 1


def test_turnover_window_keeps_boundary_change():
    """窗口首日相对前一日的权重变化必须计入,不能被 diff 首行 NaN 吃掉"""
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    w = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0], "B": [0.0, 0.0, 1.0, 1.0]}, index=idx)
    assert abs(turnover(w, start=idx[2]) - 1.0 / 2 * 252) < 1e-9
    assert turnover(w.iloc[2:]) == 0.0                       # 错误做法:换仓被吞掉


def test_fingerprint_covers_open_prices():
    """成交发生在开盘价,指纹必须对开盘价变动敏感"""
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    mk = lambda o: pd.DataFrame({"open": o, "close": [1.0, 1.1, 1.2]}, index=idx)
    prices = {"A": mk([1.0, 1.1, 1.2])}
    closes = pd.DataFrame({"A": prices["A"]["close"]})
    bumped = {"A": mk([1.5, 1.1, 1.2])}
    assert fingerprint(prices, closes) != fingerprint(bumped, closes)
    assert fingerprint(prices, closes) == fingerprint(prices, closes)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} 个测试全部通过")
