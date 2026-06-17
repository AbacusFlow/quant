"""V1 波动率目标 的稳健性确认(research-only)。

两件事:
1. 对 V1 自身参数(波动窗口 vol_lb)做敏感性:{10,20,40,60} × 动量 lookback 网格,
   看中位数是否一致变好(确认不是又一个尖峰/不是 fish 出来的窗口)。
2. Walk-forward:每年只用过去数据选动量 lookback(候选 10~60,按信号级夏普),
   对比"基线 WF" vs "V1 WF",看抗过拟合口径下 V1 是否仍占优。
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import config
import metrics as metrics_mod
import strategy
from portfolio import run_portfolio_backtest
from run_rotation import closes_table, load_pool

LBS = (10, 15, 20, 25, 30, 40, 60)
CAPITAL = 50_000.0
BUF = config.ROTATION_BUFFER
OOS = config.OOS_SPLIT


def base_rotation(closes, lookback):
    return strategy.etf_momentum_rotation(closes, lookback=lookback, buffer=BUF)


def vol_target(closes, lookback, vol_lb):
    w = base_rotation(closes, lookback)
    rets = closes.pct_change(fill_method=None).fillna(0.0)
    strat_ret = (w.shift(1).fillna(0.0) * rets).sum(axis=1)
    realized = strat_ret.rolling(vol_lb).std() * np.sqrt(252)
    target = realized.expanding(min_periods=vol_lb).median()
    scale = (target / realized).clip(upper=1.0).fillna(1.0)
    return w.mul(scale, axis=0)


def med_metrics(prices, closes, builder):
    annf, shpf, ddf, anno, shpo = [], [], [], [], []
    for lb in LBS:
        res = run_portfolio_backtest(prices, builder(closes, lb), initial_capital=CAPITAL, stamp_tax=False)
        mf = metrics_mod.equity_metrics(res.equity)
        oos = res.equity.loc[OOS:]
        mo = metrics_mod.equity_metrics(oos)
        annf.append(mf["年化收益率"]); shpf.append(mf["夏普比率"]); ddf.append(mf["最大回撤"])
        anno.append(mo["年化收益率"]); shpo.append(mo["夏普比率"])
    return (np.median(annf), np.median(shpf), np.median(ddf), np.median(anno), np.median(shpo))


def sharpe(r):
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return -np.inf
    return float(r.mean() / r.std() * np.sqrt(252))


def walk_forward(prices, closes, apply_vol, vol_lb=20, window=504, step=252):
    """每 step 日用过去 window 日信号级夏普选 lookback,向前应用;可选叠加波动率目标。"""
    rets = closes.pct_change(fill_method=None)
    wtabs = {lb: base_rotation(closes, lb) for lb in LBS}
    sig = {lb: (wtabs[lb].shift(1) * rets).sum(axis=1) for lb in LBS}
    idx = closes.index
    start = window
    wf = pd.DataFrame(0.0, index=idx, columns=closes.columns)
    i = start
    while i < len(idx):
        j = min(i + step, len(idx))
        best = max(LBS, key=lambda lb: sharpe(sig[lb].iloc[i - window:i]))
        wf.iloc[i:j] = wtabs[best].iloc[i:j].values
        i = j
    if apply_vol:
        strat_ret = (wf.shift(1).fillna(0.0) * rets.fillna(0.0)).sum(axis=1)
        realized = strat_ret.rolling(vol_lb).std() * np.sqrt(252)
        target = realized.expanding(min_periods=vol_lb).median()
        scale = (target / realized).clip(upper=1.0).fillna(1.0)
        wf = wf.mul(scale, axis=0)
    res = run_portfolio_backtest(prices, wf, initial_capital=CAPITAL, stamp_tax=False)
    eq = res.equity.iloc[start:]
    return metrics_mod.equity_metrics(eq), len(res.trades)


def main():
    prices = load_pool(config.ROTATION_START, pd.Timestamp.today().date().isoformat())
    closes = closes_table(prices)
    print(f"共同日历 {closes.index[0].date()} ~ {closes.index[-1].date()}\n")

    print("===== 1. V1 对波动窗口 vol_lb 的敏感性(中位数,跨动量lookback网格)=====")
    print(f"{'配置':<16}{'全年化':>9}{'全夏普':>9}{'全回撤':>9}{'样本外年化':>12}{'样本外夏普':>12}")
    b = med_metrics(prices, closes, base_rotation)
    print(f"{'基线(无目标)':<16}{b[0]:>8.1%}{b[1]:>9.2f}{b[2]:>8.1%}{b[3]:>11.1%}{b[4]:>12.2f}")
    for vlb in (10, 20, 40, 60):
        m = med_metrics(prices, closes, lambda c, lb, v=vlb: vol_target(c, lb, v))
        print(f"{'V1 vol_lb='+str(vlb):<16}{m[0]:>8.1%}{m[1]:>9.2f}{m[2]:>8.1%}{m[3]:>11.1%}{m[4]:>12.2f}")

    print("\n===== 2. Walk-forward(每年滚动选lookback,2年窗):基线 vs +波动率目标 =====")
    for label, av in (("WF 基线", False), ("WF + 波动率目标(vol_lb=20)", True)):
        m, nt = walk_forward(prices, closes, apply_vol=av)
        print(f"{label:<28} 年化 {m['年化收益率']:>6.1%}  夏普 {m['夏普比率']:>5.2f}  "
              f"回撤 {m['最大回撤']:>6.1%}  交易 {nt}")

    print("\n研究脚本,未改实盘。回测不代表未来。")


if __name__ == "__main__":
    main()
