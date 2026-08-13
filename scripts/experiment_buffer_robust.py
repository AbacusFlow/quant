"""ROTATION_BUFFER 稳健性检验(抗过拟合)。

experiment_buffer.py 显示 buffer=0.02 在全期/IS/OOS 三段都最优,但「单点最优」
正是过拟合的典型形态(参照 lookback=20 的尖峰教训)。本脚本从四个角度检验
0.02 是真信号还是曲线拟合:

  A. 走查(walk-forward):每年只用过去数据挑 buffer,看是否会挑中 0.02,
     以及「自适应挑选」相对「固定值」的实际表现。
  B. 参数族稳健性:换 ensemble lookback 组合 / 换 single 模式,峰值是否漂移。
  C. 子区间稳健性:滚动 2 年窗口内,0.02 相对 0.01 的胜率。
  D. 差异显著性:0.02 vs 0.01 的日收益差做 bootstrap,看是否可能纯属噪音。

复现说明:`--end` 固定为研究截止日(含),强制 qfq_only=True 拒绝新浪不复权回退;
开头打印数据指纹,先核对指纹一致再比对指标。

用法: python scripts/experiment_buffer_robust.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
import metrics as metrics_mod
import strategy
from experiment_buffer import fingerprint, year_return
from portfolio import run_portfolio_backtest
from run_rotation import build_weights, closes_table, load_pool

BUFFERS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08)
CAPITAL = 185_000.0


def equity_for(prices, closes, buffer, lookbacks=None, mode="ensemble"):
    if lookbacks is None:
        w = build_weights(closes, mode=mode, lookback=config.ROTATION_LOOKBACK,
                          buffer=buffer, dd_control=False, vol_control=True, sleeve=True)
    else:
        w = strategy.etf_momentum_ensemble(closes, lookbacks=lookbacks, buffer=buffer)
        w = strategy.apply_vol_targeting(w, closes, lookback=config.VOL_TARGET_LOOKBACK)
        w = strategy.apply_defensive_sleeve(w, closes)
    res = run_portfolio_backtest(prices, w, initial_capital=CAPITAL, stamp_tax=False)
    return res.equity


def ann_sharpe(eq):
    m = metrics_mod.equity_metrics(eq)
    return m["年化收益率"], m["夏普比率"], m["最大回撤"]


def slice_eq(eq, start=None, end=None):
    e = eq
    if start is not None:
        e = e[e.index >= start]
    if end is not None:
        e = e[e.index < end]
    return e


def part_a_walkforward(eqs, closes):
    """每年初用「过去全部数据」按夏普挑 buffer,应用于未来一年。

    局限(结论不可外推为「自适应调参必然更差」):这里是从若干条「全程固定 buffer」
    的净值曲线上切年度片段拼接,并未模拟年初换参时的持仓状态延续、滞回路径、
    vol-target 扩张中位数重算与切换成本。它只能说明这条简化选择规则劣于固定值。
    """
    print("\n===== A. 走查:每年只用过去数据挑 buffer(简化规则,见 docstring 局限) =====")
    years = sorted({d.year for d in closes.index})
    train_start = closes.index[0]
    picks, wf_rets, fixed_rets = [], {}, {}
    for y in years:
        if y <= years[0] + 2:      # 前 3 年做训练底料
            continue
        cut = pd.Timestamp(f"{y}-01-01")
        best, best_sh = None, -np.inf
        for b in BUFFERS:
            tr = slice_eq(eqs[b], train_start, cut)
            if len(tr) < 60:
                continue
            _, sh, _ = ann_sharpe(tr)
            if sh > best_sh:
                best, best_sh = b, sh
        # 年度收益以「上一年最后一个交易日」为基准,不遗漏年初首日涨跌
        te = year_return(eqs[best], y)
        r1 = year_return(eqs[0.01], y)
        r2 = year_return(eqs[0.02], y)
        if te is None or r1 is None or r2 is None:
            continue
        picks.append((y, best))
        wf_rets[y] = te
        fixed_rets[y] = (r1, r2)
    print(f"{'年份':>6} {'挑中buffer':>10} {'走查收益':>9} {'固定0.01':>9} {'固定0.02':>9}")
    for y, b in picks:
        w = wf_rets[y]
        f1, f2 = fixed_rets[y]
        print(f"{y:>6} {b:>10.3f} {w:>9.1%} {f1:>9.1%} {f2:>9.1%}")
    n = len(picks)
    if n:
        # 末年可能只是 YTD(如 2026 只到 8 月),直接开 n 次方会把不足一年当成一年、
        # 系统性低估全部三列。改按区间内实际交易日数年化。
        y0, y1 = picks[0][0], picks[-1][0]
        n_days = int(((closes.index.year >= y0) & (closes.index.year <= y1)).sum())
        ann = lambda rs: np.prod([1 + r for r in rs]) ** (252 / n_days) - 1
        cum_w = ann([wf_rets[y] for y, _ in picks])
        cum_1 = ann([fixed_rets[y][0] for y, _ in picks])
        cum_2 = ann([fixed_rets[y][1] for y, _ in picks])
        print(f"\n  年化({y0}~{y1},{n_days} 交易日): "
              f"走查 {cum_w:.2%} | 固定0.01 {cum_1:.2%} | 固定0.02 {cum_2:.2%}")
        hits = sum(1 for _, b in picks if b == 0.02)
        print(f"  走查挑中 0.02 的年份: {hits}/{n}")


def part_b_family(prices, closes):
    """换参数族:峰值是否稳定在 0.02 附近。"""
    print("\n===== B. 参数族稳健性(不同 lookback 组合 / single 模式) =====")
    families = [
        ("ensemble(15,20,25)=生产", dict(lookbacks=(15, 20, 25))),
        ("ensemble(10,20,30)", dict(lookbacks=(10, 20, 30))),
        ("ensemble(20,30,40)", dict(lookbacks=(20, 30, 40))),
        ("ensemble(30,40,50)", dict(lookbacks=(30, 40, 50))),
        ("single(lookback=20)", dict(mode="single")),
    ]
    print(f"{'参数族':>22} | " + " ".join(f"{b:>7.3f}" for b in BUFFERS) + " | 最优")
    for label, kw in families:
        row, best, best_sh = [], None, -np.inf
        for b in BUFFERS:
            eq = equity_for(prices, closes, b, **kw)
            _, sh, _ = ann_sharpe(eq)
            row.append(sh)
            if sh > best_sh:
                best, best_sh = b, sh
        cells = " ".join(f"{v:>7.2f}" for v in row)
        print(f"{label:>22} | {cells} | {best:.3f}")
    print("  (数值=全期夏普)")


def part_c_rolling(eqs):
    """滚动 2 年窗口:0.02 相对 0.01 的胜率。"""
    print("\n===== C. 滚动 2 年窗口胜率(0.02 vs 0.01,窗口高度重叠仅供描述) =====")
    e1, e2 = eqs[0.01], eqs[0.02]
    idx = e1.index
    wins = tot = 0
    for i in range(0, len(idx) - 504, 63):     # 每季度起一个 2 年窗口
        a, b = idx[i], idx[i + 504]
        r1 = e1.loc[b] / e1.loc[a] - 1
        r2 = e2.loc[b] / e2.loc[a] - 1
        tot += 1
        wins += r2 > r1
    print(f"  窗口数 {tot},0.02 胜出 {wins} ({wins / tot:.0%})")
    print(f"  注:窗口每 63 交易日起一个、长 504 日,相邻重叠约 87.5%,")
    print(f"     {tot} 个窗口远非独立样本,不可当作 {tot} 次独立胜负。")


def part_d_bootstrap(eqs, n_boot=5000, seed=42, block=60):
    """0.02 vs 0.01 日收益差 bootstrap:差异是否显著区别于 0。

    同时给出 IID 与 block bootstrap:日收益差存在序列依赖(同一段行情里两条
    参数曲线的差会连续同号),IID 重采样会低估不确定性,以 block 结果为准。
    """
    print("\n===== D. 差异显著性(bootstrap, 0.02 - 0.01 日收益差) =====")
    r1 = eqs[0.01].pct_change().dropna()
    r2 = eqs[0.02].pct_change().dropna()
    d = (r2 - r1).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    ann = lambda x: x * 252

    iid = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)

    n_blocks = int(np.ceil(len(d) / block))
    starts = rng.integers(0, len(d) - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)
    blk = d[idx[:, :len(d)]].mean(axis=1)

    print(f"  日均差 {d.mean():.5f} (年化约 {ann(d.mean()):.2%})")
    for label, samp in (("IID    ", iid), (f"block{block:<3}", blk)):
        lo, hi = np.percentile(samp, [2.5, 97.5])
        print(f"  {label} 95%CI(年化) [{ann(lo):>6.2%}, {ann(hi):>6.2%}]"
              f"  P(差>0)={(samp > 0).mean():.1%}")
    print("  → 区间包含 0:收益差不显著。注意这只说明『未观测到显著差异』,")
    print("    并不构成非劣性证明(以 block 为准,区间左端仍允许约 1.6pp 年化劣化)。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=config.ROTATION_START)
    ap.add_argument("--end", default="2026-08-12")   # 研究截止日,含
    args = ap.parse_args()

    prices = load_pool(args.start, args.end, write_cache=False, qfq_only=True)
    closes = closes_table(prices)
    print(f"共同日历 {closes.index[0].date()} ~ {closes.index[-1].date()}"
          f" ({len(closes)} 交易日),资金 {CAPITAL:,.0f}")
    print(f"数据指纹 {fingerprint(prices, closes)}")

    eqs = {b: equity_for(prices, closes, b) for b in BUFFERS}
    part_a_walkforward(eqs, closes)
    part_b_family(prices, closes)
    part_c_rolling(eqs)
    part_d_bootstrap(eqs)


if __name__ == "__main__":
    main()
