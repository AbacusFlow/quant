"""跨境 QDII 溢价研究实验(预登记见 research/2026-10-15-qdii-premium.md)。

B0 现行 V2 / A 净值动量 / B 溢价过滤 / C=A+B,成交一律按真实场内开盘价(含溢价)。
数据:akshare 前复权行情(qfq_only, write_cache=False)+ research/data/qdii_nav_close_ak.csv
(akshare 净值/新浪未复权收盘,由 scripts/fetch_qdii_nav_ak.py 生成并与妙想数据交叉核对)。

用法:
  python scripts/experiment_qdii_premium.py [--lag 2] [--capital 100000] [--x 0.05] [--end 2026-10-14]
  python scripts/experiment_qdii_premium.py --all     # 主方案 + 全部预登记敏感性
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import strategy
from experiment_buffer import fingerprint, seg_metrics, yearly
from portfolio import run_portfolio_backtest
from run_rotation import build_weights, closes_table, load_pool

QDII = ["513100", "513500", "159920"]
NAV_CSV = os.path.join(ROOT, "research", "data", "qdii_nav_close_ak.csv")  # akshare 全量;与妙想重叠 5588 行逐位一致


def load_premium(index: pd.DatetimeIndex, prev_nav: bool = False) -> pd.DataFrame:
    """逐日溢价 prem = 未复权收盘 / 单位净值 − 1,对齐到 A 股共同日历。

    净值前向填充(美股休市日 A 股照常交易);收盘价缺失视为数据缺口 → 报错(不静默)。
    prev_nav=True(稳健性,非预登记):用前一交易日净值——A 股 15:00 收盘时美股 T 日尚未开盘,
    可参照的公允价值是 T−1 净值;同日口径会把收盘后发生的美股 T 日涨跌混入溢价。
    """
    raw = pd.read_csv(NAV_CSV, dtype={"symbol": str}, parse_dates=["date"])
    nav = raw.pivot(index="date", columns="symbol", values="nav")[QDII]
    close = raw.pivot(index="date", columns="symbol", values="close")[QDII]
    full = index.union(nav.index)
    nav = nav.reindex(full).ffill().reindex(index)
    if prev_nav:
        # 份额拆分日:前一日净值是拆分前口径,须按拆分比例折算(否则出现 −80% 之类的伪溢价)。
        # 拆分识别:单日净值比 < 0.6 或 > 1.67(QDII 单日真实涨跌远小于此),比例取最近整数。
        r = nav / nav.shift(1)
        k = pd.DataFrame(1.0, index=nav.index, columns=nav.columns)
        for s in QDII:
            for d in r.index[(r[s] < 0.6) | (r[s] > 1.67)]:
                ratio = r.at[d, s]
                k.at[d, s] = 1.0 / round(1.0 / ratio) if ratio < 1 else float(round(ratio))
                print(f"  拆分/合并识别: {s} {d.date()} 净值比 {ratio:.4f} → 折算系数 {k.at[d, s]:.4f}")
        nav = nav.shift(1) * k
    close = close.reindex(index)
    gap = (close.isna() | nav.isna()).iloc[1:] if prev_nav else close.isna() | nav.isna()
    if gap.any().any():
        bad = {s: [d.date().isoformat() for d in gap.index[gap[s]]][:5] for s in QDII if gap[s].any()}
        raise ValueError(f"净值/收盘数据缺口(前5): {bad}")
    return close / nav - 1


def nav_hash() -> str:
    with open(NAV_CSV, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def weights_for(closes, prem, variant: str, lag: int, x: float) -> pd.DataFrame:
    """按预登记机制构造目标权重(全部因果:prem/adj 均 shift(lag))。"""
    kw = dict(buffer=config.ROTATION_BUFFER)
    sig = closes
    if variant in ("A", "C"):
        adj = closes[QDII] / (1 + prem)          # 去溢价 = 复权净值
        sig = closes.copy()
        sig[QDII] = adj.shift(lag)
    w = strategy.etf_momentum_ensemble(sig, lookbacks=config.ENSEMBLE_LOOKBACKS, **kw)
    w = strategy.apply_vol_targeting(w, sig, lookback=config.VOL_TARGET_LOOKBACK)
    if variant in ("B", "C"):
        known = prem.shift(lag)
        w = w.copy()
        for s in QDII:
            w.loc[known[s] > x, s] = 0.0
    return strategy.apply_defensive_sleeve(w, sig)


def premium_contrib(res, closes, prem) -> float:
    """持有跨境标的期间来自溢价变化的累计对数收益 ≈ Σ w_{t−1} · Δln(1+prem_t)。"""
    h = res.holdings
    val = h[QDII] * closes.loc[h.index, QDII]
    w = val.div(res.equity, axis=0).shift(1)
    dlp = np.log1p(prem.loc[h.index]).diff()
    return float((w * dlp).sum().sum())


def stress_final(res, closes, prem) -> float:
    """期末溢价瞬间归零:跨境持仓按净值计价的终值。"""
    d = res.equity.index[-1]
    adj = sum(res.holdings.at[d, s] * closes.at[d, s] * prem.at[d, s] / (1 + prem.at[d, s]) for s in QDII)
    return float(res.equity.iloc[-1] - adj)


def block_boot(e0, e1, n=5000, block=60, seed=42):
    d = (e1.pct_change() - e0.pct_change()).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(d) / block))
    st = rng.integers(0, len(d) - block + 1, size=(n, nb))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :len(d)]
    m = d[idx].mean(axis=1) * 252
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def rolling_win(e0, e1, years=2):
    n = 252 * years
    r0 = e0 / e0.shift(n) - 1
    r1 = e1 / e1.shift(n) - 1
    ok = r0.notna()
    return float((r1[ok] > r0[ok]).mean())


def run(prices, closes, prem, lag, capital, x, verbose=True) -> dict:
    out = {}
    for v in ("B0", "A", "B", "C"):
        w = weights_for(closes, prem, v, lag, x)
        if v == "B0":
            ref = build_weights(closes, mode="ensemble", lookback=config.ROTATION_LOOKBACK,
                                buffer=config.ROTATION_BUFFER, dd_control=False,
                                vol_control=True, sleeve=True)
            assert np.allclose(w.to_numpy(), ref.to_numpy()), "B0 与线上 V2 权重不一致"
        if v == "B":
            known = prem.shift(lag)
            base = weights_for(closes, prem, "B0", lag, x)
            over = (known[QDII] > x)
            hit = over & (base[QDII] > 1e-9)
            trig = {s: (int(over[s].sum()), int(hit[s].sum()),
                        sorted({d.year for d in hit.index[hit[s]]})) for s in QDII}
        res = run_portfolio_backtest(prices, w, initial_capital=capital, stamp_tax=False)
        eq = res.equity
        f, p22 = seg_metrics(eq), seg_metrics(eq, start="2022-01-01")
        yrs = len(eq) / 252
        pc = premium_contrib(res, closes, prem)
        ann_ex = np.exp((np.log(eq.iloc[-1] / eq.iloc[0]) - pc) / yrs) - 1
        out[v] = dict(eq=eq, f=f, p22=p22, pc=pc, ann_ex=ann_ex, stress=stress_final(res, closes, prem),
                      trades=len(res.trades) / yrs, yearly=yearly(eq),
                      hold_q=(res.holdings[QDII] > 0).any(axis=1).mean())
    if verbose:
        print(f"\n--- lag={lag}  资金={capital:,.0f}  X={x:.0%} ---")
        print(f"{'方案':<4} {'年化':>7} {'夏普':>5} {'回撤':>7} | {'22后年化':>8} {'夏普':>5} | "
              f"{'溢价贡献':>8} {'剔溢价年化':>9} | {'期末':>9} {'溢价归零终值':>11} | {'笔/年':>5} {'持跨境天%':>8}")
        for v, r in out.items():
            print(f"{v:<4} {r['f']['年化收益率']:>7.2%} {r['f']['夏普比率']:>5.2f} {r['f']['最大回撤']:>7.1%} | "
                  f"{r['p22']['年化收益率']:>8.2%} {r['p22']['夏普比率']:>5.2f} | "
                  f"{r['pc']:>+8.2%} {r['ann_ex']:>9.2%} | {r['eq'].iloc[-1]:>9,.0f} {r['stress']:>11,.0f} | "
                  f"{r['trades']:>5.0f} {r['hold_q']:>8.0%}")
        print("  B 触发:标的 超阈值天数 / 其中改变目标(B0 当日目标权重非零)天数 / 涉及年份 → "
              + "; ".join(f"{s} {a}/{h} {ys}" for s, (a, h, ys) in trig.items()))
        b0 = out["B0"]["eq"]
        for v in ("A", "B", "C"):
            lo, hi = block_boot(b0, out[v]["eq"])
            ys = [y for y in out[v]["yearly"] if y in out["B0"]["yearly"]]
            wins = sum(out[v]["yearly"][y] > out["B0"]["yearly"][y] for y in ys)
            print(f"  {v} vs B0: block bootstrap 年化差 95%CI [{lo:+.2%}, {hi:+.2%}]  "
                  f"逐年胜 {wins}/{len(ys)}  滚动2年胜率 {rolling_win(b0, out[v]['eq']):.0%}")
    return out


def _abcd(res: dict, v: str) -> tuple[bool, bool, bool, bool]:
    b0, r = res["B0"], res[v]
    lo, _ = block_boot(b0["eq"], r["eq"])
    return (r["f"]["夏普比率"] >= b0["f"]["夏普比率"] and r["f"]["最大回撤"] >= b0["f"]["最大回撤"] - 0.02,
            r["ann_ex"] >= b0["ann_ex"] - 0.01,
            r["stress"] >= b0["stress"],
            lo > -0.02)


def gate(main: dict, sens: list[dict]) -> None:
    """预登记采纳门槛 (a)~(e)。(e)=整体结论((a)~(d) 是否全过)在各敏感性口径下与主口径一致。"""
    print("\n===== 预登记门槛判定(主方案口径)=====")
    for v in ("A", "B", "C"):
        a, b, c, d = _abcd(main, v)
        e = all(all(_abcd(s, v)) == all((a, b, c, d)) for s in sens)
        print(f"  {v}: (a)夏普/回撤 {'✓' if a else '✗'}  (b)剔溢价年化 {'✓' if b else '✗'}  "
              f"(c)溢价归零终值 {'✓' if c else '✗'}  (d)bootstrap下限 {'✓' if d else '✗'}  "
              f"(e)敏感性方向一致 {'✓' if e else '✗'}  → {'通过' if all((a, b, c, d, e)) else '不通过'}")


def describe_premium(prem):
    print("\n===== 溢价概况(按年:均值 / 最大 / 最小)=====")
    for s in QDII:
        g = prem[s].groupby(prem.index.year)
        print(f"  {s}: " + "  ".join(f"{y}:{m:+.1%}/{mx:+.1%}/{mn:+.1%}"
                                     for y, m, mx, mn in zip(g.mean().index, g.mean(), g.max(), g.min())))
    jumps = prem.diff().abs()
    big = jumps[(jumps > 0.04).any(axis=1)]
    print(f"  单日溢价变动 >4pp 的天数: {len(big)}(若集中在个别日期,疑为拆分日净值/价格错位)")
    for d, row in big.head(10).iterrows():
        print(f"    {d.date()} " + " ".join(f"{s}:{prem.at[d, s]:+.1%}" for s in QDII if row[s] > 0.04))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-10-14")
    ap.add_argument("--lag", type=int, default=2)
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--x", type=float, default=0.05)
    ap.add_argument("--all", action="store_true", help="主方案 + 预登记敏感性 + 门槛判定")
    ap.add_argument("--prev-nav", action="store_true", help="稳健性(非预登记):溢价用前一日净值")
    args = ap.parse_args()

    prices = load_pool(config.ROTATION_START, args.end, write_cache=False, qfq_only=True)
    closes = closes_table(prices)
    prem = load_premium(closes.index, prev_nav=args.prev_nav)
    if args.prev_nav:
        print("【稳健性口径:溢价 = 收盘 / 前一日净值 − 1(非预登记)】")
    print(f"\n共同日历 {closes.index[0].date()} ~ {closes.index[-1].date()} ({len(closes)} 日)  "
          f"行情指纹 {fingerprint(prices, closes)}  净值数据指纹 {nav_hash()}")
    describe_premium(prem)

    main_res = run(prices, closes, prem, args.lag, args.capital, args.x)
    if args.all:
        sens = [run(prices, closes, prem, 1, args.capital, args.x),
                run(prices, closes, prem, args.lag, 185_000.0, args.x)]
        print("\n===== 阈值敏感性(仅展示,不参与选择)=====")
        for x in (0.03, 0.08, 0.12):
            run(prices, closes, prem, args.lag, args.capital, x)
        gate(main_res, sens)


if __name__ == "__main__":
    main()
