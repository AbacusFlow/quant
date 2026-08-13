"""换仓缓冲(ROTATION_BUFFER)敏感性研究。

背景:2026-08-11~08-13 出现 512100「买入→次日卖出→再买入」的一日游抖动,
怀疑 buffer=0.01 过小,排名轻微抖动即触发换仓。本脚本量化「调大 buffer」的
收益/成本权衡,不修改任何生产代码。

指标:
  - 全期 / 样本内 / 样本外 年化、夏普、最大回撤
  - 逐年收益(检查改善是否只来自个别年份)
  - 信号级换手率、成交笔数、佣金 / 滑点 / 总摩擦(真实资金口径)
  - 抖动次数:某标的信号权重从 0 变正后 <= N 个交易日又归零

复现说明:`--end` 固定为研究截止日(含),且强制 qfq_only=True 拒绝新浪不复权回退,
避免复权口径混入。脚本开头打印数据指纹(共同日历上的开盘价 + 收盘价校验和),复现时应先核对指纹
一致再比对指标;行情源历史数据偶有微调,指纹不一致时表格末位数字可能小幅漂移。

用法:
  python scripts/experiment_buffer.py [--capital 185000] [--end 2026-08-12]
"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
import metrics as metrics_mod
from portfolio import run_portfolio_backtest
from run_rotation import build_weights, closes_table, load_pool

BUFFERS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)


def fingerprint(prices: dict[str, pd.DataFrame], closes: pd.DataFrame) -> str:
    """行情指纹:共同日历上的开盘价 + 收盘价。

    信号用收盘价,但成交在次日开盘价,开盘价会改变净值/笔数/佣金/滑点。
    只哈希收盘价会漏掉这一半输入(改开盘价而指纹不变),故两者都纳入。
    """
    opens = pd.DataFrame({s: df["open"] for s, df in prices.items()})
    opens = opens.loc[closes.index, list(closes.columns)]
    payload = (f"{closes.shape}|{list(closes.columns)}|"
               f"{closes.index[0].date()}|{closes.index[-1].date()}|"
               + ",".join(f"{v:.6f}" for v in closes.to_numpy().ravel())
               + "|" + ",".join(f"{v:.6f}" for v in opens.to_numpy().ravel()))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def turnover(weights: pd.DataFrame, start=None) -> float:
    """年化信号级换手率(单边):每日权重变化绝对值之和 / 2,再年化。

    `start` 只过滤统计区间,差分仍在完整序列上计算——否则窗口首日相对
    窗口前一日的权重变化会被 diff 的首行 NaN 吃掉。
    """
    delta = weights.diff().abs().sum(axis=1).iloc[1:] / 2.0
    if start is not None:
        delta = delta[delta.index >= start]
    if len(delta) == 0:
        return float("nan")
    return delta.sum() / len(delta) * 252


def whipsaw_count(weights: pd.DataFrame, max_hold: int = 3, start=None) -> int:
    """抖动次数:某标的权重由 0 转正后,在 max_hold 个交易日内又归零。

    持有天数始终在完整序列上累计,`start` 只筛选「归零日」是否计入统计区间。
    否则窗口开始时早已持有的标的会被误当作窗口内新建仓,虚增抖动次数。

    注:持有天数从 i=1 起累计,若序列第一根就已持仓会少算一天;实际权重表因
    lookback 预热期首行必为 0,不会触发该退化。
    """
    n = 0
    idx = weights.index
    for sym in weights.columns:
        w = weights[sym].to_numpy()
        held = 0
        for i in range(1, len(w)):
            if w[i] > 0:
                held += 1
            elif held:
                if held <= max_hold and (start is None or idx[i] >= start):
                    n += 1
                held = 0
    return n


def seg_slice(series: pd.Series, start=None, end=None) -> pd.Series:
    """区间切片:把 start 前一个交易日作为基准点纳入,不遗漏区间首日收益。"""
    s = series
    if end is not None:
        s = s[s.index < end]
    if start is not None:
        pos = max(int(s.index.searchsorted(start, side="left")) - 1, 0)
        s = s.iloc[pos:]
    return s


def seg_metrics(equity: pd.Series, start=None, end=None) -> dict:
    e = seg_slice(equity, start, end)
    if len(e) < 2:
        return {}
    return metrics_mod.equity_metrics(e)


def year_return(equity: pd.Series, year: int) -> float | None:
    """某年收益:以「上一年最后一个交易日」净值为基准,不遗漏年初首日涨跌。

    首年无前一交易日,退化为用当年首日为基准(会漏掉首日,已在报告中标注)。
    """
    pos = np.flatnonzero(equity.index.year == year)
    if len(pos) < 2:
        return None
    first, last = int(pos[0]), int(pos[-1])
    base = equity.iloc[first - 1] if first > 0 else equity.iloc[first]
    return equity.iloc[last] / base - 1


def yearly(equity: pd.Series) -> dict:
    out = {}
    for y in sorted({int(v) for v in equity.index.year}):
        r = year_return(equity, y)
        if r is not None:
            out[y] = r
    return out


def slippage_cost(trades) -> float:
    """模型滑点成本:成交价已含滑点,反推与开盘价的差额 × 股数。

    买入 price = open×(1+s) → 成本 = price×shares×s/(1+s)
    卖出 price = open×(1-s) → 成本 = price×shares×s/(1-s)
    """
    s = config.SLIPPAGE_RATE
    total = 0.0
    for t in trades:
        gross = t.price * t.shares
        total += gross * (s / (1 + s) if t.side == "buy" else s / (1 - s))
    return total


def run(prices, closes, buffer: float, capital: float,
        vol_control: bool, sleeve: bool) -> dict:
    w = build_weights(closes, mode="ensemble", lookback=config.ROTATION_LOOKBACK,
                      buffer=buffer, dd_control=False,
                      vol_control=vol_control, sleeve=sleeve)
    res = run_portfolio_backtest(prices, w, initial_capital=capital, stamp_tax=False)
    eq = res.equity
    commission = sum(t.fee for t in res.trades)
    slip = slippage_cost(res.trades)
    return {
        "buffer": buffer,
        "full": seg_metrics(eq),
        "is": seg_metrics(eq, end=config.OOS_SPLIT),
        "oos": seg_metrics(eq, start=config.OOS_SPLIT),
        "trades": len(res.trades),
        "commission": commission,
        "slippage": slip,
        "friction": commission + slip,
        "turnover": turnover(w),
        "whip3": whipsaw_count(w, 3),
        "whip1": whipsaw_count(w, 1),
        "yearly": yearly(eq),
        "equity": eq,
        "weights": w,
    }


def table(rows: list[dict], title: str) -> None:
    print(f"\n===== {title} =====")
    print(f"{'buffer':>7} | {'年化':>7} {'夏普':>6} {'回撤':>7} | "
          f"{'OOS年化':>8} {'OOS夏普':>7} {'OOS回撤':>8} | "
          f"{'笔数':>5} {'佣金':>8} {'滑点':>8} {'总摩擦':>8} "
          f"{'换手':>6} {'抖<=3':>6} {'抖=1':>5}")
    for r in rows:
        f, o = r["full"], r["oos"]
        print(f"{r['buffer']:>7.3f} | {f['年化收益率']:>7.2%} {f['夏普比率']:>6.2f} "
              f"{f['最大回撤']:>7.1%} | {o['年化收益率']:>8.2%} {o['夏普比率']:>7.2f} "
              f"{o['最大回撤']:>8.1%} | {r['trades']:>5} {r['commission']:>8,.0f} "
              f"{r['slippage']:>8,.0f} {r['friction']:>8,.0f} "
              f"{r['turnover']:>6.1f} {r['whip3']:>6} {r['whip1']:>5}")


def yearly_table(rows: list[dict]) -> None:
    years = sorted({y for r in rows for y in r["yearly"]})
    print("\n===== 逐年收益(生产口径) =====")
    print("  年份 | " + " ".join(f"{r['buffer']:>7.3f}" for r in rows))
    for y in years:
        cells = " ".join(f"{r['yearly'].get(y, float('nan')):>7.1%}" for r in rows)
        print(f"  {y} | {cells}")


def recent_table(rows: list[dict], years: int = 2) -> None:
    """近 N 年切片:检查改善在当前市场状态下还剩多少(全期改善可能来自早年震荡市)。

    净值按 seg_slice 纳入窗口前一个交易日为基准;换手/抖动在完整序列上计算、
    只按日期筛选统计区间(窗口前的持仓状态必须延续,否则会虚增新建仓)。
    """
    cut = rows[0]["equity"].index[-1] - pd.DateOffset(years=years)
    print(f"\n===== 近 {years} 年({cut.date()} 起,生产口径) =====")
    print(f"{'buffer':>7} | {'年化':>7} {'夏普':>6} {'回撤':>7} "
          f"{'换手':>6} {'抖<=3':>6} {'抖=1':>5}")
    for r in rows:
        eq = seg_slice(r["equity"], start=cut)
        w = r["weights"]
        if len(eq) < 2:
            continue
        m = metrics_mod.equity_metrics(eq)
        print(f"{r['buffer']:>7.3f} | {m['年化收益率']:>7.2%} {m['夏普比率']:>6.2f} "
              f"{m['最大回撤']:>7.1%} {turnover(w, start=cut):>6.1f} "
              f"{whipsaw_count(w, 3, start=cut):>6} {whipsaw_count(w, 1, start=cut):>5}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=config.ROTATION_START)
    ap.add_argument("--end", default="2026-08-12")   # 研究截止日,含
    ap.add_argument("--capital", type=float, default=185_000.0)
    args = ap.parse_args()

    prices = load_pool(args.start, args.end, write_cache=False, qfq_only=True)
    closes = closes_table(prices)
    print(f"\n共同日历: {closes.index[0].date()} ~ {closes.index[-1].date()} "
          f"({len(closes)} 个交易日),资金 {args.capital:,.0f}")
    print(f"数据指纹: {fingerprint(prices, closes)}")

    prod = [run(prices, closes, b, args.capital, vol_control=True, sleeve=True)
            for b in BUFFERS]
    table(prod, f"生产口径 ensemble + vol-target + sleeve(资金 {args.capital:,.0f})")
    yearly_table(prod)
    recent_table(prod, years=2)

    bare = [run(prices, closes, b, args.capital, vol_control=False, sleeve=False)
            for b in BUFFERS]
    table(bare, f"纯轮动 ensemble(无 vol/sleeve,资金 {args.capital:,.0f})")

    big = [run(prices, closes, b, 1_000_000.0, vol_control=True, sleeve=True)
           for b in BUFFERS]
    table(big, "生产口径(资金 1,000,000,弱化最低佣金影响)")


if __name__ == "__main__":
    main()
