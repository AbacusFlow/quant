"""抗过拟合的结构性改进研究(research-only,不改 strategy.py/config.py/实盘逻辑)。

目标:在 walk-forward / 样本外口径下,检验几个"有经济逻辑的结构改动"能否稳健提升
风险调整后表现——而非只在 lookback=20 这个尖峰上变好(那是过拟合)。

判定标准(抗过拟合):一个改动要算"有效",必须
  (a) 在 lookback {10,15,20,25,30,40,60} 全区间的【中位数】上改善(年化或夏普),且
  (b) 样本外(2022+)也不变差。
只在尖峰变好、中位数/样本外变差的,一律视为过拟合,丢弃。

变体(均有文献依据,用标准默认参数,不做参数搜索):
  V1 波动率目标(Barroso & Santa-Clara 2015):波动飙升时降仓,无杠杆
  V2 风险调整动量排序:按 收益/波动 选,而非纯涨幅
  V3 Top-N 反波动分散:持前 N 名反波动加权(5万下佣金可忽略)
  V4 月度调仓:动量是月级效应,日度调仓拟合噪声+加成本
  V5 长期趋势闸门(Antonacci 双动量):只买 200 日均线上方标的

用法:docker run --rm --network=host -v "$PWD":/work quant python scripts/research_robust.py
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


# ---------- 变体构造(全部因果:T 日权重只用 ≤T 数据;引擎再 shift(1) 到 T+1 执行)----------

def base_rotation(closes, lookback):
    return strategy.etf_momentum_rotation(closes, lookback=lookback, buffer=BUF)


def _select_loop(closes, score, abs_mom, buffer):
    """通用 top-1 选择:按 score 选,abs_mom>0 作绝对动量过滤,buffer 抑制换手。"""
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    current = None
    for date in closes.index:
        s = score.loc[date]
        m = abs_mom.loc[date]
        if s.isna().all():
            current = None
            continue
        best = s.idxmax()
        if (current is not None and not pd.isna(s.get(current, np.nan))
                and m.get(current, 0) > 0):
            if best != current and s[best] > s[current] + buffer:
                current = best
        else:
            current = best if m.get(best, 0) > 0 else None
        if current is not None and m.get(current, 0) <= 0:
            current = None
        if current is not None:
            weights.at[date, current] = 1.0
    return weights


def v1_vol_target(closes, lookback, vol_lb=20):
    """对基线权重做波动率目标缩放:scale = min(1, 目标波动/当前波动),
    目标 = 策略自身已实现波动的因果扩张中位数(约半数时间满仓、半数降仓),无杠杆。"""
    w = base_rotation(closes, lookback)
    rets = closes.pct_change(fill_method=None).fillna(0.0)
    strat_ret = (w.shift(1).fillna(0.0) * rets).sum(axis=1)
    realized = strat_ret.rolling(vol_lb).std() * np.sqrt(252)
    target = realized.expanding(min_periods=vol_lb).median()  # 仅用过去
    scale = (target / realized).clip(upper=1.0).fillna(1.0)
    return w.mul(scale, axis=0)


def v2_risk_adj(closes, lookback):
    mom = closes.pct_change(lookback, fill_method=None)
    vol = closes.pct_change(fill_method=None).rolling(lookback).std()
    score = mom / vol.replace(0.0, np.nan)
    return _select_loop(closes, score, abs_mom=mom, buffer=BUF)


def v3_topn_invvol(closes, lookback, n=3):
    mom = closes.pct_change(lookback, fill_method=None)
    vol = closes.pct_change(fill_method=None).rolling(lookback).std()
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for date in closes.index:
        m = mom.loc[date]
        if m.isna().all():
            continue
        elig = m[m > 0]  # 绝对动量过滤
        if elig.empty:
            continue
        top = elig.nlargest(min(n, len(elig))).index
        v = vol.loc[date, top]
        inv = (1.0 / v).replace([np.inf, -np.inf], np.nan)
        if inv.isna().all():
            wv = pd.Series(1.0 / len(top), index=top)
        else:
            inv = inv.fillna(inv.max())
            wv = inv / inv.sum()
        for s in top:
            weights.at[date, s] = wv[s]
    return weights


def v4_monthly(closes, lookback):
    """月度调仓:每月最后一个交易日采用当日信号,月内持有不变。"""
    w = base_rotation(closes, lookback)
    idx = w.index
    period = idx.to_period("M")
    is_month_end = np.array(
        [period[i] != period[i + 1] if i + 1 < len(idx) else True for i in range(len(idx))])
    held = pd.Series(0.0, index=w.columns)
    rows = {}
    for i, date in enumerate(idx):
        if is_month_end[i]:
            held = w.loc[date].copy()
        rows[date] = held.copy()
    return pd.DataFrame(rows).T


def v5_trend_gate(closes, lookback, ma=200):
    mom = closes.pct_change(lookback, fill_method=None)
    trend_ok = closes > closes.rolling(ma).mean()
    masked = mom.where(trend_ok)  # 不在均线上方 -> NaN(不合格)
    return _select_loop(closes, masked, abs_mom=masked, buffer=BUF)


VARIANTS = {
    "基线(top1,lookback)": base_rotation,
    "V1 波动率目标": v1_vol_target,
    "V2 风险调整排序": v2_risk_adj,
    "V3 Top3反波动": v3_topn_invvol,
    "V4 月度调仓": v4_monthly,
    "V5 趋势闸门200": v5_trend_gate,
}


def evaluate(prices, closes):
    summary = []
    for name, fn in VARIANTS.items():
        ann_full, shp_full, dd_full = [], [], []
        ann_oos, shp_oos = [], []
        trades = []
        for lb in LBS:
            w = fn(closes, lb)
            res = run_portfolio_backtest(prices, w, initial_capital=CAPITAL, stamp_tax=False)
            mf = metrics_mod.equity_metrics(res.equity)
            oos = res.equity.loc[OOS:]
            mo = metrics_mod.equity_metrics(oos) if len(oos) >= 2 else None
            ann_full.append(mf["年化收益率"]); shp_full.append(mf["夏普比率"]); dd_full.append(mf["最大回撤"])
            if mo:
                ann_oos.append(mo["年化收益率"]); shp_oos.append(mo["夏普比率"])
            trades.append(len(res.trades))
        summary.append({
            "策略": name,
            "全区间年化(中位)": np.median(ann_full),
            "全区间夏普(中位)": np.median(shp_full),
            "全区间回撤(中位)": np.median(dd_full),
            "样本外年化(中位)": np.median(ann_oos) if ann_oos else np.nan,
            "样本外夏普(中位)": np.median(shp_oos) if shp_oos else np.nan,
            "lb=20年化": ann_full[LBS.index(20)],
            "lb=20夏普": shp_full[LBS.index(20)],
            "交易次数(中位)": int(np.median(trades)),
        })
    df = pd.DataFrame(summary)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)
    fmt = df.copy()
    for c in ["全区间年化(中位)", "全区间回撤(中位)", "样本外年化(中位)", "lb=20年化"]:
        fmt[c] = fmt[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
    for c in ["全区间夏普(中位)", "样本外夏普(中位)", "lb=20夏普"]:
        fmt[c] = fmt[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    print("\n========== 抗过拟合稳健性对比(本金5万,含费用,ETF免印花)==========")
    print("判定:看【中位数】列(全参数区间)与【样本外】列,而非 lb=20 尖峰\n")
    print(fmt.to_string(index=False))
    return df


def main():
    prices = load_pool(config.ROTATION_START, pd.Timestamp.today().date().isoformat())
    closes = closes_table(prices)
    print(f"共同日历 {closes.index[0].date()} ~ {closes.index[-1].date()},lookback 网格 {LBS}")
    evaluate(prices, closes)
    print("\n提示:此为研究脚本,未改动实盘策略;结论见报告。回测不代表未来。")


if __name__ == "__main__":
    main()
