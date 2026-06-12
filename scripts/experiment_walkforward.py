"""Walk-forward 实验:滚动窗口选 lookback(只用过去数据),与固定 lookback=20 对比。

研究用一次性脚本,不进生产流程。
  docker run --rm --network=host -v "$PWD":/work quant python scripts/experiment_walkforward.py

方法:
- 候选 lookback = (10, 15, 20, 25, 30, 40, 60)(与 --sensitivity 扫描一致)
- 每个调参日,用截至前一日的 trailing window 内"信号级日收益"
  (权重 shift(1) × 收盘收益,无费用)的夏普选出最优 lookback
- 向前应用 step 个交易日,滚动到底;拼接的权重序列喂组合回测(费用口径同实盘)
- 固定 lookback=20 在同一评估起点、同一资金下回测作对照
- 输出:各配置指标对比、选参时间线、与固定20的持仓重合度

因果性:第 t 日的选参只用 t 之前的数据;动量权重本身只依赖过去 lookback 天,
buffer 的路径依赖也只回溯历史,全程无未来函数。

口径说明(Codex 审查存档):
- buffer 状态来自"每个 lookback 专家从历史起点各自虚拟运行"的路径,切换 lookback 时
  不重置 buffer——因果成立,符合 walk-forward 选专家的设定
- sigret 是 close-to-close 近似,仅用于窗口内选参;最终指标一律走含费用组合回测

结论(2026-06-12 实跑,评估区间 2020-10~2026-06,本金1万):
WF 年化 17~21%(2年窗/年调仅3.4%)vs 固定20 的 28.1%;持仓重合 64~90%。
固定20 的超额有过拟合/运气成分,实盘合理预期年化 10~20%;WF 自身对窗口/频率
敏感且更差,不切换。操作不变,10万切 ensemble 作为参数尖峰风险的对冲。
"""
import datetime as dt
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
CAPITAL = 10_000.0


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return -np.inf
    return float(r.mean() / r.std() * np.sqrt(252))


def seg_metrics(equity: pd.Series) -> dict:
    return metrics_mod.equity_metrics(equity)


def main():
    end = dt.date.today().isoformat()
    prices = load_pool(config.ROTATION_START, end)
    closes = closes_table(prices)
    rets = closes.pct_change(fill_method=None)

    # 各 lookback 的权重表与信号级日收益(无费用,仅用于窗口内选参)
    wtabs = {lb: strategy.etf_momentum_rotation(closes, lookback=lb, buffer=config.ROTATION_BUFFER)
             for lb in LBS}
    sigret = {lb: (wtabs[lb].shift(1) * rets).sum(axis=1) for lb in LBS}

    configs = [
        ("窗口2年/季调", 504, 63),
        ("窗口2年/年调", 504, 252),
        ("窗口3年/季调", 756, 63),
    ]
    max_window = max(w for _, w, _ in configs)
    idx = closes.index
    eval_start = max_window  # 统一评估起点,保证可比
    print(f"\n共同日历 {idx[0].date()} ~ {idx[-1].date()},评估起点 {idx[eval_start].date()}"
          f"(留足最长选参窗口),本金 {CAPITAL:,.0f}\n")

    rows = []
    timelines = {}
    wf_weights_store = {}
    for label, window, step in configs:
        wf = pd.DataFrame(0.0, index=idx, columns=closes.columns)
        chosen = []
        i = eval_start
        while i < len(idx):
            j = min(i + step, len(idx))
            best = max(LBS, key=lambda lb: sharpe(sigret[lb].iloc[i - window:i]))
            wf.iloc[i:j] = wtabs[best].iloc[i:j].values
            chosen.append((idx[i].date(), best))
            i = j
        timelines[label] = chosen
        wf_weights_store[label] = wf
        result = run_portfolio_backtest(prices, wf, initial_capital=CAPITAL, stamp_tax=False)
        m = seg_metrics(result.equity.iloc[eval_start:])
        rows.append({"策略": f"WF {label}", "年化": f"{m['年化收益率']:.2%}",
                     "最大回撤": f"{m['最大回撤']:.2%}", "夏普": f"{m['夏普比率']:.2f}",
                     "交易次数": len(result.trades)})

    # 对照:固定 lookback=20,同一评估起点(起点前权重清零,同等条件起跑)
    fixed = wtabs[config.ROTATION_LOOKBACK].copy()
    fixed.iloc[:eval_start] = 0.0
    res_fixed = run_portfolio_backtest(prices, fixed, initial_capital=CAPITAL, stamp_tax=False)
    m = seg_metrics(res_fixed.equity.iloc[eval_start:])
    rows.append({"策略": f"固定 lookback={config.ROTATION_LOOKBACK}(现行)", "年化": f"{m['年化收益率']:.2%}",
                 "最大回撤": f"{m['最大回撤']:.2%}", "夏普": f"{m['夏普比率']:.2f}",
                 "交易次数": len(res_fixed.trades)})

    # 其余固定参数作参照(看 WF 是否至少不输"事后乱选"的固定参数)
    for lb in LBS:
        if lb == config.ROTATION_LOOKBACK:
            continue
        w = wtabs[lb].copy()
        w.iloc[:eval_start] = 0.0
        r = run_portfolio_backtest(prices, w, initial_capital=CAPITAL, stamp_tax=False)
        m = seg_metrics(r.equity.iloc[eval_start:])
        rows.append({"策略": f"固定 lookback={lb}", "年化": f"{m['年化收益率']:.2%}",
                     "最大回撤": f"{m['最大回撤']:.2%}", "夏普": f"{m['夏普比率']:.2f}",
                     "交易次数": len(r.trades)})

    print("========== 指标对比(同一评估区间、含费用)==========")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n========== 选参时间线 ==========")
    for label, chosen in timelines.items():
        # 压缩连续相同选择
        seq = []
        for d, lb in chosen:
            if not seq or seq[-1][1] != lb:
                seq.append((d, lb))
        print(f"{label}: " + " -> ".join(f"{d}起lb={lb}" for d, lb in seq))

    print("\n========== 与固定20的持仓重合度(评估区间)==========")
    fixed_hold = fixed.iloc[eval_start:].idxmax(axis=1).where(fixed.iloc[eval_start:].max(axis=1) > 0, "现金")
    for label in timelines:
        wf = wf_weights_store[label].iloc[eval_start:]
        wf_hold = wf.idxmax(axis=1).where(wf.max(axis=1) > 0, "现金")
        same = (wf_hold == fixed_hold).mean()
        print(f"{label}: 持仓相同天数占比 {same:.1%}")


if __name__ == "__main__":
    main()
