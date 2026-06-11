"""模拟盘每日信号:收盘后运行,输出最新目标持仓并记录到 output/signal_log.csv。

用法(每个交易日收盘后):
  python daily_signal.py

输出:
- 最新目标权重(多周期集成、不带回撤控制,与 run_rotation.py 默认回测口径一致)
- 与上次记录的信号对比,给出调仓指令(次日开盘执行)
- 信号历史追加到 output/signal_log.csv
"""
import datetime as dt
import os

import pandas as pd

import config
import data
from run_rotation import build_weights, closes_table

LOG_PATH = os.path.join(config.OUTPUT_DIR, "signal_log.csv")


def latest_weights(end: str) -> pd.Series:
    prices = {}
    for symbol, name in config.ETF_POOL.items():
        prices[symbol] = data.get_etf_daily(symbol, config.ROTATION_START, end)
    closes = closes_table(prices)
    weights = build_weights(
        closes, mode="ensemble", lookback=config.ROTATION_LOOKBACK,
        buffer=config.ROTATION_BUFFER, dd_control=False,
    )
    return weights.iloc[-1]


def describe(w: pd.Series) -> str:
    held = w[w > 0.005]
    if held.empty:
        return "空仓持现金"
    parts = [f"{config.ETF_POOL[s]}({s}) {v:.0%}" for s, v in held.items()]
    cash = 1 - held.sum()
    if cash > 0.005:
        parts.append(f"现金 {cash:.0%}")
    return ", ".join(parts)


def main():
    end = dt.date.today().isoformat()
    w = latest_weights(end)
    signal_date = w.name.date()

    print(f"\n========== 最新信号 (数据截至 {signal_date}) ==========")
    print(f"目标持仓: {describe(w)}")

    # 与上次记录对比
    prev = None
    if os.path.exists(LOG_PATH):
        log = pd.read_csv(LOG_PATH)
        if not log.empty:
            prev = log.iloc[-1]

    if prev is not None and str(prev["signal_date"]) == str(signal_date):
        print("(信号与日志中最新记录同一天,不重复记录)")
        return

    if prev is not None:
        print("\n--- 调仓指令(次日开盘执行)---")
        changed = False
        for s in config.ETF_POOL:
            old = float(prev.get(s, 0.0))
            new = float(w.get(s, 0.0))
            if abs(new - old) > 0.005:
                action = "买入" if new > old else "卖出"
                print(f"  {action} {config.ETF_POOL[s]}({s}): 目标权重 {old:.0%} -> {new:.0%}")
                changed = True
        if not changed:
            print("  无需调仓")

    # 追加日志
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    row = {"run_date": end, "signal_date": str(signal_date)}
    row.update({s: round(float(w.get(s, 0.0)), 4) for s in config.ETF_POOL})
    row["desc"] = describe(w)
    pd.DataFrame([row]).to_csv(
        LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH),
        index=False, encoding="utf-8-sig",
    )
    print(f"\n信号已记录: {LOG_PATH}")


if __name__ == "__main__":
    main()
