"""回撤事件触发告警:实盘净值回撤触线当天就推送,不必等月度/半年定时检查。

每个交易日 CI 跑完信号后运行(与 reminders.yml 的定时提醒互补):
- 实盘回撤 ≥ 回测最大回撤        → ⚠ 已达回测极限水平(正常区间边缘,密切关注)
- 实盘回撤 ≥ 回测最大回撤 × 1.5  → ⚠ 暂停加仓 + 深度复盘(2026-06 约定的事件 trigger)
回测基线与 report_web 同口径(默认 single 模式、无回撤控制、全区间)。

只告警不阻断:任何异常都打印 ⚠ 行并 exit 0,由 CI 解析 ⚠ 决定是否推 Telegram。
"""
import argparse
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

import config
import data
import metrics as metrics_mod
from daily_signal import data_end_date
from portfolio import run_portfolio_backtest
from report_web import load_executions, real_equity_series
from run_rotation import build_weights, closes_table, load_pool

DD_ALERT_MULT = 1.5  # 实盘回撤超过回测最大回撤的 1.5 倍 → 暂停加仓、深度复盘


def check(mode: str, capital: float, vol_control: bool = False,
          sleeve: bool = False) -> None:
    execs = load_executions()
    confirmed = execs[execs["status"] != "计划"] if execs is not None else pd.DataFrame()
    if confirmed.empty:
        print("尚无实盘已成交记录,跳过回撤检查")
        return

    now = dt.datetime.now(ZoneInfo("Asia/Shanghai"))
    end = data_end_date(now).isoformat()
    # 读侧路径:write_cache=False 绝不落盘(缓存由 daily_signal 权威写入);
    # qfq_only=True 拒绝新浪不复权回退——回撤告警基线须与信号同复权口径
    prices = load_pool(config.ROTATION_START, end, write_cache=False, qfq_only=True)
    closes = closes_table(prices).loc[:end]
    if confirmed["date"].max() > closes.index[-1]:
        # 盘中手动运行时今日成交尚无收盘行情;CI 开盘前运行不会出现,不算告警
        print(f"流水含 {confirmed['date'].max().date()} 成交而最新行情为 "
              f"{closes.index[-1].date()},行情未更新,跳过回撤检查")
        return

    # 回测基线最大回撤(与 report_web/线上同口径:无回撤控制,波动率目标随线上开关)
    weights = build_weights(closes, mode=mode, lookback=config.ROTATION_LOOKBACK,
                            buffer=config.ROTATION_BUFFER, dd_control=False,
                            vol_control=vol_control, sleeve=sleeve)
    bt_equity = run_portfolio_backtest(prices, weights, initial_capital=capital,
                                       stamp_tax=False).equity
    bt_maxdd = -float(metrics_mod.equity_metrics(bt_equity)["最大回撤"])  # 转为正数

    # 实盘份额化净值(口径同 report_web「实盘 vs 模拟」);池外标的按需补收盘价
    extra = sorted(set(confirmed.loc[confirmed["action"].isin(("buy", "sell")), "symbol"])
                   - set(closes.columns))
    if extra:
        closes = closes.copy()
        for s in extra:
            px = data.get_etf_daily(s, config.ROTATION_START, end,
                                    write_cache=False, qfq_only=True)["close"]
            closes[s] = px.reindex(closes.index).ffill()
    _, navs, _ = real_equity_series(confirmed, closes)
    cur_dd = 1 - float(navs.iloc[-1]) / float(navs.max())

    print(f"实盘当前回撤 {cur_dd:.1%}(净值 {navs.iloc[-1]:.4f} / 峰值 {navs.max():.4f}),"
          f"回测最大回撤 {bt_maxdd:.1%},告警线 {DD_ALERT_MULT * bt_maxdd:.1%}")
    if bt_maxdd <= 0:
        print("⚠ 回测最大回撤异常(<=0),无法比较,请检查回测数据")
    elif cur_dd >= DD_ALERT_MULT * bt_maxdd:
        print(f"⚠ 实盘回撤 {cur_dd:.1%} 已超过回测最大回撤的 {DD_ALERT_MULT} 倍"
              f"({DD_ALERT_MULT * bt_maxdd:.1%}):按约定 暂停加仓,启动深度复盘。"
              f"先查执行环节与数据,不要情绪化改策略")
    elif cur_dd >= bt_maxdd:
        print(f"⚠ 实盘回撤 {cur_dd:.1%} 已达到回测最大回撤水平({bt_maxdd:.1%}):"
              f"仍在历史发生过的范围内,密切关注;超过 {DD_ALERT_MULT * bt_maxdd:.1%} 才触发暂停加仓")
    else:
        print("回撤在正常范围内")


def main() -> int:
    parser = argparse.ArgumentParser(description="实盘回撤事件检查(只告警不阻断)")
    parser.add_argument("--mode", choices=("single", "ensemble"), default="single")
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--vol-target", action=argparse.BooleanOptionalAction,
                        default=config.VOL_TARGET_ENABLED,
                        help="波动率目标覆盖层(默认随 config.VOL_TARGET_ENABLED),"
                             "回撤告警基线须与线上策略同口径")
    parser.add_argument("--sleeve", action=argparse.BooleanOptionalAction,
                        default=config.SLEEVE_ENABLED,
                        help="防御 sleeve(默认随 config.SLEEVE_ENABLED),基线须与线上同口径")
    args = parser.parse_args()
    try:
        check(args.mode, args.capital, vol_control=args.vol_target, sleeve=args.sleeve)
    except Exception as e:
        print(f"⚠ 回撤检查脚本异常: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
