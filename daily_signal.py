"""模拟盘每日信号:每个交易日早上开盘前运行(基于前一交易日收盘数据),
输出最新目标持仓并记录到 output/signal_log.csv,调仓当日开盘执行。

用法(每个交易日开盘前;收盘后运行也可,但数据源当日日线通常夜间才发布):
  python daily_signal.py [--mode single|ensemble] [--capital 10000]

模式说明:
- single(默认):整仓切换,小资金(约1万)下单笔金额大,摊薄 5 元最低佣金
- ensemble:多周期集成,本金加到 10 万以上后切换,更抗参数过拟合

输出:
- 最新目标权重(不带回撤控制,与 run_rotation.py 对应模式回测口径一致)
- 与上次记录的信号对比,给出调仓指令(信号日的下一交易日开盘执行,早上运行即当日开盘)
- 信号历史追加到 output/signal_log.csv(含 mode 列,口径变更可追溯)
"""
import argparse
import datetime as dt
import math
import os
from zoneinfo import ZoneInfo

import pandas as pd

import config
import data
from run_rotation import build_weights, closes_table

LOG_PATH = os.path.join(config.OUTPUT_DIR, "signal_log.csv")
EXEC_PATH = os.path.join(config.OUTPUT_DIR, "executions.csv")
EXEC_COLS = ["date", "action", "symbol", "price", "shares", "amount", "note", "status"]


def data_end_date(now: dt.datetime) -> dt.date:
    """取数截止日:A股收盘前(上海时间 15:05 前)运行时,当日 K 线是盘中未完成数据,
    截止到昨天(取数阶段即排除,避免脏数据写入缓存);收盘后截止到今天。"""
    if now.time() < dt.time(15, 5):
        return now.date() - dt.timedelta(days=1)
    return now.date()


def latest_weights(end: str, mode: str) -> tuple[pd.Series, pd.Series]:
    """返回 (最新目标权重, 最新收盘价),用于信号与下单股数估算"""
    prices = {}
    for symbol, name in config.ETF_POOL.items():
        prices[symbol] = data.get_etf_daily(symbol, config.ROTATION_START, end)
    closes = closes_table(prices)
    closes = closes.loc[:end]  # 防御:缓存若混入 end 之后的脏数据也不参与信号
    weights = build_weights(
        closes, mode=mode, lookback=config.ROTATION_LOOKBACK,
        buffer=config.ROTATION_BUFFER, dd_control=False,
    )
    return weights.iloc[-1], closes.iloc[-1]


def describe(w: pd.Series) -> str:
    held = w[w > 0.005]
    if held.empty:
        return "空仓持现金"
    parts = [f"{config.ETF_POOL[s]}({s}) {v:.0%}" for s, v in held.items()]
    cash = 1 - held.sum()
    if cash > 0.005:
        parts.append(f"现金 {cash:.0%}")
    return ", ".join(parts)


def _load_executions_raw() -> pd.DataFrame:
    if not os.path.exists(EXEC_PATH):
        return pd.DataFrame(columns=EXEC_COLS)
    df = pd.read_csv(EXEC_PATH, dtype={"symbol": str}, encoding="utf-8-sig")
    if "status" not in df.columns:
        df["status"] = ""
    df["status"] = df["status"].fillna("").astype(str).str.strip()  # 与 report_web.py 口径一致
    return df.reindex(columns=EXEC_COLS)


def current_holdings(execs: pd.DataFrame) -> dict[str, int]:
    """已成交流水推算当前持仓(计划行不算)"""
    hold: dict[str, int] = {}
    for _, r in execs[execs["status"] != "计划"].iterrows():
        a = str(r["action"]).strip().lower()
        a = {"买入": "buy", "卖出": "sell"}.get(str(r["action"]).strip(), a)
        if a in ("buy", "sell"):
            s = str(r["symbol"]).strip()
            n = int(float(r["shares"]))
            hold[s] = hold.get(s, 0) + (n if a == "buy" else -n)
    return {s: n for s, n in hold.items() if n > 0}


def next_trade_date(after) -> dt.date:
    """信号日之后的第一个 A 股交易日;交易日历不可用时退化为下一工作日"""
    try:
        cal = data.get_trade_dates()
        nxt = cal[cal > pd.Timestamp(after)]
        if len(nxt):
            return nxt[0].date()
    except Exception:
        pass
    return (pd.Timestamp(after) + pd.tseries.offsets.BDay(1)).date()


def write_planned(signal_date, orders: list[tuple[str, str, int]], last_close: pd.Series,
                  exec_date: dt.date):
    """把执行日(信号日的下一交易日)调仓计划写入 executions.csv(status=计划),
    用户成交后改为已成交。

    - 价格为信号日收盘估算,股数为预估,实际以用户回填为准
    - 旧计划行(过期或被新信号取代)先全部清除,再写入本次计划,天然幂等
    """
    df = _load_executions_raw()
    stale = int((df["status"] == "计划").sum())
    df = df[df["status"] != "计划"]  # 清除所有旧计划(过期或被新信号取代)
    if not orders and not stale:
        return  # 无新计划也无旧计划,不动文件
    rows = []
    for action, symbol, shares in orders:
        rows.append({"date": str(exec_date), "action": action, "symbol": symbol,
                     "price": round(float(last_close[symbol]), 3), "shares": shares,
                     "amount": "", "note": f"按 {signal_date} 信号,价格为信号日收盘估算,成交后请改实际价格/股数(日期有出入也请修正)并将 status 改为 已成交",
                     "status": "计划"})
    out = pd.concat([df, pd.DataFrame(rows, columns=EXEC_COLS)], ignore_index=True)
    out["shares"] = pd.to_numeric(out["shares"], errors="coerce").astype("Int64")
    out.to_csv(EXEC_PATH, index=False, encoding="utf-8", float_format="%.10g")
    print(f"操作计划已写入: {EXEC_PATH}({len(rows)} 条,执行日 {exec_date})")


def main():
    parser = argparse.ArgumentParser(description="模拟盘每日信号")
    parser.add_argument("--mode", choices=("single", "ensemble"), default="single",
                        help="single=整仓切换(小资金默认), ensemble=多周期集成(本金≥10万)")
    parser.add_argument("--capital", type=float, default=None,
                        help="账户资金(元),提供后调仓指令附带预估下单股数(按昨收估算)")
    args = parser.parse_args()
    if args.capital is not None and (not math.isfinite(args.capital) or args.capital <= 0):
        parser.error("--capital 必须是正数")

    now = dt.datetime.now(ZoneInfo("Asia/Shanghai"))  # 统一用A股时区,避免本地/CI时区不一致
    today = now.date()
    try:
        # 非交易日(节假日/周末)跳过:避免假期早上推送"今日执行"、又因幂等
        # 挡掉真正开市日的推送;日历不可用时保持原行为(幂等仍兜底)
        if pd.Timestamp(today) not in data.get_trade_dates():
            print(f"{today} 非A股交易日,跳过信号记录与计划生成")
            return
    except Exception as e:
        print(f"(交易日历不可用,继续运行: {e})")

    end = data_end_date(now).isoformat()
    w, last_close = latest_weights(end, args.mode)
    signal_date = w.name.date()

    print(f"\n========== 最新信号 (mode={args.mode}, 数据截至 {signal_date}) ==========")
    print(f"目标持仓: {describe(w)}")

    # 读取历史日志(兼容无 mode 列的旧格式:旧脚本固定 ensemble 口径)
    log = pd.DataFrame()
    if os.path.exists(LOG_PATH):
        log = pd.read_csv(LOG_PATH, dtype={"signal_date": str}, encoding="utf-8-sig")
        if not log.empty and "mode" not in log.columns:
            log["mode"] = "ensemble"

    prev = log.iloc[-1] if not log.empty else None

    # 幂等:同一信号日期 + 同一模式在日志任意位置已存在则不重复记录
    if not log.empty and ((log["signal_date"] == str(signal_date)) & (log["mode"] == args.mode)).any():
        print("(同一信号日期、同一模式已记录,不重复记录)")
        return

    # 执行日 = 信号日的下一交易日;早上开盘前运行时即今天
    exec_date = next_trade_date(signal_date)
    day_word = "今日" if exec_date == today else str(exec_date)

    orders: list[tuple[str, str, int]] = []
    changed = False
    if prev is not None:
        print(f"\n--- 调仓指令({day_word}开盘 09:30 执行)---")
        holdings = current_holdings(_load_executions_raw())
        for s in config.ETF_POOL:
            old = float(prev.get(s, 0.0))
            new = float(w.get(s, 0.0))
            if abs(new - old) > 0.005:
                action = "买入" if new > old else "卖出"
                if action == "卖出":
                    # 卖出按已成交流水的实际持仓;部分减仓按权重比例折算,无持仓则不生成计划
                    held = holdings.get(s, 0)
                    if new <= 0.005:
                        est = held  # 清仓
                    else:
                        est = int(held * (old - new) / old // 100) * 100 if old > 0 else 0
                else:
                    est = int((args.capital or 0) * abs(new - old) / last_close[s] // 100) * 100
                line = f"  {action} {config.ETF_POOL[s]}({s}): 目标权重 {old:.0%} -> {new:.0%}"
                if est >= 100:
                    line += f",约 {est} 股(按昨收 {last_close[s]:.3f} 估算,以{day_word}开盘价为准)"
                    orders.append((action, s, est))
                elif args.capital:
                    line += f"(金额不足 1 手/100 股,可忽略)"
                print(line)
                changed = True
        # 持仓与目标对齐:即使信号未变,实际持仓和目标不一致也生成计划
        # (覆盖"清空流水重新开始"建仓、上次计划未确认成交等场景,计划行每日重写直到确认)
        for s in config.ETF_POOL:
            tgt = float(w.get(s, 0.0))
            held = holdings.get(s, 0)
            if tgt > 0.005 and held == 0:
                # 实际空仓时按完整目标权重建仓;差异循环生成的增量买入不足以对齐,予以替换
                est = int((args.capital or 0) * tgt / last_close[s] // 100) * 100
                prior = next((o for o in orders if o[0] == "买入" and o[1] == s), None)
                if prior:
                    orders.remove(prior)
                if est >= 100:
                    orders.append(("买入", s, est))
                    if prior is None or prior[2] != est:
                        print(f"  买入 {config.ETF_POOL[s]}({s}): 建仓至目标权重 {tgt:.0%},"
                              f"约 {est} 股(按昨收 {last_close[s]:.3f} 估算,以{day_word}开盘价为准)")
                        changed = True
            elif tgt <= 0.005 and held > 0 and not any(o[0] == "卖出" and o[1] == s for o in orders):
                orders.append(("卖出", s, held))
                print(f"  卖出 {config.ETF_POOL[s]}({s}): 目标权重 0,清仓 {held} 股")
                changed = True
        if not changed:
            print("  无需调仓")

    # 追加日志(整体重写,保证旧格式迁移后表头与数据列对齐)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    row = {"run_date": str(today), "signal_date": str(signal_date), "mode": args.mode}
    row.update({s: round(float(w.get(s, 0.0)), 4) for s in config.ETF_POOL})
    row["desc"] = describe(w)
    cols = ["run_date", "signal_date", "mode", *config.ETF_POOL, "desc"]
    out = pd.concat([log, pd.DataFrame([row])], ignore_index=True).reindex(columns=cols)
    out.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
    print(f"\n信号已记录: {LOG_PATH}")

    # 即使本次无可执行计划,也要清除已被新信号取代的旧计划行
    write_planned(signal_date, orders, last_close, exec_date)


if __name__ == "__main__":
    main()
