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
import contextlib
import datetime as dt
import fcntl
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
LEDGER_LOCK = os.path.join(config.OUTPUT_DIR, ".ledger.lock")


@contextlib.contextmanager
def ledger_lock():
    """账本互斥锁:与 record_trade.py 串行,防并发「读→改→整表重写」丢已成交行。

    锁次序约定:.daily_local.lock(shell 编排层)→ .ledger.lock(账本层)。
    本脚本只取账本层锁——daily_local.sh 调用本脚本时父 shell 已持编排锁,
    子进程再取同一把会死锁,故用独立的第二把;record_trade 两把都按序取。
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    fd = os.open(LEDGER_LOCK, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def account_state(last_close: pd.Series) -> tuple[float, float] | None:
    """真实账户状态:已成交流水回放持仓/现金 + 最新收盘计价,返回 (总资产, 现金)。

    用于替代静态 --capital 估算下单股数(资产随行情波动,.env 快照会系统性偏差)。

    失败语义(fail-closed):账本**完整性异常**(CSV 解析失败/负现金/负持仓等
    ValueError)一律向上传播 → 信号非零退出 → shell 推「信号脚本运行失败」告警。
    账本已不可信时绝不静默降级继续按过期本金生成真实调仓计划。
    仅以下明确可降级情形返回 None(回退 --capital):
    - 账本文件不存在 / 尚无已成交行(从零开始,估算基准本就只有 --capital);
    - 某持仓标的最新收盘无价(池外持仓,暂时无法计价,不算账本损坏)。

    合法的零资产(如全部出金后)返回 0.0 而非 None——空账户绝不能回退
    --capital 生成不存在资金的建仓买单。
    """
    from report_web import load_executions, replay_positions
    execs = load_executions()  # 解析/校验异常 → 传播,不吞
    if execs is None:
        return None
    confirmed = execs[execs["status"] != "计划"]
    if confirmed.empty:
        return None
    pos, cash = replay_positions(confirmed)  # 负现金/负持仓 → 传播,不吞
    total = cash
    for s, n in pos.items():
        if not n:
            continue
        px = last_close.get(s)
        if px is None or not math.isfinite(float(px)) or float(px) <= 0:
            return None
        total += n * float(px)
    if not math.isfinite(total) or total < -0.01:
        # 非有限/显著为负的总资产 = 账本或行情数据异常,fail-closed:抛错而非回退 --capital
        raise ValueError(f"账本回放总资产异常({total!r}),请检查流水与行情")
    # 现金浮点残差(如 -2e-13)夹到 0,合法零资产如实返回
    return max(total, 0.0), max(cash, 0.0)


def data_end_date(now: dt.datetime) -> dt.date:
    """取数截止日:A股收盘前(上海时间 15:05 前)运行时,当日 K 线是盘中未完成数据,
    截止到昨天(取数阶段即排除,避免脏数据写入缓存);收盘后截止到今天。"""
    if now.time() < dt.time(15, 5):
        return now.date() - dt.timedelta(days=1)
    return now.date()


def latest_weights(end: str, mode: str, vol_control: bool = False,
                   sleeve: bool = False) -> tuple[pd.Series, pd.Series]:
    """返回 (最新目标权重, 最新收盘价),用于信号与下单股数估算"""
    prices = {}
    for symbol, name in config.ETF_POOL.items():
        # money 信号路径:qfq_only 拒绝新浪不复权回退(宁可当日无信号,幂等次日补)
        prices[symbol] = data.get_etf_daily(symbol, config.ROTATION_START, end, qfq_only=True)
    closes = closes_table(prices)
    closes = closes.loc[:end]  # 防御:缓存若混入 end 之后的脏数据也不参与信号
    weights = build_weights(
        closes, mode=mode, lookback=config.ROTATION_LOOKBACK,
        buffer=config.ROTATION_BUFFER, dd_control=False, vol_control=vol_control,
        sleeve=sleeve,
    )
    return weights.iloc[-1], closes.iloc[-1]


PLAN_MARGIN = 0.01  # 现金封顶的隔夜跳空/滑点安全垫:卖出回款按 -1%、买入成本按 +1% 估


def _fee(amount: float) -> float:
    """佣金口径与账本/回测引擎一致(report_web._apply_exec_row / portfolio._commission)"""
    return max(amount * config.ETF_COMMISSION_RATE, config.COMMISSION_MIN)


def plan_orders(target_w: pd.Series, holdings: dict[str, int], last_close: pd.Series,
                equity: float | None, cash: float | None,
                band: float = config.REBALANCE_BAND,
                margin: float = PLAN_MARGIN) -> tuple[list[tuple[str, str, int]], list[str]]:
    """按回测引擎口径规划调仓订单:目标股数 desired=⌊equity×目标权重/px⌋整手,
    与**实际持仓**之差生成订单(先卖后买)。返回 (orders, notes)。

    对比旧「前后信号权重差」方案的关键差异:部分成交/现金不足导致的仓位漂移会被
    直接对齐(与 portfolio.py 引擎 desired-vs-shares 同构),不再依赖 prev 信号
    准确代表实际仓位;加仓后欠配的已持仓标的也会生成补仓单(带宽防漂移刷屏)。

    规则:
    - 带宽:|desired-held|×px < band×equity 不交易(费用拖累);目标权重 ≤0.005 的
      真清仓、池外持仓清仓不受带宽限制(与引擎的不对称行为一致)
    - 现金封顶(cash 非 None 时):avail = cash + Σ[卖出估值×(1−margin) − 佣金],
      买入按 px×(1+margin)+佣金 依次装入,超出则整手缩量、不足一手放弃;
      margin 为隔夜跳空/滑点安全垫,佣金用真实公式(万0.5 最低5元)
    - equity 为 None(账本与 --capital 均不可用)→ 只生成清仓类订单
    - notes 为诊断说明行,措辞不含「买入/卖出」(daily_local.sh 用该关键词 grep 指令行)
    """
    orders_sell: list[tuple[str, str, int]] = []
    buy_cands: list[tuple[str, int]] = []
    notes: list[str] = []

    def px_of(s):
        v = last_close.get(s)
        return float(v) if s in last_close.index and pd.notna(v) else None

    band_value = band * equity if equity else 0.0
    # 池外持仓(目标权重表无此列)一律清仓,不受带宽限制、无需价格
    for s, held in holdings.items():
        if held > 0 and s not in target_w.index:
            orders_sell.append(("卖出", s, held))
    for s in target_w.index:
        tgt = float(target_w[s]) if pd.notna(target_w[s]) else 0.0
        held = holdings.get(s, 0)
        if tgt <= 0.005:
            if held > 0:
                orders_sell.append(("卖出", s, held))  # 真清仓:不受带宽限制
            continue
        px = px_of(s)
        if px is None or px <= 0 or equity is None:
            if held == 0 and equity is not None:
                notes.append(f"({s} 无有效昨收,暂无法估算建仓股数)")
            continue
        desired = int(equity * tgt / px // 100) * 100
        diff = desired - held
        if diff == 0:
            continue
        if abs(diff) * px < band_value:
            if abs(diff) >= 100:
                notes.append(f"({s} 与目标差约 {abs(diff)} 股(≈{abs(diff) * px:,.0f} 元),"
                             f"低于再平衡带宽 {band_value:,.0f} 元,不动)")
            continue
        if diff < 0:
            orders_sell.append(("卖出", s, -diff))
        elif diff >= 100:
            buy_cands.append((s, diff))

    if cash is not None:
        # 现金封顶:卖出回款按下浮价扣佣金、买入按上浮价加佣金,留隔夜跳空余地。
        # 预算分配与引擎同口径(portfolio.py):按各标的待买金额**比例**切分现金快照,
        # 佣金计入各自预算——分配结果与标的顺序无关,不会让排前的标的挤占排后的资金
        avail = cash
        for _, s, n in orders_sell:
            px = px_of(s)
            if px:
                gross = n * px * (1 - margin)
                avail += gross - _fee(gross)
        total_buy_value = sum(n * float(last_close[s]) * (1 + margin) for s, n in buy_cands)
        remaining = avail
        capped: list[tuple[str, int]] = []
        for s, est in buy_cands:
            px = float(last_close[s]) * (1 + margin)

            def cost(n: int) -> float:
                return n * px + _fee(n * px)

            budget = avail * (est * px / total_buy_value) if total_buy_value > 0 else 0.0
            fit = min(est, int(budget / (px * (1 + config.ETF_COMMISSION_RATE)) // 100) * 100)
            while fit > 0 and (cost(fit) > budget or cost(fit) > remaining):
                fit -= 100
            if fit < 100:
                notes.append(f"(资金不足,略过 {s} 补仓 {est} 股;待现金回笼后由后续信号重算)")
                continue
            if fit < est:
                notes.append(f"({s} 补仓按可用资金调整:{est} → {fit} 股)")
            remaining -= cost(fit)
            capped.append((s, fit))
        buy_cands = capped

    return orders_sell + [("买入", s, n) for s, n in buy_cands], notes


def describe(w: pd.Series) -> str:
    held = w[w > 0.005]
    if held.empty:
        return "空仓持现金"
    parts = [f"{s} {config.ETF_POOL[s]} {v:.0%}" for s, v in held.items()]
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
        px = last_close.get(symbol)  # 池外标的(误买清仓)无系统价格,留空
        rows.append({"date": str(exec_date), "action": action, "symbol": symbol,
                     "price": round(float(px), 3) if px is not None else "", "shares": shares,
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
    parser.add_argument("--vol-target", action=argparse.BooleanOptionalAction,
                        default=config.VOL_TARGET_ENABLED,
                        help="波动率目标覆盖层(默认随 config.VOL_TARGET_ENABLED);"
                             "影子用法:--vol-target 看开启后的调仓指令")
    parser.add_argument("--sleeve", action=argparse.BooleanOptionalAction,
                        default=config.SLEEVE_ENABLED,
                        help="防御 sleeve:残余现金金债各半(默认随 config.SLEEVE_ENABLED)")
    args = parser.parse_args()
    if args.capital is not None and (not math.isfinite(args.capital) or args.capital <= 0):
        parser.error("--capital 必须是正数")

    now = dt.datetime.now(ZoneInfo("Asia/Shanghai"))  # 统一用A股时区,避免本地/CI时区不一致
    today = now.date()

    end = data_end_date(now).isoformat()
    w, last_close = latest_weights(end, args.mode, vol_control=args.vol_target,
                                   sleeve=args.sleeve)
    signal_date = w.name.date()

    vt_word = "开" if args.vol_target else "关"
    sl_word = "开" if args.sleeve else "关"
    print(f"\n========== 最新信号 (mode={args.mode}, 波动率目标={vt_word}, 防御sleeve={sl_word}, 数据截至 {signal_date}) ==========")
    print(f"目标持仓: {describe(w)}")

    # 账本/信号日志互斥:signal_log 与 executions 的「读→判→整表重写」全程持锁,
    # 与 record_trade、以及另一手动 daily_signal 实例串行——读在锁外会拿到旧快照,
    # 后写者覆盖丢掉先写者刚追加的行(账本/信号日志都是不可再生数据)
    with ledger_lock():
        # 读取历史日志(兼容无 mode 列的旧格式:旧脚本固定 ensemble 口径)
        log = pd.DataFrame()
        if os.path.exists(LOG_PATH):
            log = pd.read_csv(LOG_PATH, dtype={"signal_date": str}, encoding="utf-8-sig")
            if not log.empty and "mode" not in log.columns:
                log["mode"] = "ensemble"

        prev = log.iloc[-1] if not log.empty else None

        # 幂等:同一信号日期 + 同一模式在日志任意位置已存在则不重复记录信号,
        # 但计划生成仍要跑(流水可能在两次运行之间被修正,如误买后改持仓)
        already = (not log.empty
                   and ((log["signal_date"] == str(signal_date)) & (log["mode"] == args.mode)).any())

        # 跳过逻辑:仅当"今天非交易日 且 该信号已记录"时跳过(无新信号可恢复)。
        # 用数据驱动的 signal_date 判断,而非运行时 today:定时任务可能迟到数小时,
        # 交易日晚间任务可能被推迟到次日(周末/假期凌晨)才跑,此时 today 虽非交易日,
        # 但上一交易日的信号正是次日开盘前要送达的,必须补出(否则周一/节后开盘漏信号);
        # 同时避免真正的周末/假期空跑重复推送(已记录则跳过)。日历不可用时不跳过,交给幂等兜底。
        try:
            trading_today = pd.Timestamp(today) in data.get_trade_dates()
        except Exception as e:
            print(f"(交易日历不可用,继续运行: {e})")
            trading_today = True
        if not trading_today and already:
            print(f"{today} 非A股交易日,且最新信号({signal_date})已记录,跳过")
            return

        # 执行日 = 信号日的下一交易日;早上开盘前运行时即今天
        exec_date = next_trade_date(signal_date)
        day_word = "今日" if exec_date == today else str(exec_date)

        # 下单股数估算基准:优先真实账户资产+现金(流水回放+最新收盘计价),
        # 静态 --capital 是手工快照,行情波动后估算股数会系统性偏差;回退 --capital
        state = account_state(last_close)
        if state is not None:
            est_capital, avail_cash = state
            print(f"(股数估算基准:实际账户资产 {est_capital:,.0f} 元,其中现金 {avail_cash:,.2f} 元)")
        else:
            est_capital, avail_cash = args.capital, None

        orders: list[tuple[str, str, int]] = []
        if prev is not None:
            print(f"\n--- 调仓指令({day_word}开盘 09:30 执行,先卖后买)---")
            holdings = current_holdings(_load_executions_raw())
            # 按引擎口径规划:目标股数 vs 实际持仓(部分成交/漂移自动对齐),
            # 带 REBALANCE_BAND 带宽与现金封顶(详见 plan_orders)
            orders, notes = plan_orders(w, holdings, last_close,
                                        equity=est_capital, cash=avail_cash)
            for line in notes:
                print(f"  {line}")
            for action, s, est in orders:
                name = config.ETF_POOL.get(s, "(池外)")
                tgt = float(w.get(s, 0.0)) if s in w.index else 0.0
                px = last_close.get(s)
                px_s = (f"(按昨收 {float(px):.3f} 估算,以{day_word}开盘价为准)"
                        if s in last_close.index and pd.notna(px) else "")
                print(f"  {action} {s}({name}): 约 {est} 股,对齐目标权重 {tgt:.0%}{px_s}")
            if not orders:
                print("  无需调仓")

        if already:
            print("\n(同一信号日期、同一模式已记录,不重复记录)")
        else:
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
