"""portfolio_status.py — 持仓偏离目标 + 账户简况(供每日 Telegram 消息展示,不产生告警)。

口径与 report_web/check_risk 严格一致:load_pool→closes→build_weights(线上 flag)→
real_equity_series(份额化净值/净入金/回撤)+ replay_positions(持仓/现金)。

输出一个可直接嵌进消息的文本块:
    持仓偏离目标(>5pp):
    • 510500 超配 +19pp → 卖约 4000 股
    • 511260 欠配 -19pp → 买约 300 股
    账户 184,347 元 | 盈亏 +490(+0.3%) | 回撤 4.2%

任何异常/数据未就绪 → 打印空串(消息里就不带该块),绝不中断每日流程。
数据拉取的「拉取 …」日志被 redirect 吞掉,只输出这一块。
"""
import argparse
import contextlib
import datetime as dt
import io
import json
import math
import os
import sys
import traceback
from zoneinfo import ZoneInfo

import pandas as pd

import config
import data
from daily_signal import data_end_date
from report_web import (load_executions, overnight_held_symbols, real_equity_series,
                        replay_positions)
from run_rotation import build_weights, closes_table, load_pool

DEV_THRESHOLD = 0.05  # 偏离绝对值 >5pp 才在消息里单列(2% 内本就可忽略,5% 才提示手动)


def _round_lot(x: float) -> int:
    """四舍五入到 100 股整手,对称(远离零方向取整),保证建议必为整手。"""
    sign = 1 if x >= 0 else -1
    return sign * int(math.floor(abs(x) / 100 + 0.5) * 100)


def _finite(x) -> bool:
    try:
        return bool(pd.notna(x)) and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _apply_mx_fallback(prices: dict, path: str) -> pd.Timestamp | None:
    """收盘后档专用:把妙想(mx_data)当日收盘**追加为最新一根 bar**,in-memory 补齐同日行情。

    仅用于展示档(账户小结/持仓偏离);money 信号路径永不调用。硬约束:
    - 只追加 `end` 当日一根 bar(o/h/l=close,volume=0),`end` 必须严格晚于现有最新 bar;
    - 绝不写磁盘缓存(防「名新实旧」堵住 akshare 后续权威拉取);
    - 只补池内已加载的标的;JSON 缺失/损坏/字段缺 → 静默忽略,优雅降级回 akshare。
    部分填充自然降级:若非全部标的补到 end,closes_table 的 align_prices 交集会截回上一交易日;
    调用方据返回的 mx end 与实际最新交易日比对,不一致则弃用(避免用旧价冒充 mx 日期)。

    返回:成功解析并尝试追加时返回 mx 的 end(pd.Timestamp);无 JSON/损坏/字段缺返回 None。
    """
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        end = payload.get("end")
        closes = payload.get("closes")
        if not end or not isinstance(closes, dict):
            return None
        ts = pd.Timestamp(end)
        if pd.isna(ts):
            return None
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    for sym, df in prices.items():
        if sym not in closes:
            continue
        try:
            px = float(closes[sym])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(px) or px <= 0:
            continue
        if df is None or df.empty or ts <= df.index[-1]:
            continue  # 只在严格更新时追加;akshare 已出当日 bar 则不动
        row = pd.DataFrame(
            {"open": [px], "high": [px], "low": [px], "close": [px], "volume": [0.0]},
            index=pd.DatetimeIndex([ts], name=df.index.name),
        )
        prices[sym] = pd.concat([df, row])
    return ts


def build_status_block(mode: str, vol_control: bool, sleeve: bool,
                       dev_threshold: float = DEV_THRESHOLD,
                       include_dev: bool = True,
                       mx_fallback_path: str | None = None) -> str:
    execs = load_executions()
    confirmed = execs[execs["status"] != "计划"] if execs is not None else pd.DataFrame()
    if confirmed.empty:
        return ""

    now = dt.datetime.now(ZoneInfo("Asia/Shanghai"))
    end = data_end_date(now).isoformat()

    # 数据拉取会往 stdout 打「拉取 …」;吞掉,只留最终文本块
    with contextlib.redirect_stdout(io.StringIO()):
        # 纯展示路径:write_cache=False,绝不落盘(end=today 但数据源只到昨天时,写
        # `{name}_..._{today}.csv` 会以文件名命中堵住后续 akshare 权威拉取——见 data.py)
        prices = load_pool(config.ROTATION_START, end, write_cache=False)
        # 收盘后档:若有妙想当日收盘 JSON,追加为最新一根 bar(in-memory,不落盘)
        mx_end = _apply_mx_fallback(prices, mx_fallback_path)
        closes = closes_table(prices).loc[:end]
        if closes.empty or confirmed["date"].max() > closes.index[-1]:
            return ""  # 行情未更新(晚于最新收盘)则不展示,避免错配
        # mx 部分覆盖时 align_prices 交集会把实际日期截回上一交易日,但调用方(postclose)
        # 仍以 mx 的 end 作 KEY/回执 → 会用旧价冒充 mx 日期。故要求实际最新日==mx end 才展示。
        if mx_end is not None and closes.index[-1] != mx_end:
            return ""
        weights = build_weights(closes, mode=mode, lookback=config.ROTATION_LOOKBACK,
                                buffer=config.ROTATION_BUFFER, dd_control=False,
                                vol_control=vol_control, sleeve=sleeve)
        pos, cash = replay_positions(confirmed)
        held = {s for s, sh in pos.items() if sh}  # 当前仍持有(非零)的标的
        # 池外标的(如误买错代码后的持有)按需补收盘价,口径同 check_risk:
        # 只补「曾隔夜持有」的标的(当日买卖光的纠错标的日终恒 0 不参与计价)
        extra = sorted(overnight_held_symbols(confirmed, closes.index) - set(closes.columns))
        if extra:
            closes = closes.copy()
            for s in extra:
                px = data.get_etf_daily(s, config.ROTATION_START, end, write_cache=False)["close"]
                # mx 兜底日:仍持有的池外标的必须有 mx_end 当日**真实**收盘,否则 ffill 会用旧价
                # 冒充 mx 日期(账户市值/盈亏含旧价却标 mx 日)→ 弃用整块。已清仓的池外历史标的
                # (pos=0,如误买后卖光的 513300)不参与当前计价,不触发弃用,避免误伤正常展示。
                if mx_end is not None and s in held and mx_end not in px.index:
                    return ""
                closes[s] = px.reindex(closes.index).ffill()
        equity, navs, net_deposit = real_equity_series(confirmed, closes)

    last = closes.iloc[-1]
    total = float(equity.iloc[-1])
    if not _finite(total) or total <= 0:
        return ""
    cur_dd = 1 - float(navs.iloc[-1]) / float(navs.max())
    abs_pnl = total - net_deposit
    target = weights.iloc[-1]

    symbols = sorted(
        set(pos) | {s for s, w in target.items() if _finite(w) and float(w) > 0.005},
        key=lambda s: -(float(target[s]) if s in target.index and _finite(target[s]) else 0.0),
    )
    lines = []
    if include_dev:
        dev_lines = []
        for s in symbols:
            px = float(last[s]) if s in last.index and _finite(last[s]) else float("nan")
            if not _finite(px) or px <= 0:
                continue
            shares = pos.get(s, 0)
            actual = shares * px / total
            tgt = float(target[s]) if s in target.index and _finite(target[s]) else 0.0
            dev = actual - tgt
            if abs(dev) <= dev_threshold:  # 恰好 5pp 不列(与「>5pp」文案一致)
                continue
            delta = _round_lot(tgt * total / px - shares)
            if delta < 0:
                delta = max(delta, -shares)  # 卖出不得超过实际持仓(允许非整百的卖光)
            word = "超配" if dev > 0 else "欠配"
            act = f"买约 {delta:,} 股" if delta > 0 else (f"卖约 {-delta:,} 股" if delta < 0 else "基本到位")
            dev_lines.append(f"• {s} {word} {dev * 100:+.0f}pp → {act}")
        if dev_lines:
            lines.append("持仓偏离目标(>5pp):")
            lines.extend(dev_lines)
        else:
            lines.append("持仓与目标基本一致(偏离均≤5pp)")
    ret_s = f"{abs_pnl / net_deposit:+.1%}" if net_deposit > 0 else "—"
    lines.append(f"账户 {total:,.0f} 元 | 盈亏 {abs_pnl:+,.0f}({ret_s}) | 回撤 {cur_dd:.1%}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="持仓偏离目标 + 账户简况(供每日消息展示,不告警)")
    p.add_argument("--mode", choices=("single", "ensemble"), default="single")
    p.add_argument("--capital", type=float, default=10000, help="接口对齐用,本脚本按真实流水计算,不使用")
    p.add_argument("--vol-target", action=argparse.BooleanOptionalAction,
                   default=config.VOL_TARGET_ENABLED, help="须与线上策略同口径")
    p.add_argument("--sleeve", action=argparse.BooleanOptionalAction,
                   default=config.SLEEVE_ENABLED, help="须与线上策略同口径")
    p.add_argument("--account-only", action="store_true",
                   help="只输出账户简况行,不列持仓偏离(调仓日用,避免与调仓指令重复)")
    p.add_argument("--mx-fallback", default=None,
                   help="收盘后档:妙想当日收盘 JSON 路径,追加为最新一根 bar(in-memory,不落盘)")
    args = p.parse_args()
    try:
        block = build_status_block(args.mode, vol_control=args.vol_target, sleeve=args.sleeve,
                                   include_dev=not args.account_only,
                                   mx_fallback_path=args.mx_fallback)
    except Exception as e:
        # 每日主流程不因本块出错而中断:仍返回空块(stdout),但把异常写 stderr 供 cron 日志留痕,
        # 避免数据口径错误/CSV 损坏/代码回归长期无声消失(daily_local.sh 会把 stderr 收进 cron_signal.log)
        print(f"portfolio_status 生成失败(已跳过,不影响主流程): {e}", file=sys.stderr)
        traceback.print_exc()
        block = ""
    if block:
        print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
