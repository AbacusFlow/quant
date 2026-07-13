"""昨收校验:系统前复权"上一交易日收盘" vs 独立行情源的真实昨收。

校验思路:系统数据(前复权)中上一交易日的收盘价,与独立行情源给出的同一天真实
(不复权)收盘逐一比对。一致=数据源正常;不一致=数据源异常或标的近期除权除息
(前复权历史被整体调整,属正常,人工确认即可)。

行情源按可靠性排序、逐源回退,任一可用即用("腾讯不行就试新浪"):
1. 新浪日K线 JSON(带日期标签,可精确取上一交易日,盘前同样可靠)—— 首选、权威
2. 腾讯实时昨收(qt.gtimg.cn 字段[4])—— 盘前(集合竞价前)可能尚未滚动到最近交易日
3. 新浪实时昨收(hq.sinajs.cn 字段[2])—— 同上

盘前陷阱:腾讯/新浪"实时昨收"在开盘前可能仍停在"上上交易日",其值会等于系统的
上上交易日收盘 —— 据此识别为"实时源盘前未滚动",降级为提示而非告警,避免每天盘前
误报(2026-06-18 早间安全网即踩此坑)。只有当权威带日期源、或实时源给出"既非上一日
也非上上一日"的值时,才判为真实异常并告警(⚠)。

只告警不阻断(永远 exit 0):含"⚠"的行被 CI 追加到 Telegram 推送中。
"""
import datetime as dt
import json
import re
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import config
import data
from daily_signal import data_end_date

TOL_MIN = 0.001   # 容差下限(元):三位小数行情的最后一位舍入
TOL_REL = 0.0005  # 相对容差 0.05%:池内价位跨两个数量级(159920 ~1.4元 vs 511260 ~135元),
                  # 绝对容差对高价 ETF 过严(0.0015 元只有 0.001%,正常舍入即误报)、
                  # 对低价 ETF 过松,须按价位缩放


def _tol(px: float) -> float:
    """按价位的比对容差:max(下限, 价格×0.05%)"""
    return max(TOL_MIN, abs(px) * TOL_REL)
SINA_HEADERS = {"Referer": "https://finance.sina.com.cn"}  # 新浪接口需 Referer 否则 403


def _tag(s: str) -> str:
    return ("sh" if s.startswith(("5", "6")) else "sz") + s


def fetch_sina_kline_prevclose(symbols, prev_trade) -> dict[str, float]:
    """新浪日K线(带日期标签,不复权):返回 {symbol: 上一交易日收盘}。

    按日期精确取值,无"实时昨收"的盘前滚动歧义;盘前/盘后/节假日均可靠。
    单标的取数失败只跳过该标的,不影响其它标的回退。
    """
    out: dict[str, float] = {}
    target = prev_trade.isoformat()
    for s in symbols:
        try:
            url = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                   f"CN_MarketData.getKLineData?symbol={_tag(s)}&scale=240&ma=no&datalen=6")
            r = requests.get(url, headers=SINA_HEADERS, timeout=10)
            r.raise_for_status()
            rows = json.loads(r.content.decode("gbk", errors="replace"))
            for row in rows:
                if str(row.get("day", ""))[:10] == target:
                    out[s] = float(row["close"])
                    break
        except Exception:
            continue
    return out


def fetch_tencent_yest(symbols) -> dict[str, float]:
    """腾讯实时昨收(qt.gtimg.cn 字段[4]);盘前可能尚未滚动。"""
    codes = ",".join(_tag(s) for s in symbols)
    r = requests.get(f"https://qt.gtimg.cn/q={codes}", timeout=10)
    r.raise_for_status()
    text = r.content.decode("gbk", errors="replace")
    out: dict[str, float] = {}
    for line in text.strip().splitlines():
        f = line.split("~")  # v_sh510300="1~名称~代码~现价~昨收~今开~...
        if len(f) > 4 and f[2]:
            try:
                out[f[2]] = float(f[4])
            except ValueError:
                pass
    return out


def fetch_sina_yest(symbols) -> dict[str, float]:
    """新浪实时昨收(hq.sinajs.cn 字段[2]);盘前可能尚未滚动。"""
    url = "https://hq.sinajs.cn/list=" + ",".join(_tag(s) for s in symbols)
    r = requests.get(url, headers=SINA_HEADERS, timeout=10)
    r.raise_for_status()
    text = r.content.decode("gbk", errors="replace")
    out: dict[str, float] = {}
    for line in text.strip().splitlines():
        # var hq_str_sh510300="名称,今开,昨收,现价,最高,...";
        m = re.match(r'var hq_str_(?:sh|sz)(\d{6})="([^"]*)"', line)
        if not m:
            continue
        parts = m.group(2).split(",")
        if len(parts) > 2 and parts[2]:
            try:
                out[m.group(1)] = float(parts[2])
            except ValueError:
                pass
    return out


def verdict(sys_prev: float, sys_pp: float | None,
            dated_val: float | None, realtime_vals: dict[str, float]):
    """单标的校验判定。返回 (level, detail):

    - ("ok", 源名)            : 与权威/实时源一致
    - ("mismatch", 对比串)    : 真实不一致(异常或除权),需告警
    - ("nodata", None)        : 所有源都没给昨收,无法校验,告警
    - ("skip", None)          : 实时源盘前未滚动昨收(全部等于上上交易日),非异常

    优先采信带日期的权威源(dated_val);无则回退实时源,并用"上上交易日"
    启发式区分"盘前未滚动"(skip)与"真实不一致"(mismatch)。
    """
    if dated_val is not None:
        if abs(sys_prev - dated_val) <= _tol(sys_prev):
            return "ok", "新浪日K线"
        return "mismatch", f"新浪日K线 {dated_val:.3f}"

    if not realtime_vals:
        return "nodata", None

    # 先把每个实时源分类:match(=上一交易日)/ stale(=上上交易日,盘前未滚动)/ diff(都不是)
    # 特例:若上一交易日与上上交易日同价(差<容差),实时源即使等于该价也无法证明
    # 它已滚动到正确日期(可能是盘前未滚动的陈值),按"未验证"处理,留给 skip,
    # 不计 match(否则会用一个无法区分新旧的值谎报已校验);带日期权威源存在时已提前返回,不受影响
    ambiguous = sys_pp is not None and abs(sys_prev - sys_pp) <= _tol(sys_prev)
    matched, diff = [], {}
    for src, v in realtime_vals.items():
        near_prev = abs(v - sys_prev) <= _tol(sys_prev)
        near_pp = sys_pp is not None and abs(v - sys_pp) <= _tol(sys_pp)
        if near_pp and ambiguous:
            continue  # 同价歧义:既不算 match 也不算 diff,留作 skip
        if near_prev:
            matched.append(src)
        elif near_pp:
            pass  # 盘前未滚动,忽略
        else:
            diff[src] = v

    # 优先级 mismatch > ok > skip:任一源给出"既非上一日也非上上一日"的值即告警,
    # 即使另一源恰好匹配(源间矛盾不可静默,宁可误报也不漏报)
    if diff:
        return "mismatch", ", ".join(f"{src} {v:.3f}" for src, v in diff.items())
    if matched:
        return "ok", "/".join(matched)
    # 全部等于上上交易日 → 实时源盘前尚未滚动昨收,非异常
    return "skip", None


def check() -> None:
    now = dt.datetime.now(ZoneInfo("Asia/Shanghai"))
    end = data_end_date(now)
    try:
        cal = data.get_trade_dates()
        prev_trade = cal[cal < pd.Timestamp(now.date())][-1].date()
    except Exception as e:
        print(f"(交易日历不可用,跳过校验: {e})")
        return
    if now.time() >= dt.time(9, 30) and pd.Timestamp(now.date()) in cal:
        # 交易时段开始后实时接口的"昨收"指今日的前一交易日,但系统数据可能已含
        # 今日(15:05 后),口径易混;校验设计在开盘前运行,盘中/盘后直接跳过
        print(f"(已过开盘时间 {now:%H:%M},昨收校验仅开盘前有意义,跳过)")
        return

    symbols = list(config.ETF_POOL)

    # 首选:带日期标签的新浪日K线(精确取上一交易日,盘前无滚动歧义)
    try:
        dated = fetch_sina_kline_prevclose(symbols, prev_trade)
    except Exception as e:
        dated = {}
        print(f"(新浪日K线源不可用: {e})")

    # 回退:两路实时昨收(盘前可能未滚动,verdict 用上上交易日启发式识别)
    realtime_sources: dict[str, dict[str, float]] = {}
    for name, fn in (("腾讯实时", fetch_tencent_yest), ("新浪实时", fetch_sina_yest)):
        try:
            d = fn(symbols)
            if d:
                realtime_sources[name] = d
        except Exception as e:
            print(f"({name}接口不可用: {e})")

    if not dated and not realtime_sources:
        # 三源全挂:无法校验数据正确性,必须告警(而非静默跳过)
        print("⚠ 所有行情源均不可用(新浪日K线/腾讯实时/新浪实时),本轮无法校验昨收")
        return
    if not dated:
        # 权威带日期源缺失,只能靠实时源+启发式,可靠性下降,提示但不阻断
        print("(注意:新浪日K线权威源不可用,本轮仅靠实时源校验,盘前可能多为暂跳过)")

    bad = 0
    skipped = 0
    for s, name in config.ETF_POOL.items():
        df = data.get_etf_daily(s, config.ROTATION_START, end.isoformat())
        sub = df.loc[:pd.Timestamp(prev_trade)]
        if sub.empty or sub.index[-1].date() != prev_trade:
            last = sub.index[-1].date() if not sub.empty else "无"
            print(f"⚠ {name}({s}): 系统缺少上一交易日 {prev_trade} 的K线(最新 {last},数据源滞后)")
            bad += 1
            continue
        sys_prev = float(sub["close"].iloc[-1])
        sys_pp = float(sub["close"].iloc[-2]) if len(sub) >= 2 else None
        realtime_vals = {src: d[s] for src, d in realtime_sources.items() if s in d}

        level, detail = verdict(sys_prev, sys_pp, dated.get(s), realtime_vals)
        if level == "ok":
            print(f"  {name}({s}): 昨收 {sys_prev:.3f} 校验一致({detail})")
        elif level == "mismatch":
            print(f"⚠ {name}({s}): 系统昨收 {sys_prev:.3f} ≠ {detail}"
                  f"(数据源异常或近期除权,请人工确认)")
            bad += 1
        elif level == "nodata":
            print(f"⚠ {name}({s}): 各行情源均未返回昨收,无法校验")
            bad += 1
        else:  # skip
            print(f"  {name}({s}): 实时源盘前昨收尚未滚动(=上上交易日),本轮跳过(开盘后自动一致)")
            skipped += 1

    ok = len(config.ETF_POOL) - bad - skipped
    if bad:
        print("昨收校验:发现异常,见上方 ⚠ 行")
    elif skipped and ok == 0:
        # 无任何标的完成校验(权威日K线源缺失 + 实时源盘前全未滚动)→ 告警提醒,本轮等于没校验
        print(f"⚠ 昨收校验:本轮 {skipped} 标的均未能校验"
              f"(权威日K线源缺失且实时源盘前未滚动),数据正确性未确认,请关注")
    elif skipped:
        print(f"昨收校验:{ok} 标的一致,{skipped} 标的本轮未能校验"
              f"(权威源缺失且实时源盘前未滚动,非异常,开盘后自动一致)")
    else:
        print("昨收校验:全部一致")


def main() -> int:
    try:
        check()
    except Exception as e:  # 顶层兜底:崩溃也要产出 ⚠ 行供 CI 推送,且不阻断流程
        print(f"⚠ 昨收校验脚本异常: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
