"""昨收校验:系统前复权"上一交易日收盘" vs 腾讯实时接口的真实昨收。

实时接口的"昨收"在任意时刻都指上一交易日收盘价(盘中/收盘后/节假日均如此),
与系统数据中上一交易日的收盘逐一比对:一致说明数据源正常;不一致说明数据源
异常或标的近期除权除息(前复权历史被整体调整,属正常,人工确认即可)。
输出含"⚠"的行会被 CI 追加到 Telegram 推送中告警;只告警不阻断流程(永远 exit 0)。
"""
import datetime as dt
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import config
import data
from daily_signal import data_end_date


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

    def prefix(s: str) -> str:
        return "sh" if s.startswith(("5", "6")) else "sz"

    codes = ",".join(f"{prefix(s)}{s}" for s in config.ETF_POOL)
    try:
        resp = requests.get(f"https://qt.gtimg.cn/q={codes}", timeout=10)
        resp.raise_for_status()
        text = resp.content.decode("gbk")
    except Exception as e:
        print(f"(实时接口不可用,跳过校验: {e})")
        return

    realtime = {}  # symbol -> 真实昨收
    for line in text.strip().splitlines():
        # 行格式: v_sh513100="1~纳指ETF~513100~现价~昨收~今开~...
        fields = line.split("~")
        if len(fields) > 4:
            realtime[fields[2]] = float(fields[4])

    bad = 0
    for s, name in config.ETF_POOL.items():
        df = data.get_etf_daily(s, config.ROTATION_START, end.isoformat())
        sub = df.loc[:pd.Timestamp(prev_trade)]
        rt = realtime.get(s)
        if sub.empty or sub.index[-1].date() != prev_trade:
            last = sub.index[-1].date() if not sub.empty else "无"
            print(f"⚠ {name}({s}): 系统缺少上一交易日 {prev_trade} 的K线(最新 {last},数据源滞后)")
            bad += 1
        elif rt is None:
            print(f"⚠ {name}({s}): 实时接口未返回昨收,无法校验")
            bad += 1
        elif abs(float(sub["close"].iloc[-1]) - rt) > 0.0015:
            print(f"⚠ {name}({s}): 系统昨收 {float(sub['close'].iloc[-1]):.3f} ≠ 真实昨收 {rt:.3f}"
                  f"(数据源异常或近期除权,请人工确认)")
            bad += 1
        else:
            print(f"  {name}({s}): 昨收 {float(sub['close'].iloc[-1]):.3f} 校验一致")
    print("昨收校验:发现异常,见上方 ⚠ 行" if bad else "昨收校验:全部一致")


def main() -> int:
    try:
        check()
    except Exception as e:  # 顶层兜底:崩溃也要产出 ⚠ 行供 CI 推送,且不阻断流程
        print(f"⚠ 昨收校验脚本异常: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
