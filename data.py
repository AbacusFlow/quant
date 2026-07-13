"""数据获取:akshare 拉取 A 股日线(前复权)与沪深300基准,本地 CSV 缓存"""
import datetime as dt
import os
import time
from zoneinfo import ZoneInfo

import pandas as pd

import config


def _retry(fetch, attempts: int = 3, wait: float = 3.0):
    """接口限流/断连时重试"""
    last_exc = None
    for i in range(attempts):
        try:
            return fetch()
        except Exception as e:  # noqa: BLE001 akshare 抛出的异常类型不稳定
            last_exc = e
            if i < attempts - 1:
                time.sleep(wait * (i + 1))
    raise last_exc


def _cache_path(name: str, start: str, end: str) -> str:
    return os.path.join(config.DATA_DIR, f"{name}_{start}_{end}.csv")


def _load_cache(path: str, end: str | None = None) -> pd.DataFrame | None:
    """读缓存;传入 end 时对命中做完整性检查:历史上已写入的「名新实旧」缓存
    (文件名 end=今天而内容只到昨天,升级/踩坑前遗留)视为未命中,交给上游重拉,
    数据源补齐当日 K 线后正常写回即自愈——防线不能只挡新写入,还要挡旧毒缓存。"""
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        if not df.empty and not (end is not None and _cache_would_lie(df, end)):
            return df
    return None


def _cache_would_lie(df: pd.DataFrame, end: str) -> bool:
    """「名新实旧」检测:end 是今天/未来、而数据实际未到位时,以 end 命名的缓存
    会在数据源随后补齐当日 K 线后仍按文件名永久命中,堵住后续权威重拉
    (2026-07-10 实际踩坑)。此类缓存拒写也拒读,防线下沉到本模块,
    调用方无需逐点记得传 write_cache=False。

    「到位」按交易日历判定:数据最后一根 bar ≥「end 之前(含)最后一个交易日」
    即完整——end 落在周末/节假日时(如周六跑 end=周六),到周五的数据就是完整的,
    不算说谎,照常读写缓存(否则非交易日每次绕过缓存重拉,数据源临时不可用时
    qfq_only 任务会无谓失败)。end 在过去(历史区间)数据不会再增长,恒可缓存。
    日历不可用/解析异常时保守拒缓存(只损失效率,不损失正确性)。
    """
    try:
        end_ts = pd.Timestamp(end)
        today = pd.Timestamp(dt.datetime.now(ZoneInfo("Asia/Shanghai")).date())
        if end_ts < today:
            return False
        if df.index[-1] >= end_ts:
            return False
        try:
            cal = get_trade_dates()
            # 日历必须覆盖到 end 才可据此判「完整」:截断日历(如只到上一交易日)
            # 无法证明 end 当天不是交易日,若径直放行会把真毒缓存误判为完整
            if len(cal) and cal.max() >= end_ts:
                expected = cal[cal <= end_ts]
                if len(expected) and df.index[-1] >= expected[-1]:
                    return False  # 已到 end 前最后一个交易日 → 数据完整(end 是非交易日)
        except Exception:
            pass  # 日历不可用 → 落到保守拒缓存
        return True
    except Exception:
        return True


def get_stock_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    """A股日线前复权数据,列: open/high/low/close/volume,索引为日期"""
    path = _cache_path(f"stock_{symbol}", start, end)
    cached = _load_cache(path, end)
    if cached is not None:
        return cached

    import akshare as ak

    raw = _retry(lambda: ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        adjust="qfq",
    ))
    if raw is None or raw.empty:
        raise RuntimeError(f"akshare 未返回 {symbol} 的数据,请检查代码或接口版本")

    df = raw.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
        }
    )[["date", "open", "high", "low", "close", "volume"]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    if not _cache_would_lie(df, end):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        df.to_csv(path)
    return df


def _exchange_prefix(symbol: str) -> str:
    return "sh" if symbol.startswith(("5", "6")) else "sz"


def _fetch_etf_tencent(symbol: str, start: str, end: str) -> pd.DataFrame:
    """腾讯前复权日线(无需 akshare,海外 IP 下通常可用)。

    接口单次最多返回区间内最近 640 根 K 线,按结束日期向前翻页拼接。
    """
    import requests

    code = f"{_exchange_prefix(symbol)}{symbol}"
    all_rows: list[list] = []
    cur_end = end
    prev_earliest = None
    # 页数上限:40页×640根≈100年,正常永远够;防接口异常(忽略 cur_end/重复回同页)死循环
    for _ in range(40):
        url = (
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            f"?param={code},day,{start},{cur_end},640,qfq"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()["data"]
        if not isinstance(data, dict) or code not in data:
            raise RuntimeError(f"腾讯接口未返回 ETF {symbol} 的数据")
        # 仅接受前复权数据,避免不复权数据污染 qfq 缓存
        rows = data[code].get("qfqday")
        if not rows:
            break
        earliest = rows[0][0]
        # 无进展保护:接口若忽略 cur_end 重复返回同页,累积会死循环且数据重复
        if prev_earliest is not None and earliest >= prev_earliest:
            raise RuntimeError(
                f"腾讯接口翻页无进展(ETF {symbol}: {earliest} >= {prev_earliest}),中止以防死循环")
        prev_earliest = earliest
        all_rows = rows + all_rows
        if earliest <= start:
            break
        # 不能以 len(rows)<640 判定"历史已尽":接口限流时可能少给,提前终止会把
        # 被截断的残缺历史写进缓存永久固化。继续向前翻页,靠空页自然终止
        # (标的上市日之前无数据 → 下一页 qfqday 为空;earliest 每轮严格递减,必然收敛)。
        cur_end = (dt.date.fromisoformat(earliest) - dt.timedelta(days=1)).isoformat()
    else:
        raise RuntimeError(f"腾讯接口翻页超过页数上限(ETF {symbol}),疑似接口异常,中止")
    if not all_rows:
        raise RuntimeError(f"腾讯接口未返回 ETF {symbol} 的数据")
    # 行格式: [date, open, close, high, low, volume, ...]
    df = pd.DataFrame([r[:6] for r in all_rows],
                      columns=["date", "open", "close", "high", "low", "volume"])
    df = df.drop_duplicates(subset="date", keep="first")
    df[["open", "close", "high", "low", "volume"]] = \
        df[["open", "close", "high", "low", "volume"]].astype(float)
    return df[["date", "open", "high", "low", "close", "volume"]]


def get_etf_daily(symbol: str, start: str, end: str, write_cache: bool = True,
                  qfq_only: bool = False) -> pd.DataFrame:
    """场内 ETF 日线前复权数据,列: open/high/low/close/volume,索引为日期。

    数据源优先级:东财(qfq)→ 腾讯(qfq)→ 新浪(不复权,独立缓存键)。

    write_cache=False:只读缓存、绝不落盘(供纯展示/兜底路径用)。write_cache=True
    时仍有内置防线 `_cache_would_lie`:end=today 而数据只到昨天的「名新实旧」缓存
    一律拒写,避免以文件名命中堵住后续 akshare 权威拉取。

    qfq_only=True:东财/腾讯双双失败时**拒绝**回退新浪不复权数据,直接抛错。
    money 信号路径(daily_signal/check_risk)必须开启——不复权价在分红标的除息日
    有向下跳空,动量被人为压低,且逐标的回退会造成横截面比较口径不一(9 只里
    1 只不复权、8 只 qfq);宁可当日无信号(幂等机制次日自动补),不可静默混口径。
    展示路径可保持 False 优雅降级。
    """
    path = _cache_path(f"etf_{symbol}", start, end)
    cached = _load_cache(path, end)
    if cached is not None:
        return cached

    import akshare as ak

    try:
        raw = _retry(lambda: ak.fund_etf_hist_em(
            symbol=symbol,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq",
        ))
        if raw is None or raw.empty:
            raise RuntimeError(f"东财未返回 ETF {symbol} 的数据")
        df = raw.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            }
        )[["date", "open", "high", "low", "close", "volume"]]
    except Exception:
        try:
            # 回退1:腾讯,同为前复权,可共用 qfq 缓存键
            df = _retry(lambda: _fetch_etf_tencent(symbol, start, end))
        except Exception:
            if qfq_only:
                raise RuntimeError(
                    f"ETF {symbol} 前复权源(东财/腾讯)均不可用;qfq_only 模式拒绝"
                    f"新浪不复权回退(money 信号路径不可混复权口径,等数据源恢复后重跑)")
            # 回退2:新浪为不复权数据,用独立缓存键避免污染 qfq 缓存
            path = _cache_path(f"etf_{symbol}_sina", start, end)
            cached = _load_cache(path, end)
            if cached is not None:
                return cached
            raw = _retry(lambda: ak.fund_etf_hist_sina(symbol=f"{_exchange_prefix(symbol)}{symbol}"))
            df = raw[["date", "open", "high", "low", "close", "volume"]].copy()

    if df is None or df.empty:
        raise RuntimeError(f"akshare 未返回 ETF {symbol} 的数据,请检查代码或接口版本")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[start:end]
    if df.empty:
        raise RuntimeError(f"ETF {symbol} 在 {start} ~ {end} 区间内无数据")

    if write_cache and not _cache_would_lie(df, end):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        df.to_csv(path)
    return df


def get_trade_dates() -> pd.DatetimeIndex:
    """A股交易日历(新浪源,含未来已公布日期),本地缓存;缓存覆盖不足未来30天时刷新"""
    path = os.path.join(config.DATA_DIR, "trade_dates.csv")
    today = pd.Timestamp.today().normalize()
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["trade_date"])
        if not df.empty and df["trade_date"].max() >= today + pd.Timedelta(days=30):
            return pd.DatetimeIndex(df["trade_date"])

    import akshare as ak

    raw = _retry(lambda: ak.tool_trade_date_hist_sina())
    df = pd.DataFrame({"trade_date": pd.to_datetime(raw["trade_date"])}).sort_values("trade_date")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    return pd.DatetimeIndex(df["trade_date"])


def get_benchmark_daily(start: str, end: str) -> pd.DataFrame:
    """沪深300指数日线,列: close"""
    path = _cache_path(f"index_{config.BENCHMARK_SYMBOL}", start, end)
    cached = _load_cache(path, end)
    if cached is not None:
        return cached

    import akshare as ak

    try:
        raw = _retry(lambda: ak.index_zh_a_hist(
            symbol=config.BENCHMARK_SYMBOL,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
        ))
        df = raw.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]]
    except Exception:
        # 东财接口失败时回退到新浪接口
        raw = _retry(lambda: ak.stock_zh_index_daily(symbol=f"sh{config.BENCHMARK_SYMBOL}"))
        df = raw[["date", "close"]].copy()

    if df is None or df.empty:
        raise RuntimeError("akshare 未返回沪深300指数数据,请检查接口版本")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[start:end]
    if df.empty:
        raise RuntimeError(f"沪深300指数在 {start} ~ {end} 区间内无数据")

    if not _cache_would_lie(df, end):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        df.to_csv(path)
    return df
