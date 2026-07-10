"""数据获取:akshare 拉取 A 股日线(前复权)与沪深300基准,本地 CSV 缓存"""
import os
import time

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


def _load_cache(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        if not df.empty:
            return df
    return None


def get_stock_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    """A股日线前复权数据,列: open/high/low/close/volume,索引为日期"""
    path = _cache_path(f"stock_{symbol}", start, end)
    cached = _load_cache(path)
    if cached is not None:
        return cached

    import akshare as ak

    raw = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        adjust="qfq",
    )
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

    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_csv(path)
    return df


def _exchange_prefix(symbol: str) -> str:
    return "sh" if symbol.startswith(("5", "6")) else "sz"


def _fetch_etf_tencent(symbol: str, start: str, end: str) -> pd.DataFrame:
    """腾讯前复权日线(无需 akshare,海外 IP 下通常可用)。

    接口单次最多返回区间内最近 640 根 K 线,按结束日期向前翻页拼接。
    """
    import datetime as dt

    import requests

    code = f"{_exchange_prefix(symbol)}{symbol}"
    all_rows: list[list] = []
    cur_end = end
    while True:
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
        all_rows = rows + all_rows
        earliest = rows[0][0]
        if earliest <= start or len(rows) < 640:
            break
        cur_end = (dt.date.fromisoformat(earliest) - dt.timedelta(days=1)).isoformat()
    if not all_rows:
        raise RuntimeError(f"腾讯接口未返回 ETF {symbol} 的数据")
    # 行格式: [date, open, close, high, low, volume, ...]
    df = pd.DataFrame([r[:6] for r in all_rows],
                      columns=["date", "open", "close", "high", "low", "volume"])
    df = df.drop_duplicates(subset="date", keep="first")
    df[["open", "close", "high", "low", "volume"]] = \
        df[["open", "close", "high", "low", "volume"]].astype(float)
    return df[["date", "open", "high", "low", "close", "volume"]]


def get_etf_daily(symbol: str, start: str, end: str, write_cache: bool = True) -> pd.DataFrame:
    """场内 ETF 日线前复权数据,列: open/high/low/close/volume,索引为日期。

    数据源优先级:东财(qfq)→ 腾讯(qfq)→ 新浪(不复权,独立缓存键)。

    write_cache=False:只读缓存、绝不落盘(供纯展示/兜底路径用,避免写「名新实旧」缓存
    ——当日盘后 end=today 但数据源只到昨天时,写 `{name}_..._{today}.csv` 会以文件名命中
    堵住后续 akshare 权威拉取)。
    """
    path = _cache_path(f"etf_{symbol}", start, end)
    cached = _load_cache(path)
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
            # 回退2:新浪为不复权数据,用独立缓存键避免污染 qfq 缓存
            path = _cache_path(f"etf_{symbol}_sina", start, end)
            cached = _load_cache(path)
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

    if write_cache:
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
    cached = _load_cache(path)
    if cached is not None:
        return cached

    import akshare as ak

    try:
        raw = ak.index_zh_a_hist(
            symbol=config.BENCHMARK_SYMBOL,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
        )
        df = raw.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]]
    except Exception:
        # 东财接口失败时回退到新浪接口
        raw = ak.stock_zh_index_daily(symbol=f"sh{config.BENCHMARK_SYMBOL}")
        df = raw[["date", "close"]].copy()

    if df is None or df.empty:
        raise RuntimeError("akshare 未返回沪深300指数数据,请检查接口版本")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[start:end]
    if df.empty:
        raise RuntimeError(f"沪深300指数在 {start} ~ {end} 区间内无数据")

    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_csv(path)
    return df
