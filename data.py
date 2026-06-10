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


def get_etf_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    """场内 ETF 日线前复权数据,列: open/high/low/close/volume,索引为日期"""
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
        # 东财接口失败时回退到新浪(注意:新浪为不复权数据,用独立缓存键避免污染 qfq 缓存)
        path = _cache_path(f"etf_{symbol}_sina", start, end)
        cached = _load_cache(path)
        if cached is not None:
            return cached
        prefix = "sh" if symbol.startswith(("5", "6")) else "sz"
        raw = _retry(lambda: ak.fund_etf_hist_sina(symbol=f"{prefix}{symbol}"))
        df = raw[["date", "open", "high", "low", "close", "volume"]].copy()

    if df is None or df.empty:
        raise RuntimeError(f"akshare 未返回 ETF {symbol} 的数据,请检查代码或接口版本")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[start:end]
    if df.empty:
        raise RuntimeError(f"ETF {symbol} 在 {start} ~ {end} 区间内无数据")

    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_csv(path)
    return df


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
