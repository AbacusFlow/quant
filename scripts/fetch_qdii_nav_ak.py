"""研究用:用 akshare 抓跨境 QDII ETF 的全历史「单位净值 + 未复权收盘价」,并与妙想数据交叉核对。

背景:妙想免费版每日 150 次调用上限,无法分批抓全 9 年;akshare 无配额、可在 Docker 内复现。
- 单位净值:ak.fund_open_fund_info_em(symbol, indicator="单位净值走势")(东方财富)
- 未复权收盘:ak.fund_etf_hist_sina(symbol="sh513100")(新浪,不复权)
- 交叉核对:与 research/data/qdii_nav_close.csv(妙想)重叠日期逐日比对,报告最大偏差。

输出 research/data/qdii_nav_close_ak.csv(date,symbol,nav,close)。
用法(Docker,需网络):python scripts/fetch_qdii_nav_ak.py
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDII = {"513100": "sh513100", "513500": "sh513500", "159920": "sz159920"}
OUT = os.path.join(ROOT, "research", "data", "qdii_nav_close_ak.csv")
MX = os.path.join(ROOT, "research", "data", "qdii_nav_close.csv")
START = "2017-06-01"


def fetch() -> pd.DataFrame:
    import akshare as ak
    frames = []
    for sym, sina in QDII.items():
        nav = ak.fund_open_fund_info_em(symbol=sym, indicator="单位净值走势")
        nav = nav.rename(columns={"净值日期": "date", "单位净值": "nav"})[["date", "nav"]]
        px = ak.fund_etf_hist_sina(symbol=sina).rename(columns={"close": "close"})[["date", "close"]]
        for df in (nav, px):
            df["date"] = pd.to_datetime(df["date"])
        m = pd.merge(px, nav, on="date", how="outer").sort_values("date")
        m = m[m["date"] >= START]
        m["symbol"] = sym
        frames.append(m[["date", "symbol", "nav", "close"]])
        print(f"{sym}: 净值 {nav['date'].min().date()}~{nav['date'].max().date()} {len(nav)} 行;"
              f" 收盘 {px['date'].min().date()}~{px['date'].max().date()} {len(px)} 行")
    return pd.concat(frames, ignore_index=True)


def cross_check(ak_df: pd.DataFrame) -> None:
    if not os.path.exists(MX):
        print("(无妙想数据,跳过交叉核对)")
        return
    mx = pd.read_csv(MX, dtype={"symbol": str}, parse_dates=["date"])
    j = pd.merge(ak_df, mx, on=["date", "symbol"], suffixes=("_ak", "_mx")).dropna()
    print(f"\n===== 与妙想交叉核对(重叠 {len(j)} 行)=====")
    for sym, g in j.groupby("symbol"):
        dn = (g["nav_ak"] / g["nav_mx"] - 1).abs()
        dc = (g["close_ak"] / g["close_mx"] - 1).abs()
        print(f"  {sym}: {len(g)} 行  净值最大偏差 {dn.max():.4%}(>0.1% 的 {int((dn > 0.001).sum())} 行)"
              f"  收盘最大偏差 {dc.max():.4%}(>0.1% 的 {int((dc > 0.001).sum())} 行)")
        bad = g[(dn > 0.001) | (dc > 0.001)]
        for _, r in bad.head(5).iterrows():
            print(f"     {r['date'].date()} nav ak/mx {r['nav_ak']}/{r['nav_mx']}  close ak/mx {r['close_ak']}/{r['close_mx']}")


def main() -> int:
    df = fetch()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    df.sort_values(["date", "symbol"]).to_csv(tmp, index=False, date_format="%Y-%m-%d")
    os.replace(tmp, OUT)
    print(f"写入 {OUT}({len(df)} 行)")
    cross_check(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
