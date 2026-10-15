"""研究用:从妙想(mx_data)抓跨境 QDII ETF 的历史每日「单位净值 + 未复权收盘价」。

用途:溢价研究(research/2026-10-qdii-premium)。溢价 = 收盘价 / 单位净值 - 1,两者同为
未复权、同一份额口径,份额拆分/分红在比值中自动抵消,无需单独复权。

宿主机运行(需 MX_APIKEY + 网络 + requests),只依赖标准库 + requests:
    MX_APIKEY=... python3 scripts/mx_fetch_nav.py [--start 2017-06-01] [--end 2026-10-14]

- 单次查询约 60 行上限 → 按 2 个月窗口分批,逐标的查询。
- 可断点续抓:已有输出中某(标的,窗口)两列都有数据则跳过。
- 输出 research/data/qdii_nav_close.csv(date,symbol,nav,close),原子写,按日期排序。
  该文件纳入版本控制,使研究在无 API key 时也可复现。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import time

import requests

BASE_URL = "https://mkapi2.dfcfs.com/finskillshub/api/claw/query"
SYMBOLS = ["513100", "513500", "159920"]
OUT = os.path.join(os.path.dirname(__file__), "..", "research", "data", "qdii_nav_close.csv")
FIELDS = {"单位净值": "nav", "收盘价": "close"}


def _query(api_key: str, text: str) -> dict:
    for attempt in range(3):
        try:
            r = requests.post(BASE_URL, headers={"Content-Type": "application/json", "apikey": api_key},
                              json={"toolQuery": text}, timeout=60)
            r.raise_for_status()
            j = r.json()
            if j.get("status") not in (0, None):  # 如 113=免费版日配额(150 次)用尽:立即停止,不空转
                raise SystemExit(f"妙想接口拒绝(status={j.get('status')}): {j.get('message')}")
            return j
        except (requests.RequestException, ValueError) as e:
            if attempt == 2:
                raise
            print(f"  重试({e})", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    return {}


def _parse(payload: dict, sym: str) -> dict[str, dict[str, float]]:
    """{date: {"nav": x, "close": y}};只收实体名含该代码的块,指标按 nameMap 中文名识别。"""
    out: dict[str, dict[str, float]] = {}
    try:
        blocks = payload["data"]["data"]["searchDataResultDTO"]["dataTableDTOList"]
    except (KeyError, TypeError):
        return out
    for b in blocks or []:
        if sym not in str(b.get("entityName") or ""):
            continue
        raw, names = b.get("rawTable") or {}, b.get("nameMap") or {}
        dates = raw.get("headName") or []
        for key, vals in raw.items():
            col = FIELDS.get(str(names.get(key, "")))
            if key == "headName" or col is None or not isinstance(vals, list):
                continue
            for d, v in zip(dates, vals):
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    continue
                if x > 0 and len(str(d)) == 10:
                    out.setdefault(str(d), {})[col] = x
    return out


def _windows(start: dt.date, end: dt.date):
    a = start
    while a <= end:
        m = a.month + 2
        b = dt.date(a.year + (m - 1) // 12, (m - 1) % 12 + 1, 1) - dt.timedelta(days=1)
        yield a, min(b, end)
        a = b + dt.timedelta(days=1)


def _load() -> dict[tuple[str, str], dict[str, float]]:
    rows: dict[tuple[str, str], dict[str, float]] = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[(r["symbol"], r["date"])] = {k: float(r[k]) for k in ("nav", "close") if r[k]}
    return rows


def _save(rows) -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "symbol", "nav", "close"])
        for (s, d), v in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            w.writerow([d, s, v.get("nav", ""), v.get("close", "")])
    os.replace(tmp, OUT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-06-01")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    a = ap.parse_args()
    key = os.environ.get("MX_APIKEY")
    if not key:
        print("缺少 MX_APIKEY", file=sys.stderr)
        return 2
    rows = _load()
    start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    for sym in SYMBOLS:
        for w0, w1 in _windows(start, end):
            have = [d for (s, d), v in rows.items() if s == sym and w0.isoformat() <= d <= w1.isoformat()
                    and "nav" in v and "close" in v]
            if len(have) >= 25:  # 2 个月窗口约 40 交易日;已有足量完整行即视为已抓
                continue
            text = f"{sym} {w0:%Y年%m月%d日}到{w1:%Y年%m月%d日}每天的单位净值和收盘价"
            got = _parse(_query(key, text), sym)
            for d, v in got.items():
                rows.setdefault((sym, d), {}).update(v)
            n_full = sum(1 for v in got.values() if "nav" in v and "close" in v)
            print(f"{sym} {w0}~{w1}: {len(got)} 天, 两列齐全 {n_full}")
            _save(rows)
            time.sleep(0.5)
    print(f"完成,共 {len(rows)} 行 → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
