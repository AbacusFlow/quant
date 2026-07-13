#!/usr/bin/env python3
"""mx_fetch_latest.py — 宿主机运行,从东方财富妙想(mx_data)拉取池内 ETF 的「最新一根收盘」。

用途:收盘后档(postclose_local.sh,17:03)的**同日兜底数据源**。akshare 收盘后要 ~22:00
才出当日 K,妙想 16:30 即有当日收盘。本脚本只抓「最新一根 bar」的收盘价,写成一个精简
JSON 供 portfolio_status.py --mx-fallback 追加到 in-memory 行情最后一行(绝不落盘、绝不
拉全历史——门槛验证已知 mx 历史分红标的与 akshare qfq 复权口径发散,只有近锚点最新收盘一致)。

自包含:只依赖标准库 + requests,**不 import 项目 config/pandas、不依赖 skill 路径**
(契合 DOCKER.md:host 编排取数,容器跑项目运行时)。symbols 由调用方从 config.ETF_POOL
传入(单一真相)。

用法:
  MX_APIKEY=... python3 scripts/mx_fetch_latest.py 510300 510500 ... \
      --end 2026-07-10 --out output/mx_latest.json

输出 JSON(拿到部分就写部分;整体失败不写文件、退出码非零,调用方容错):
  {"end": "2026-07-10", "closes": {"510500": 8.698, "510300": 4.829, ...}}
  end = mx 返回的最新交易日(≤ 请求 end,天然处理节假日=上一交易日)。
"""
import argparse
import datetime as dt
import json
import os
import re
import sys

import requests

BASE_URL = "https://mkapi2.dfcfs.com/finskillshub/api/claw/query"
BATCH_SIZE = 5          # 单次自然语言 query 的实体数上限(超过会被截断)
WINDOW_DAYS = 7         # 查询窗口 [end-7d, end],覆盖节假日仍能取到最近交易日
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")  # 输出的 end 会被调用方用作文件路径组件,必须是干净日期


def _query(api_key: str, tool_query: str) -> dict:
    resp = requests.post(
        BASE_URL,
        headers={"Content-Type": "application/json", "apikey": api_key},
        json={"toolQuery": tool_query},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_batch(payload: dict, want: list[str]) -> dict[str, dict[str, str]]:
    """从一次查询响应里抽出 {symbol: {date: close_str}}。

    妙想返回两种 rawTable 形态,均在此处理:
    - 多标的:key = 实体名(含 6 位代码,如 "南方中证500ETF(510500.SH)"),value = 收盘数组;
    - 单标的:key = 指标 ID(如 "325898"),实体代码在 block.entityName。
    统一按「请求代码是否出现在 key/entityName 里」归属,避免正则歧义。
    """
    out: dict[str, dict[str, str]] = {}
    if payload.get("status") != 0:
        return out
    try:
        dto_list = payload["data"]["data"]["searchDataResultDTO"]["dataTableDTOList"]
    except (KeyError, TypeError):
        return out
    if not isinstance(dto_list, list):
        return out

    for block in dto_list:
        if not isinstance(block, dict):
            continue
        raw = block.get("rawTable")
        if not isinstance(raw, dict):
            continue
        dates = raw.get("headName")
        if not isinstance(dates, list) or not dates:
            continue
        entity_name = str(block.get("entityName") or "")
        entity_sym = next((s for s in want if s in entity_name), None)

        for key, vals in raw.items():
            if key == "headName" or not isinstance(vals, list):
                continue
            sym = next((s for s in want if s in str(key)), None)
            if sym is None:
                sym = entity_sym  # 单标的形态:key 是指标 ID,代码在 entityName
            if sym is None:
                continue
            bucket = out.setdefault(sym, {})
            for d, v in zip(dates, vals):
                ds = str(d).strip()
                vs = str(v).strip()
                if ds and vs:
                    bucket[ds] = vs
    return out


def fetch_latest(api_key: str, symbols: list[str], end: str) -> dict:
    """返回 {"end": <最新交易日>, "closes": {sym: float}};无任何数据时抛异常。"""
    start = (dt.date.fromisoformat(end) - dt.timedelta(days=WINDOW_DAYS)).isoformat()
    collected: dict[str, dict[str, str]] = {}
    errors: list[str] = []

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        q = f"{' '.join(batch)} 这{len(batch)}只ETF {start}到{end}每天的收盘价"
        try:
            payload = _query(api_key, q)
        except Exception as e:  # noqa: BLE001 网络/接口异常不稳定
            errors.append(f"batch {batch}: {e}")
            continue
        for sym, series in _parse_batch(payload, batch).items():
            collected.setdefault(sym, {}).update(series)

    if not collected:
        raise RuntimeError("mx_data 未返回任何收盘数据: " + "; ".join(errors) if errors
                           else "mx_data 未返回任何收盘数据")

    # 最新交易日 = 所有标的里 ≤ end 的最大日期(节假日时自动回退到上一交易日)。
    # 必须先过 DATE_RE:日期来自外部 API 响应,会被 postclose_local.sh 直接拼进
    # 回执/spool 文件路径——不校验格式则恶意/异常响应(如 "../../x")可路径逃逸
    all_dates = {d for series in collected.values() for d in series
                 if DATE_RE.match(d) and d <= end}
    if not all_dates:
        raise RuntimeError("mx_data 返回的日期均晚于请求 end")
    latest = max(all_dates)

    closes: dict[str, float] = {}
    for sym, series in collected.items():
        v = series.get(latest)
        if v is None:
            continue
        try:
            px = float(v)
        except (TypeError, ValueError):
            continue
        if px > 0:
            closes[sym] = px
    if not closes:
        raise RuntimeError(f"mx_data 在 {latest} 无有效收盘价")
    return {"end": latest, "closes": closes}


def main() -> int:
    p = argparse.ArgumentParser(description="从妙想拉取池内 ETF 最新收盘(收盘后档兜底,宿主机运行)")
    p.add_argument("symbols", nargs="+", help="ETF 6 位代码(由调用方从 config.ETF_POOL 传入)")
    p.add_argument("--end", required=True, help="请求截止日 YYYY-MM-DD")
    p.add_argument("--out", required=True, help="输出 JSON 路径")
    args = p.parse_args()

    api_key = os.getenv("MX_APIKEY")
    if not api_key:
        print("MX_APIKEY 环境变量未设置", file=sys.stderr)
        return 1
    try:
        result = fetch_latest(api_key, args.symbols, args.end)
    except Exception as e:  # noqa: BLE001 尽力而为:整体失败不写文件
        print(f"mx_fetch_latest 失败(不写文件): {e}", file=sys.stderr)
        return 1

    # 原子写:先写临时文件再 rename,避免读方读到半截 JSON
    tmp = f"{args.out}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    os.replace(tmp, args.out)
    print(f"mx_fetch_latest: end={result['end']} 收到 {len(result['closes'])}/{len(args.symbols)} 只 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
