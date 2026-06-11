"""生成 GitHub Pages 静态报告 site/index.html。

内容(回答"策略做了什么、后果如何、不操作会怎样"):
- 结论横幅:最新信号 + 策略 vs 基准的样本外年化对比
- 净值曲线:策略 vs 沪深300买入持有 vs ETF池等权持有(全区间 + 近一年)
- 操作明细表:每段持仓(切换即一次操作)的区间收益 vs 同期沪深300,红绿标注
- 指标卡片:全区间/样本外 年化、回撤、夏普
- 模拟盘信号日志(output/signal_log.csv 末尾若干条)

用法:
  python report_web.py [--mode single] [--capital 10000] [--end YYYY-MM-DD]
"""
import argparse
import base64
import datetime as dt
import html
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
import data
import metrics as metrics_mod
from run_rotation import build_weights, closes_table, load_pool
from portfolio import run_portfolio_backtest

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SITE_DIR = "site"


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def held_name(row: pd.Series) -> str:
    """该日目标持仓的描述(single 模式为单一标的或空仓)"""
    held = row[row > 0.005]
    if held.empty:
        return "空仓(现金)"
    return " + ".join(f"{config.ETF_POOL[s]} {v:.0%}" for s, v in held.items())


def holding_segments(weights: pd.DataFrame, equity: pd.Series, bench: pd.Series) -> list[dict]:
    """按持仓变化切分区间,统计每段的策略收益与同期基准收益。

    weights 为 T 日收盘信号,T+1 开盘执行,故标签后移 1 日与实际持仓对齐;
    段收益以"切换前一日收盘净值"为基准,切换日的开盘成交成本(佣金/滑点)
    归入新段,避免操作明细偏乐观。
    """
    executed = weights.shift(1).dropna(how="all")
    labels = executed.apply(held_name, axis=1).reindex(equity.index).ffill().bfill()
    segments = []  # (基准日(前段末日), 段末日, 标签)
    base = labels.index[0]
    cur = labels.iloc[0]
    for i in range(1, len(labels)):
        if labels.iloc[i] != cur:
            segments.append((base, labels.index[i - 1], cur))
            base, cur = labels.index[i - 1], labels.iloc[i]
    segments.append((base, labels.index[-1], cur))

    rows = []
    for b0, e, label in segments:
        seg_eq = equity.loc[b0:e]
        seg_b = bench.loc[b0:e].dropna()
        if len(seg_eq) < 2:
            continue
        rows.append({
            "start": seg_eq.index[1].date(), "end": e.date(), "label": label,
            "days": len(seg_eq) - 1,
            "ret": seg_eq.iloc[-1] / seg_eq.iloc[0] - 1,
            "bench_ret": seg_b.iloc[-1] / seg_b.iloc[0] - 1 if len(seg_b) >= 2 else float("nan"),
        })
    return rows


def pct(v: float, signed: bool = True) -> str:
    if pd.isna(v):
        return "-"
    return f"{v:+.2%}" if signed else f"{v:.2%}"


def color_cls(v: float) -> str:
    if pd.isna(v):
        return ""
    return "pos" if v > 0 else ("neg" if v < 0 else "")


def metric_cards(title: str, m: dict) -> str:
    items = [
        ("年化收益率", pct(m["年化收益率"], signed=False), color_cls(m["年化收益率"])),
        ("总收益率", pct(m["总收益率"], signed=False), color_cls(m["总收益率"])),
        ("最大回撤", pct(m["最大回撤"], signed=False), "neg"),
        ("夏普比率", f"{m['夏普比率']:.2f}", ""),
    ]
    cards = "".join(
        f'<div class="card"><div class="card-label">{k}</div>'
        f'<div class="card-value {cls}">{v}</div></div>'
        for k, v, cls in items
    )
    return f'<h3>{title}</h3><div class="cards">{cards}</div>'


def build_equity_chart(equity: pd.Series, bench: pd.Series, ew: pd.Series,
                       title: str, log_scale: bool) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(equity.index, equity / equity.iloc[0], label="ETF动量轮动(本策略)", linewidth=1.8, color="#d62728")
    b = bench.dropna()
    ax.plot(b.index, b / b.iloc[0], label="沪深300 买入持有(不操作)", linewidth=1.3, color="#7f7f7f")
    ax.plot(ew.index, ew / ew.iloc[0], label="ETF池等权 买入持有(不操作)", linewidth=1.3, color="#1f77b4", alpha=0.8)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig_to_b64(fig)


def main():
    parser = argparse.ArgumentParser(description="生成静态网页报告")
    parser.add_argument("--mode", choices=("single", "ensemble"), default="single")
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--end", default=dt.date.today().isoformat())
    args = parser.parse_args()

    prices = load_pool(config.ROTATION_START, args.end)
    closes = closes_table(prices)
    weights = build_weights(closes, mode=args.mode, lookback=config.ROTATION_LOOKBACK,
                            buffer=config.ROTATION_BUFFER, dd_control=False)
    result = run_portfolio_backtest(prices, weights, initial_capital=args.capital, stamp_tax=False)
    equity = result.equity

    benchmark = data.get_benchmark_daily(config.ROTATION_START, args.end)
    bench = benchmark["close"].reindex(equity.index).ffill()
    # ETF池等权持有(各标的归一后均值)
    ew = (closes / closes.iloc[0]).mean(axis=1).reindex(equity.index)

    m_all = metrics_mod.equity_metrics(equity)
    oos = equity.loc[config.OOS_SPLIT:]
    m_oos = metrics_mod.equity_metrics(oos) if len(oos) >= 2 else None
    b = bench.dropna()
    m_bench = metrics_mod.equity_metrics(b)
    m_bench_oos = metrics_mod.equity_metrics(b.loc[config.OOS_SPLIT:])

    # 图表
    img_full = build_equity_chart(equity, bench, ew,
                                  f"全区间净值对比({equity.index[0].date()} ~ {equity.index[-1].date()},对数坐标)",
                                  log_scale=True)
    one_year = equity.index[-1] - pd.Timedelta(days=365)
    img_1y = build_equity_chart(equity.loc[one_year:], bench.loc[one_year:], ew.loc[one_year:],
                                "近一年净值对比", log_scale=False)

    # 操作明细(近一年)
    segs = holding_segments(weights, equity, bench)
    recent_segs = [r for r in segs if pd.Timestamp(r["end"]) >= one_year]
    seg_rows = "".join(
        f'<tr><td>{r["start"]} ~ {r["end"]}</td><td>{html.escape(r["label"])}</td>'
        f'<td>{r["days"]}</td>'
        f'<td class="{color_cls(r["ret"])}">{pct(r["ret"])}</td>'
        f'<td class="{color_cls(r["bench_ret"])}">{pct(r["bench_ret"])}</td>'
        f'<td class="{color_cls(r["ret"] - r["bench_ret"])}">{pct(r["ret"] - r["bench_ret"])}</td></tr>'
        for r in reversed(recent_segs)
    )
    seg_wins = sum(1 for r in segs if pd.notna(r["bench_ret"]) and r["ret"] > r["bench_ret"])
    seg_total = sum(1 for r in segs if pd.notna(r["bench_ret"]))

    # 最新信号
    latest = held_name(weights.iloc[-1])
    signal_date = weights.index[-1].date()

    # 信号日志
    log_rows = ""
    log_path = os.path.join(config.OUTPUT_DIR, "signal_log.csv")
    if os.path.exists(log_path):
        log = pd.read_csv(log_path, encoding="utf-8-sig")
        for _, r in log.tail(15).iloc[::-1].iterrows():
            log_rows += (f'<tr><td>{html.escape(str(r.get("signal_date", "")))}</td>'
                         f'<td>{html.escape(str(r.get("mode", "")))}</td>'
                         f'<td>{html.escape(str(r.get("desc", "")))}</td></tr>')

    # 结论
    edge_oos = (m_oos["年化收益率"] - m_bench_oos["年化收益率"]) if m_oos else float("nan")
    verdict = ("策略当前优于不操作" if pd.notna(edge_oos) and edge_oos > 0 else "策略当前未跑赢不操作")

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ETF 动量轮动 - 模拟盘报告</title>
<style>
body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
       max-width: 1000px; margin: 0 auto; padding: 16px; color: #222; background: #fafafa; }}
h1 {{ font-size: 1.5em; }} h2 {{ font-size: 1.2em; border-bottom: 2px solid #ddd; padding-bottom: 6px; margin-top: 36px; }}
h3 {{ font-size: 1em; color: #555; margin-bottom: 8px; }}
.banner {{ background: #fff; border-left: 6px solid #d62728; padding: 14px 18px; border-radius: 6px;
          box-shadow: 0 1px 3px rgba(0,0,0,.08); margin: 16px 0; }}
.banner .big {{ font-size: 1.25em; font-weight: 600; }}
.cards {{ display: flex; gap: 12px; flex-wrap: wrap; }}
.card {{ background: #fff; border-radius: 8px; padding: 12px 18px; min-width: 130px;
        box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
.card-label {{ font-size: .8em; color: #888; }}
.card-value {{ font-size: 1.35em; font-weight: 600; margin-top: 4px; }}
.pos {{ color: #d62728; }} .neg {{ color: #2e7d32; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; font-size: .9em;
        box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; }}
th {{ background: #f0f0f0; }}
img {{ max-width: 100%; background: #fff; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
.note {{ color: #888; font-size: .85em; }}
</style>
</head>
<body>
<h1>ETF 动量轮动 — 模拟盘报告</h1>
<p class="note">更新于 {now}(数据截至 {signal_date})· mode={args.mode} · 本金 {args.capital:,.0f} 元 · 回测含佣金/滑点/整手约束 · A股红涨绿跌</p>

<div class="banner">
  <div class="big">最新信号:{html.escape(latest)}</div>
  <div style="margin-top:6px">{verdict}:样本外({config.OOS_SPLIT} 至今)策略年化
  <b class="{color_cls(m_oos['年化收益率']) if m_oos else ''}">{pct(m_oos['年化收益率'], signed=False) if m_oos else '-'}</b>
  vs 沪深300买入持有 <b>{pct(m_bench_oos['年化收益率'], signed=False)}</b>
  (超额 <b class="{color_cls(edge_oos)}">{pct(edge_oos)}</b>/年)</div>
</div>

<h2>净值对比:操作 vs 不操作</h2>
<img src="data:image/png;base64,{img_full}" alt="全区间净值">
<img src="data:image/png;base64,{img_1y}" alt="近一年净值">

<h2>策略操作明细(近一年,每次切换为一次操作)</h2>
<p class="note">每段持仓的区间收益与同期沪深300对比。全历史 {seg_total} 段持仓中,跑赢同期沪深300的有 {seg_wins} 段({seg_wins / seg_total:.0%})。</p>
<table>
<tr><th>持仓区间</th><th>持有标的</th><th>天数</th><th>本段收益</th><th>同期沪深300</th><th>超额</th></tr>
{seg_rows}
</table>

<h2>绩效指标</h2>
{metric_cards(f"策略 · 全区间({equity.index[0].date()} 起)", m_all)}
{metric_cards(f"策略 · 样本外({config.OOS_SPLIT} 起,更接近真实预期)", m_oos) if m_oos else ""}
{metric_cards("沪深300 买入持有 · 全区间(不操作的对照)", m_bench)}

<h2>模拟盘信号日志(最近 15 条)</h2>
<table>
<tr><th>信号日期</th><th>模式</th><th>目标持仓</th></tr>
{log_rows or '<tr><td colspan="3">暂无记录</td></tr>'}
</table>

<p class="note">免责声明:回测与模拟盘结果不代表未来收益,本页面仅供学习记录。</p>
</body>
</html>
"""
    os.makedirs(SITE_DIR, exist_ok=True)
    out = os.path.join(SITE_DIR, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"报告已生成: {out}")


if __name__ == "__main__":
    main()
