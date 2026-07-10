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
import math
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


def holding_segments(weights: pd.DataFrame, equity: pd.Series, bench: pd.Series,
                     ew: pd.Series) -> list[dict]:
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
        seg_w = ew.loc[b0:e].dropna()
        if len(seg_eq) < 2:
            continue
        rows.append({
            "start": seg_eq.index[1].date(), "end": e.date(), "label": label,
            "days": len(seg_eq) - 1,
            "ret": seg_eq.iloc[-1] / seg_eq.iloc[0] - 1,
            "bench_ret": seg_b.iloc[-1] / seg_b.iloc[0] - 1 if len(seg_b) >= 2 else float("nan"),
            "ew_ret": seg_w.iloc[-1] / seg_w.iloc[0] - 1 if len(seg_w) >= 2 else float("nan"),
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
                       title: str, log_scale: bool,
                       extras: list[tuple[str, pd.Series]] | None = None,
                       main_label: str = "ETF动量轮动(本策略)") -> str:
    """净值对比图;extras 为历史版本变体曲线 [(标签, 净值序列)],细虚线绘制。

    最新线上策略永远是红色粗线,变体/基准用其他颜色区分。
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(equity.index, equity / equity.iloc[0], label=main_label, linewidth=1.8, color="#d62728")
    extra_colors = ["#999999", "#4c72b0", "#8c564b"]
    for i, (label, series) in enumerate(extras or []):
        s = series.dropna()
        if len(s) >= 2:
            ax.plot(s.index, s / s.iloc[0], label=label, linewidth=1.1,
                    linestyle="--", color=extra_colors[i % len(extra_colors)])
    b = bench.dropna()
    ax.plot(b.index, b / b.iloc[0], label="沪深300 买入持有(不操作)", linewidth=1.3, color="#7f7f7f")
    ax.plot(ew.index, ew / ew.iloc[0], label="ETF池等权 买入持有(不操作)", linewidth=1.3, color="#1f77b4", alpha=0.8)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return fig_to_b64(fig)


EXEC_PATH = os.path.join(config.OUTPUT_DIR, "executions.csv")
ACTION_ALIASES = {
    "买入": "buy", "buy": "buy",
    "卖出": "sell", "sell": "sell",
    "入金": "deposit", "deposit": "deposit",
    "出金": "withdraw", "withdraw": "withdraw",
}


def load_executions() -> pd.DataFrame | None:
    """读取手工维护的实盘成交记录;只有表头(无数据行)视为尚未实盘。

    列: date,action,symbol,price,shares,amount,note
    - action: 入金/出金(amount 必填)、买入/卖出(symbol/price/shares 必填,
      amount 选填 = 券商实际发生金额,不填则按 价格*股数±默认佣金 估算)
    手工输入是边界,这里做显式校验,错误信息带行号方便在 GitHub 上改。
    """
    if not os.path.exists(EXEC_PATH):
        return None
    df = pd.read_csv(EXEC_PATH, encoding="utf-8-sig", dtype={"symbol": str})
    if df.empty:
        return None
    if "status" not in df.columns:
        df["status"] = ""
    df["status"] = df["status"].fillna("").astype(str).str.strip()
    raw_action = df["action"].astype(str).str.strip()
    df["action"] = raw_action.str.lower().map(ACTION_ALIASES)
    df["symbol"] = df["symbol"].astype(str).str.strip()
    parsed_date = pd.to_datetime(df["date"], errors="coerce")
    for col in ("price", "shares", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for i, r in df.iterrows():
        rowno = i + 2  # 含表头的 CSV 行号
        if pd.isna(r["action"]):
            raise ValueError(f"第 {rowno} 行 action 无效: {raw_action[i]}(应为 买入/卖出/入金/出金)")
        if pd.isna(parsed_date[i]):
            raise ValueError(f"第 {rowno} 行日期无效: {r['date']}")
        if pd.notna(r["amount"]) and float(r["amount"]) <= 0:
            raise ValueError(f"第 {rowno} 行 amount 必须为正数")
        if r["action"] in ("deposit", "withdraw"):
            if pd.isna(r["amount"]):
                raise ValueError(f"第 {rowno} 行 {raw_action[i]} 必须填正数 amount(金额)")
        else:
            # 允许池外标的(如误买错代码后的持有/清仓),只校验代码格式
            if not (r["symbol"].isdigit() and len(r["symbol"]) == 6):
                raise ValueError(f"第 {rowno} 行 symbol 无效(应为6位代码): {r['symbol']}")
            if pd.isna(r["shares"]) or float(r["shares"]) <= 0:
                raise ValueError(f"第 {rowno} 行买卖必须填正数 shares")
            # 计划行允许 price 留空(池外标的系统无估算价);已成交必须有真实价格
            if r["status"] != "计划" and (pd.isna(r["price"]) or float(r["price"]) <= 0):
                raise ValueError(f"第 {rowno} 行买卖必须填正数 price")
            if float(r["shares"]) != int(r["shares"]):
                raise ValueError(f"第 {rowno} 行 shares 必须为整数股")
    df["date"] = parsed_date
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def real_equity_series(execs: pd.DataFrame, closes: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    """根据成交流水重建真实账户。返回 (每日总资产, 份额化净值 NAV, 累计净入金)。

    NAV 采用份额化(TWR)口径(标准基金记账):入金/出金按"日初"折算份额
    (用前一日 NAV 计价),使新资本与既有份额一起承担当日涨跌,再按当日收盘
    总资产 eq/units 计 NAV。追加/抽回资金不扭曲历史收益曲线,可与模拟盘/基准
    直接比较。

    为何按日初而非日终:实盘现金往往当日开盘即部署为持仓、按收盘计价;若按
    日终"剔除净流入"折份额(nav=(eq-flow)/units),会把这笔新钱盘中 open→close
    的涨跌错记到既有份额上,虚增净值(如大额加仓当天所持标的其实几乎没动,
    净值却跳涨)。日初口径下,当日收益由新老份额按比例分摊,无此错配。
    """
    last_quote = closes.index[-1]
    future = execs[execs["date"] > last_quote]
    if not future.empty:
        raise ValueError(f"流水日期 {future['date'].iloc[0].date()} 晚于最新行情 {last_quote.date()},"
                         "请等行情更新后再录入或修正日期")
    cash = 0.0
    net_deposit = 0.0
    units = 0.0  # 份额
    nav = 1.0
    pos: dict[str, int] = {}  # 含池外标的(误买),按需建键
    first = execs["date"].min()
    days = closes.index[closes.index >= first.normalize()]
    if days.empty:
        days = closes.index[-1:]
    applied = pd.Series(False, index=execs.index)
    equity = pd.Series(dtype=float)
    navs = pd.Series(dtype=float)
    for day in days:
        todo = execs.index[(~applied) & (execs["date"] <= day)]
        flow_today = 0.0  # 当日净流入,按日初(前一日 NAV)折份额
        for i in todo:
            r = execs.loc[i]
            if r["action"] == "deposit":
                amt = float(r["amount"])
                cash += amt; net_deposit += amt; flow_today += amt
            elif r["action"] == "withdraw":
                amt = float(r["amount"])
                cash -= amt; net_deposit -= amt; flow_today -= amt
                if cash < -0.01:
                    raise ValueError(f"{r['date'].date()} 出金后现金为负,请检查流水")
            else:
                gross = float(r["price"]) * int(r["shares"])
                fee_default = max(gross * config.ETF_COMMISSION_RATE, config.COMMISSION_MIN)
                if r["action"] == "buy":
                    cost = float(r["amount"]) if pd.notna(r["amount"]) else gross + fee_default
                    cash -= cost
                    pos[r["symbol"]] = pos.get(r["symbol"], 0) + int(r["shares"])
                    if cash < -0.01:
                        raise ValueError(f"{r['date'].date()} 买入 {r['symbol']} 后现金为负,"
                                         "请检查流水(是否漏记入金或金额多写)")
                else:
                    proceeds = float(r["amount"]) if pd.notna(r["amount"]) else gross - fee_default
                    cash += proceeds
                    pos[r["symbol"]] = pos.get(r["symbol"], 0) - int(r["shares"])
                    if pos[r["symbol"]] < 0:
                        raise ValueError(f"{r['date'].date()} 卖出 {r['symbol']} 后持仓为负,请检查流水")
            applied[i] = True
        eq = cash + sum(n * float(closes.at[day, s]) for s, n in pos.items() if n)
        equity.at[day] = eq
        # 入金/出金按日初折份额:此处 nav 仍为前一日 NAV(首日为 1.0),新资本
        # 据此计价并入份额,随后与既有份额一起承担当日涨跌 → 无盘中错配虚增。
        if flow_today != 0.0:
            units += flow_today / nav
            if units < -1e-9:
                raise ValueError(f"{day.date()} 出金超过账户份额,请检查流水")
        if units > 1e-9:
            nav = eq / units  # 当日收益由新老份额按比例分摊
        navs.at[day] = nav
    return equity, navs, net_deposit


def exec_table(execs: pd.DataFrame) -> str:
    """成交流水表(计划行灰显标注)"""
    act_cn = {"buy": "买入", "sell": "卖出", "deposit": "入金", "withdraw": "出金"}
    rows = ""
    for _, r in execs.iloc[::-1].head(30).iterrows():
        name = config.ETF_POOL.get(str(r.get("symbol", "")), "(池外)")
        price_s = "市价" if pd.isna(r.get("price")) else r["price"]
        detail = (f"{r['symbol']} {name} {price_s} x {int(r['shares'])}股"
                  if r["action"] in ("buy", "sell") else f"{float(r['amount']):,.0f} 元")
        note = "" if pd.isna(r.get("note")) else str(r["note"])
        planned = r["status"] == "计划"
        status = "待执行(计划)" if planned else "已成交"
        style = ' style="color:#999"' if planned else ""
        rows += (f'<tr{style}><td>{r["date"].date()}</td><td>{act_cn[r["action"]]}</td>'
                 f'<td>{html.escape(detail)}</td><td>{status}</td><td>{html.escape(note)}</td></tr>')
    return (f'<table><tr><th>日期</th><th>操作</th><th>明细</th><th>状态</th><th>备注</th></tr>'
            f'{rows}</table>')


def real_account_html(closes: pd.DataFrame, sim_equity: pd.Series, bench: pd.Series,
                      ew: pd.Series) -> str:
    """实盘板块 HTML;无记录给操作指引,解析失败给出错误而不中断整页生成。"""
    guide = ('<p class="note">记录方法:在 GitHub 编辑 <code>output/executions.csv</code>,'
             '列为 date,action,symbol,price,shares,amount,note,status。'
             '系统在有调仓信号时会自动写入 status=计划 的行(价格为昨收估算);'
             '你实际成交后把该行价格/股数改成真实值、status 改为 已成交(或留空)即可。'
             '计划行不参与净值计算,过期未确认的计划行会被下一次信号自动清除。'
             '入金如 <code>2026-06-12,入金,,,,10000,初始入金,</code>'
             '(amount 选填=券商实际发生金额,含手续费更准)。没操作就不动。</p>')
    try:
        execs = load_executions()
    except (ValueError, KeyError) as e:
        return (f'<h2>实盘 vs 模拟</h2><p style="color:#c00">executions.csv 解析失败:'
                f'{html.escape(str(e))}</p>{guide}')
    if execs is None:
        return f'<h2>实盘 vs 模拟</h2><p class="note">尚无实盘成交记录。</p>{guide}'

    confirmed = execs[execs["status"] != "计划"]
    if confirmed.empty:
        return (f'<h2>实盘 vs 模拟</h2><p class="note">尚无已成交记录(仅有待执行计划,'
                f'见下表)。</p>{exec_table(execs)}{guide}')

    # 池外标的(误买)的收盘价不在轮动 closes 表里,按需补列用于净值计算
    extra = sorted(set(confirmed.loc[confirmed["action"].isin(("buy", "sell")), "symbol"])
                   - set(closes.columns))
    if extra:
        closes = closes.copy()
        for s in extra:
            try:
                px = data.get_etf_daily(s, config.ROTATION_START,
                                        str(closes.index[-1].date()))["close"]
            except Exception as e:
                return (f'<h2>实盘 vs 模拟</h2><p style="color:#c00">池外标的 {s} 行情获取失败:'
                        f'{html.escape(str(e))}</p>{exec_table(execs)}{guide}')
            closes[s] = px.reindex(closes.index).ffill()

    try:
        real_eq, navs, net_deposit = real_equity_series(confirmed, closes)
    except ValueError as e:
        return (f'<h2>实盘 vs 模拟</h2><p style="color:#c00">实盘净值计算失败:'
                f'{html.escape(str(e))}</p>{guide}')

    start = real_eq.index[0]
    sim = sim_equity.loc[start:]
    b = bench.loc[start:].dropna()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(navs.index, navs, label="实盘账户(份额化净值,追加资金不扭曲)", linewidth=1.8, color="#d62728")
    if len(sim) >= 2:
        ax.plot(sim.index, sim / sim.iloc[0], label="模拟盘(策略理论执行)", linewidth=1.3,
                color="#ff9896", linestyle="--")
    if len(b) >= 2:
        ax.plot(b.index, b / b.iloc[0], label="沪深300(不操作)", linewidth=1.2, color="#7f7f7f")
    e = ew.loc[start:].dropna()
    if len(e) >= 2:
        ax.plot(e.index, e / e.iloc[0], label="ETF池等权(不操作)", linewidth=1.2,
                color="#1f77b4", alpha=0.8)
    ax.set_title(f"实盘净值对比(自 {start.date()},起始净值 1.0)")
    ax.legend(); ax.grid(alpha=0.3)
    img = fig_to_b64(fig)

    real_ret = navs.iloc[-1] - 1  # 份额化(TWR)收益,不受入金/出金时点影响
    sim_ret = sim.iloc[-1] / sim.iloc[0] - 1 if len(sim) >= 2 else float("nan")
    abs_pnl = float(real_eq.iloc[-1]) - net_deposit  # 绝对盈亏 = 口袋里的钱
    abs_ret = abs_pnl / net_deposit if net_deposit > 0 else float("nan")
    cards = (f'<div class="cards">'
             f'<div class="card"><div class="card-label">绝对盈亏(总资产-净入金)</div>'
             f'<div class="card-value {color_cls(abs_pnl)}">{abs_pnl:+,.0f} 元({pct(abs_ret)})</div></div>'
             f'<div class="card"><div class="card-label">当前总资产</div>'
             f'<div class="card-value">{real_eq.iloc[-1]:,.0f} 元</div></div>'
             f'<div class="card"><div class="card-label">累计净入金</div>'
             f'<div class="card-value">{net_deposit:,.0f} 元</div></div>'
             f'<div class="card"><div class="card-label">实盘累计收益(份额化)</div>'
             f'<div class="card-value {color_cls(real_ret)}">{pct(real_ret)}</div></div>'
             f'<div class="card"><div class="card-label">同期模拟盘</div>'
             f'<div class="card-value {color_cls(sim_ret)}">{pct(sim_ret)}</div></div>'
             f'<div class="card"><div class="card-label">执行偏差(实盘-模拟)</div>'
             f'<div class="card-value {color_cls(real_ret - sim_ret)}">{pct(real_ret - sim_ret)}</div></div></div>'
             f'<p class="note">「绝对盈亏」= 你实际赚/亏的钱(简单差额,大额入金晚到则早期涨幅贡献小);'
             f'「份额化收益」= 策略每份资金的历史表现(入金时点不影响),用于与模拟盘/基准公平对比。'
             f'两者可能一正一负,都是对的,回答的问题不同。</p>')

    return (f'<h2>实盘 vs 模拟</h2>{cards}'
            f'<img src="data:image/png;base64,{img}" alt="实盘净值">'
            f'<h3>成交流水(最近 30 条,含待执行计划)</h3>'
            f'{exec_table(execs)}'
            f'{guide}')


def replay_positions(confirmed: pd.DataFrame) -> tuple[dict[str, int], float]:
    """回放已成交流水,返回 (最终持仓 {symbol: shares(!=0)}, 现金余额)。

    口径与 real_equity_series 的账本回放严格一致:买入现金减 amount(缺则
    gross+费),卖出现金加 amount(缺则 gross-费),入金/出金直接加减现金,
    并沿用同款负现金/负持仓校验(异常流水抛 ValueError,交由上层吞掉)。
    仅用于展示当前持仓与策略目标的对照,不做净值折份额。
    """
    cash = 0.0
    pos: dict[str, int] = {}
    for _, r in confirmed.iterrows():
        if r["action"] == "deposit":
            cash += float(r["amount"])
        elif r["action"] == "withdraw":
            cash -= float(r["amount"])
            if cash < -0.01:
                raise ValueError(f"{r['date'].date()} 出金后现金为负,请检查流水")
        else:
            gross = float(r["price"]) * int(r["shares"])
            fee_default = max(gross * config.ETF_COMMISSION_RATE, config.COMMISSION_MIN)
            if r["action"] == "buy":
                cost = float(r["amount"]) if pd.notna(r["amount"]) else gross + fee_default
                cash -= cost
                pos[r["symbol"]] = pos.get(r["symbol"], 0) + int(r["shares"])
                if cash < -0.01:
                    raise ValueError(f"{r['date'].date()} 买入 {r['symbol']} 后现金为负,请检查流水")
            else:
                proceeds = float(r["amount"]) if pd.notna(r["amount"]) else gross - fee_default
                cash += proceeds
                pos[r["symbol"]] = pos.get(r["symbol"], 0) - int(r["shares"])
                if pos[r["symbol"]] < 0:
                    raise ValueError(f"{r['date'].date()} 卖出 {r['symbol']} 后持仓为负,请检查流水")
    return {s: n for s, n in pos.items() if n}, cash


def holdings_vs_target_html(closes: pd.DataFrame, weights: pd.DataFrame) -> str:
    """实际持仓 vs 策略目标对照 + 参考调整建议。

    实际持仓来自 executions.csv 已成交流水回放;策略目标为最新信号权重
    (与页面「最新信号」横幅同口径:当前线上 mode/vol-target/sleeve)。
    用户已选择维持现状,此段仅作提醒,不代表必须调仓。任何解析/回放/取价
    异常都返回空串,绝不中断整页生成(与其他 *_html 容错约定一致)。
    """
    try:
        execs = load_executions()
        if execs is None:
            return ""
        confirmed = execs[execs["status"] != "计划"]
        if confirmed.empty or closes.empty or weights.empty:
            return ""
        # 流水日期晚于最新行情 → 无法用当日收盘可靠计价(同 real_equity_series 口径),返回空串
        if (confirmed["date"] > closes.index[-1]).any():
            return ""
        pos, cash = replay_positions(confirmed)

        last = closes.iloc[-1]
        target = weights.iloc[-1]

        def finite(x) -> bool:
            try:
                return pd.notna(x) and math.isfinite(float(x))
            except (TypeError, ValueError):
                return False

        def price_of(s: str) -> float:
            if s in last.index and finite(last[s]):
                return float(last[s])
            return float("nan")

        # 任何非零持仓拿不到有限正价格 → 无法可靠计价,整段返回空串(不静默低估总资产)
        prices = {s: price_of(s) for s in pos}
        if any(not finite(px) or px <= 0 for px in prices.values()):
            return ""

        etf_value = sum(n * prices[s] for s, n in pos.items())
        total = etf_value + cash
        if not finite(total) or total <= 0:
            return ""

        # 100 股整手:对"调整量"本身向最近整手四舍五入(半手向上,避免银行家舍入),
        # 从而即便实际持仓非整手,给出的买/卖建议也一定是整手
        def round_lot(x: float) -> int:
            sign = 1 if x >= 0 else -1
            return sign * int(math.floor(abs(x) / 100 + 0.5) * 100)

        symbols = sorted(
            set(pos) | {s for s, w in target.items() if finite(w) and float(w) > 0.005},
            key=lambda s: -(float(target[s]) if s in target.index and finite(target[s]) else 0.0),
        )

        rows = ""
        for s in symbols:
            name = config.ETF_POOL.get(s, "(池外)")
            px = price_of(s)
            shares = pos.get(s, 0)
            tgt_pct = float(target[s]) if s in target.index and finite(target[s]) else 0.0
            if finite(px) and px > 0:
                value = shares * px
                actual_pct = value / total
                dev = actual_pct - tgt_pct
                value_s = f"{value:,.0f} 元"
                actual_s = pct(actual_pct, signed=False)
                dev_s = pct(dev)
                delta = round_lot(tgt_pct * total / px - shares)  # 目标股数 - 实际股数,整手
                if abs(delta) < 100:
                    adj = "维持"
                elif delta > 0:
                    adj = f"买入约 {delta:,} 股"
                else:
                    adj = f"卖出约 {-delta:,} 股"
            else:
                value_s = actual_s = dev_s = "-"
                adj = "无行情,暂无法计算"
            rows += (f'<tr><td>{html.escape(str(s))} {html.escape(str(name))}</td>'
                     f'<td>{shares:,} 股</td><td>{value_s}</td><td>{actual_s}</td>'
                     f'<td>{pct(tgt_pct, signed=False)}</td><td>{dev_s}</td><td>{adj}</td></tr>')
        if cash > 0.5:
            rows += (f'<tr><td>现金</td><td>-</td><td>{cash:,.0f} 元</td>'
                     f'<td>{pct(cash / total, signed=False)}</td><td>-</td><td>-</td><td>-</td></tr>')

        return (f'<h2>实际持仓 vs 策略目标(据此决定是否手动调仓)</h2>'
                f'<p class="note">「实际持仓」来自已成交流水,「策略目标」为最新信号权重'
                f'(当前线上口径)。你已选择<b>维持现状</b>,下表仅作提醒:若要对齐策略,'
                f'可参考「参考调整」列手动下单(T+1 开盘、100 股整手;偏离在 2% 总资产以内可忽略)。'
                f'总资产 {total:,.0f} 元。</p>'
                f'<table><tr><th>标的</th><th>实际持仓</th><th>实际市值</th><th>实际占比</th>'
                f'<th>策略目标占比</th><th>偏离</th><th>参考调整</th></tr>{rows}</table>')
    except Exception:
        return ""


ASSETS_DIR = "assets"


def strategy_explainer_html(mode: str, vol_target: bool, sleeve: bool) -> str:
    """页首策略说明:用大白话解释本策略是什么、各版本(V0/V1/V2)的含义。"""
    if vol_target and sleeve:
        live = "V2"
    elif vol_target:
        live = "V1"
    elif sleeve:
        live = "V0+防御sleeve(非标准组合)"
    else:
        live = "V0"
    mode_txt = ("三个周期(15/20/25日)各自选最强、结果平均(ensemble,权重会出现 1/3、2/3 等分数)"
                if mode == "ensemble" else "单一 20 日周期选最强、整仓切换(single)")
    return f"""
<h2>本策略是什么(先读我)</h2>
<div style="background:#fff;border-radius:8px;padding:14px 18px;box-shadow:0 1px 3px rgba(0,0,0,.08);font-size:.92em;line-height:1.7">
<p><b>核心思路(动量轮动)</b>:每天收盘后,在 9 只 ETF(沪深300/中证500/创业板/纳指/标普/恒生/黄金/国债/中证1000)里比较"过去一段时间谁涨得最多",持有最强者,次日开盘调仓;若所有候选近期都在跌,则空仓避险。当前口径:{mode_txt}。</p>
<p><b>版本演进</b>(图表中的 V0/V1/V2,<b style="color:#d62728">红色曲线永远是当前最新版</b>):</p>
<table>
<tr><th>版本</th><th>做了什么</th><th>解决什么问题</th></tr>
<tr><td>V0 基线</td><td>纯动量轮动(如上)</td><td>捕捉趋势,但大跌时回撤深(约 -25%)</td></tr>
<tr><td>V1 +波动率目标</td><td>当组合近期波动明显高于自身历史水平时,按比例降低仓位(剩余持现金),波动回落后自动加回</td><td>动量策略的崩盘集中在高波动期 → 回撤 -25.5%→-18.4%,收益基本不变</td></tr>
<tr><td>V2 +防御sleeve</td><td>V1 降仓/空仓留下的闲置现金,按各半买入黄金ETF+国债ETF,不再干躺</td><td>现金零收益 → 债吃利息、金对冲股票熊市,年化 +1.5pp</td></tr>
</table>
<p class="note">当前线上运行:<b>{live}</b>。执行规则:信号用 T 日收盘计算,T+1 日开盘成交(无未来数据);
回测含佣金(ETF 万0.5、最低5元)、滑点、100股整手约束。下方对比基准"不操作"=期初买入后一直持有。</p>
</div>
"""


def variants_comparison_html() -> str:
    """策略演进对比段:V0 基线 → V1 波动率目标 → V2 防御 sleeve(最新,红色)。

    静态研究记录(图与数字为一次性回测固定结果,见 scripts/experiment_sleeve.py);
    资产缺失时返回空串,不影响整页生成。
    """
    img_path = os.path.join(ASSETS_DIR, "compare_variants.png")
    if not os.path.exists(img_path):
        return ""
    with open(img_path, "rb") as f:
        img = base64.b64encode(f.read()).decode()
    return (
        '<h2>策略演进:各版本实验结果对比</h2>'
        '<p class="note">同窗口(2017-08-24 ~ 2026-07-01)、同引擎、11万本金真实费用。'
        '<b style="color:#d62728">红色 = 当前最新版 V2</b>(波动率目标 + 防御 sleeve)。</p>'
        '<table><tr><th>版本</th><th>年化</th><th>夏普</th><th>最大回撤</th>'
        '<th>样本外年化</th><th>样本外夏普</th></tr>'
        '<tr><td>V0 基线 ensemble</td><td>16.8%</td><td>0.75</td><td>-25.5%</td>'
        '<td>27.9%</td><td>1.05</td></tr>'
        '<tr><td>V1 +波动率目标</td><td>16.9%</td><td>0.96</td><td>-18.4%</td>'
        '<td>28.8%</td><td>1.40</td></tr>'
        '<tr style="color:#d62728;font-weight:bold"><td>V2 +防御sleeve(金债各半)【最新】</td>'
        '<td>18.4%</td><td>1.00</td><td>-18.4%</td><td>32.2%</td><td>1.47</td></tr></table>'
        f'<img src="data:image/png;base64,{img}" alt="策略演进对比">'
        '<p class="note">V1 波动率目标:高波动期降仓,回撤 -25.5%→-18.4%,夏普 0.75→0.96'
        '(收益基本不变,赚在"稳")。V2 防御 sleeve:滤空/降仓留下的闲置现金按金债各半路由'
        '(518880 黄金 + 511260 国债),零参数;年化 +1.5pp、样本外 +3.4pp,'
        '黄金在 2018/2022 股票熊年亦为正贡献。样本外 = 2022-01-01 起。</p>'
        f'{sleeve_rules_html()}'
    )


def sleeve_rules_html() -> str:
    """防御 sleeve 规则筛选子段:四种残余现金路由方式的对比与选型理由。

    静态研究记录(scripts/experiment_sleeve.py);资产缺失时返回空串。
    """
    img_path = os.path.join(ASSETS_DIR, "compare_sleeve_rules.png")
    if not os.path.exists(img_path):
        return ""
    with open(img_path, "rb") as f:
        img = base64.b64encode(f.read()).decode()
    return (
        '<h3>附:sleeve 规则筛选(为什么选"金债各半")</h3>'
        '<table><tr><th>路由规则</th><th>年化</th><th>夏普</th><th>最大回撤</th>'
        '<th>样本外年化</th><th>样本外夏普</th></tr>'
        '<tr><td>V1 无 sleeve(对照)</td><td>16.9%</td><td>0.96</td><td>-18.4%</td>'
        '<td>28.8%</td><td>1.40</td></tr>'
        '<tr><td>全债 511260</td><td>17.3%</td><td>0.97</td><td>-18.6%</td>'
        '<td>29.3%</td><td>1.41</td></tr>'
        '<tr><td>全金 518880</td><td>19.5%</td><td>1.00</td><td>-18.2%</td>'
        '<td>35.1%</td><td>1.49</td></tr>'
        '<tr><td>金债择优(20日动量)</td><td>17.8%</td><td>0.95</td><td>-18.6%</td>'
        '<td>33.0%</td><td>1.45</td></tr>'
        '<tr style="color:#d62728;font-weight:bold"><td>金债各半【选定】</td>'
        '<td>18.4%</td><td>1.00</td><td>-18.4%</td><td>32.2%</td><td>1.47</td></tr></table>'
        f'<img src="data:image/png;base64,{img}" alt="sleeve规则筛选对比">'
        '<p class="note">选型理由:"全金"账面最好,但领先集中在 2024-25 黄金牛市'
        '(2025 单年比 V1 多 10.8pp),有吃行情嫌疑,单押未来黄金走势;"全债"最保守'
        '(纯利息 carry)但增益最小;"择优"引入动量参数且分年不稳(2020 落后 4.8pp)。'
        '"金债各半"零参数、sleeve 内部自分散,黄金在 2018/2022 股票熊年为正贡献'
        '(非纯牛市运气),夏普与全金并列最高 → 按抗过拟合原则选定。</p>'
    )


def etrade_comparison_html() -> str:
    """一次性研究对比段:本策略 vs 第三方 eTrade Dynamic TopN V2。

    静态内容(不进 CI、不每日重算、不 vendoring 第三方策略源):图片为预先
    生成、提交在 assets/ 下的同窗口归一化净值对比;数字为该次回测的固定结果。
    资产缺失时返回空串,不影响整页生成。
    """
    img_path = os.path.join(ASSETS_DIR, "compare_etrade.png")
    if not os.path.exists(img_path):
        return ""
    with open(img_path, "rb") as f:
        img = base64.b64encode(f.read()).decode()
    cards = (
        '<div class="cards">'
        '<div class="card"><div class="card-label">本策略 年化</div>'
        '<div class="card-value pos">34.8%</div></div>'
        '<div class="card"><div class="card-label">eTrade 年化</div>'
        '<div class="card-value pos">32.1%</div></div>'
        '<div class="card"><div class="card-label">本策略 夏普</div>'
        '<div class="card-value">1.19</div></div>'
        '<div class="card"><div class="card-label">eTrade 夏普</div>'
        '<div class="card-value">1.03</div></div>'
        '<div class="card"><div class="card-label">本策略 最大回撤</div>'
        '<div class="card-value neg">-25.5%</div></div>'
        '<div class="card"><div class="card-label">eTrade 最大回撤</div>'
        '<div class="card-value neg">-32.5%</div></div>'
        '</div>'
    )
    return (
        '<h2>研究对比:本策略 vs 第三方 eTrade Dynamic TopN V2</h2>'
        '<p class="note">一次性研究记录(非实时跟踪)。用本仓库同一套数据管道、独立复现'
        ' eTrade 公开代码(13 只跨境/商品/债券交易池),回测区间 2023-01-03 ~ 2026-06-26'
        ',与本策略 ensemble 在同窗口、同口径对齐归一化对比。</p>'
        f'{cards}'
        f'<img src="data:image/png;base64,{img}" alt="本策略 vs eTrade 对比">'
        '<p class="note">要点:① 同窗口下本策略四项(收益/年化/夏普/回撤)全胜;'
        '② eTrade 公开代码独立复现的最大回撤 -32.5%、年化波动 32%,显著高于其报告宣称的'
        ' -9% / 18.6% 年化——其低回撤数字疑似来自挑选的验证窗口与其自有 parquet 数据,'
        '不可在独立数据上复现;③ 值得借鉴的是其防御性 sleeve(金/债)、多因子打分与'
        ' TopN≥2 分散的思路,而非其偏门 QDII/商品交易池(有溢价/限购的实盘摩擦,'
        '且其回测未建模最低佣金/整手)。</p>'
    )


def main():
    parser = argparse.ArgumentParser(description="生成静态网页报告")
    parser.add_argument("--mode", choices=("single", "ensemble"), default="single")
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--end", default=dt.date.today().isoformat())
    parser.add_argument("--vol-target", action=argparse.BooleanOptionalAction,
                        default=config.VOL_TARGET_ENABLED,
                        help="波动率目标覆盖层(默认随 config.VOL_TARGET_ENABLED),需与线上口径一致")
    parser.add_argument("--sleeve", action=argparse.BooleanOptionalAction,
                        default=config.SLEEVE_ENABLED,
                        help="防御 sleeve(默认随 config.SLEEVE_ENABLED),需与线上口径一致")
    args = parser.parse_args()

    prices = load_pool(config.ROTATION_START, args.end)
    closes = closes_table(prices)
    weights = build_weights(closes, mode=args.mode, lookback=config.ROTATION_LOOKBACK,
                            buffer=config.ROTATION_BUFFER, dd_control=False,
                            vol_control=args.vol_target, sleeve=args.sleeve)
    result = run_portfolio_backtest(prices, weights, initial_capital=args.capital, stamp_tax=False)
    equity = result.equity

    # 历史版本变体曲线(近一年图用):仅当线上开了覆盖层才有对比意义
    variant_extras = []
    if args.vol_target or args.sleeve:
        w_v0 = build_weights(closes, mode=args.mode, lookback=config.ROTATION_LOOKBACK,
                             buffer=config.ROTATION_BUFFER, dd_control=False)
        eq_v0 = run_portfolio_backtest(prices, w_v0, initial_capital=args.capital,
                                       stamp_tax=False).equity
        variant_extras.append(("V0 基线(无覆盖层)", eq_v0))
    if args.vol_target and args.sleeve:
        w_v1 = build_weights(closes, mode=args.mode, lookback=config.ROTATION_LOOKBACK,
                             buffer=config.ROTATION_BUFFER, dd_control=False, vol_control=True)
        eq_v1 = run_portfolio_backtest(prices, w_v1, initial_capital=args.capital,
                                       stamp_tax=False).equity
        variant_extras.append(("V1 +波动率目标", eq_v1))

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
    w = ew.dropna()
    m_ew = metrics_mod.equity_metrics(w)
    m_ew_oos = metrics_mod.equity_metrics(w.loc[config.OOS_SPLIT:])

    # 图表
    img_full = build_equity_chart(equity, bench, ew,
                                  f"全区间净值对比({equity.index[0].date()} ~ {equity.index[-1].date()},对数坐标)",
                                  log_scale=True)
    one_year = equity.index[-1] - pd.Timedelta(days=365)
    if args.vol_target and args.sleeve:
        main_label_1y = "V2 本策略(波动目标+防御sleeve)【最新】"
    elif args.vol_target:
        main_label_1y = "V1 本策略(+波动率目标)【最新】"
    elif args.sleeve:
        main_label_1y = "本策略(+防御sleeve)【最新】"
    else:
        main_label_1y = "ETF动量轮动(本策略)"
    img_1y = build_equity_chart(equity.loc[one_year:], bench.loc[one_year:], ew.loc[one_year:],
                                "近一年净值对比(含历史版本变体)" if variant_extras else "近一年净值对比",
                                log_scale=False,
                                extras=[(lb, s.loc[one_year:]) for lb, s in variant_extras],
                                main_label=main_label_1y)

    # 操作明细(近一年)
    segs = holding_segments(weights, equity, bench, ew)
    recent_segs = [r for r in segs if pd.Timestamp(r["end"]) >= one_year]
    seg_rows = "".join(
        f'<tr><td>{r["start"]} ~ {r["end"]}</td><td>{html.escape(r["label"])}</td>'
        f'<td>{r["days"]}</td>'
        f'<td class="{color_cls(r["ret"])}">{pct(r["ret"])}</td>'
        f'<td class="{color_cls(r["bench_ret"])}">{pct(r["bench_ret"])}</td>'
        f'<td class="{color_cls(r["ret"] - r["bench_ret"])}">{pct(r["ret"] - r["bench_ret"])}</td>'
        f'<td class="{color_cls(r["ew_ret"])}">{pct(r["ew_ret"])}</td>'
        f'<td class="{color_cls(r["ret"] - r["ew_ret"])}">{pct(r["ret"] - r["ew_ret"])}</td></tr>'
        for r in reversed(recent_segs)
    )
    seg_wins = sum(1 for r in segs if pd.notna(r["bench_ret"]) and r["ret"] > r["bench_ret"])
    seg_wins_ew = sum(1 for r in segs if pd.notna(r["ew_ret"]) and r["ret"] > r["ew_ret"])
    seg_total = sum(1 for r in segs if pd.notna(r["bench_ret"]))
    seg_total_ew = sum(1 for r in segs if pd.notna(r["ew_ret"]))

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
    edge_ew_oos = (m_oos["年化收益率"] - m_ew_oos["年化收益率"]) if m_oos else float("nan")
    beat_300 = pd.notna(edge_oos) and edge_oos > 0
    beat_ew = pd.notna(edge_ew_oos) and edge_ew_oos > 0
    if beat_300 and beat_ew:
        verdict = "策略当前优于不操作(同时跑赢沪深300与ETF池等权)"
    elif beat_300:
        verdict = "策略跑赢沪深300、但未跑赢ETF池等权"
    elif beat_ew:
        verdict = "策略跑赢ETF池等权、但未跑赢沪深300"
    else:
        verdict = "策略当前未跑赢不操作"

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

{strategy_explainer_html(args.mode, args.vol_target, args.sleeve)}

{real_account_html(closes, equity, bench, ew)}

{holdings_vs_target_html(closes, weights)}

<div class="banner">
  <div class="big">最新信号:{html.escape(latest)}</div>
  <div style="margin-top:6px">{verdict}:样本外({config.OOS_SPLIT} 至今)策略年化
  <b class="{color_cls(m_oos['年化收益率']) if m_oos else ''}">{pct(m_oos['年化收益率'], signed=False) if m_oos else '-'}</b>
  vs 沪深300买入持有 <b>{pct(m_bench_oos['年化收益率'], signed=False)}</b>
  vs ETF池等权持有 <b>{pct(m_ew_oos['年化收益率'], signed=False)}</b>
  (超额沪深300 <b class="{color_cls(edge_oos)}">{pct(edge_oos)}</b>/年,
  超额等权 <b class="{color_cls(edge_ew_oos)}">{pct(edge_ew_oos)}</b>/年)</div>
</div>

<h2>净值对比:操作 vs 不操作</h2>
<img src="data:image/png;base64,{img_full}" alt="全区间净值">
<img src="data:image/png;base64,{img_1y}" alt="近一年净值">

<h2>策略操作明细(近一年,每次切换为一次操作)</h2>
<p class="note">每段持仓的区间收益与同期基准对比。全历史跑赢同期沪深300的有 {seg_wins}/{seg_total} 段({seg_wins / seg_total:.0%}),跑赢同期ETF池等权的有 {seg_wins_ew}/{seg_total_ew} 段({seg_wins_ew / seg_total_ew:.0%})。</p>
<table>
<tr><th>持仓区间</th><th>持有标的</th><th>天数</th><th>本段收益</th><th>同期沪深300</th><th>超额(vs沪深300)</th><th>同期等权</th><th>超额(vs等权)</th></tr>
{seg_rows}
</table>

<h2>绩效指标</h2>
{metric_cards(f"策略 · 全区间({equity.index[0].date()} 起)", m_all)}
{metric_cards(f"策略 · 样本外({config.OOS_SPLIT} 起,更接近真实预期)", m_oos) if m_oos else ""}
{metric_cards("沪深300 买入持有 · 全区间(不操作的对照)", m_bench)}
{metric_cards("ETF池等权 买入持有 · 全区间(持有全部标的不轮动的对照)", m_ew)}

<h2>模拟盘信号日志(最近 15 条)</h2>
<table>
<tr><th>信号日期</th><th>模式</th><th>目标持仓</th></tr>
{log_rows or '<tr><td colspan="3">暂无记录</td></tr>'}
</table>

{variants_comparison_html()}
{etrade_comparison_html()}

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
