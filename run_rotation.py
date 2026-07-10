"""ETF 动量轮动策略回测入口。

用法:
  python run_rotation.py [--start 2015-01-01] [--end 2026-06-09] [--lookback 20]
  python run_rotation.py --sensitivity   # 参数敏感性扫描

输出:全区间 / 样本内 / 样本外指标,净值曲线,交易记录。
"""
import argparse
import datetime as dt
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
import data
import metrics as metrics_mod
import strategy
from portfolio import align_prices, run_portfolio_backtest

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_pool(start: str, end: str, write_cache: bool = True) -> dict[str, pd.DataFrame]:
    prices = {}
    for symbol, name in config.ETF_POOL.items():
        print(f"拉取 {name}({symbol}) ...")
        prices[symbol] = data.get_etf_daily(symbol, start, end, write_cache=write_cache)
    return prices


def closes_table(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """收盘价表,只保留所有标的都有行情的共同交易日,与回测引擎日历一致"""
    common = align_prices(prices)
    return pd.DataFrame({s: df["close"] for s, df in prices.items()}).loc[common]


def report_segment(equity: pd.Series, label: str) -> dict | None:
    if len(equity) < 2:
        print(f"\n--- {label}: 区间内无足够数据,跳过 ---")
        return None
    m = metrics_mod.equity_metrics(equity)
    print(f"\n--- {label} ({equity.index[0].date()} ~ {equity.index[-1].date()}) ---")
    for k, v in m.items():
        print(f"  {k:<8}: {v:>10.2%}" if k != "夏普比率" else f"  {k:<8}: {v:>10.2f}")
    return m


def build_weights(closes: pd.DataFrame, mode: str, lookback: int, buffer: float,
                  dd_control: bool, vol_control: bool = False,
                  sleeve: bool = False) -> pd.DataFrame:
    """根据模式生成目标权重(可选叠加回撤控制、波动率目标、防御 sleeve)。

    叠加顺序:回撤控制(dd_control)→ 波动率目标(vol_control)→ 防御 sleeve
    (sleeve 最后:把前两层留下的现金空档路由到金/债)。
    """
    if mode == "single":
        weights = strategy.etf_momentum_rotation(closes, lookback=lookback, buffer=buffer)
    else:
        weights = strategy.etf_momentum_ensemble(closes, lookbacks=config.ENSEMBLE_LOOKBACKS, buffer=buffer)
    if dd_control:
        weights = strategy.apply_drawdown_control(
            weights, closes, ma_window=config.DD_MA_WINDOW, scale=config.DD_SCALE)
    if vol_control:
        weights = strategy.apply_vol_targeting(
            weights, closes, lookback=config.VOL_TARGET_LOOKBACK)
    if sleeve:
        weights = strategy.apply_defensive_sleeve(weights, closes)
    return weights


def run_once(prices: dict[str, pd.DataFrame], lookback: int, buffer: float,
             mode: str = "single", dd_control: bool = False,
             capital: float = config.INITIAL_CAPITAL, vol_control: bool = False) -> tuple:
    closes = closes_table(prices)
    weights = build_weights(closes, mode, lookback, buffer, dd_control, vol_control)
    result = run_portfolio_backtest(prices, weights, initial_capital=capital, stamp_tax=False)
    return result, weights


def main():
    parser = argparse.ArgumentParser(description="ETF 动量轮动回测")
    parser.add_argument("--start", default=config.ROTATION_START)
    parser.add_argument("--end", default=dt.date.today().isoformat())
    parser.add_argument("--lookback", type=int, default=config.ROTATION_LOOKBACK)
    parser.add_argument("--buffer", type=float, default=config.ROTATION_BUFFER)
    parser.add_argument("--capital", type=float, default=config.INITIAL_CAPITAL,
                        help="初始资金(元),小资金下最低佣金与整手约束影响显著")
    parser.add_argument("--sensitivity", action="store_true", help="lookback 参数敏感性扫描")
    parser.add_argument("--mode", choices=("single", "ensemble"), default="ensemble",
                        help="single=单一lookback, ensemble=多周期集成(默认)")
    parser.add_argument("--dd", action="store_true",
                        help="开启回撤控制(回测显示年化损耗约3%%、回撤仅改善约2pp,默认关闭)")
    parser.add_argument("--vol-target", action="store_true",
                        help="开启波动率目标覆盖层(研究显示夏普↑/回撤↓、年化基本不变,默认关闭)")
    parser.add_argument("--compare", action="store_true", help="对比 单一/集成/集成+回撤控制")
    args = parser.parse_args()

    prices = load_pool(args.start, args.end)

    if args.compare:
        print("\n========== 策略变体对比 ==========")
        variants = [
            (f"单一lookback{args.lookback}", "single", False),
            ("集成15/20/25", "ensemble", False),
            ("集成+回撤控制", "ensemble", True),
        ]
        rows = []
        for label, mode, dd in variants:
            result, _ = run_once(prices, args.lookback, args.buffer, mode=mode, dd_control=dd,
                                 capital=args.capital, vol_control=args.vol_target)
            m = metrics_mod.equity_metrics(result.equity)
            oos = result.equity.loc[config.OOS_SPLIT:]
            m_oos = metrics_mod.equity_metrics(oos) if len(oos) >= 2 else None
            rows.append({
                "策略": label,
                "全区间年化": f"{m['年化收益率']:.2%}",
                "全区间回撤": f"{m['最大回撤']:.2%}",
                "夏普": f"{m['夏普比率']:.2f}",
                "样本外年化": f"{m_oos['年化收益率']:.2%}" if m_oos else "-",
                "样本外回撤": f"{m_oos['最大回撤']:.2%}" if m_oos else "-",
                "交易次数": len(result.trades),
            })
        print(pd.DataFrame(rows).to_string(index=False))
        return

    if args.sensitivity:
        # 扫描单一 lookback 的敏感性,固定 single 模式(ensemble 的 lookback 组合是固定的,扫描无意义)
        print(f"\n========== 参数敏感性: lookback 扫描 (mode=single) ==========")
        rows = []
        for lb in (10, 15, 20, 25, 30, 40, 60):
            result, _ = run_once(prices, lb, args.buffer, mode="single", capital=args.capital,
                                 vol_control=args.vol_target)
            m = metrics_mod.equity_metrics(result.equity)
            oos = result.equity.loc[config.OOS_SPLIT:]
            m_oos = metrics_mod.equity_metrics(oos) if len(oos) >= 2 else None
            rows.append({
                "lookback": lb,
                "全区间年化": f"{m['年化收益率']:.2%}",
                "全区间回撤": f"{m['最大回撤']:.2%}",
                "夏普": f"{m['夏普比率']:.2f}",
                "样本外年化": f"{m_oos['年化收益率']:.2%}" if m_oos else "-",
                "样本外回撤": f"{m_oos['最大回撤']:.2%}" if m_oos else "-",
                "交易次数": len(result.trades),
            })
        print(pd.DataFrame(rows).to_string(index=False))
        return

    dd_control = args.dd
    result, weights = run_once(prices, args.lookback, args.buffer, mode=args.mode, dd_control=dd_control,
                               capital=args.capital, vol_control=args.vol_target)
    equity = result.equity

    desc = (f"mode={args.mode}, 回撤控制={'开' if dd_control else '关'}, "
            f"波动率目标={'开' if args.vol_target else '关'}, buffer={args.buffer}, 本金={args.capital:,.0f}")
    if args.mode == "single":
        desc += f", lookback={args.lookback}"
    print(f"\n========== ETF 动量轮动 ({desc}) ==========")
    report_segment(equity, "全区间")
    report_segment(equity.loc[: config.OOS_SPLIT], "样本内")
    report_segment(equity.loc[config.OOS_SPLIT:], "样本外")

    # 基准:沪深300
    benchmark = data.get_benchmark_daily(args.start, args.end)
    bench = benchmark["close"].reindex(equity.index).ffill()
    report_segment(bench.dropna(), "沪深300基准")

    # 净值曲线
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity.index, equity / equity.iloc[0], label="ETF动量轮动", linewidth=1.5)
    norm_bench = bench / bench.dropna().iloc[0]
    ax.plot(norm_bench.index, norm_bench, label="沪深300", linewidth=1.2, alpha=0.8)
    ax.axvline(pd.Timestamp(config.OOS_SPLIT), color="gray", linestyle="--", alpha=0.6, label="样本外分割")
    ax.set_title("ETF动量轮动 vs 沪深300")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(config.OUTPUT_DIR, "rotation_equity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n净值曲线已保存: {path}")

    trades_df = result.trades_df()
    trades_path = os.path.join(config.OUTPUT_DIR, "rotation_trades.csv")
    trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
    print(f"交易记录已保存: {trades_path} (共 {len(trades_df)} 笔)")

    # 当前持仓信号(用于实盘跟踪)
    last_w = weights.iloc[-1]
    held = last_w[last_w > 0]
    if held.empty:
        print(f"\n最新信号 ({weights.index[-1].date()}): 空仓持现金")
    else:
        names = ", ".join(f"{config.ETF_POOL[s]}({s}) {w:.0%}" for s, w in held.items())
        cash_w = 1 - held.sum()
        suffix = f",现金 {cash_w:.0%}" if cash_w > 0.005 else ""
        print(f"\n最新信号 ({weights.index[-1].date()}): {names}{suffix}")

    print("\n提示: 回测结果不代表未来收益。")


if __name__ == "__main__":
    main()
