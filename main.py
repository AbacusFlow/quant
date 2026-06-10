"""A股双均线策略回测 CLI 入口

用法:
  python main.py --symbol 600519 --start 2020-01-01 --end 2025-12-31 [--short 5 --long 20]
"""
import argparse

import config
import data
import metrics as metrics_mod
import report
import strategy
from backtest import run_backtest


def main():
    parser = argparse.ArgumentParser(description="A股双均线策略回测")
    parser.add_argument("--symbol", default=config.DEFAULT_SYMBOL, help="股票代码,如 600519")
    parser.add_argument("--start", default=config.DEFAULT_START, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=config.DEFAULT_END, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--short", type=int, default=config.MA_SHORT, help="短均线周期")
    parser.add_argument("--long", type=int, default=config.MA_LONG, help="长均线周期")
    args = parser.parse_args()

    print(f"拉取 {args.symbol} 日线数据 ({args.start} ~ {args.end}) ...")
    df = data.get_stock_daily(args.symbol, args.start, args.end)
    print(f"共 {len(df)} 个交易日")

    print("拉取沪深300基准数据 ...")
    benchmark = data.get_benchmark_daily(args.start, args.end)

    position = strategy.dual_ma_signal(df, args.short, args.long)
    result = run_backtest(df, position)

    trades_df = result.trades_df()
    m = metrics_mod.compute_metrics(result.equity, trades_df)
    report.print_metrics(m, f"MA{args.short}/MA{args.long}", args.symbol)
    report.plot_equity(result.equity, benchmark, args.symbol)
    report.save_trades(trades_df)

    print("\n提示: 回测结果不代表未来收益。")


if __name__ == "__main__":
    main()
