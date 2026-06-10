"""结果输出:终端指标表、净值曲线图、交易记录 CSV"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

# 中文字体(Docker 镜像内置 Noto CJK)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def print_metrics(metrics: dict, strategy_name: str, symbol: str):
    print(f"\n========== 回测结果: {strategy_name} ({symbol}) ==========")
    for k, v in metrics.items():
        if isinstance(v, float):
            if k in ("总收益率", "年化收益率", "最大回撤", "胜率"):
                print(f"  {k:<10}: {v:>10.2%}")
            else:
                print(f"  {k:<10}: {v:>10.2f}")
        else:
            print(f"  {k:<10}: {v:>10}")
    print("=" * 50)


def plot_equity(equity: pd.Series, benchmark: pd.DataFrame, symbol: str):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    norm_strategy = equity / equity.iloc[0]

    bench = benchmark["close"].reindex(equity.index).ffill()
    norm_bench = bench / bench.dropna().iloc[0]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(norm_strategy.index, norm_strategy, label=f"双均线策略 ({symbol})", linewidth=1.5)
    ax.plot(norm_bench.index, norm_bench, label="沪深300", linewidth=1.2, alpha=0.8)
    ax.set_title("策略净值 vs 沪深300")
    ax.set_ylabel("归一化净值")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(config.OUTPUT_DIR, "equity_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"净值曲线已保存: {path}")


def save_trades(trades_df: pd.DataFrame):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, "trades.csv")
    trades_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"交易记录已保存: {path} (共 {len(trades_df)} 笔)")
