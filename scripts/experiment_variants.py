"""三版本对比(V0 基线 / V1 波动率目标 / V2 防御 sleeve)一次性复现脚本。

README「当前线上配置与三版本对比」表的唯一生成入口。三行共用同一口径:
同一份行情、同一 buffer、同一本金、同一回测引擎,只切换两层覆盖层的开关。

与 `run_rotation.py` 的差异(为什么单独一个脚本):
  - 强制 `qfq_only=True`,拒绝新浪不复权回退,避免复权口径混入;
  - `write_cache=False`,研究运行绝不污染 `data/` 缓存;
  - 打印数据指纹(开盘价 + 收盘价),复现时先核对指纹再比对数字;
  - 样本外指标把区间起点前一个交易日纳入作基准(见 experiment_buffer.seg_slice),
    `run_rotation.py` 直接切片,样本外数字会因少算首日而略低。

用法:
  python scripts/experiment_variants.py [--capital 185000] [--end 2026-08-12]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from experiment_buffer import fingerprint, seg_metrics
from portfolio import run_portfolio_backtest
from run_rotation import build_weights, closes_table, load_pool

VARIANTS = (
    ("V0 基线 ensemble", False, False),
    ("V1 +波动率目标", True, False),
    ("V2 +防御sleeve(当前)", True, True),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=config.ROTATION_START)
    ap.add_argument("--end", default="2026-08-12")   # 研究截止日,含
    ap.add_argument("--capital", type=float, default=185_000.0)
    args = ap.parse_args()

    prices = load_pool(args.start, args.end, write_cache=False, qfq_only=True)
    closes = closes_table(prices)
    print(f"\n共同日历 {closes.index[0].date()} ~ {closes.index[-1].date()} "
          f"({len(closes)} 个交易日)")
    print(f"数据指纹 {fingerprint(prices, closes)}  "
          f"buffer={config.ROTATION_BUFFER}  资金 {args.capital:,.0f}")
    print(f"\n{'版本':<24} {'年化':>8} {'夏普':>6} {'回撤':>8} "
          f"{'OOS年化':>9} {'OOS夏普':>8} {'OOS回撤':>8}")
    for label, vol_control, sleeve in VARIANTS:
        w = build_weights(closes, mode="ensemble", lookback=config.ROTATION_LOOKBACK,
                          buffer=config.ROTATION_BUFFER, dd_control=False,
                          vol_control=vol_control, sleeve=sleeve)
        eq = run_portfolio_backtest(prices, w, initial_capital=args.capital,
                                    stamp_tax=False).equity
        f = seg_metrics(eq)
        o = seg_metrics(eq, start=config.OOS_SPLIT)
        print(f"{label:<24} {f['年化收益率']:>8.2%} {f['夏普比率']:>6.2f} "
              f"{f['最大回撤']:>8.1%} {o['年化收益率']:>9.2%} "
              f"{o['夏普比率']:>8.2f} {o['最大回撤']:>8.1%}")


if __name__ == "__main__":
    main()
