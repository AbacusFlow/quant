"""防御 sleeve 实验:残余现金(绝对动量滤空 + 波动目标降仓)路由到防御资产。

对比变体(同窗口、同引擎、真实费用):
  base      : ensemble(15/20/25)                       —— 旧基线
  vt        : ensemble + 波动率目标                     —— 当前线上
  vt+bond   : vt 残余现金 → 511260 十年国债
  vt+gold   : vt 残余现金 → 518880 黄金
  vt+best   : vt 残余现金 → 金/债中 20 日动量更高且为正者,否则现金
  vt+split  : vt 残余现金 → 金债各半

评估:全区间 / 样本外(config.OOS_SPLIT) 年化、夏普、最大回撤 + 分年收益。
用法: python scripts/experiment_sleeve.py [--end 2026-07-01] [--capital 110000]
"""
import argparse
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
import metrics as metrics_mod
from portfolio import run_portfolio_backtest
from run_rotation import build_weights, closes_table, load_pool

DEFENSIVE = {"gold": "518880", "bond": "511260"}


def apply_sleeve(weights: pd.DataFrame, closes: pd.DataFrame, rule: str,
                 mom_lookback: int = 20) -> pd.DataFrame:
    """残余现金路由到防御资产(因果:T 日仅用 ≤T 收盘价,引擎再 shift 到 T+1)。

    residual = 1 - row_sum(≥0)。防御资产若已被动量选中,只增配 residual 部分。
    """
    w = weights.copy()
    residual = (1.0 - w.sum(axis=1)).clip(lower=0.0)
    gold, bond = DEFENSIVE["gold"], DEFENSIVE["bond"]
    if rule == "bond":
        w[bond] = w[bond] + residual
    elif rule == "gold":
        w[gold] = w[gold] + residual
    elif rule == "split":
        w[gold] = w[gold] + residual / 2
        w[bond] = w[bond] + residual / 2
    elif rule == "best":
        mom = closes[[gold, bond]].pct_change(mom_lookback, fill_method=None)
        pick_gold = (mom[gold] >= mom[bond]) & (mom[gold] > 0)
        pick_bond = (~pick_gold) & (mom[bond] > 0)
        w[gold] = w[gold] + residual.where(pick_gold, 0.0)
        w[bond] = w[bond] + residual.where(pick_bond, 0.0)
    else:
        raise ValueError(rule)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all(), "行权重和不得超过 1"
    return w


def yearly_returns(equity: pd.Series) -> pd.Series:
    yearly = equity.resample("YE").last()
    first = equity.iloc[0]
    prev = pd.concat([pd.Series([first]), yearly[:-1]])
    prev.index = yearly.index
    return yearly / prev - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", default=dt.date.today().isoformat())
    parser.add_argument("--capital", type=float, default=110000)
    parser.add_argument("--save-curves", default=None, help="净值曲线 CSV 输出路径")
    args = parser.parse_args()

    prices = load_pool(config.ROTATION_START, args.end)
    closes = closes_table(prices)

    base = build_weights(closes, mode="ensemble", lookback=config.ROTATION_LOOKBACK,
                         buffer=config.ROTATION_BUFFER, dd_control=False, vol_control=False)
    vt = build_weights(closes, mode="ensemble", lookback=config.ROTATION_LOOKBACK,
                       buffer=config.ROTATION_BUFFER, dd_control=False, vol_control=True)
    variants = {
        "base(ensemble)": base,
        "vt(当前线上)": vt,
        "vt+bond": apply_sleeve(vt, closes, "bond"),
        "vt+gold": apply_sleeve(vt, closes, "gold"),
        "vt+best": apply_sleeve(vt, closes, "best"),
        "vt+split": apply_sleeve(vt, closes, "split"),
    }

    curves = {}
    rows = []
    for name, w in variants.items():
        result = run_portfolio_backtest(prices, w, initial_capital=args.capital,
                                        stamp_tax=False)
        eq = result.equity
        curves[name] = eq
        m_all = metrics_mod.equity_metrics(eq)
        oos = eq.loc[config.OOS_SPLIT:]
        m_oos = metrics_mod.equity_metrics(oos)
        rows.append({
            "变体": name,
            "年化": m_all["年化收益率"], "夏普": m_all["夏普比率"], "回撤": m_all["最大回撤"],
            "OOS年化": m_oos["年化收益率"], "OOS夏普": m_oos["夏普比率"], "OOS回撤": m_oos["最大回撤"],
            "交易数": len(result.trades),
        })

    df = pd.DataFrame(rows).set_index("变体")
    pd.set_option("display.width", 160)
    for c in ("年化", "回撤", "OOS年化", "OOS回撤"):
        df[c] = (df[c] * 100).round(1)
    for c in ("夏普", "OOS夏普"):
        df[c] = df[c].round(2)
    print(f"\n===== 全区间 {closes.index[0].date()} ~ {closes.index[-1].date()} | "
          f"OOS {config.OOS_SPLIT}+ | capital={args.capital:,.0f}(真实费用) =====")
    print(df.to_string())

    print("\n===== 分年收益(%) =====")
    yr = pd.DataFrame({n: yearly_returns(eq) * 100 for n, eq in curves.items()})
    yr.index = yr.index.year
    print(yr.round(1).to_string())

    if args.save_curves:
        pd.DataFrame(curves).to_csv(args.save_curves)
        print(f"\n曲线已保存: {args.save_curves}")


if __name__ == "__main__":
    main()
