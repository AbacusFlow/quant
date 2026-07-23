"""daily_signal.plan_orders 订单规划回归测试(纯函数,无 IO)。

锚定:目标股数 vs 实际持仓的引擎口径、再平衡带宽不对称、部分成交漂移对齐、
现金封顶(安全垫/真实佣金/缩量/放弃)、池外清仓、说明行不含指令关键词。
运行: python test_plan_orders.py
"""
import pandas as pd

import config
from daily_signal import plan_orders

PX = pd.Series({"A1": 1.0, "B2": 10.0, "C3": 100.0})


def _w(**kw) -> pd.Series:
    base = {"A1": 0.0, "B2": 0.0, "C3": 0.0}
    base.update(kw)
    return pd.Series(base)


def test_band_skips_small_deviation():
    """偏离 < band×equity 不出单(有说明行);偏离足够大出单"""
    # equity 100k,band 2% → 2000 元;A1 目标 50%=50000 股,持 49000 → 差 1000 元 < 2000
    orders, notes = plan_orders(_w(A1=0.5), {"A1": 49000}, PX, equity=100_000, cash=0.0)
    assert orders == []
    assert any("再平衡带宽" in n for n in notes)
    # 持 40000 → 差 10000 元 > 2000 → 补仓 10000 股(现金足够时)
    orders, _ = plan_orders(_w(A1=0.5), {"A1": 40000}, PX, equity=100_000, cash=20_000)
    assert orders == [("买入", "A1", 10000)]


def test_zero_target_full_sell_ignores_band():
    """目标权重归零 → 清仓不受带宽限制(哪怕市值远小于带宽)"""
    orders, _ = plan_orders(_w(), {"A1": 100}, PX, equity=100_000, cash=0.0)
    assert orders == [("卖出", "A1", 100)]


def test_partial_fill_drift_generates_topup_not_sell():
    """H1 场景:上日计划因现金不足只成交一部分(实际仓位低于目标),
    今日目标虽下调但仍高于实际 → 应补仓而非按信号差卖出"""
    # 目标 50%(50000 股),实际只有 40000(上日只买到 80%)
    orders, _ = plan_orders(_w(A1=0.5), {"A1": 40000}, PX, equity=100_000, cash=50_000)
    assert orders == [("买入", "A1", 10000)]


def test_out_of_pool_cleared_without_price():
    """池外持仓(目标表无此列)一律清仓,无需价格、不受带宽限制"""
    orders, _ = plan_orders(_w(A1=0.5), {"A1": 50000, "999999": 300}, PX,
                            equity=100_000, cash=0.0)
    assert ("卖出", "999999", 300) in orders


def test_sell_before_buy_order():
    """输出顺序:先卖后买(卖出回款供买入使用)"""
    orders, _ = plan_orders(_w(A1=0.5, B2=0.5), {"A1": 100000}, PX,
                            equity=100_000, cash=0.0)
    actions = [a for a, _, _ in orders]
    assert actions == ["卖出", "买入"]
    assert orders[0][1] == "A1" and orders[1][1] == "B2"


def test_cash_cap_shrinks_buy():
    """现金封顶:买入超出「现金+卖出回款(含安全垫/佣金)」→ 整手缩量并留说明"""
    # 无卖出;现金 10000,B2@10 上浮 1% 后一手成本 ~1010+5;目标 50%=5000 股(50000 元)远超现金
    orders, notes = plan_orders(_w(B2=0.5), {}, PX, equity=100_000, cash=10_000)
    assert len(orders) == 1 and orders[0][0] == "买入" and orders[0][1] == "B2"
    shares = orders[0][2]
    px = 10.0 * 1.01
    assert shares % 100 == 0 and shares * px + max(shares * px * config.ETF_COMMISSION_RATE,
                                                   config.COMMISSION_MIN) <= 10_000
    assert shares >= 900  # 接近上限,不过度保守
    assert any("按可用资金调整" in n for n in notes)


def test_cash_cap_drops_unaffordable_buy():
    """连一手都买不起 → 放弃并留说明(说明行不含指令关键词)"""
    orders, notes = plan_orders(_w(C3=0.5), {}, PX, equity=100_000, cash=50.0)
    assert orders == []
    assert any("资金不足" in n for n in notes)


def test_sell_proceeds_fund_buys_with_margin():
    """卖出回款按 (1-margin) 折价扣佣金后计入可用资金"""
    # 卖 A1 10000 股@1.0 → 回款 ~10000×0.99−5 ≈ 9895;现金 0;B2 买入应被限在 ~9800 元内
    orders, _ = plan_orders(_w(A1=0.4, B2=0.1), {"A1": 50000}, PX,
                            equity=100_000, cash=0.0)
    sells = [(s, n) for a, s, n in orders if a == "卖出"]
    buys = [(s, n) for a, s, n in orders if a == "买入"]
    assert sells == [("A1", 10000)]
    assert len(buys) == 1 and buys[0][0] == "B2"
    n = buys[0][1]
    assert n * 10.0 * 1.01 <= 10000 * 0.99  # 买入成本不超过折价回款
    assert n >= 900


def test_real_fee_formula_not_flat_min():
    """佣金按 max(金额×万0.5, 5) 而非固定 5 元:大额订单边界不得超买"""
    # 现金 200,000;C3 目标 100% → desired 2000 股×100 元=200,000,加佣金/安全垫必超
    orders, notes = plan_orders(_w(C3=1.0), {}, PX, equity=200_000, cash=200_000)
    assert len(orders) == 1
    n = orders[0][2]
    px = 100.0 * 1.01
    gross = n * px
    assert gross + max(gross * config.ETF_COMMISSION_RATE, config.COMMISSION_MIN) <= 200_000


def test_multi_buy_proportional_order_independent():
    """多标的现金缩量按待买金额比例分配预算(引擎同口径),与索引顺序无关"""
    def run(order):
        w = pd.Series({s: 0.5 for s in order})
        orders, _ = plan_orders(w, {}, PX, equity=100_000, cash=50_000)
        return {s: n for a, s, n in orders if a == "买入"}

    r1, r2 = run(["A1", "B2"]), run(["B2", "A1"])
    assert r1 == r2, f"缩量结果依赖顺序: {r1} vs {r2}"
    # 各按 ~50% 预算成交(现金 50000 → 各 ~25000):A1@1.0×1.01 ≈ 24700+,B2@10×1.01 ≈ 2400
    assert 24000 <= r1["A1"] <= 24800 and 2400 <= r1["B2"] <= 2500
    # 总成本不超现金(佣金用与生产一致的真实公式:max(金额×万0.5, 5))
    def _cost(s, n):
        gross = n * PX[s] * 1.01
        return gross + max(gross * config.ETF_COMMISSION_RATE, config.COMMISSION_MIN)
    assert sum(_cost(s, n) for s, n in r1.items()) <= 50_000


def test_no_equity_only_liquidations():
    """equity 不可用(账本+capital 都缺)→ 只出清仓类订单,不猜买入"""
    orders, _ = plan_orders(_w(A1=0.5), {"999999": 300, "A1": 0}, PX,
                            equity=None, cash=None)
    assert orders == [("卖出", "999999", 300)]


def test_notes_never_contain_instruction_keywords():
    """说明行不得含「买入/卖出」——daily_local.sh 用 grep -E "买入|卖出" 抓指令行"""
    _, n1 = plan_orders(_w(A1=0.5), {"A1": 49000}, PX, equity=100_000, cash=0.0)
    _, n2 = plan_orders(_w(C3=0.5), {}, PX, equity=100_000, cash=50.0)
    _, n3 = plan_orders(_w(B2=0.5), {}, PX, equity=100_000, cash=10_000)
    for note in n1 + n2 + n3:
        assert "买入" not in note and "卖出" not in note, f"说明行混入指令关键词: {note}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} 个测试全部通过")
