"""账本现金流规则(report_web._apply_exec_row)与 account_state fail-closed 回归测试。

_apply_exec_row 是 real_equity_series / replay_positions / record_trade 校验共用的
唯一实现,本文件锚定其口径:amount 覆盖语义、默认佣金、资金流、负现金/负持仓校验。
运行: python test_ledger.py
"""
import os
import tempfile

import pandas as pd

import config
import daily_signal
import report_web
from report_web import _apply_exec_row, load_executions, overnight_held_symbols

FEE_MIN = config.COMMISSION_MIN  # 5 元最低佣金,测试金额较小时恒命中


def _row(action, symbol="", price=float("nan"), shares=float("nan"),
         amount=float("nan"), date="2026-01-05"):
    return pd.Series({"date": pd.Timestamp(date), "action": action, "symbol": symbol,
                      "price": price, "shares": shares, "amount": amount})


def test_deposit_withdraw_flow():
    """入金/出金:现金与外部资金流一致;买卖资金流为 0"""
    pos = {}
    cash, flow = _apply_exec_row(_row("deposit", amount=10000.0), pos, 0.0)
    assert cash == 10000.0 and flow == 10000.0
    cash, flow = _apply_exec_row(_row("withdraw", amount=3000.0), pos, cash)
    assert cash == 7000.0 and flow == -3000.0


def test_buy_default_fee_and_position():
    """买入缺 amount:现金减 gross+默认佣金(最低5元),持仓增加,flow=0"""
    pos = {}
    cash, _ = _apply_exec_row(_row("deposit", amount=10000.0), pos, 0.0)
    cash, flow = _apply_exec_row(_row("buy", "510300", 1.0, 1000), pos, cash)
    assert abs(cash - (10000.0 - 1000.0 - FEE_MIN)) < 1e-9
    assert flow == 0.0 and pos["510300"] == 1000


def test_amount_override():
    """amount 填写(券商实际金额)时精确覆盖默认估算"""
    pos = {}
    cash, _ = _apply_exec_row(_row("deposit", amount=10000.0), pos, 0.0)
    cash, _ = _apply_exec_row(_row("buy", "510300", 1.0, 1000, amount=1003.3), pos, cash)
    assert abs(cash - (10000.0 - 1003.3)) < 1e-9
    cash, _ = _apply_exec_row(_row("sell", "510300", 1.0, 1000, amount=996.2), pos, cash)
    assert abs(cash - (10000.0 - 1003.3 + 996.2)) < 1e-9 and pos["510300"] == 0


def test_sell_default_fee():
    """卖出缺 amount:现金加 gross-默认佣金"""
    pos = {}
    cash, _ = _apply_exec_row(_row("deposit", amount=10000.0), pos, 0.0)
    cash, _ = _apply_exec_row(_row("buy", "510300", 1.0, 1000), pos, cash)
    cash, flow = _apply_exec_row(_row("sell", "510300", 1.0, 1000), pos, cash)
    assert abs(cash - (10000.0 - 2 * FEE_MIN)) < 1e-9 and flow == 0.0


def test_negative_cash_raises():
    pos = {}
    try:
        _apply_exec_row(_row("buy", "510300", 1.0, 1000), pos, 100.0)
        raise AssertionError("现金不足的买入应抛 ValueError")
    except ValueError as e:
        assert "现金为负" in str(e)


def test_negative_position_raises():
    pos = {}
    try:
        _apply_exec_row(_row("sell", "510300", 1.0, 100), pos, 0.0)
        raise AssertionError("无持仓的卖出应抛 ValueError")
    except ValueError as e:
        assert "持仓为负" in str(e)


def test_withdraw_overdraft_raises():
    try:
        _apply_exec_row(_row("withdraw", amount=100.0), {}, 50.0)
        raise AssertionError("出金透支应抛 ValueError")
    except ValueError as e:
        assert "现金为负" in str(e)


def _with_ledger(content):
    """临时替换 report_web.EXEC_PATH 指向给定内容的 CSV,返回还原函数"""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                      encoding="utf-8")
    tmp.write(content)
    tmp.close()
    old = report_web.EXEC_PATH
    report_web.EXEC_PATH = tmp.name

    def restore():
        report_web.EXEC_PATH = old
        os.unlink(tmp.name)
    return restore


HEADER = "date,action,symbol,price,shares,amount,note,status\n"


def test_account_equity_fail_closed_on_corrupt_ledger():
    """损坏账本(无持仓即卖出→负持仓)→ ValueError 必须向上传播,
    绝不静默回退 --capital 继续生成计划(fail-closed)"""
    restore = _with_ledger(HEADER + "2026-01-05,卖出,510300,1.0,100,,,已成交\n")
    try:
        daily_signal.account_state(pd.Series({"510300": 1.0}))
        raise AssertionError("损坏账本应抛 ValueError")
    except ValueError:
        pass
    finally:
        restore()


def test_blank_amount_still_defaults_fee():
    """正向锚定:买卖行 amount 真正留空仍合法,走 价格×股数+默认佣金 估算"""
    restore = _with_ledger(HEADER
                           + "2026-01-05,入金,,,,10000,,已成交\n"
                           + "2026-01-06,买入,510300,1.0,1000,,,已成交\n")
    try:
        state = daily_signal.account_state(pd.Series({"510300": 1.0}))
        # 现金 10000-1000-5,持仓 1000×1.0
        assert state is not None
        total, cash = state
        assert abs(total - 9995.0) < 1e-6 and abs(cash - 8995.0) < 1e-6
    finally:
        restore()


def test_account_equity_valid_ledger():
    """正常账本:总资产 = 现金 + 持仓×最新收盘"""
    restore = _with_ledger(HEADER
                           + "2026-01-05,入金,,,,10000,,已成交\n"
                           + "2026-01-06,买入,510300,1.0,1000,,,已成交\n")
    try:
        state = daily_signal.account_state(pd.Series({"510300": 2.0}))
        # 现金 10000-1000-5 = 8995,持仓 1000×2.0 = 2000
        assert state is not None
        total, cash = state
        assert abs(total - 10995.0) < 1e-6 and abs(cash - 8995.0) < 1e-6
    finally:
        restore()


def test_account_equity_zero_is_zero_not_none():
    """合法零资产(全部出金)必须返回 0.0 而非 None:
    空账户绝不能回退 --capital 生成不存在资金的建仓买单"""
    restore = _with_ledger(HEADER
                           + "2026-01-05,入金,,,,10000,,已成交\n"
                           + "2026-01-06,出金,,,,10000,,已成交\n")
    try:
        state = daily_signal.account_state(pd.Series({"510300": 1.0}))
        assert state == (0.0, 0.0), f"应为 (0.0, 0.0)(而非 None 回退 --capital),实得 {state!r}"
    finally:
        restore()


def test_account_equity_rejects_infinite_amount():
    """账本含 inf 金额:load_executions 边界校验必须抛错(fail-closed),
    绝不静默回退 --capital"""
    restore = _with_ledger(HEADER + "2026-01-05,入金,,,,inf,,已成交\n")
    try:
        daily_signal.account_state(pd.Series({"510300": 1.0}))
        raise AssertionError("inf 金额应抛 ValueError")
    except ValueError as e:
        assert "非有限" in str(e)
    finally:
        restore()


def test_account_equity_rejects_garbage_amount():
    """买卖行 amount/price 为乱值文本("abc"):解析失败必须抛错,
    不得被 coerce 成 NaN 静默当作"未填"走默认佣金估算"""
    for col_csv in ("2026-01-06,买入,510300,1.0,1000,abc,,已成交\n",     # amount 乱值
                    "2026-01-06,买入,510300,1.0,1000,NULL,,已成交\n",    # pandas 默认 NA 词表值
                    "2026-01-06,买入,510300,1.0,1000,nan,,已成交\n",     # 字面 nan 文本
                    "2026-01-06,买入,510300,abc,1000,,,已成交\n"):       # price 乱值
        restore = _with_ledger(HEADER + "2026-01-05,入金,,,,10000,,已成交\n" + col_csv)
        try:
            daily_signal.account_state(pd.Series({"510300": 1.0}))
            raise AssertionError("乱值应抛 ValueError")
        except ValueError as e:
            assert "无法解析" in str(e)
        finally:
            restore()


def test_account_equity_degrades_gracefully():
    """明确可降级情形返回 None:账本不存在 / 持仓标的无最新价(池外)"""
    old = report_web.EXEC_PATH
    report_web.EXEC_PATH = "/nonexistent/executions.csv"
    try:
        assert daily_signal.account_state(pd.Series({"510300": 1.0})) is None
    finally:
        report_web.EXEC_PATH = old
    restore = _with_ledger(HEADER
                           + "2026-01-05,入金,,,,10000,,已成交\n"
                           + "2026-01-06,买入,512880,1.0,1000,,,已成交\n")
    try:
        assert daily_signal.account_state(pd.Series({"510300": 1.0})) is None
    finally:
        restore()


def test_overnight_held_symbols():
    """池外补价范围:当日买入又卖光(日终恒0,如误买纠错的513300)不需要行情;
    隔夜持有过的(即使后来清仓)需要——qfq_only 下多拉一个冷门标的都可能拖垮检查"""
    restore = _with_ledger(
        HEADER
        + "2026-01-05,入金,,,,100000,,已成交\n"
        + "2026-01-06,买入,513300,2.672,3700,,,已成交\n"   # 当日
        + "2026-01-06,卖出,513300,2.65,3700,,,已成交\n"    # 当日卖光 → 日终0,不需要
        + "2026-01-06,买入,513100,2.194,4400,,,已成交\n"   # 隔夜持有
        + "2026-01-08,卖出,513100,2.254,4400,,,已成交\n")  # 后清仓 → 仍需历史行情
    try:
        execs = load_executions()
        held = overnight_held_symbols(execs[execs["status"] != "计划"])
        assert held == {"513100"}, f"应只含隔夜持有过的 513100,实得 {held}"
    finally:
        restore()


def test_overnight_held_uses_valuation_calendar():
    """判定必须按估值日历而非自然日:成交日不在 closes.index 时,买卖会被并到
    下一估值日一次性应用互相抵消(该标的实际不需要行情);无日历时保守按自然日"""
    restore = _with_ledger(
        HEADER
        + "2026-01-05,入金,,,,100000,,已成交\n"
        + "2026-01-06,买入,513300,2.672,3700,,,已成交\n"   # 01-06 不在估值日历
        + "2026-01-07,卖出,513300,2.65,3700,,,已成交\n")   # 下一估值日前已清零
    try:
        execs = load_executions()
        conf = execs[execs["status"] != "计划"]
        vi = pd.DatetimeIndex([pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-07"),
                               pd.Timestamp("2026-01-08")])
        # 买(01-06)卖(01-07)都映射到估值日 01-07 → 净 0 → 不需要行情
        assert overnight_held_symbols(conf, vi) == set()
        # 无估值日历:退化为自然日 → 保守判为持有过(超集,绝不漏价)
        assert overnight_held_symbols(conf) == {"513300"}
        # 成交晚于最后估值日 → 归越界哨兵桶:买卖同落桶净0排除
        vi_short = pd.DatetimeIndex([pd.Timestamp("2026-01-05")])
        assert overnight_held_symbols(conf, vi_short) == set()
        # 哨兵桶净非零 → 保守计入(只有越界买入未卖)
        only_buy = conf[~((conf["action"] == "sell") & (conf["symbol"] == "513300"))]
        assert overnight_held_symbols(only_buy, vi_short) == {"513300"}
    finally:
        restore()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} 个测试全部通过")
