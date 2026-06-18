"""check_prices.verdict 单元测试:多源昨收校验的判定逻辑(含盘前未滚动启发式)。

运行: python test_check_prices.py
"""
from check_prices import verdict

PREV = 4.958      # 上一交易日(06-17)真实收盘
PP = 4.910        # 上上交易日(06-16)收盘


def test_dated_source_agrees():
    """权威带日期源与系统一致 → ok,忽略实时源"""
    level, detail = verdict(PREV, PP, dated_val=4.958, realtime_vals={"腾讯实时": 4.910})
    assert level == "ok" and "日K线" in detail


def test_dated_source_disagrees():
    """权威带日期源给出不同值(真异常/除权)→ mismatch"""
    level, detail = verdict(PREV, PP, dated_val=4.800, realtime_vals={})
    assert level == "mismatch" and "4.800" in detail


def test_realtime_match_when_no_dated():
    """无带日期源,实时昨收等于上一交易日 → ok"""
    level, detail = verdict(PREV, PP, dated_val=None,
                            realtime_vals={"腾讯实时": 4.958, "新浪实时": 4.958})
    assert level == "ok"


def test_realtime_all_stale_is_skip_not_warn():
    """盘前陷阱:实时昨收全部等于上上交易日 → skip(非异常,不告警)"""
    level, detail = verdict(PREV, PP, dated_val=None,
                            realtime_vals={"腾讯实时": 4.910, "新浪实时": 4.910})
    assert level == "skip"


def test_realtime_genuine_mismatch():
    """实时昨收既非上一日也非上上一日 → mismatch(真异常)"""
    level, detail = verdict(PREV, PP, dated_val=None, realtime_vals={"腾讯实时": 4.700})
    assert level == "mismatch" and "4.700" in detail


def test_mismatch_takes_priority_over_match():
    """源间矛盾:一源匹配、另一源给陌生值 → mismatch 优先(不被静默)"""
    level, detail = verdict(PREV, PP, dated_val=None,
                            realtime_vals={"腾讯实时": 4.958, "新浪实时": 4.700})
    assert level == "mismatch" and "4.700" in detail and "新浪实时" in detail


def test_one_stale_one_fresh_is_ok():
    """一源盘前未滚动、另一源已滚动且匹配 → ok(回退冗余救场)"""
    level, detail = verdict(PREV, PP, dated_val=None,
                            realtime_vals={"腾讯实时": 4.910, "新浪实时": 4.958})
    assert level == "ok" and "新浪实时" in detail


def test_ambiguous_prev_equals_pp_is_skip():
    """上一日与上上日同价:实时源等于该价无法证明已滚动 → skip(不谎报 ok)"""
    level, _ = verdict(4.910, 4.910, dated_val=None,
                       realtime_vals={"腾讯实时": 4.910, "新浪实时": 4.910})
    assert level == "skip"


def test_ambiguous_prev_equals_pp_but_dated_validates():
    """同价歧义下,只要带日期权威源在且一致 → 仍判 ok"""
    level, detail = verdict(4.910, 4.910, dated_val=4.910, realtime_vals={"腾讯实时": 4.910})
    assert level == "ok" and "日K线" in detail


def test_ambiguous_prev_equals_pp_with_real_diff_still_warns():
    """同价歧义下,实时源给出陌生值仍要 mismatch(不被歧义吞掉)"""
    level, detail = verdict(4.910, 4.910, dated_val=None, realtime_vals={"腾讯实时": 4.700})
    assert level == "mismatch" and "4.700" in detail


def test_no_source_is_nodata():
    """所有源都没给昨收 → nodata(无法校验,告警)"""
    level, detail = verdict(PREV, PP, dated_val=None, realtime_vals={})
    assert level == "nodata"


def test_pp_none_does_not_crash():
    """系统仅一根K线(无上上交易日)时不崩溃,陌生值判 mismatch"""
    level, detail = verdict(PREV, None, dated_val=None, realtime_vals={"腾讯实时": 4.910})
    assert level == "mismatch"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} 个测试全部通过")
