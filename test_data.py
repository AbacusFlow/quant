"""data.py 数据层回归测试:缓存防线(名新实旧)/ 腾讯翻页 / qfq_only。

全部离线:伪造 akshare / requests 模块,不碰网络、不碰真实 data/ 目录。
运行: python test_data.py
"""
import datetime as dt
import os
import shutil
import sys
import tempfile
import types
from zoneinfo import ZoneInfo

import pandas as pd

import config
import data

TODAY = dt.datetime.now(ZoneInfo("Asia/Shanghai")).date()


def _mk_df(last_day: dt.date, days: int = 5) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp(last_day), periods=days, freq="D")
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                         "volume": 0.0}, index=pd.DatetimeIndex(idx, name="date"))


def _em_frame(last_day: dt.date, days: int = 5) -> pd.DataFrame:
    dates = [(last_day - dt.timedelta(days=i)).isoformat() for i in range(days)][::-1]
    return pd.DataFrame({"日期": dates, "开盘": 1.0, "最高": 1.0, "最低": 1.0,
                         "收盘": 1.0, "成交量": 0.0})


class _Sandbox:
    """临时 DATA_DIR + 可选伪造 akshare/requests + 关闭重试 sleep,用完全部还原。"""

    def __init__(self, fake_ak=None, fake_requests=None):
        self.fake_ak, self.fake_requests = fake_ak, fake_requests

    def __enter__(self):
        self.tmp = tempfile.mkdtemp()
        self.old_dir, config.DATA_DIR = config.DATA_DIR, self.tmp
        self.old_sleep, data.time.sleep = data.time.sleep, lambda s: None
        self.old_ak = sys.modules.get("akshare")
        self.old_req = sys.modules.get("requests")
        if self.fake_ak is not None:
            sys.modules["akshare"] = self.fake_ak
        if self.fake_requests is not None:
            sys.modules["requests"] = self.fake_requests
        return self.tmp

    def __exit__(self, *exc):
        config.DATA_DIR = self.old_dir
        data.time.sleep = self.old_sleep
        for name, old in (("akshare", self.old_ak), ("requests", self.old_req)):
            if old is not None:
                sys.modules[name] = old
            elif name in sys.modules and sys.modules[name] in (self.fake_ak, self.fake_requests):
                del sys.modules[name]
        shutil.rmtree(self.tmp, ignore_errors=True)
        return False


def test_cache_would_lie_boundaries():
    """名新实旧判定边界:历史区间可写 / end=今天数据滞后拒写 / 数据已到今天可写 / 未来拒写"""
    hist = _mk_df(TODAY - dt.timedelta(days=30))
    assert not data._cache_would_lie(hist, (TODAY - dt.timedelta(days=20)).isoformat())
    assert data._cache_would_lie(_mk_df(TODAY - dt.timedelta(days=1)), TODAY.isoformat())
    assert not data._cache_would_lie(_mk_df(TODAY), TODAY.isoformat())
    assert data._cache_would_lie(_mk_df(TODAY), (TODAY + dt.timedelta(days=3)).isoformat())


def test_poisoned_cache_bypassed_then_healed():
    """升级前遗留的旧毒缓存(文件名 end=今天、内容只到前天):
    命中侧防线应绕过它重拉;源仍滞后时不覆盖写;源补齐当日后写回自愈并恢复命中"""
    fake = types.ModuleType("akshare")
    with _Sandbox(fake_ak=fake) as tmp:
        end = TODAY.isoformat()
        path = os.path.join(tmp, f"etf_510300_2026-01-01_{end}.csv")
        _mk_df(TODAY - dt.timedelta(days=2)).to_csv(path)  # 植入毒缓存

        # 源仍滞后(只到昨天):绕过毒缓存、返回新数据、且不写缓存(仍会说谎)
        fake.fund_etf_hist_em = lambda **kw: _em_frame(TODAY - dt.timedelta(days=1))
        df = data.get_etf_daily("510300", "2026-01-01", end)
        assert df.index[-1].date() == TODAY - dt.timedelta(days=1)
        on_disk = pd.read_csv(path, parse_dates=["date"])
        assert on_disk["date"].max().date() == TODAY - dt.timedelta(days=2), "毒缓存不应被滞后数据覆盖"

        # 源补齐(到今天):写回自愈
        fake.fund_etf_hist_em = lambda **kw: _em_frame(TODAY)
        df = data.get_etf_daily("510300", "2026-01-01", end)
        assert df.index[-1].date() == TODAY
        healed = pd.read_csv(path, parse_dates=["date"])
        assert healed["date"].max().date() == TODAY, "缓存应已自愈"

        # 自愈后正常命中,不再走网络
        def _no_fetch(**kw):
            raise AssertionError("自愈后应命中缓存,不该再拉取")
        fake.fund_etf_hist_em = _no_fetch
        df = data.get_etf_daily("510300", "2026-01-01", end)
        assert df.index[-1].date() == TODAY


def test_historical_cache_still_hits():
    """历史区间缓存(end 在过去)不受防线影响,照常命中"""
    fake = types.ModuleType("akshare")

    def _no_fetch(**kw):
        raise AssertionError("历史缓存应命中,不该拉取")
    fake.fund_etf_hist_em = _no_fetch
    with _Sandbox(fake_ak=fake) as tmp:
        end = (TODAY - dt.timedelta(days=10)).isoformat()
        path = os.path.join(tmp, f"etf_510300_2026-01-01_{end}.csv")
        _mk_df(TODAY - dt.timedelta(days=12)).to_csv(path)  # 数据没到 end(周末)也合法
        df = data.get_etf_daily("510300", "2026-01-01", end)
        assert df.index[-1].date() == TODAY - dt.timedelta(days=12)


def test_weekend_complete_cache_accepted():
    """end=非交易日(如周六)而数据已到 end 前最后一个交易日 → 缓存完整,照常命中;
    end=交易日且当日 bar 未出 → 仍判名新实旧拒缓存"""
    fake = types.ModuleType("akshare")

    def _no_fetch(**kw):
        raise AssertionError("完整缓存应命中,不该拉取")
    fake.fund_etf_hist_em = _no_fetch
    old_cal = data.get_trade_dates
    friday = TODAY - dt.timedelta(days=2)
    nxt = TODAY + dt.timedelta(days=5)   # 日历必须覆盖 end 之后,判定才可信
    try:
        with _Sandbox(fake_ak=fake) as tmp:
            # 日历:end 之前最后交易日 = TODAY-2(把"今天"当周末场景),且覆盖到未来
            data.get_trade_dates = lambda: pd.DatetimeIndex(
                [pd.Timestamp(friday), pd.Timestamp(nxt)])
            end = TODAY.isoformat()
            path = os.path.join(tmp, f"etf_510300_2026-01-01_{end}.csv")
            _mk_df(friday).to_csv(path)
            df = data.get_etf_daily("510300", "2026-01-01", end)  # 应命中,不触网
            assert df.index[-1].date() == friday
            # 日历包含"今天"(交易日)而数据只到 friday → 应拒缓存(判定为说谎)
            data.get_trade_dates = lambda: pd.DatetimeIndex(
                [pd.Timestamp(friday), pd.Timestamp(TODAY), pd.Timestamp(nxt)])
            assert data._cache_would_lie(_mk_df(friday), end)
            # 截断日历(未覆盖 end):无法证明 end 非交易日 → 保守拒缓存,不得放行
            data.get_trade_dates = lambda: pd.DatetimeIndex([pd.Timestamp(friday)])
            assert data._cache_would_lie(_mk_df(friday), end)
            # 日历不可用:同样保守拒缓存
            def _cal_boom():
                raise RuntimeError("日历接口挂了")
            data.get_trade_dates = _cal_boom
            assert data._cache_would_lie(_mk_df(friday), end)
    finally:
        data.get_trade_dates = old_cal


def test_qfq_only_rejects_sina_fallback():
    """东财/腾讯双挂 + qfq_only=True → 抛错拒绝新浪不复权回退(money 路径不可混口径)"""
    fake_ak = types.ModuleType("akshare")

    def _em_boom(**kw):
        raise RuntimeError("东财挂了")

    def _sina_forbidden(**kw):
        raise AssertionError("qfq_only 不应走到新浪回退")
    fake_ak.fund_etf_hist_em = _em_boom
    fake_ak.fund_etf_hist_sina = _sina_forbidden
    fake_req = types.ModuleType("requests")

    def _get_boom(*a, **kw):
        raise RuntimeError("腾讯挂了")
    fake_req.get = _get_boom
    with _Sandbox(fake_ak=fake_ak, fake_requests=fake_req):
        try:
            data.get_etf_daily("510300", "2026-01-01", TODAY.isoformat(), qfq_only=True)
            raise AssertionError("应当抛错")
        except RuntimeError as e:
            assert "qfq_only" in str(e) or "前复权" in str(e)


def test_tencent_paging_no_progress_raises():
    """腾讯翻页接口忽略 cur_end 重复回同页 → 抛错中止,不死循环不重复累积"""
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"sh510300": {"qfqday": [
                ["2020-01-02", 1, 1, 1, 1, 0], ["2020-01-03", 1, 1, 1, 1, 0]]}}}

    fake_req = types.ModuleType("requests")
    fake_req.get = lambda *a, **kw: _Resp()
    with _Sandbox(fake_requests=fake_req):
        try:
            data._fetch_etf_tencent("510300", "2015-01-01", "2020-12-31")
            raise AssertionError("应当抛错(翻页无进展)")
        except RuntimeError as e:
            assert "无进展" in str(e)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} 个测试全部通过")
