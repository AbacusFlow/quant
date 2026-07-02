"""趋势/动量策略库。

统一接口:每个策略接收日线 DataFrame(open/high/low/close/volume),
返回目标仓位序列(0=空仓, 1=满仓),信号基于当日收盘数据计算,次日开盘执行(T+1)。
"""
import numpy as np
import pandas as pd


def dual_ma_signal(df: pd.DataFrame, short: int = 5, long: int = 20) -> pd.Series:
    """双均线:MA(short) 上穿 MA(long) 买入(金叉),下穿卖出(死叉)。"""
    ma_s = df["close"].rolling(short).mean()
    ma_l = df["close"].rolling(long).mean()
    position = pd.Series(0, index=df.index, dtype=int)
    position[ma_s > ma_l] = 1
    position[ma_l.isna() | ma_s.isna()] = 0
    position.name = "position"
    return position


def macd_signal(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD:DIF 上穿 DEA 且 DIF>0 买入,DIF 下穿 DEA 卖出。"""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()

    position = pd.Series(0, index=df.index, dtype=int)
    holding = 0
    for i in range(slow, len(df)):
        if holding == 0 and dif.iloc[i] > dea.iloc[i] and dif.iloc[i] > 0:
            holding = 1
        elif holding == 1 and dif.iloc[i] < dea.iloc[i]:
            holding = 0
        position.iloc[i] = holding
    position.name = "position"
    return position


def bollinger_signal(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """布林带突破:收盘价上穿上轨买入,跌破中轨卖出。"""
    mid = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    upper = mid + n_std * std

    position = pd.Series(0, index=df.index, dtype=int)
    holding = 0
    for i in range(window, len(df)):
        c = df["close"].iloc[i]
        if holding == 0 and c > upper.iloc[i]:
            holding = 1
        elif holding == 1 and c < mid.iloc[i]:
            holding = 0
        position.iloc[i] = holding
    position.name = "position"
    return position


def momentum_signal(df: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """动量(ROC):过去 lookback 日收益 > 0 持仓,否则空仓。"""
    roc = df["close"].pct_change(lookback)
    position = pd.Series(0, index=df.index, dtype=int)
    position[roc > 0] = 1
    position[roc.isna()] = 0
    position.name = "position"
    return position


def buy_and_hold_signal(df: pd.DataFrame) -> pd.Series:
    """买入持有:第一天起一直满仓,作为对照。"""
    position = pd.Series(1, index=df.index, dtype=int)
    position.name = "position"
    return position


def etf_momentum_rotation(
    closes: pd.DataFrame,
    lookback: int = 20,
    buffer: float = 0.01,
) -> pd.DataFrame:
    """ETF 动量轮动:持有过去 lookback 日涨幅最高的 ETF(top-1)。

    - 绝对动量过滤:所有候选动量 <= 0 时空仓持现金
    - 换仓缓冲:新候选动量需超过当前持仓动量 buffer 以上才切换,降低换手

    closes: 收盘价表(日期 x symbol)
    返回:目标权重表(日期 x symbol),每行至多一个 1.0
    """
    momentum = closes.pct_change(lookback, fill_method=None)
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

    current: str | None = None
    for date in closes.index:
        mom = momentum.loc[date]
        if mom.isna().all():
            current = None
            continue
        best = mom.idxmax()
        best_mom = mom[best]

        if current is not None and not pd.isna(mom[current]) and mom[current] > 0:
            # 仅当新候选显著更强时才切换
            if best != current and best_mom > mom[current] + buffer:
                current = best
        else:
            current = best if best_mom > 0 else None

        if current is not None and mom[current] <= 0:
            current = None
        if current is not None:
            weights.at[date, current] = 1.0
    return weights


def etf_momentum_ensemble(
    closes: pd.DataFrame,
    lookbacks: tuple[int, ...] = (15, 20, 25),
    buffer: float = 0.01,
) -> pd.DataFrame:
    """多周期集成轮动:多个 lookback 子策略各占等权,降低单一参数过拟合风险。

    返回:目标权重表,每行权重和 <= 1(子策略空仓时对应份额持现金)
    """
    tables = [etf_momentum_rotation(closes, lookback=lb, buffer=buffer) for lb in lookbacks]
    combined = tables[0].copy()
    for t in tables[1:]:
        combined += t
    return combined / len(lookbacks)


def apply_drawdown_control(
    weights: pd.DataFrame,
    closes: pd.DataFrame,
    ma_window: int = 60,
    scale: float = 0.5,
) -> pd.DataFrame:
    """回撤控制:策略虚拟净值跌破其 ma_window 日均线时,目标仓位乘以 scale。

    虚拟净值由未缩放权重的收盘-收盘收益构造(权重移1日对齐 T+1 执行),
    T 日的控制信号只用到 T 日及之前的收盘价,无前视。
    """
    rets = closes.pct_change(fill_method=None).fillna(0.0)
    strat_ret = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)
    virtual = (1 + strat_ret).cumprod()
    ma = virtual.rolling(ma_window).mean()

    factor = pd.Series(1.0, index=weights.index)
    factor[virtual < ma] = scale
    return weights.mul(factor, axis=0)


def apply_vol_targeting(
    weights: pd.DataFrame,
    closes: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """波动率目标:策略自身已实现波动超过其历史中位时按比例降仓,封顶 1.0(无杠杆)。

    经济逻辑(Barroso & Santa-Clara 2015):动量崩盘集中在高波动期,据此降敞口
    可显著改善稳定性(walk-forward 夏普↑、最大回撤↓),年化基本不变。

    - 自适应目标:用因果扩张中位数(expanding median),不引入固定 target_vol 旋钮
    - 无前视:scale_t 仅用 ≤T 的收盘价(与 apply_drawdown_control 同口径);
      引擎再 shift(1) 到 T+1 执行
    - scale ∈ (0, 1],只降不加杠杆;暖机期/波动低于历史中位时 scale==1(权重不变)
    """
    rets = closes.pct_change(fill_method=None).fillna(0.0)
    strat_ret = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)
    realized = strat_ret.rolling(lookback).std() * np.sqrt(252)
    target = realized.expanding(min_periods=lookback).median()
    scale = (target / realized).clip(upper=1.0).fillna(1.0)
    return weights.mul(scale, axis=0)


def apply_defensive_sleeve(
    weights: pd.DataFrame,
    closes: pd.DataFrame,
    gold: str = "518880",
    bond: str = "511260",
) -> pd.DataFrame:
    """防御 sleeve:残余现金(绝对动量滤空 + 波动目标降仓后的空档)金债各半。

    经济逻辑(借鉴 eTrade 防御 sleeve 对比研究):闲置现金不产生收益,
    路由到国债(利息 carry)+ 黄金(股票熊市对冲,2018/2022 年均为正贡献)。
    零参数(固定 50/50),无数据依赖 → 天然无前视;引擎再 shift(1) 到 T+1 执行。
    行权重和仍 ≤ 1(只填充空档,不加杠杆)。

    回测(2017-08~2026-07,11万真实费用):年化 16.9%→18.4%、夏普 0.96→1.00、
    回撤 -18.4% 基本不变;样本外年化 28.8%→32.2%(见 scripts/experiment_sleeve.py)。
    """
    w = weights.copy()
    residual = (1.0 - w.sum(axis=1)).clip(lower=0.0)
    w[gold] = w[gold] + residual / 2
    w[bond] = w[bond] + residual / 2
    return w


# 策略注册表:名称 -> (函数, 说明)
STRATEGIES = {
    "dual_ma": (dual_ma_signal, "双均线金叉/死叉"),
    "macd": (macd_signal, "MACD DIF/DEA 交叉"),
    "boll": (bollinger_signal, "布林带上轨突破"),
    "momentum": (momentum_signal, "60日动量"),
    "hold": (buy_and_hold_signal, "买入持有(基准)"),
}
