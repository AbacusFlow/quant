"""趋势/动量策略库。

统一接口:每个策略接收日线 DataFrame(open/high/low/close/volume),
返回目标仓位序列(0=空仓, 1=满仓),信号基于当日收盘数据计算,次日开盘执行(T+1)。
"""
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


# 策略注册表:名称 -> (函数, 说明)
STRATEGIES = {
    "dual_ma": (dual_ma_signal, "双均线金叉/死叉"),
    "macd": (macd_signal, "MACD DIF/DEA 交叉"),
    "boll": (bollinger_signal, "布林带上轨突破"),
    "momentum": (momentum_signal, "60日动量"),
    "hold": (buy_and_hold_signal, "买入持有(基准)"),
}
