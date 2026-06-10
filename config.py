"""回测系统全局配置"""

# 默认标的与区间
DEFAULT_SYMBOL = "600519"
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2025-12-31"

# 基准指数(沪深300)
BENCHMARK_SYMBOL = "000300"

# 双均线参数
MA_SHORT = 5
MA_LONG = 20

# 初始资金
INITIAL_CAPITAL = 1_000_000.0

# 交易费用(A股)
COMMISSION_RATE = 0.00025   # 佣金 万2.5,双边
COMMISSION_MIN = 5.0        # 最低佣金 5 元
STAMP_TAX_RATE = 0.0005     # 印花税 万5,仅卖出

# 滑点(单边,按成交价比例;流动性好的宽基ETF约万5)
SLIPPAGE_RATE = 0.0005

# ETF 轮动策略
# ETF 池:相关性弱的大类资产(宽基/成长/跨境/商品)
ETF_POOL = {
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "159915": "创业板ETF",
    "513100": "纳指ETF",
    "518880": "黄金ETF",
}
ROTATION_LOOKBACK = 20      # 动量回看天数
ROTATION_BUFFER = 0.01      # 换仓缓冲:新标的动量需超过当前持仓 1% 才切换
ROTATION_START = "2015-01-01"
OOS_SPLIT = "2022-01-01"    # 样本内/样本外分割点

# 路径
DATA_DIR = "data"
OUTPUT_DIR = "output"
