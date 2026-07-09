#!/usr/bin/env bash
# 本地每日信号编排(取代已停用的 GitHub Actions daily.yml)。
# GitHub 账号 suspended 后,自动化改为本机 cron 直接跑,信号照推 Telegram。
#
# 流程(对齐原 daily.yml,信号日 T 收盘算信号、T+1 开盘执行):
#   1. 读 .env 白名单键(mode/capital/vol-target/sleeve;不把 .env 当 shell 执行)
#   2. daily_signal.py 出信号(幂等:同信号日期+模式不重复记录;非交易日跳过)
#   3. check_prices.py 昨收校验(只告警不阻断)
#   4. check_risk.py 回撤检查(与线上同口径,vol-target/sleeve 一并传)
#   5. 本地 git 提交信号日志与流水(路径受限、只提交这两文件;无远端,不 push)
#   6. Telegram spool 推送(成功才出队,失败留待下次;两文件有变更 或 有告警 才推送)
#
# cron(交易日晚上,机器 TZ=Asia/Hong_Kong;22:37 主 + 23:37 兜底 + 次日 08:37 安全网):
#   37 22 * * 1-5  /home/logan/Projects/quant/scripts/daily_local.sh
#   37 23 * * 1-5  /home/logan/Projects/quant/scripts/daily_local.sh
#   37 8  * * 1-5  /home/logan/Projects/quant/scripts/daily_local.sh
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DOCKER_RUN=(docker run --rm --network=host -v "$PROJECT_DIR":/work quant python)
LEDGER=(output/signal_log.csv output/executions.csv)

LOG_DIR="output"
SPOOL="$LOG_DIR/push_spool"        # 待推送消息队列:每条一个文件,推送成功才删除
mkdir -p "$LOG_DIR" "$SPOOL"
RUN_LOG="$LOG_DIR/cron_signal.log"

# 并发护栏:三档 cron 若前一档卡住不与后一档抢 CSV/git 索引/spool(取代原 workflow 的 concurrency)
exec 9>"$LOG_DIR/.daily_local.lock"
if ! flock -n 9; then
    echo "$(date '+%F %T %Z') 另一实例运行中,跳过" >> "$RUN_LOG"
    exit 0
fi

# 只解析白名单键,绝不把 .env 当 shell 执行(防注入)
envval() { [ -f .env ] && grep -E "^$1=" .env | tail -1 | cut -d= -f2-; }
TG_BOT_TOKEN="$(envval TG_BOT_TOKEN || true)"
TG_CHAT_ID="$(envval TG_CHAT_ID || true)"
MODE="$(envval SIGNAL_MODE || true)"; MODE="${MODE:-single}"
CAPITAL="$(envval SIGNAL_CAPITAL || true)"; CAPITAL="${CAPITAL:-10000}"
VOL_TARGET="$(envval SIGNAL_VOL_TARGET || true)"; VOL_TARGET="${VOL_TARGET:-false}"
SLEEVE="$(envval SIGNAL_SLEEVE || true)"; SLEEVE="${SLEEVE:-false}"

# 开关 → flag(与线上一致:daily_signal / check_risk 同口径)
if [ "$VOL_TARGET" = "true" ]; then VOL_FLAG="--vol-target"; else VOL_FLAG="--no-vol-target"; fi
if [ "$SLEEVE" = "true" ]; then SLEEVE_FLAG="--sleeve"; else SLEEVE_FLAG="--no-sleeve"; fi

echo "===== $(date '+%F %T %Z') mode=$MODE capital=$CAPITAL vol=$VOL_TARGET sleeve=$SLEEVE =====" >> "$RUN_LOG"

# 入队一条消息(整块正文,首行作标题),永不覆盖已有未送达消息
enqueue() { printf '%s\n' "$1" > "$SPOOL/$(date +%s%N).msg"; }

# 按时间顺序推送队列中所有消息;失败即停,留待下次运行重试。
# 仅当 Telegram 返回 ok=true 才视为送达;未配置 token 视为投递失败,队列保留。
flush_spool() {
    local f body resp
    for f in $(ls "$SPOOL"/*.msg 2>/dev/null | sort); do
        if [ -z "$TG_BOT_TOKEN" ] || [ -z "$TG_CHAT_ID" ]; then
            echo "(未配置 TG_BOT_TOKEN/TG_CHAT_ID,消息保留待补发: $f)" >> "$RUN_LOG"
            return 1
        fi
        body=$(cat "$f")
        resp=$(curl -sS --max-time 20 "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" \
            --data-urlencode "chat_id=${TG_CHAT_ID}" --data-urlencode "text=$body" 2>&1)
        if echo "$resp" | grep -q '"ok" *: *true'; then
            echo "推送成功: $(head -1 "$f")" >> "$RUN_LOG"
            rm -f "$f"
        else
            echo "推送失败(下次运行重试): $resp" >> "$RUN_LOG"
            return 1
        fi
    done
    return 0
}

# 先补发历史欠账,再产生新消息
flush_spool; BACKLOG_STATUS=$?

# ---- 1. 每日信号 ----
SIGNAL_OUT=$("${DOCKER_RUN[@]}" daily_signal.py --mode "$MODE" --capital "$CAPITAL" "$VOL_FLAG" "$SLEEVE_FLAG" 2>&1)
SIGNAL_STATUS=$?
echo "$SIGNAL_OUT" >> "$RUN_LOG"

if [ $SIGNAL_STATUS -ne 0 ]; then
    enqueue "【ETF量化】信号脚本运行失败
$SIGNAL_OUT"
    flush_spool
    exit 1
fi

# ---- 2. 昨收校验(只告警不阻断)----
PRICE_OUT=$("${DOCKER_RUN[@]}" check_prices.py 2>&1) \
    || PRICE_OUT="$PRICE_OUT
⚠ 昨收校验脚本启动失败,见日志"
echo "$PRICE_OUT" >> "$RUN_LOG"
WARN=$(echo "$PRICE_OUT" | grep '⚠' || true)

# ---- 3. 回撤检查(与线上同口径,只告警不阻断)----
RISK_OUT=$("${DOCKER_RUN[@]}" check_risk.py --mode "$MODE" --capital "$CAPITAL" "$VOL_FLAG" "$SLEEVE_FLAG" 2>&1) \
    || RISK_OUT="$RISK_OUT
⚠ 回撤检查脚本启动失败,见日志"
echo "$RISK_OUT" >> "$RUN_LOG"
RISK=$(echo "$RISK_OUT" | grep '⚠' || true)

# 组装告警块
ALERTS=""
[ -n "$WARN" ] && ALERTS="数据校验告警:
$WARN"
[ -n "$RISK" ] && ALERTS="${ALERTS:+$ALERTS

}回撤告警:
$RISK"

# ---- 4. 持久化优先:两文件相对 HEAD 有变更才提交(路径受限,不裹挟其他暂存改动)----
# CHANGED 等价于原 daily.yml 的 new_signal 判据(信号日志/流水变化),而非 stdout 标记,
# 覆盖"信号已记录但 write_planned 重写了 executions 计划行"的情形。
CHANGED=false
if ! git diff --quiet HEAD -- "${LEDGER[@]}" 2>/dev/null; then
    CHANGED=true
fi
if [ "$CHANGED" = "true" ]; then
    if git commit -q -m "信号日志 $(TZ=Asia/Hong_Kong date '+%F')" -- "${LEDGER[@]}" >> "$RUN_LOG" 2>&1; then
        echo "本地已提交" >> "$RUN_LOG"
    else
        # 持久化失败:不推送"已出信号",改推失败告警(含已探到的风控/校验告警),留待人工
        echo "本地提交失败" >> "$RUN_LOG"
        enqueue "【ETF量化】本地提交失败,信号未持久化,请手动检查${ALERTS:+

$ALERTS}"
        flush_spool
        exit 1
    fi
fi

# ---- 5. 判定是否推送信号 ----
# NEW_SIGNAL = 两文件已成功提交(CHANGED) 且能解析到目标持仓行
SIGNAL_LINE=$(echo "$SIGNAL_OUT" | grep "目标持仓" | head -1)
if [ "$CHANGED" = "true" ] && [ -n "$SIGNAL_LINE" ]; then
    NEW_SIGNAL=true
else
    NEW_SIGNAL=false
fi

# ---- 6. 组装并推送(有新信号 或 有告警 才推送)----
if [ "$NEW_SIGNAL" = "true" ]; then
    if echo "$SIGNAL_OUT" | grep -qE "买入|卖出"; then
        ORDERS=$(echo "$SIGNAL_OUT" | grep -E "买入|卖出")
        # 执行日措辞由 daily_signal.py 给出(今日/具体日期),不在此硬编码
        HEADER=$(echo "$SIGNAL_OUT" | grep "调仓指令" | head -1 | sed 's/^-*[[:space:]]*//; s/[[:space:]]*-*$//')
        TEXT="【ETF量化】开盘需调仓
━━━━━━━━━━━━
$SIGNAL_LINE

${HEADER:-调仓指令:}
$ORDERS

⚠ 请按6位证券代码下单,成交前核对代码一致(同指数基金名称相近,勿凭名称选)"
    else
        TEXT="【ETF量化】今日无操作
━━━━━━━━━━━━
$SIGNAL_LINE"
    fi
    [ -n "$ALERTS" ] && TEXT="$TEXT

$ALERTS"
    enqueue "$TEXT"
    flush_spool
elif [ -n "$ALERTS" ]; then
    enqueue "【ETF量化】告警
━━━━━━━━━━━━
$ALERTS"
    flush_spool
else
    # 无新信号也无告警:退出码反映历史欠账补发结果
    exit $BACKLOG_STATUS
fi
