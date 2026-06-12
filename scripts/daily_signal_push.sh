#!/usr/bin/env bash
# 模拟盘信号自动跑 + Telegram 机器人推送(手动备用;正式自动化在 GitHub Actions)。
# cron(交易日早上开盘前,基于前一交易日收盘;09:05 为兜底:信号重复时仅补发未送达的推送):
#   35 8 * * 1-5  /home/logan/Projects/quant/scripts/daily_signal_push.sh
#   5  9 * * 1-5  /home/logan/Projects/quant/scripts/daily_signal_push.sh
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# 只解析白名单键,不把 .env 当 shell 执行
envval() { [ -f .env ] && grep -E "^$1=" .env | tail -1 | cut -d= -f2-; }
TG_BOT_TOKEN="$(envval TG_BOT_TOKEN || true)"
TG_CHAT_ID="$(envval TG_CHAT_ID || true)"
MODE="$(envval SIGNAL_MODE || true)"; MODE="${MODE:-single}"
CAPITAL="$(envval SIGNAL_CAPITAL || true)"; CAPITAL="${CAPITAL:-10000}"

LOG_DIR="output"
SPOOL="$LOG_DIR/push_spool"        # 待推送消息队列:每条一个文件,推送成功才删除
mkdir -p "$LOG_DIR" "$SPOOL"
RUN_LOG="$LOG_DIR/cron_signal.log"

echo "===== $(date '+%F %T') mode=$MODE capital=$CAPITAL =====" >> "$RUN_LOG"

# 入队一条消息(第1行=标题,其余=正文),永不覆盖已有未送达消息
enqueue() {
    printf '%s\n' "$1" > "$SPOOL/$(date +%s%N).msg"
}

# 按时间顺序推送队列中所有消息;失败即停,留待下次运行重试。
# 仅当 Telegram 返回 ok=true 才视为送达;未配置 TG_BOT_TOKEN/TG_CHAT_ID 视为投递失败,
# 队列保留(填好 .env 后下次运行自动补发)。
flush_spool() {
    local f title body resp
    for f in $(ls "$SPOOL"/*.msg 2>/dev/null | sort); do
        if [ -z "$TG_BOT_TOKEN" ] || [ -z "$TG_CHAT_ID" ]; then
            echo "(未配置 TG_BOT_TOKEN/TG_CHAT_ID,消息保留待补发: $f)" >> "$RUN_LOG"
            return 1
        fi
        title=$(head -1 "$f")
        body=$(cat "$f")
        resp=$(curl -sS --fail --max-time 15 "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" \
            --data-urlencode "chat_id=${TG_CHAT_ID}" \
            --data-urlencode "text=$body" 2>&1)
        if [ $? -eq 0 ] && echo "$resp" | grep -q '"ok" *: *true'; then
            echo "推送成功: $title" >> "$RUN_LOG"
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

OUTPUT=$(docker run --rm --network=host -v "$PROJECT_DIR":/work quant \
    python daily_signal.py --mode "$MODE" --capital "$CAPITAL" 2>&1)
STATUS=$?
echo "$OUTPUT" >> "$RUN_LOG"

if [ $STATUS -ne 0 ]; then
    enqueue "[量化] 信号脚本运行失败
$OUTPUT"
    flush_spool
    exit 1
fi

# 信号已记录过(兜底运行)或非交易日 → 无新消息,退出码反映补发结果
if echo "$OUTPUT" | grep -qE "不重复记录|非A股交易日"; then
    exit $BACKLOG_STATUS
fi

SIGNAL=$(echo "$OUTPUT" | grep "目标持仓" | head -1)
# 未解析到目标持仓(意外输出格式)→ 不入队,避免推空消息
if [ -z "$SIGNAL" ]; then
    echo "(未解析到目标持仓,跳过推送)" >> "$RUN_LOG"
    exit $BACKLOG_STATUS
fi
if echo "$OUTPUT" | grep -qE "买入|卖出"; then
    ORDERS=$(echo "$OUTPUT" | grep -E "买入|卖出")
    # 执行日措辞由 daily_signal.py 给出(今日/具体日期),不在此硬编码
    HEADER=$(echo "$OUTPUT" | grep "调仓指令" | head -1 | sed 's/^-*[[:space:]]*//; s/[[:space:]]*-*$//')
    enqueue "[量化] 开盘需调仓!
$SIGNAL

${HEADER:-调仓指令:}
$ORDERS"
else
    enqueue "[量化] 今日无操作
$SIGNAL"
fi
flush_spool
