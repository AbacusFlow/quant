#!/usr/bin/env bash
# 模拟盘信号自动跑 + Server酱微信推送。
# cron(交易日收盘后,17:35 为兜底:信号重复时仅补发未送达的推送):
#   35 16 * * 1-5  /home/logan/Projects/quant/scripts/daily_signal_push.sh
#   35 17 * * 1-5  /home/logan/Projects/quant/scripts/daily_signal_push.sh
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# 只解析白名单键,不把 .env 当 shell 执行
envval() { [ -f .env ] && grep -E "^$1=" .env | tail -1 | cut -d= -f2-; }
SENDKEY="$(envval SENDKEY || true)"
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
# 仅当 Server酱 应用层返回 code=0 才视为送达;未配置 SENDKEY 视为投递失败,
# 队列保留(填好 .env 后下次运行自动补发)。
flush_spool() {
    local f title body resp
    for f in $(ls "$SPOOL"/*.msg 2>/dev/null | sort); do
        if [ -z "$SENDKEY" ]; then
            echo "(未配置 SENDKEY,消息保留待补发: $f)" >> "$RUN_LOG"
            return 1
        fi
        title=$(head -1 "$f")
        body=$(tail -n +2 "$f")
        resp=$(curl -sS --fail --max-time 15 "https://sctapi.ftqq.com/${SENDKEY}.send" \
            --data-urlencode "title=$title" \
            --data-urlencode "desp=$body" 2>&1)
        if [ $? -eq 0 ] && echo "$resp" | grep -q '"code" *: *0'; then
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

# 信号已记录过(节假日/兜底运行)→ 无新消息,退出码反映补发结果
if echo "$OUTPUT" | grep -q "不重复记录"; then
    exit $BACKLOG_STATUS
fi

SIGNAL=$(echo "$OUTPUT" | grep "目标持仓" | head -1)
if echo "$OUTPUT" | grep -qE "买入|卖出"; then
    ORDERS=$(echo "$OUTPUT" | grep -E "买入|卖出")
    enqueue "[量化] 明早需调仓!
$SIGNAL

调仓指令(明日开盘执行):
$ORDERS"
else
    enqueue "[量化] 今日无操作
$SIGNAL"
fi
flush_spool
