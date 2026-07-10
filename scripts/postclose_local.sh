#!/usr/bin/env bash
# 收盘后档(交易日 17:03,机器 TZ=Asia/Hong_Kong):今日整体盈亏 = 账户小结 + 持仓偏离。
#
# 与早 8 点档(daily_local.sh,只讲今日操作)职责分离:
#   - 早 8 点档:用昨收信号讲「今日调仓指令 / 无需调仓」;
#   - 收盘后档(本脚本):用**妙想(mx_data)当日收盘**兜底(akshare 收盘后 ~22:00 才出当日 K),
#     算今日账户小结 + 持仓偏离,仅推 Telegram(网页仍走晚上 akshare + 自动刷新)。
#
# 硬约束:mx_data 只用于本展示档;money 信号路径(daily_signal/check_risk/8点档)永不碰 mx。
#   mx 只追加「end 当日一根 bar」到 in-memory 行情,绝不写 data/*.csv(见 mx_fetch_latest.py)。
#
# 与 daily_local 队列完全隔离:独立 flock / spool / 回执,不碰 daily_local 的 SPOOL/RECEIPT_DIR。
#
# cron(交易日每天 17:03):
#   3 17 * * 1-5  /home/logan/Projects/quant/scripts/postclose_local.sh >> .../output/cron_wrapper.log 2>&1
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DOCKER_RUN=(docker run --rm --network=host -v "$PROJECT_DIR":/work quant python)

LOG_DIR="output"
SPOOL="$LOG_DIR/postclose_spool"          # 独立队列,不碰 daily_local 的 push_spool
RECEIPT_DIR="$LOG_DIR/.pushed_postclose"  # 每个收盘日一份回执(幂等/节假日跳过)
MX_JSON="$LOG_DIR/mx_latest.json"
mkdir -p "$LOG_DIR" "$SPOOL" "$RECEIPT_DIR"
RUN_LOG="$LOG_DIR/cron_postclose.log"

# 独立并发护栏:与 daily_local 分锁,互不阻塞
exec 8>"$LOG_DIR/.postclose.lock"
if ! flock -n 8; then
    echo "$(date '+%F %T %Z') 另一收盘后实例运行中,跳过" >> "$RUN_LOG"
    exit 0
fi

# 只解析白名单键,绝不把 .env 当 shell 执行(防注入);新增 MX_APIKEY(仅本档用)
envval() { [ -f .env ] && grep -E "^$1=" .env | tail -1 | cut -d= -f2-; }
TG_BOT_TOKEN="$(envval TG_BOT_TOKEN || true)"
TG_CHAT_ID="$(envval TG_CHAT_ID || true)"
MODE="$(envval SIGNAL_MODE || true)"; MODE="${MODE:-single}"
CAPITAL="$(envval SIGNAL_CAPITAL || true)"; CAPITAL="${CAPITAL:-10000}"
VOL_TARGET="$(envval SIGNAL_VOL_TARGET || true)"; VOL_TARGET="${VOL_TARGET:-false}"
SLEEVE="$(envval SIGNAL_SLEEVE || true)"; SLEEVE="${SLEEVE:-false}"
MX_APIKEY="$(envval MX_APIKEY || true)"

if [ "$VOL_TARGET" = "true" ]; then VOL_FLAG="--vol-target"; else VOL_FLAG="--no-vol-target"; fi
if [ "$SLEEVE" = "true" ]; then SLEEVE_FLAG="--sleeve"; else SLEEVE_FLAG="--no-sleeve"; fi

echo "===== $(date '+%F %T %Z') postclose mode=$MODE capital=$CAPITAL vol=$VOL_TARGET sleeve=$SLEEVE =====" >> "$RUN_LOG"

REPORT_LINE="报告 https://abacusflow.github.io/quant/"

# 推送 spool 中所有消息;失败即停,留待下次重试。仅 Telegram ok=true 才视为送达。
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

# ---- 1. 池内标的(单一真相:config.ETF_POOL)----
SYMBOLS=$("${DOCKER_RUN[@]}" -c "import config;print(' '.join(config.ETF_POOL))" 2>>"$RUN_LOG")
if [ -z "$SYMBOLS" ]; then
    echo "无法读取 ETF_POOL,跳过(不影响 money 档)" >> "$RUN_LOG"
    exit 0
fi

# ---- 2. 妙想当日收盘兜底(宿主机跑,失败不影响任何 money 档)----
END=$(TZ=Asia/Hong_Kong date +%F)
if [ -z "$MX_APIKEY" ]; then
    echo "未配置 MX_APIKEY,跳过收盘后档(需妙想密钥拿当日收盘)" >> "$RUN_LOG"
    exit 0
fi
# shellcheck disable=SC2086
if ! MX_APIKEY="$MX_APIKEY" python3 scripts/mx_fetch_latest.py $SYMBOLS --end "$END" --out "$MX_JSON" >> "$RUN_LOG" 2>&1; then
    echo "mx_fetch_latest 失败,跳过收盘后档(等晚间 akshare + 自动刷新网页)" >> "$RUN_LOG"
    exit 0   # 尽力而为:拿不到当日收盘就不发,绝不影响 money 档
fi

# ---- 3. 幂等/节假日跳过:回执按 mx 实际最新交易日键 ----
KEY=$(python3 -c "import json;print(json.load(open('$MX_JSON'))['end'])" 2>>"$RUN_LOG")
if [ -z "$KEY" ]; then
    echo "无法解析 mx_latest.json 的 end,跳过" >> "$RUN_LOG"
    exit 0
fi
# 回执目录轻量清理:删 30 天前旧回执
find "$RECEIPT_DIR" -type f -mtime +30 -delete 2>/dev/null || true
if [ -f "$RECEIPT_DIR/$KEY" ]; then
    echo "收盘日 $KEY 已推送过,跳过(幂等)" >> "$RUN_LOG"
    exit 0
fi

# ---- 4. 账户小结 + 持仓偏离(注入 mx 当日 bar)----
BLOCK=$("${DOCKER_RUN[@]}" portfolio_status.py --mode "$MODE" --capital "$CAPITAL" \
    "$VOL_FLAG" "$SLEEVE_FLAG" --mx-fallback "$MX_JSON" 2>>"$RUN_LOG" || true)
if [ -z "$BLOCK" ]; then
    echo "收盘后小结为空($KEY,数据未就绪/流水晚于收盘),跳过" >> "$RUN_LOG"
    exit 0
fi

# ---- 5. 组装并原子入队 + 推送 ----
TEXT="【ETF量化】收盘小结($KEY)
━━━━━━━━━━━━
$BLOCK

$REPORT_LINE"

SPOOL_FILE="$SPOOL/postclose-${KEY}.msg"
rm -f "$SPOOL"/.tmp-* 2>/dev/null || true   # 清理上轮孤儿临时档
[ -e "$SPOOL_FILE" ] && [ ! -s "$SPOOL_FILE" ] && rm -f "$SPOOL_FILE"   # 异常空档自愈
if [ ! -e "$SPOOL_FILE" ]; then
    TMP_SPOOL=$(mktemp "$SPOOL/.tmp-XXXXXX") || TMP_SPOOL=""
    if [ -n "$TMP_SPOOL" ] && printf '%s\n' "$TEXT" > "$TMP_SPOOL"; then
        ln "$TMP_SPOOL" "$SPOOL_FILE" 2>/dev/null || true   # 目标已存在(上轮滞留)则失败,无害
    fi
    [ -n "$TMP_SPOOL" ] && rm -f "$TMP_SPOOL"
fi

flush_spool
# 消息确已送达(spool 档已被 flush 删除)才写回执;未送达(creds 缺/网络失败)保留队列下次重试
if [ ! -e "$SPOOL_FILE" ]; then
    : > "$RECEIPT_DIR/$KEY"
    echo "收盘后小结已投递并写回执: $KEY" >> "$RUN_LOG"
fi
exit 0
