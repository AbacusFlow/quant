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
#   6. Telegram spool 推送(成功才出队,失败留待下次;交易日有信号行即推每日提醒,或有告警才推)
#
# cron(交易日每天早 8 点一次,机器 TZ=Asia/Hong_Kong;开盘前拿到昨收信号+持仓简况):
#   3 8 * * 1-5  /home/logan/Projects/quant/scripts/daily_local.sh
# 交易日每天推一条消息(含信号/持仓偏离/账户简况/报告链接);非交易日 daily_signal 无信号行→不推。
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DOCKER_RUN=(docker run --rm --network=host -v "$PROJECT_DIR":/work quant python)
LEDGER=(output/signal_log.csv output/executions.csv)

LOG_DIR="output"
SPOOL="$LOG_DIR/push_spool"          # 待推送消息队列:每条一个文件,推送成功才删除
RECEIPT_DIR="$LOG_DIR/.pushed_signals"  # 每个已投递信号键一个空回执文件(投递幂等 + 跨日自愈)
mkdir -p "$LOG_DIR" "$SPOOL" "$RECEIPT_DIR"
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
# 参数 $1="generic":只发非键控档(告警/错误欠账),跳过 signal-*.msg。用于"今日最新信号尚未算出"
#   的早期时点(脚本开头补发、以及信号/提交失败的错误告警),避免抢先发出昨天的陈旧信号提醒。
#   缺省/"all":全发(键控信号 + 告警),用于第 5 步已确定最新键并淘汰陈旧键之后。
flush_spool() {
    local only="${1:-all}" f body resp base
    for f in $(ls "$SPOOL"/*.msg 2>/dev/null | sort); do
        base=$(basename "$f")
        # generic 模式跳过键控信号档(signal-*.msg),留待最新键确定后再发
        if [ "$only" = "generic" ] && [ "${base#signal-}" != "$base" ]; then
            continue
        fi
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

# 先补发历史告警/错误欠账(不含键控信号——今日最新信号尚未算出,勿抢先发陈旧信号)
flush_spool generic; BACKLOG_STATUS=$?

# ---- 1. 每日信号 ----
SIGNAL_OUT=$("${DOCKER_RUN[@]}" daily_signal.py --mode "$MODE" --capital "$CAPITAL" "$VOL_FLAG" "$SLEEVE_FLAG" 2>&1)
SIGNAL_STATUS=$?
echo "$SIGNAL_OUT" >> "$RUN_LOG"

if [ $SIGNAL_STATUS -ne 0 ]; then
    enqueue "【ETF量化】信号脚本运行失败
$SIGNAL_OUT"
    flush_spool generic   # 只发本条错误告警,勿抢先发未确认的陈旧键控信号
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
# CHANGED 只管"是否需要本地提交账本"(新信号行/计划行重写/record_trade 手动改动都算),
# 与"是否推送每日提醒"解耦(推送见第 5 步的投递标记判据)。
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
        flush_spool generic   # 只发本条失败告警,勿抢先发未确认的陈旧键控信号
        exit 1
    fi
fi

# ---- 5. 组装并推送(每交易日一条新信号提醒;非交易日/重复运行 → 仅有告警才推)----
# 投递用"每个信号键一份回执"(RECEIPT_DIR/键),而非单值标记:
#   身份键 = signal_log.csv 的 (signal_date,mode);daily_signal 每交易日至多追加一条新
#   signal_date 行(同日期+模式幂等)。
#   1) 跨日自愈:扫描近期所有合法键(非仅末行)。只要"最新键"还没回执 **且本轮确实产出了当前信号**
#      (SIGNAL_LINE 目标持仓行在)就推当前消息;推送落盘后把这批近期键一并写回执对账,旧漏推键不会
#      永久悬空、也不会被当作陈旧提醒补发。这是刻意取舍:保证用户总能收到"最新可执行消息至少一次",
#      被取代的中间信号不再单独复活。
#   2) **必须 SIGNAL_LINE 在才构建新消息**:LATEST_KEY 取自历史 signal_log,非交易日(工作日休市,如
#      国庆/春节)其值仍是上一交易日的键、非空;而 daily_signal 此时早退不产出当前信号(SIGNAL_LINE 空)。
#      若仅凭"最新键无回执"就构建,会用休市当天的空 SIGNAL_OUT 拼出空信号消息并误写回执,把真实信号
#      永久吞掉。故 push 分支加 SIGNAL_LINE 非空这一闸:交易日重跑必打印目标持仓→自愈成立;休市日
#      不产信号→不伪造、不写回执,留待下个交易日的真实信号自然取代(supersede)。
#   3) 入队幂等:spool 文件按键命名 + 原子 ln 发布 → 上轮已入队(写回执前被杀)则文件已存在,不重复
#      入队。回执仅在"键命名 spool 确为非空常规文件(已完整落盘)"后才写,入队失败/无当前信号不误标已投递。
#   4) 陈旧淘汰:supersede_stale_signals 删掉所有非最新键的滞留信号档,确保只会送出最新那份。
#   注:本地 spool 只能保证"至少送达一次"。若 Telegram 已接收但删档前进程被杀,极端下会重发一次
#      (无 Telegram 侧幂等无法根除);对每日一次的个人提醒可接受。
SIGNAL_LINE=$(echo "$SIGNAL_OUT" | grep "目标持仓" | head -1)
REPORT_LINE="报告 https://abacusflow.github.io/quant/"

# 回执目录轻量清理:删 30 天前的旧回执,避免无限增长(文件为空,影响极小)
find "$RECEIPT_DIR" -type f -mtime +30 -delete 2>/dev/null || true

# 近期所有合法信号键(校验 signal_date 为 YYYY-MM-DD 且 mode == 本次运行的 $MODE):
# 只认当前模式的键——本脚本推的消息/持仓简况全用 $MODE,若混入另一模式的行(模式切换过渡期
# single↔ensemble 并存),全局取末行会张冠李戴:另一模式已有回执会误跳过本模式提醒,或把本模式
# 内容挂到另一模式键下。限定 $3==mode 即杜绝跨模式串键。同时天然剔除表头行、残缺末行(伪键
# ","/",ensemble")、含 /、..、空白等危险字段(防 spool/回执路径逃逸)。取最后 50 行足够覆盖跨日漏推窗口。
mapfile -t RECENT_KEYS < <(tail -n 50 output/signal_log.csv 2>/dev/null \
    | awk -F, -v mode="$MODE" '$2 ~ /^[0-9]{4}-[0-9]{2}-[0-9]{2}$/ && $3==mode {print $2","$3}')
LATEST_KEY=""
[ "${#RECENT_KEYS[@]}" -gt 0 ] && LATEST_KEY="${RECENT_KEYS[-1]}"
LATEST_SAFE=$(printf '%s' "$LATEST_KEY" | tr ',' '_')

# 对账写回执。分两类键处理,防止把"从未成功入队的最新真实信号"提前结清而永久吞掉:
#   • 非最新键:已被最新键取代(supersede 已删其 spool),按"只发最新、中间信号不复活"约定
#     直接结清,不再悬空。
#   • 最新键:仅当"已入队(非空键控 spool 档在)或已有回执"时才结清。否则——信号日 spool 发布前
#     进程被杀 / mktemp/printf/ln 失败,导致该键既无 spool 又无回执——保持悬空,待下个交易日拿到
#     真实 SIGNAL_LINE 重建投递,绝不在非 push 分支(休市日/仅告警)提前伪造回执把它永久跳过。
reconcile_recent() {
    local k safe
    for k in "${RECENT_KEYS[@]}"; do
        safe=$(printf '%s' "$k" | tr ',' '_')
        if [ "$k" = "$LATEST_KEY" ]; then
            [ -s "$SPOOL/signal-${safe}.msg" ] || [ -f "$RECEIPT_DIR/$safe" ] || continue
        fi
        : > "$RECEIPT_DIR/$safe"
    done
}

# 淘汰陈旧键控信号档:删掉所有非"当前最新键"的 signal-*.msg(它们对应已被最新信号取代的旧信号,
# 上一轮送达失败而滞留;若此刻直接 flush 会先发出昨天的陈旧提醒)。仅保留最新键那份(可能是本轮
# 刚入队的,也可能是上一轮送达失败滞留的同键档,需继续重试)。LATEST_KEY 为空(非交易日/无本模式
# 键)时不淘汰——此时滞留的键控档是"最后一次已知信号"、尚未确认送达,应保留重试而非删除。
supersede_stale_signals() {
    [ -n "$LATEST_SAFE" ] || return 0
    local f base keep="signal-${LATEST_SAFE}.msg"
    for f in "$SPOOL"/signal-*.msg; do
        [ -e "$f" ] || continue
        base=$(basename "$f")
        [ "$base" = "$keep" ] || { rm -f "$f"; echo "淘汰陈旧信号档: $base" >> "$RUN_LOG"; }
    done
}
supersede_stale_signals   # 无论走哪个分支,先淘汰被最新键取代的陈旧信号档,确保只会发最新那份

# 推送最新信号的前置条件三连:
#   1) LATEST_KEY 非空 —— 本模式历史上有过信号键;
#   2) 该键尚无回执 —— 还没投递过(幂等/跨日自愈);
#   3) SIGNAL_LINE 非空 —— 本轮 daily_signal 确实产出了当日「目标持仓」信号行。
# 第 3 条是关键:LATEST_KEY 取自历史 signal_log.csv,交易日休市(工作日节假日)时它仍是
# 上一交易日的键(非空),但 daily_signal 当天不产信号 → SIGNAL_LINE 为空。若缺此闸,
# 休市日会用空信号 SIGNAL_OUT 现场构建并「回执」一份 signal-$LATEST_SAFE.msg,
# 把真正的历史信号伪装成已投递(内容还是空的)。故必须 SIGNAL_LINE 在才构建新消息。
# 休市日/无新信号但仍有滞留键控档需重试的情形,交给下方 else 分支 reconcile+flush 补发。
if [ -n "$LATEST_KEY" ] && [ ! -f "$RECEIPT_DIR/$LATEST_SAFE" ] && [ -n "$SIGNAL_LINE" ]; then
    if echo "$SIGNAL_OUT" | grep -qE "买入|卖出"; then
        # 调仓日:信号变化 → 展示调仓指令(权威操作);账户简况只带一行,不再列持仓偏离
        # (避免"调仓指令"与"持仓偏离"给出两套买卖股数造成困惑,Codex 审查项/防杂乱)
        ORDERS=$(echo "$SIGNAL_OUT" | grep -E "买入|卖出")
        # 执行日措辞由 daily_signal.py 给出(今日/具体日期),不在此硬编码
        HEADER=$(echo "$SIGNAL_OUT" | grep "调仓指令" | head -1 | sed 's/^-*[[:space:]]*//; s/[[:space:]]*-*$//')
        STATUS_BLOCK=$("${DOCKER_RUN[@]}" portfolio_status.py --mode "$MODE" --capital "$CAPITAL" \
            "$VOL_FLAG" "$SLEEVE_FLAG" --account-only 2>>"$RUN_LOG" || true)
        TEXT="【ETF量化】开盘需调仓
━━━━━━━━━━━━
$SIGNAL_LINE

${HEADER:-调仓指令:}
$ORDERS

⚠ 请按6位证券代码下单,成交前核对代码一致(同指数基金名称相近,勿凭名称选)"
    else
        # 无信号变化(含 commit 后漏推的自愈补推):展示持仓偏离(>5pp 才列)+ 账户简况,
        # 给用户可执行的漂移纠正建议(自愈补推时原调仓指令因幂等已不复现,持仓偏离同样可操作)
        STATUS_BLOCK=$("${DOCKER_RUN[@]}" portfolio_status.py --mode "$MODE" --capital "$CAPITAL" \
            "$VOL_FLAG" "$SLEEVE_FLAG" 2>>"$RUN_LOG" || true)
        TEXT="【ETF量化】每日提醒
━━━━━━━━━━━━
$SIGNAL_LINE"
    fi
    [ -n "$STATUS_BLOCK" ] && TEXT="$TEXT

$STATUS_BLOCK"
    [ -n "$ALERTS" ] && TEXT="$TEXT

$ALERTS"
    TEXT="$TEXT

$REPORT_LINE"
    # 按键命名原子入队:先写全临时文件、再 ln 硬链到位(同目录同文件系统,ln 不覆盖已存在目标)。
    # 不用 `printf > $SPOOL_FILE` 直写——那样 redirection 先建空文件,若 printf 未写完就被杀/磁盘满,
    # 会留下空档/半档,下轮 `-e` 判定已入队却永远发不出或发残缺。原子发布保证 spool 要么完整、要么不存在。
    SPOOL_FILE="$SPOOL/signal-${LATEST_SAFE}.msg"
    rm -f "$SPOOL"/.tmp-* 2>/dev/null || true   # 清理上轮 mktemp 后被杀留下的孤儿临时档(非 *.msg,不入队)
    # 异常空档(外部篡改/历史遗留;原子发布下本不该出现)自愈:删掉以便下方原子重建,不让空文件卡死
    [ -e "$SPOOL_FILE" ] && [ ! -s "$SPOOL_FILE" ] && rm -f "$SPOOL_FILE"
    if [ ! -e "$SPOOL_FILE" ]; then
        TMP_SPOOL=$(mktemp "$SPOOL/.tmp-XXXXXX") || TMP_SPOOL=""
        if [ -n "$TMP_SPOOL" ] && printf '%s\n' "$TEXT" > "$TMP_SPOOL"; then
            ln "$TMP_SPOOL" "$SPOOL_FILE" 2>/dev/null || true   # 目标已存在(并发/上轮)则失败,无害
        fi
        [ -n "$TMP_SPOOL" ] && rm -f "$TMP_SPOOL"
    fi
    # 仅当键命名 spool 确为非空常规文件才对账(原子发布下即"已完整入队"),否则留待下次重推,绝不误标已投递
    if [ -s "$SPOOL_FILE" ]; then
        reconcile_recent   # 当前消息取代所有更早漏推键 → 一并结清回执
    fi
    flush_spool
elif [ -n "$ALERTS" ]; then
    # 最新信号已投递(或今日无信号),但仍有告警 → 单独推告警;顺带结清任何旧漏推键
    reconcile_recent
    enqueue "【ETF量化】告警
━━━━━━━━━━━━
$ALERTS

$REPORT_LINE"
    flush_spool
else
    # 非交易日/最新信号已投递且无告警:结清旧键;若最新键那份键控档仍滞留(上轮送达失败),
    # 经上面 supersede 后只剩它,再 flush 一次重试补发(不会发出陈旧信号,陈旧的已被淘汰)。
    reconcile_recent
    flush_spool
    exit $?
fi
