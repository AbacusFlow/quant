#!/usr/bin/env bash
# 月度复盘提醒(取代已停用的 GitHub reminders.yml,本地 cron 版)。
#
# cron(每月 1 号 09:47,机器 TZ=Asia/Hong_Kong):
#   47 9 1 * *  /home/logan/Projects/quant/scripts/monthly_review_local.sh >> .../output/cron_wrapper.log 2>&1
#
# 行为:组一条复盘检查清单消息(1/4/7/10 月附季度检查项,6/12 月附半年评估及格线),
# 直接推 Telegram;推送失败或未配置 creds 时原子写入 daily_local 的 push_spool
# (非 signal-*.msg 属 generic 档,下个交易日早 8 点档 flush_spool 自动补发)。
# 纯提醒,不碰账本/信号/git,恒 exit 0(初始化/入队失败写 stderr 供 cron_wrapper.log 留痕)。
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || { echo "monthly_review: 定位项目目录失败" >&2; exit 0; }
cd "$PROJECT_DIR" || { echo "monthly_review: cd $PROJECT_DIR 失败" >&2; exit 0; }

LOG_DIR="output"
SPOOL="$LOG_DIR/push_spool"          # 与 daily_local 共用队列(generic 档)
mkdir -p "$LOG_DIR" "$SPOOL" || { echo "monthly_review: 创建 spool 目录失败" >&2; exit 0; }
RUN_LOG="$LOG_DIR/cron_signal.log"

# 只解析白名单键,绝不把 .env 当 shell 执行(防注入;同 daily_local.sh)
envval() { [ -f .env ] && grep -E "^$1=" .env | tail -1 | cut -d= -f2-; }
TG_BOT_TOKEN="$(envval TG_BOT_TOKEN || true)"
TG_CHAT_ID="$(envval TG_CHAT_ID || true)"

MONTH=$(TZ=Asia/Hong_Kong date +%m)   # POSIX 两位月,不用 GNU 扩展 %-m
EXTRA=""
case "$MONTH" in
    01|04|07|10) EXTRA="

【季度检查】① 季度 TWR vs 沪深300/ETF池等权 ② 换手与费用占比 ③ 资金规模是否需要调整 mode/capital";;
    06|12) EXTRA="

【半年评估(6/12月)】及格线=年化落在 10-20% 区间且跑赢沪深300与ETF池等权(TWR口径)。不及格再讨论调整;勿凭短期浮亏中途换策略";;
esac

TEXT="【ETF量化】月度复盘提醒($(TZ=Asia/Hong_Kong date +%Y-%m))
━━━━━━━━━━━━
① 看网页报告:TWR/绝对盈亏/当前回撤 vs 回测最大回撤 -18.4%(达 1.0 倍密切关注,1.5 倍暂停加仓)
② 实际持仓 vs 策略目标:偏离是否 ≤5pp(收盘后档每日也在提示)
③ 本月是否有偏离信号的主观操作?记录原因,复盘其损益$EXTRA

报告 https://abacusflow.github.io/quant/"

if [ -n "$TG_BOT_TOKEN" ] && [ -n "$TG_CHAT_ID" ]; then
    # token 经 -K- 配置管道传入,不上命令行;响应写日志前脱敏(同 daily_local.sh)
    resp=$(curl -sS --max-time 20 -K- \
        --data-urlencode "chat_id=${TG_CHAT_ID}" --data-urlencode "text=$TEXT" \
        <<<"url = \"https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage\"" 2>&1)
    resp=${resp//"$TG_BOT_TOKEN"/<TG_BOT_TOKEN>}
    if echo "$resp" | grep -q '"ok" *: *true'; then
        echo "$(date '+%F %T %Z') 月度复盘提醒已推送" >> "$RUN_LOG"
        exit 0
    fi
    echo "$(date '+%F %T %Z') 月度复盘推送失败,入队待早8点档补发: $resp" >> "$RUN_LOG"
else
    echo "$(date '+%F %T %Z') 未配置 TG creds,月度复盘提醒入队待补发" >> "$RUN_LOG"
fi

# 原子入队(同 daily_local 信号档模式):临时文件写全 → ln 硬链发布 → 删临时文件。
# 直接重定向到最终 .msg 会在写半截/磁盘满时留下残档被 flush 误发;mktemp 同时保证文件名唯一。
TMP_SPOOL=$(mktemp "$SPOOL/.tmp-monthly-XXXXXX") || { echo "monthly_review: mktemp 失败,提醒未入队" >&2; exit 0; }
if printf '%s\n' "$TEXT" > "$TMP_SPOOL"; then
    DEST="$SPOOL/monthly-$(TZ=Asia/Hong_Kong date +%Y-%m)-$$.msg"
    ln "$TMP_SPOOL" "$DEST" 2>/dev/null || echo "monthly_review: 发布入队失败($DEST)" >&2
else
    echo "monthly_review: 写临时档失败,提醒未入队" >&2
fi
rm -f "$TMP_SPOOL"
exit 0
