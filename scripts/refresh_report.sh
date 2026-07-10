#!/usr/bin/env bash
# 重建静态 Pages 报告并推送到 AbacusFlow,保持 https://abacusflow.github.io/quant/ 最新。
# 由 git post-commit 钩子在"账本/策略/报告"相关文件提交后后台调用;也可手动运行。
#
# 设计要点:
#   - 尽力而为:报告生成/推送失败绝不影响已完成的提交(始终 exit 0)。
#   - 合并(coalesce):用 .pending 标记 + 单持锁者循环,连续快速提交只重建到最新 HEAD,
#     不会因"锁被占用即退出"而丢刷新(Codex 审查项)。
#   - 快照隔离:从已提交 HEAD 的临时 detached worktree 生成,绝不读触发后新增的未提交改动,
#     也不覆盖主工作区里未提交的 docs/index.html(Codex 审查项)。
#   - 防挂死:docker 生成与 git push 均加 timeout,并 GIT_TERMINAL_PROMPT=0,避免持锁挂起。
#   - 递归护栏:自身提交 docs 时带 REFRESH_REPORT_RUNNING=1,钩子见此即跳过。
#   - 口径与线上一致:mode/capital/vol/sleeve 从 .env 白名单读(同 daily_local.sh)。
set -uo pipefail

# cron/钩子环境 PATH 可能很精简,补上 docker/git/flock 的常见位置
export PATH="/home/linuxbrew/.linuxbrew/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
# 离线/无凭据时 git 不弹交互,直接失败返回(配合 timeout 防持锁挂死)
export GIT_TERMINAL_PROMPT=0

# 由 post-commit 钩子后台拉起时,会继承 git 为钩子导出的 GIT_DIR/GIT_INDEX_FILE 等
# 本地仓库环境变量;它们指向主仓索引,会让本脚本内的 `git worktree add` 误把 .git/index
# 当路径打开而报 "index file open failed: Not a directory",导致每次钩子触发都必失败。
# 用 git 自己的权威清单 `git rev-parse --local-env-vars`(随 git 版本自适应,避免手写漏项)
# 清空全部继承变量,让脚本内的 git 命令按 cwd 正常解析仓库(手动运行时它们本就不存在)。
# 该清单是静态名单、无需身处仓库即可输出;git 缺失时 unset 无参数为无害空操作。
unset $(git rev-parse --local-env-vars 2>/dev/null) 2>/dev/null || true

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 0

LOG="output/refresh_report.log"
MARK="output/.refresh_report.pending"
mkdir -p output

# 标记"有待办":谁持锁谁负责把标记清掉前的所有请求合并做掉
: > "$MARK"

# 串行护栏:一次只跑一个重建。此处**阻塞等待**锁(非 -n):抢不到的调用不会立即退出,
# 而是排队成为下一个持锁者,拿到锁后检查标记——若已被前者处理则空转退出,否则补跑。
# 这样彻底杜绝"前者刚判无标记、后者在其释放锁前 touch 标记"造成的 lost-wakeup 孤儿标记
# (Codex 审查项)。持锁者内部的阻塞操作(docker/push)均有 timeout,故不会无限占锁。
exec 8>"output/.refresh_report.lock"
flock 8

# 只解析白名单键,绝不把 .env 当 shell 执行
envval() { [ -f .env ] && grep -E "^$1=" .env | tail -1 | cut -d= -f2-; }

# 合并循环:认领标记 → 重建最新 HEAD → 提交/推送;期间来的新提交会重置标记 → 再来一轮
while [ -f "$MARK" ]; do
    rm -f "$MARK"

    MODE="$(envval SIGNAL_MODE || true)"; MODE="${MODE:-ensemble}"
    CAPITAL="$(envval SIGNAL_CAPITAL || true)"; CAPITAL="${CAPITAL:-168000}"
    VOL_TARGET="$(envval SIGNAL_VOL_TARGET || true)"; VOL_TARGET="${VOL_TARGET:-false}"
    SLEEVE="$(envval SIGNAL_SLEEVE || true)"; SLEEVE="${SLEEVE:-false}"
    if [ "$VOL_TARGET" = "true" ]; then VOL_FLAG="--vol-target"; else VOL_FLAG="--no-vol-target"; fi
    if [ "$SLEEVE" = "true" ]; then SLEEVE_FLAG="--sleeve"; else SLEEVE_FLAG="--no-sleeve"; fi

    echo "===== $(date '+%F %T %Z') 重建报告 mode=$MODE capital=$CAPITAL vol=$VOL_TARGET sleeve=$SLEEVE =====" >> "$LOG"

    # 从已提交 HEAD 的干净快照生成(隔离主工作区未提交改动)
    WT="$(mktemp -d)"
    if ! git worktree add --detach --quiet "$WT" HEAD >> "$LOG" 2>&1; then
        echo "$(date '+%F %T %Z') 创建临时 worktree 失败,放弃本轮" >> "$LOG"
        rm -rf "$WT"
        continue
    fi

    ok=true
    if ! timeout 360 docker run --rm --network=host -v "$WT":/work quant \
            python report_web.py --mode "$MODE" --capital "$CAPITAL" "$VOL_FLAG" "$SLEEVE_FLAG" >> "$LOG" 2>&1; then
        echo "$(date '+%F %T %Z') 报告生成失败/超时,放弃本轮" >> "$LOG"
        ok=false
    fi
    if [ "$ok" = true ] && [ ! -f "$WT/site/index.html" ]; then
        echo "$(date '+%F %T %Z') 未产出 site/index.html,放弃本轮" >> "$LOG"
        ok=false
    fi

    # 发布前护栏(必须在移除临时 worktree 前完成拷贝,因源文件在 $WT 内):
    # 主工作区 docs/index.html 若有未提交/已暂存改动则跳过,避免覆盖(docs 本是机器生成、
    # 正常干净;此判定兑现"不覆盖未提交 docs"的承诺,Codex 审查项)。
    if [ "$ok" = true ]; then
        if ! git diff --quiet -- docs/index.html 2>/dev/null || ! git diff --cached --quiet -- docs/index.html 2>/dev/null; then
            echo "$(date '+%F %T %Z') 主工作区 docs/index.html 有未提交改动,跳过发布以免覆盖" >> "$LOG"
            ok=false
        else
            cp "$WT/site/index.html" docs/index.html
        fi
    fi

    git worktree remove --force "$WT" >> "$LOG" 2>&1 || rm -rf "$WT"
    [ "$ok" = true ] || continue

    # docs 有变化才提交(路径受限);递归护栏避免钩子再次触发
    if git diff --quiet HEAD -- docs/index.html 2>/dev/null; then
        echo "$(date '+%F %T %Z') 报告无变化,不提交" >> "$LOG"
    else
        if REFRESH_REPORT_RUNNING=1 git commit -q -m "更新静态报告 $(TZ=Asia/Hong_Kong date '+%F %T')" -- docs/index.html >> "$LOG" 2>&1; then
            echo "$(date '+%F %T %Z') 已提交 docs/index.html" >> "$LOG"
        else
            echo "$(date '+%F %T %Z') docs 提交失败" >> "$LOG"
        fi
    fi

    # 推送所有待推提交(账本+报告);离线/失败不致命,留待下次
    if timeout 60 git push origin main >> "$LOG" 2>&1; then
        echo "$(date '+%F %T %Z') 已 push 到 AbacusFlow" >> "$LOG"
    else
        echo "$(date '+%F %T %Z') push 失败(下次提交或手动重试)" >> "$LOG"
    fi
done

exit 0
