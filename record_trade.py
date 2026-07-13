"""record_trade.py — 便捷记录一笔实盘成交并即时校验,追加到 output/executions.csv。

口径与 report_web 严格一致:用 load_executions 读现有流水、replay_positions 回放校验
(负现金/负持仓/出金透支即报错拒绝写入)。写入后打印最新持仓与现金。

用法(action 支持中英文 买入/卖出/入金/出金 或 buy/sell/deposit/withdraw):
    python record_trade.py sell 510500 8.865 4000 --note "减仓中证500超配"
    python record_trade.py buy  511260 134.947 200
    python record_trade.py deposit 10000 --note "加仓入金"
    python record_trade.py withdraw 5000
    # 券商实际发生金额与 价格*股数±默认佣金 不同,可显式覆盖:
    python record_trade.py buy 510500 8.852 3700 --amount 32763.4
    # 日期默认今天(上海);--date 覆盖;--dry-run 只校验不写;--commit 写入后提交(触发报告刷新)

安全:
- 与 daily_local.sh 共用 flock,避免并发下 daily_signal 重写计划行时互相覆盖。
- 用 csv.writer 写行 + 原子替换(临时文件 os.replace);写后重新按 load_executions
  解析实际落盘的 CSV 并回放校验,失败则回滚,绝不留下损坏账本。
- note 内换行/半角逗号会破坏 CSV,先净化;以 = + - @ 开头的 note 前置 ' 防公式注入。
- 价格/股数/金额拒绝 NaN/Inf。
"""
import argparse
import csv
import datetime as dt
import fcntl
import io
import math
import os
import subprocess
import tempfile
from zoneinfo import ZoneInfo

import pandas as pd

import config
from report_web import ACTION_ALIASES, EXEC_PATH, load_executions, replay_positions

CN = {"buy": "买入", "sell": "卖出", "deposit": "入金", "withdraw": "出金"}
HEADER = "date,action,symbol,price,shares,amount,note,status"
LOCK_PATH = os.path.join(config.OUTPUT_DIR, ".daily_local.lock")  # 与 daily_local.sh 同一把锁
LEDGER_LOCK = os.path.join(config.OUTPUT_DIR, ".ledger.lock")  # 账本层锁:与 daily_signal 写账本互斥


def _fmt_shares(pos: dict, cash: float) -> str:
    lines = [f"  {s}: {n:,} 股" for s, n in sorted(pos.items())]
    lines.append(f"  现金: {cash:,.2f} 元")
    return "\n".join(lines)


def _sanitize_note(note: str) -> str:
    """净化 note:换行/制表→空格、半角逗号→分号;公式前缀加 ' 防注入。"""
    note = note.replace("\r", " ").replace("\n", " ").replace("\t", " ").replace(",", ";").strip()
    if note[:1] in ("=", "+", "-", "@"):
        note = "'" + note
    return note


def _row_csv(cells: list[str]) -> str:
    """用 csv.writer 序列化一行(自动处理引号),返回不含换行的单行字符串。"""
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(cells)
    return buf.getvalue()


def _atomic_append(raw_line: str) -> None:
    """把 raw_line 原子追加到 EXEC_PATH:读旧内容→拼新行→临时文件→os.replace。

    文件不存在时以标准表头新建。调用方须已持有 flock。
    """
    if os.path.exists(EXEC_PATH):
        with open(EXEC_PATH, "r", encoding="utf-8-sig") as f:
            content = f.read()
        if content and not content.endswith("\n"):
            content += "\n"
    else:
        content = HEADER + "\n"
    new_content = content + raw_line + "\n"
    d = os.path.dirname(EXEC_PATH) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".exec_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(new_content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, EXEC_PATH)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _restore(content: str | None) -> None:
    """把原始内容(str 或 None=删除)原子写回,用于写后校验失败回滚。"""
    d = os.path.dirname(EXEC_PATH) or "."
    if content is None:
        if os.path.exists(EXEC_PATH):
            os.unlink(EXEC_PATH)
        return
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".exec_", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, EXEC_PATH)


def _norm_sym(s) -> str:
    """标的代码归一:NaN/空/字面 'nan' → ""(入金/出金无代码)。

    load_executions 对 symbol 列 astype(str),不同 pandas 版本下空值可能落成真正的
    NaN(pd.isna 为真)或字面字符串 'nan'(pd.isna 为假)——两者都归一为空串,
    否则重复入金/出金检测会因 'nan' != '' 而漏判(最危险的重复记账场景)。
    """
    if pd.isna(s):
        return ""
    s = str(s).strip()
    return "" if s.lower() == "nan" else s


def _is_duplicate(existing, cand: dict) -> bool:
    """判断候选行是否与已有「已成交」行完全相同(日期/方向/代码/价格/股数/金额一致)。

    仅比对非计划行(计划行不算真实成交)。数值用 NaN 相等语义 + isclose 容差,
    覆盖入金重复(不触发负现金校验)这一最危险的重复记账场景。
    """
    if existing is None or getattr(existing, "empty", True):
        return False
    conf = existing[existing["status"] != "计划"]
    if conf.empty:
        return False

    def _eqnum(a, b) -> bool:
        an, bn = pd.isna(a), pd.isna(b)
        if an and bn:
            return True
        if an != bn:
            return False
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)

    cd = cand["date"].date() if hasattr(cand["date"], "date") else cand["date"]
    csym = _norm_sym(cand["symbol"])
    for _, r in conf.iterrows():
        rd = r["date"].date() if hasattr(r["date"], "date") else r["date"]
        if (rd == cd and r["action"] == cand["action"]
                and _norm_sym(r.get("symbol")) == csym
                and _eqnum(r.get("price"), cand["price"])
                and _eqnum(r.get("shares"), cand["shares"])
                and _eqnum(r.get("amount"), cand["amount"])):
            return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(description="记录一笔实盘成交并校验(追加 executions.csv)")
    p.add_argument("action", help="买入/卖出/入金/出金 或 buy/sell/deposit/withdraw")
    p.add_argument("args", nargs="*",
                   help="买卖: 代码 价格 股数;入金/出金: 金额")
    p.add_argument("--date", help="成交日期 YYYY-MM-DD,默认今天(上海)")
    p.add_argument("--symbol")
    p.add_argument("--price", type=float)
    p.add_argument("--shares", type=int)
    p.add_argument("--amount", type=float, help="券商实际发生金额;买卖不填则按 价格*股数±默认佣金 估算")
    p.add_argument("--note", default="")
    p.add_argument("--dry-run", action="store_true", help="只校验并预览,不写入")
    p.add_argument("--allow-duplicate", action="store_true",
                   help="允许写入与已有记录完全相同的成交(默认拒绝,防手滑/重跑重复记账)")
    p.add_argument("--commit", action="store_true", help="写入后 git 提交 executions.csv(触发报告刷新)")
    a = p.parse_args()

    action = ACTION_ALIASES.get(a.action.strip().lower())
    if action is None:
        print(f"✗ 未知 action: {a.action}(应为 买入/卖出/入金/出金)")
        return 2

    symbol = (a.symbol or "").strip()
    price, shares, amount = a.price, a.shares, a.amount
    rest = list(a.args)
    try:
        if action in ("buy", "sell"):
            if not symbol and rest:
                symbol = rest.pop(0).strip()
            if price is None and rest:
                price = float(rest.pop(0))
            if shares is None and rest:
                shares = int(rest.pop(0))
        else:  # deposit / withdraw
            if amount is None and rest:
                amount = float(rest.pop(0))
    except ValueError as e:
        print(f"✗ 参数解析失败(价格/金额须为数字、股数须为整数): {e}")
        return 2
    if rest:
        print(f"✗ 多余参数: {rest}")
        return 2

    # 日期:严格 YYYY-MM-DD(拒绝带时分/时区,避免回放错配),默认今天上海
    if a.date:
        try:
            date = dt.datetime.strptime(a.date.strip(), "%Y-%m-%d").date().isoformat()
        except ValueError:
            print(f"✗ 日期无效(应为 YYYY-MM-DD): {a.date}")
            return 2
    else:
        date = dt.datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()

    note = _sanitize_note(a.note)

    # 数值必须有限(拒绝 NaN/Inf:argparse float 会解析 nan/inf 且 nan<=0 为假)
    if amount is not None and not math.isfinite(amount):
        print("✗ --amount 必须为有限数值")
        return 2

    # 字段校验(与 load_executions 同规则)
    if action in ("buy", "sell"):
        if not (symbol.isdigit() and len(symbol) == 6):
            print(f"✗ 代码无效(应为6位): {symbol}")
            return 2
        if price is None or not math.isfinite(price) or price <= 0:
            print("✗ 价格必须为正的有限数值")
            return 2
        if shares is None or shares <= 0:
            print("✗ 股数必须为正整数")
            return 2
        if amount is not None and amount <= 0:
            print("✗ --amount(券商实际金额)必须为正数")
            return 2
        if shares % 100 != 0:
            print(f"⚠ 股数 {shares} 非整百手(A股ETF按100股整手,请确认)")
        amt_cell = "" if amount is None else f"{amount}"
        cells = [date, CN[action], symbol, f"{price}", f"{shares}", amt_cell, note, "已成交"]
        cand = {"date": pd.Timestamp(date), "action": action, "symbol": symbol,
                "price": float(price), "shares": int(shares),
                "amount": float(amount) if amount is not None else float("nan"),
                "note": note, "status": "已成交"}
    else:
        if amount is None or amount <= 0:
            print("✗ 金额必须为正数")
            return 2
        cells = [date, CN[action], "", "", "", f"{amount}", note, "已成交"]
        cand = {"date": pd.Timestamp(date), "action": action, "symbol": "",
                "price": float("nan"), "shares": float("nan"),
                "amount": float(amount), "note": note, "status": "已成交"}
    raw_line = _row_csv(cells)

    # 全流程持双锁:读→内存校验→原子写→重解析校验期间串行。
    # 锁次序约定(防死锁):.daily_local.lock(编排层,先)→ .ledger.lock(账本层,后)。
    # 编排锁挡住 daily_local.sh 整条流水线;账本锁挡住手动直跑的 daily_signal.py
    # (其 write_planned 整表重写,若无此锁会覆盖丢掉本脚本刚追加的成交行)。
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    lock_fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    ledger_fd = os.open(LEDGER_LOCK, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        fcntl.flock(ledger_fd, fcntl.LOCK_EX)

        # 业务幂等:拒绝与已有「已成交」行完全相同的记录(防手滑/重跑重复记账,尤重入金)
        existing = load_executions()
        if not a.allow_duplicate and _is_duplicate(existing, cand):
            print("✗ 检测到与已有记录完全相同的成交(日期/方向/代码/价格/股数/金额一致),已拒绝。")
            print("  若确为两笔独立成交,加 --allow-duplicate 覆盖;否则很可能是重复记账。")
            return 3

        # 内存预校验:现有流水 + 候选行一起回放,负现金/负持仓即拒绝
        combined = pd.concat([existing, pd.DataFrame([cand])], ignore_index=True) \
            if existing is not None else pd.DataFrame([cand])
        combined = combined.sort_values("date", kind="stable").reset_index(drop=True)
        confirmed = combined[combined["status"] != "计划"]
        try:
            pos, cash = replay_positions(confirmed)
        except ValueError as e:
            print(f"✗ 校验失败(未写入): {e}")
            return 1

        print(f"待记录: {raw_line}")
        print(f"回放后持仓/现金:\n{_fmt_shares(pos, cash)}")

        if a.dry_run:
            print("(dry-run,未写入)")
            return 0

        # 原子追加,再重新解析实际落盘的 CSV 做校验;失败回滚
        backup = None
        if os.path.exists(EXEC_PATH):
            with open(EXEC_PATH, "r", encoding="utf-8-sig") as f:
                backup = f.read()
        _atomic_append(raw_line)
        try:
            reloaded = load_executions()
            if reloaded is None:
                raise ValueError("落盘后重新读取为空")
            replay_positions(reloaded[reloaded["status"] != "计划"])
        except Exception as e:
            _restore(backup)
            print(f"✗ 落盘后校验失败,已回滚: {e}")
            return 1
        print(f"✓ 已写入 {EXEC_PATH}")

        if a.commit:
            msg = f"记录成交 {date} {CN[action]}" + (f" {symbol}" if symbol else "")
            r = subprocess.run(["git", "commit", "-q", "-m", msg, "--", "output/executions.csv"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                print(f"✓ 已提交(post-commit 钩子将后台刷新报告): {msg}")
            else:
                # 账本已安全写入,仅 git 提交失败:返回非零让调用方/监控可感知(可手动补交)
                print(f"⚠ 已写入账本但 git 提交失败(可手动 git commit -- output/executions.csv): "
                      f"{r.stderr.strip() or r.stdout.strip()}")
                return 4
        else:
            print("提示: 加 --commit 可直接提交并触发报告刷新")
        return 0
    finally:
        fcntl.flock(ledger_fd, fcntl.LOCK_UN)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(ledger_fd)
        os.close(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
