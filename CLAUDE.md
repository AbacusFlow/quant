# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A-share (Chinese stock market) backtesting system. Two backtest paths: a single-asset dual-MA demo (`main.py`) and the primary **ETF momentum rotation** strategy (`run_rotation.py`) with a multi-asset portfolio engine. Fetches daily OHLCV via akshare, benchmarks against CSI 300, outputs metrics, equity curve PNG, and trades CSV. Comments and output are in Chinese.

**Project goal:** evolve into a live A-share trading system (planned route: miniQMT/xtquant). Backtest validity (no look-ahead, realistic fees/slippage, out-of-sample checks) is the top priority for all changes.

## Commands

Local Python has no dependencies installed — run everything through Docker:

```bash
docker build -f .docker/Dockerfile -t quant .

# ETF momentum rotation (primary): full / in-sample / out-of-sample metrics
docker run --rm -v "$PWD":/work quant python run_rotation.py --end 2026-06-09
docker run --rm -v "$PWD":/work quant python run_rotation.py --sensitivity   # lookback scan
docker run --rm -v "$PWD":/work quant python run_rotation.py --compare       # single vs ensemble vs ensemble+dd

# Paper-trading daily signal (run before market open, uses previous close; appends to output/signal_log.csv)
docker run --rm -v "$PWD":/work quant python daily_signal.py

# Single-asset dual-MA demo
docker run --rm -v "$PWD":/work quant python main.py --symbol 600519

# Unit tests
docker run --rm -v "$PWD":/work quant python test_portfolio.py   # portfolio engine
docker run --rm -v "$PWD":/work quant python test_strategy.py    # rotation/ensemble/drawdown control
```

No linters configured.

## Architecture

```
data.py → strategy.py → backtest.py (single-asset)  → metrics.py / report.py
                      ↘ portfolio.py (multi-asset)  ↗        (run_rotation.py)
```

- **portfolio.py** — multi-asset portfolio engine: target-weight DataFrame in, T+1 open execution with slippage, proportional cash budgeting across simultaneous buys (order-independent), 100-share lots, ETF trades exempt from stamp tax (`stamp_tax=False`), rebalance band (`config.REBALANCE_BAND`: skip trades when target vs. current deviates by <2% of total assets). `align_prices()` defines the common trading calendar — signals and execution must use the same calendar.
- **run_rotation.py** — rotation CLI: loads the `config.ETF_POOL`, reports full / in-sample / out-of-sample (`config.OOS_SPLIT`) metrics. Modes: `--mode ensemble` (default, multi-lookback average per `config.ENSEMBLE_LOOKBACKS`) or `--mode single`; `--dd` adds drawdown control (off by default — backtest showed ~3% annualized cost for ~2pp drawdown improvement); `--vol-target` adds the volatility-targeting overlay (off by default — improves stability not return); `--sensitivity` scans lookbacks; `--compare` tabulates the three variants. `build_weights(closes, mode, lookback, buffer, dd_control, vol_control=False)`/`closes_table()` are reused by daily_signal.py / report_web.py / check_risk.py — keep their signatures stable.
- **Volatility-targeting overlay (V1, default OFF)** — `strategy.apply_vol_targeting()` is an optional post-processing layer (same pattern as `apply_drawdown_control`): scales target weights down when the strategy's own realized vol (annualized rolling std, `config.VOL_TARGET_LOOKBACK`=20) exceeds its causal expanding-median target, capped at 1.0 (no leverage). Adaptive (no fixed `target_vol` knob), causal (scale_t uses only data ≤T; engine then shifts to T+1). Research (`.omc/research/2026-06-17-robust-improvement.md`) showed walk-forward Sharpe 0.40→0.59 and max drawdown -40.6%→-26.4% with ~unchanged return; it produces continuous fractional weights → more frequent partial rebalances. Gated everywhere by `config.VOL_TARGET_ENABLED` (False locally) and the CI repo Variable `SIGNAL_VOL_TARGET` (`true`/`false`); flip the Variable to enable in production without a code change. `daily_signal.py`/`report_web.py`/`check_risk.py` accept `--vol-target/--no-vol-target` (default = config); CI threads the same flag to all three so the drawdown-alert baseline stays in the same regime as the live strategy.
- **daily_signal.py** — paper-trading signal tracker: recomputes the latest target weights (`--mode single` default for small capital, `ensemble` once capital ≥100k; no dd control; volatility targeting off by default, `--vol-target` for shadow runs), diffs against the last logged signal to print rebalance instructions (execute next-day open), appends to `output/signal_log.csv` with a `mode` column. Idempotent per signal date.
- **test_portfolio.py** — unit tests for the portfolio engine (T+1, lots, fees, slippage, stamp tax, cash conservation) using hand-built price data. Run them after touching portfolio.py.
- **test_strategy.py** — unit tests for rotation/ensemble/drawdown control, including a no-look-ahead test (truncating future data must not change historical control coefficients). Run them after touching strategy.py.

- **data.py** — fetches stock/ETF daily bars (前复权/qfq) and CSI 300 benchmark via akshare with retry; caches as CSV in `data/` keyed by `{name}_{start}_{end}.csv` (delete to force refresh; empty caches are treated as misses). Eastmoney failures fall back to Sina — Sina ETF data is **unadjusted** and cached under a separate `_sina` key to avoid contaminating qfq caches.
- **strategy.py** — single-asset strategies take an OHLCV DataFrame and return a 0/1 position Series; `etf_momentum_rotation()` takes a closes table and returns a target-weight DataFrame (top-1 momentum, absolute-momentum filter to cash, switch buffer); `etf_momentum_ensemble()` averages rotation weights across `lookbacks` (weights become fractional, row sum ≤ 1); `apply_drawdown_control()` scales weights by `DD_SCALE` when the strategy's virtual NAV drops below its `DD_MA_WINDOW`-day MA — must stay causal (T-day coefficient uses only data ≤ T). Signals are computed on close; execution is next-day open (T+1). Use `pct_change(..., fill_method=None)` to avoid forward-filling stale prices.
- **backtest.py** — `run_backtest()` simulates day by day: shifts the position series by 1 day (T+1), trades at the open price in 100-share lots, and models A-share fees (commission with 5 CNY minimum, stamp tax on sells only). Returns `BacktestResult` with daily equity and a `Trade` list.
- **metrics.py** — total/annualized return, Sharpe (252 trading days), max drawdown, win rate per completed buy→sell round.
- **report.py** — prints metrics, saves `output/equity_curve.png` (strategy vs. normalized benchmark) and `output/trades.csv`. Uses matplotlib `Agg` backend with CJK fonts configured for Chinese labels.
- **config.py** — all defaults: symbol, date range, MA periods, initial capital (1M CNY), fee rates, `data/` and `output/` paths.
- **portfolio_status.py** — builds the「持仓偏离目标 + 账户简况」text block for daily Telegram messages (same 口径 as check_risk: load_pool→closes→build_weights→real_equity_series+replay_positions). Any error/未就绪→空串,不中断。Optional `--mx-fallback PATH`: for the收盘后档 only, appends the妙想(mx_data)当日收盘 as the latest in-memory bar so the same-day account block renders before akshare's ~22:00 K-line publish. See「双档 cron + mx_data 兜底」below.
- **scripts/mx_fetch_latest.py** (host-only) + **scripts/postclose_local.sh** (host, 17:03 cron) — the收盘后展示档: mx_data fetch runs on host (needs `MX_APIKEY`+network), pandas/status runs in Docker. Isolated flock/spool/receipt from daily_local. Money signal path (daily_signal/check_risk/8点档) NEVER touches mx_data.

## 双档 cron + mx_data 收盘后兜底(仅展示,money 路径不用)

- **早 8 点档** (`scripts/daily_local.sh`, `3 8 * * 1-5`): only今日操作 — 调仓指令 or「今日无需调仓,维持现有持仓」, using 昨收 akshare signal. No account/pnl block (moved to 收盘后档).
- **收盘后档** (`scripts/postclose_local.sh`, `3 17 * * 1-5`): 账户小结 + 持仓偏离, Telegram-only (网页仍走晚间 akshare + post-commit auto-refresh, no mx). Uses `mx_fetch_latest.py` → `output/mx_latest.json` → `portfolio_status.py --mx-fallback`.
- **mx_data 硬约束** (门槛验证 2026-07-10): 近端最新收盘逐位匹配 akshare qfq, but 历史分红标的复权口径发散 (mx AdjustFlag=3 因子法 ≠ akshare 锚定最新). 故只**追加 `end` 当日一根 bar** (o/h/l=close, volume=0, only when `end` > 现有最新 bar), **绝不拉全历史、绝不写 `data/*.csv`**. Partial fill degrades naturally (align_prices 交集截回上一交易日). mx JSON missing/corrupt → 静默降级回 akshare.
- `MX_APIKEY` is a `.env` whitelist key (gitignored), read by postclose_local's `envval`; never passed into Docker.

## Key Conventions

- All trading logic assumes A-share rules: T+1 execution, 100-share lot sizes, stamp tax on sells (ETF exempt). Commission is broker VIP rate (ETF 万0.5) with a 5 CNY minimum — the minimum dominates costs at small capital, so `single` mode (full-position switches) is preferred below ~100k capital and `ensemble` above (see `run_rotation.py --capital`).
- Single-asset position series must contain only 0/1 (no partial positions or shorting). Multi-asset target weights may be fractional but each row must sum to ≤ 1 (remainder is cash); no shorting or leverage.
- akshare is imported lazily inside data-fetching functions, so cached runs work without network access.
