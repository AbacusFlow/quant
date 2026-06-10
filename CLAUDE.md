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

# Single-asset dual-MA demo
docker run --rm -v "$PWD":/work quant python main.py --symbol 600519

# Unit tests (portfolio engine)
docker run --rm -v "$PWD":/work quant python test_portfolio.py
```

No linters configured.

## Architecture

```
data.py → strategy.py → backtest.py (single-asset)  → metrics.py / report.py
                      ↘ portfolio.py (multi-asset)  ↗        (run_rotation.py)
```

- **portfolio.py** — multi-asset portfolio engine: target-weight DataFrame in, T+1 open execution with slippage, proportional cash budgeting across simultaneous buys (order-independent), 100-share lots, ETF trades exempt from stamp tax (`stamp_tax=False`). `align_prices()` defines the common trading calendar — signals and execution must use the same calendar.
- **run_rotation.py** — rotation CLI: loads the `config.ETF_POOL`, reports full / in-sample / out-of-sample (`config.OOS_SPLIT`) metrics, `--sensitivity` runs a lookback parameter scan. Prints the latest target holding for live tracking.
- **test_portfolio.py** — unit tests for the portfolio engine (T+1, lots, fees, slippage, stamp tax, cash conservation) using hand-built price data. Run them after touching portfolio.py.

- **data.py** — fetches stock/ETF daily bars (前复权/qfq) and CSI 300 benchmark via akshare with retry; caches as CSV in `data/` keyed by `{name}_{start}_{end}.csv` (delete to force refresh; empty caches are treated as misses). Eastmoney failures fall back to Sina — Sina ETF data is **unadjusted** and cached under a separate `_sina` key to avoid contaminating qfq caches.
- **strategy.py** — single-asset strategies take an OHLCV DataFrame and return a 0/1 position Series; `etf_momentum_rotation()` takes a closes table and returns a target-weight DataFrame (top-1 momentum, absolute-momentum filter to cash, switch buffer). Signals are computed on close; execution is next-day open (T+1). Use `pct_change(..., fill_method=None)` to avoid forward-filling stale prices.
- **backtest.py** — `run_backtest()` simulates day by day: shifts the position series by 1 day (T+1), trades at the open price in 100-share lots, and models A-share fees (commission with 5 CNY minimum, stamp tax on sells only). Returns `BacktestResult` with daily equity and a `Trade` list.
- **metrics.py** — total/annualized return, Sharpe (252 trading days), max drawdown, win rate per completed buy→sell round.
- **report.py** — prints metrics, saves `output/equity_curve.png` (strategy vs. normalized benchmark) and `output/trades.csv`. Uses matplotlib `Agg` backend with CJK fonts configured for Chinese labels.
- **config.py** — all defaults: symbol, date range, MA periods, initial capital (1M CNY), fee rates, `data/` and `output/` paths.

## Key Conventions

- All trading logic assumes A-share rules: T+1 execution, 100-share lot sizes, stamp tax on sells.
- Position series must contain only 0/1 (no partial positions or shorting).
- akshare is imported lazily inside data-fetching functions, so cached runs work without network access.
