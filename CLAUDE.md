# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A-share (Chinese stock market) backtesting system for trend/momentum strategies. Fetches daily OHLCV data via akshare, runs a single-asset full-in/full-out backtest against the CSI 300 (沪深300) benchmark, and outputs metrics, an equity curve PNG, and a trades CSV. Comments and output are in Chinese.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run a backtest (defaults: 600519, 2020-01-01 ~ 2025-12-31, MA5/MA20)
python main.py --symbol 600519 --start 2020-01-01 --end 2025-12-31 --short 5 --long 20

# Docker (image includes Noto CJK fonts for Chinese chart labels)
docker build -f .docker/Dockerfile -t quant .
docker run --rm -v "$PWD":/work quant python main.py
```

There are no tests or linters configured.

## Architecture

Data flow through `main.py`:

```
data.py → strategy.py → backtest.py → metrics.py / report.py
```

- **data.py** — fetches stock daily bars (前复权/qfq adjusted) and CSI 300 benchmark via akshare; caches as CSV in `data/` keyed by `{name}_{start}_{end}.csv`. Delete the cache file to force a refresh. Benchmark fetch falls back from the Eastmoney API to the Sina API.
- **strategy.py** — strategy library with a uniform interface: takes an OHLCV DataFrame, returns a target position Series (0 = flat, 1 = fully invested). Signals are computed on close; execution is next-day open (T+1). New strategies should be registered in the `STRATEGIES` dict (name → (function, description)). Note: `main.py` currently hardcodes `dual_ma_signal` and does not use the registry.
- **backtest.py** — `run_backtest()` simulates day by day: shifts the position series by 1 day (T+1), trades at the open price in 100-share lots, and models A-share fees (commission with 5 CNY minimum, stamp tax on sells only). Returns `BacktestResult` with daily equity and a `Trade` list.
- **metrics.py** — total/annualized return, Sharpe (252 trading days), max drawdown, win rate per completed buy→sell round.
- **report.py** — prints metrics, saves `output/equity_curve.png` (strategy vs. normalized benchmark) and `output/trades.csv`. Uses matplotlib `Agg` backend with CJK fonts configured for Chinese labels.
- **config.py** — all defaults: symbol, date range, MA periods, initial capital (1M CNY), fee rates, `data/` and `output/` paths.

## Key Conventions

- All trading logic assumes A-share rules: T+1 execution, 100-share lot sizes, stamp tax on sells.
- Position series must contain only 0/1 (no partial positions or shorting).
- akshare is imported lazily inside data-fetching functions, so cached runs work without network access.
