# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

Think in English, but always respond in Japanese.

## Project Overview

Stock expected-value screening tool (株価期待値スクリーニングツール) for Tokyo Stock Exchange Prime market. It identifies stocks with statistically positive expected returns at next-day open entry. This is a decision-support web app — no automated trading.

Design documents are in `docs/` (Japanese). The canonical spec is `docs/01_requirements_basic_design.md`.

## Tech Stack

- **Language**: Python
- **Data**: yfinance (daily OHLCV for TSE Prime stocks, 5-year lookback)
- **UI**: Streamlit (deployed on Streamlit Community Cloud, mobile-responsive)
- **Automation**: GitHub Actions (21:00 JST nightly batch, 08:50 JST morning pre-market check)

## Architecture (6 Phases)

1. **Data Acquisition** — Fetch daily price data via yfinance for all TSE Prime tickers; maintain a Prime stock master list
2. **Feature Engineering** — Compute: 5-day turnover ratio, ATR(14)/price ratio, 20-day high breakout flag, 3-day return, liquidity flag
3. **Backtest Engine** — Calculate n-day (n=1..20) open-to-open returns `R(n) = Open(n)/Open(0) - 1` with exponential decay weighting `weight = exp(-0.3 * years_elapsed)`; output weighted median return, weighted win rate, weighted max-DD median
4. **Screening Engine** — Apply liquidity filters (volume >= 500K, turnover >= 500M JPY, price 800-6000 JPY), capital inflow conditions (turnover ratio >= 1.8x, 20-day high break, bullish candle, ATR >= 2%, 3-day return <= 12%), and exclusions (limit-up next day, gap >= +5%); rank by expected return
5. **Streamlit UI** — Main screen (ranked candidates table sorted by expected return), detail page (return/win-rate charts by holding period, DD distribution), portfolio management tab (holding days, forced-exit countdown)
6. **GitHub Actions** — Nightly data update + screening pipeline at 21:00 JST; pre-market gap check at 08:50 JST

## Key Domain Rules

- Entry threshold defaults: 1-day expected return >= +0.35%, win rate >= 57%, DD median >= -1.8%
- Recommended holding period: the n (1-20) that maximizes expected return; warn if DD exceeds -3%
- Max simultaneous holdings: 5 stocks, equal-weight allocation
- Max holding period: 20 business days (forced exit)
- Pre-market gap check: <= +3% OK, +3-5% caution, >= +5% exclude
- Commissions and taxes are not modeled
- All screening thresholds are externalized in `config/settings.yaml`

## Development Commands

```bash
uv sync                  # Install dependencies
uv run pytest            # Run all tests
uv run pytest tests/test_foo.py::test_bar  # Run a single test
uv run ruff check .      # Lint
uv run ruff format .     # Format
uv run streamlit run src/app/main.py  # Run the Streamlit app locally
```

## Project Structure

```
src/
  data/       # Data acquisition & ticker master (Phase 1)
  features/   # Feature engineering (Phase 2)
  backtest/   # Backtest engine (Phase 3)
  screening/  # Screening logic (Phase 4)
  app/        # Streamlit UI (Phase 5)
config/       # YAML settings (thresholds, parameters)
tests/        # pytest tests
scripts/      # GitHub Actions batch scripts (Phase 6)
docs/         # Design documents (Japanese)
```
