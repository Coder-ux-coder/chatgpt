# PSX Autonomous Stock AI Trader

A detailed Python trading bot focused **only on Pakistan Stock Exchange (PSX)** symbols.

## What it includes

- Autonomous buy/sell decisions from a multi-factor signal engine:
  - Trend (SMA relationship)
  - Momentum
  - RSI mean reversion
  - Volatility penalty
  - Liquidity/volume scoring
- Built-in risk controls:
  - Position size caps
  - Portfolio drawdown guard
  - Stop-loss, trailing stop, take-profit
  - Minimum cash buffer
- Portfolio logic:
  - Top-N signal ranking
  - Rebalancing and de-risking weak symbols
- Brokerage simulation:
  - Paper trading ledger
  - Fees and slippage modeling
  - P&L, equity, position tracking
- Modes:
  - Backtest mode from CSV OHLCV data
  - Simulated live loop (replace feed with real PSX data + broker API)

## Files

- `psx_autonomous_trader.py` — full bot implementation
- `config.psx.json` — strategy/risk configuration

## Expected data format for backtesting

Create one CSV per symbol in `data/`:

`data/OGDC.csv` (example)

```csv
ts,open,high,low,close,volume
2026-01-02T09:30:00+05:00,218.0,219.5,217.5,219.0,120000
2026-01-02T09:35:00+05:00,219.0,220.1,218.7,219.9,98000
```

## Usage

Backtest:

```bash
python3 psx_autonomous_trader.py --mode backtest --data-dir data --config config.psx.json
```

Simulated live loop:

```bash
python3 psx_autonomous_trader.py --mode live --config config.psx.json
```

## Important safety/legal note

This project is educational and defaults to paper trading. If you connect it to real execution:
- validate logic with long forward testing,
- add production-grade monitoring and kill switches,
- ensure compliance with PSX and SECP rules and your broker's terms.
