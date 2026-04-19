# PSX Autonomous Trader (Detailed, Fully Runnable)

This is a **Pakistan Stock Exchange (PSX)-only** autonomous trading framework with:

- detailed multi-factor alpha model,
- position sizing based on volatility (ATR),
- portfolio-level risk controls,
- deterministic backtesting,
- built-in sample-data generation so it runs end-to-end immediately.

> Default mode is paper/backtest. No real broker orders are sent unless you add broker integration.

## Algorithm (detailed)

The strategy combines several layers:

1. **Regime filter**
   - Uses SMA-20 vs SMA-50 spread to estimate bullish/bearish market regime.
   - Regime scales alpha aggression up/down.

2. **Multi-factor alpha score**
   - Trend (price vs SMA-50)
   - Momentum (SMA-10 vs SMA-30)
   - Mean reversion (RSI-14)
   - Volatility penalty (std dev)
   - Liquidity impulse (volume vs 20-bar mean)
   - Breakout pressure (price vs 20-bar high)

3. **Confidence-gated entries**
   - Enter only if score crosses a minimum threshold.
   - Add-on buys require a stronger threshold.

4. **Risk sizing**
   - Position size starts from max position % of equity.
   - Adjusts by ATR-driven volatility normalization and signal confidence.

5. **Exit engine**
   - ATR hard stop
   - ATR trailing stop
   - ATR take-profit partial
   - Time stop (max holding bars)
   - Weak-alpha de-risking (reduce size on negative signal)

6. **Portfolio guardrails**
   - Max gross exposure cap
   - Min cash reserve
   - Max position count
   - Drawdown kill-switch (halts new entries when breached)

## Files

- `psx_autonomous_trader.py` — full engine, backtest runner, sample-data generator.
- `config.psx.json` — symbols + risk/alpha/execution tuning.

## Quick start (works immediately)

### 1) Generate sample PSX-like data

```bash
python3 psx_autonomous_trader.py --generate-sample-data --data-dir data --config config.psx.json --bars 1200
```

### 2) Run backtest

```bash
python3 psx_autonomous_trader.py --data-dir data --config config.psx.json
```

You’ll get a metrics JSON including:
- return %, CAGR %, max drawdown,
- sharpe-like score,
- win rate, profit factor,
- trade count, turnover,
- kill-switch status.

## Real PSX usage notes

To use this with real execution:
- replace/extend data feed with real PSX market data,
- implement a broker adapter in `PaperBroker` style,
- add operational controls (circuit breaker, order reconciliation, alerting),
- ensure compliance with broker terms and SECP/PSX regulations.
