#!/usr/bin/env python3
"""
Autonomous PSX trader (Pakistan Stock Exchange only).

Features:
- Multi-factor signal engine (trend, momentum, mean-reversion, volatility, liquidity)
- Risk controls (position sizing, max drawdown guard, stop-loss, take-profit)
- Paper broker for safe simulation by default
- Backtest mode from CSV bars
- Live loop mode (pluggable data and broker adapters)
- Trade journal + portfolio snapshots

NOTE: This is educational software and defaults to paper trading. Integrate a licensed
broker API before using with real money, and comply with Pakistan regulations.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PKT = dt.timezone(dt.timedelta(hours=5))  # Pakistan Standard Time (UTC+5)


@dataclass
class RiskLimits:
    max_position_pct: float = 0.08
    max_portfolio_drawdown_pct: float = 0.18
    stop_loss_pct: float = 0.04
    take_profit_pct: float = 0.09
    min_cash_buffer_pct: float = 0.10
    max_symbols: int = 8


@dataclass
class StrategyWeights:
    trend: float = 0.30
    momentum: float = 0.25
    reversion: float = 0.15
    volatility: float = 0.15
    liquidity: float = 0.15


@dataclass
class BotConfig:
    mode: str = "paper"
    symbols: List[str] = field(default_factory=lambda: ["OGDC", "LUCK", "MCB", "HBL", "ENGRO", "PSO"])
    initial_cash_pkr: float = 2_000_000.0
    fee_pct: float = 0.002
    slippage_pct: float = 0.001
    bar_interval_seconds: int = 10
    risk: RiskLimits = field(default_factory=RiskLimits)
    weights: StrategyWeights = field(default_factory=StrategyWeights)

    @staticmethod
    def from_json(path: Path) -> "BotConfig":
        raw = json.loads(path.read_text())
        risk = RiskLimits(**raw.get("risk", {}))
        weights = StrategyWeights(**raw.get("weights", {}))
        merged = {k: v for k, v in raw.items() if k not in {"risk", "weights"}}
        return BotConfig(**merged, risk=risk, weights=weights)


@dataclass
class Bar:
    ts: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Position:
    symbol: str
    qty: int
    avg_price: float
    peak_price: float


class PaperBroker:
    def __init__(self, cash: float, fee_pct: float, slippage_pct: float):
        self.cash = cash
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.positions: Dict[str, Position] = {}
        self.realized_pnl = 0.0

    def equity(self, prices: Dict[str, float]) -> float:
        pos_value = sum(p.qty * prices.get(s, p.avg_price) for s, p in self.positions.items())
        return self.cash + pos_value

    def buy(self, symbol: str, qty: int, px: float) -> bool:
        if qty <= 0:
            return False
        fill = px * (1 + self.slippage_pct)
        cost = fill * qty
        fee = cost * self.fee_pct
        total = cost + fee
        if total > self.cash:
            return False
        self.cash -= total
        p = self.positions.get(symbol)
        if p:
            new_qty = p.qty + qty
            p.avg_price = ((p.avg_price * p.qty) + (fill * qty)) / new_qty
            p.qty = new_qty
            p.peak_price = max(p.peak_price, fill)
        else:
            self.positions[symbol] = Position(symbol, qty, fill, fill)
        return True

    def sell(self, symbol: str, qty: int, px: float) -> bool:
        p = self.positions.get(symbol)
        if not p or qty <= 0:
            return False
        qty = min(qty, p.qty)
        fill = px * (1 - self.slippage_pct)
        proceeds = fill * qty
        fee = proceeds * self.fee_pct
        self.cash += (proceeds - fee)
        self.realized_pnl += (fill - p.avg_price) * qty - fee
        p.qty -= qty
        if p.qty == 0:
            del self.positions[symbol]
        return True


class CsvMarketData:
    """Reads bars from ./data/<SYMBOL>.csv (ts,open,high,low,close,volume)."""

    def __init__(self, data_dir: Path, symbols: Iterable[str]):
        self.series: Dict[str, List[Bar]] = {}
        self.cursor = 0
        for sym in symbols:
            path = data_dir / f"{sym}.csv"
            bars: List[Bar] = []
            with path.open() as f:
                r = csv.DictReader(f)
                for row in r:
                    bars.append(
                        Bar(
                            ts=dt.datetime.fromisoformat(row["ts"]).astimezone(PKT),
                            open=float(row["open"]),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=float(row["close"]),
                            volume=float(row["volume"]),
                        )
                    )
            if not bars:
                raise ValueError(f"No bars for {sym} in {path}")
            self.series[sym] = bars
        self.length = min(len(v) for v in self.series.values())

    def next_snapshot(self) -> Optional[Dict[str, Bar]]:
        if self.cursor >= self.length:
            return None
        snap = {s: bars[self.cursor] for s, bars in self.series.items()}
        self.cursor += 1
        return snap


def _sma(values: List[float], n: int) -> float:
    if len(values) < n:
        return statistics.mean(values)
    return statistics.mean(values[-n:])


def _std(values: List[float], n: int) -> float:
    window = values[-n:] if len(values) >= n else values
    return statistics.pstdev(window) if len(window) > 1 else 0.0


def _rsi(values: List[float], n: int = 14) -> float:
    if len(values) < 2:
        return 50.0
    diffs = [values[i] - values[i - 1] for i in range(max(1, len(values) - n + 1), len(values))]
    gains = sum(x for x in diffs if x > 0)
    losses = -sum(x for x in diffs if x < 0)
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - (100 / (1 + rs))


class SignalEngine:
    def __init__(self, weights: StrategyWeights):
        self.weights = weights

    def score(self, closes: List[float], volumes: List[float]) -> float:
        px = closes[-1]
        sma10 = _sma(closes, 10)
        sma30 = _sma(closes, 30)
        rsi14 = _rsi(closes, 14)
        vol20 = _std(closes, 20)
        avg_vol = _sma(volumes, 20)

        trend = (px - sma30) / max(sma30, 1e-6)
        momentum = (px - sma10) / max(sma10, 1e-6)
        reversion = (50 - rsi14) / 50
        volatility = -vol20 / max(px, 1e-6)
        liquidity = (volumes[-1] - avg_vol) / max(avg_vol, 1e-6)

        total = (
            self.weights.trend * trend
            + self.weights.momentum * momentum
            + self.weights.reversion * reversion
            + self.weights.volatility * volatility
            + self.weights.liquidity * liquidity
        )
        return max(min(total, 1.0), -1.0)


class PSXAutonomousTrader:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.broker = PaperBroker(cfg.initial_cash_pkr, cfg.fee_pct, cfg.slippage_pct)
        self.engine = SignalEngine(cfg.weights)
        self.closes: Dict[str, List[float]] = {s: [] for s in cfg.symbols}
        self.volumes: Dict[str, List[float]] = {s: [] for s in cfg.symbols}
        self.start_equity = cfg.initial_cash_pkr
        self.max_equity = cfg.initial_cash_pkr
        self.journal: List[Dict[str, object]] = []

    def _is_market_hours(self, now: dt.datetime) -> bool:
        # Approximate PSX regular hours, Monday-Friday.
        if now.weekday() >= 5:
            return False
        t = now.time()
        return dt.time(9, 30) <= t <= dt.time(15, 30)

    def on_bar(self, snapshot: Dict[str, Bar], allow_trades: bool = True) -> None:
        prices = {s: b.close for s, b in snapshot.items()}
        for s, b in snapshot.items():
            self.closes[s].append(b.close)
            self.volumes[s].append(b.volume)
            if s in self.broker.positions:
                self.broker.positions[s].peak_price = max(self.broker.positions[s].peak_price, b.close)

        equity = self.broker.equity(prices)
        self.max_equity = max(self.max_equity, equity)
        drawdown = (self.max_equity - equity) / max(self.max_equity, 1e-6)
        if drawdown > self.cfg.risk.max_portfolio_drawdown_pct:
            allow_trades = False

        # Exit rules first.
        for sym, pos in list(self.broker.positions.items()):
            px = prices.get(sym, pos.avg_price)
            stop = pos.avg_price * (1 - self.cfg.risk.stop_loss_pct)
            take = pos.avg_price * (1 + self.cfg.risk.take_profit_pct)
            trailing = pos.peak_price * (1 - self.cfg.risk.stop_loss_pct / 2)
            if px <= stop or px >= take or px <= trailing:
                self.broker.sell(sym, pos.qty, px)
                self.journal.append({"action": "SELL_ALL", "symbol": sym, "qty": pos.qty, "price": px, "reason": "risk_exit"})

        if not allow_trades:
            return

        # Rank opportunities.
        scored: List[Tuple[str, float]] = []
        for sym in self.cfg.symbols:
            if len(self.closes[sym]) < 5:
                continue
            scored.append((sym, self.engine.score(self.closes[sym], self.volumes[sym])))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Rebalance with simple top-N logic.
        target = [s for s, sc in scored[: self.cfg.risk.max_symbols] if sc > 0.05]
        prices = {s: b.close for s, b in snapshot.items()}
        equity = self.broker.equity(prices)
        min_cash = equity * self.cfg.risk.min_cash_buffer_pct

        for sym in target:
            px = prices[sym]
            score = dict(scored)[sym]
            desired_value = equity * min(self.cfg.risk.max_position_pct, 0.03 + score * 0.06)
            cur_qty = self.broker.positions.get(sym, Position(sym, 0, px, px)).qty
            cur_value = cur_qty * px
            diff = desired_value - cur_value
            if diff <= px:
                continue
            buy_qty = int(diff / px)
            if buy_qty <= 0:
                continue
            if self.broker.cash - (buy_qty * px) < min_cash:
                continue
            if self.broker.buy(sym, buy_qty, px):
                self.journal.append({"action": "BUY", "symbol": sym, "qty": buy_qty, "price": px, "score": round(score, 4)})

        # De-risk low score holdings.
        for sym, pos in list(self.broker.positions.items()):
            if sym in target:
                continue
            if sym in dict(scored) and dict(scored)[sym] < -0.05:
                qty = max(1, pos.qty // 2)
                px = prices.get(sym, pos.avg_price)
                if self.broker.sell(sym, qty, px):
                    self.journal.append({"action": "SELL", "symbol": sym, "qty": qty, "price": px, "reason": "weak_signal"})

    def summary(self, prices: Dict[str, float]) -> Dict[str, object]:
        eq = self.broker.equity(prices)
        return {
            "equity": round(eq, 2),
            "cash": round(self.broker.cash, 2),
            "realized_pnl": round(self.broker.realized_pnl, 2),
            "return_pct": round(((eq / self.start_equity) - 1) * 100, 2),
            "positions": {s: p.qty for s, p in self.broker.positions.items()},
            "trades": len(self.journal),
        }


def run_backtest(cfg: BotConfig, data_dir: Path) -> None:
    feed = CsvMarketData(data_dir, cfg.symbols)
    bot = PSXAutonomousTrader(cfg)
    last_prices: Dict[str, float] = {}
    while True:
        snap = feed.next_snapshot()
        if snap is None:
            break
        bot.on_bar(snap, allow_trades=True)
        last_prices = {s: b.close for s, b in snap.items()}

    print(json.dumps(bot.summary(last_prices), indent=2))
    print("\nRecent trades:")
    for t in bot.journal[-20:]:
        print(t)


def run_live_loop(cfg: BotConfig) -> None:
    bot = PSXAutonomousTrader(cfg)
    base = {s: random.uniform(120, 500) for s in cfg.symbols}
    print("Running simulated live mode (replace random feed with PSX provider)")
    for _ in range(120):
        now = dt.datetime.now(tz=PKT)
        snap: Dict[str, Bar] = {}
        for s in cfg.symbols:
            drift = random.uniform(-0.01, 0.01)
            base[s] *= (1 + drift)
            p = max(base[s], 1.0)
            snap[s] = Bar(now, p * 0.99, p * 1.01, p * 0.98, p, random.uniform(5_000, 500_000))
        bot.on_bar(snap, allow_trades=bot._is_market_hours(now))
        print(json.dumps(bot.summary({s: b.close for s, b in snap.items()})))
        time.sleep(cfg.bar_interval_seconds)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autonomous PSX stock AI trader")
    p.add_argument("--config", type=Path, default=Path("config.psx.json"), help="Path to JSON config")
    p.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="CSV directory for backtests")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BotConfig.from_json(args.config) if args.config.exists() else BotConfig()
    if args.mode == "backtest":
        run_backtest(cfg, args.data_dir)
    else:
        run_live_loop(cfg)


if __name__ == "__main__":
    main()
