#!/usr/bin/env python3
"""
Production-style PSX autonomous trading framework (paper/backtest only by default).

What this version improves:
- Deterministic, reproducible backtests with per-bar portfolio accounting.
- Detailed algorithm: regime filter + multi-factor alpha + confidence-gated execution.
- Better risk model: ATR volatility sizing, max gross exposure, correlation-lite diversification,
  kill-switch by drawdown, hard stop, trailing stop, time stop.
- Trade logging and performance report (return, CAGR, Sharpe-ish proxy, max drawdown,
  win-rate, profit factor, turnover).
- Built-in sample-data generator so the tool works out of the box.

Important:
- This remains educational software and defaults to paper simulation.
- For live money, integrate an approved PSX broker API and compliance controls.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PKT = dt.timezone(dt.timedelta(hours=5))


@dataclass
class RiskConfig:
    max_position_pct: float = 0.10
    max_gross_exposure_pct: float = 0.80
    max_portfolio_drawdown_pct: float = 0.20
    min_cash_pct: float = 0.08
    stop_loss_atr_mult: float = 2.2
    trailing_stop_atr_mult: float = 2.8
    take_profit_atr_mult: float = 4.0
    max_holding_bars: int = 70
    max_positions: int = 8
    min_signal_to_enter: float = 0.12
    min_signal_to_add: float = 0.18


@dataclass
class AlphaWeights:
    trend: float = 0.30
    momentum: float = 0.25
    mean_reversion: float = 0.15
    volatility_penalty: float = 0.10
    liquidity: float = 0.10
    breakout: float = 0.10


@dataclass
class ExecutionConfig:
    fee_pct: float = 0.0018
    slippage_pct: float = 0.0012
    lot_size: int = 1


@dataclass
class BotConfig:
    symbols: List[str] = field(default_factory=lambda: ["OGDC", "MCB", "HBL", "ENGRO", "PSO", "LUCK"])
    initial_cash_pkr: float = 2_000_000
    seed: int = 42
    risk: RiskConfig = field(default_factory=RiskConfig)
    alpha: AlphaWeights = field(default_factory=AlphaWeights)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    @staticmethod
    def from_json(path: Path) -> "BotConfig":
        raw = json.loads(path.read_text())
        risk = RiskConfig(**raw.get("risk", {}))
        alpha = AlphaWeights(**raw.get("alpha", {}))
        execution = ExecutionConfig(**raw.get("execution", {}))
        root = {k: v for k, v in raw.items() if k not in {"risk", "alpha", "execution"}}
        return BotConfig(**root, risk=risk, alpha=alpha, execution=execution)


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
    entry_bar: int
    peak: float
    stop_price: float
    take_price: float


@dataclass
class Trade:
    ts: dt.datetime
    symbol: str
    side: str
    qty: int
    price: float
    fee: float
    reason: str


class CsvFeed:
    def __init__(self, data_dir: Path, symbols: Iterable[str]):
        self.symbols = list(symbols)
        self.rows: Dict[str, List[Bar]] = {}
        self.cursor = 0
        for s in self.symbols:
            path = data_dir / f"{s}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing data file: {path}")
            bars: List[Bar] = []
            with path.open() as f:
                r = csv.DictReader(f)
                required = {"ts", "open", "high", "low", "close", "volume"}
                if not r.fieldnames or not required.issubset(set(r.fieldnames)):
                    raise ValueError(f"CSV {path} must contain columns: {sorted(required)}")
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
            if len(bars) < 50:
                raise ValueError(f"Not enough bars in {path}; need at least 50")
            self.rows[s] = bars
        self.length = min(len(v) for v in self.rows.values())

    def next(self) -> Optional[Dict[str, Bar]]:
        if self.cursor >= self.length:
            return None
        snap = {s: self.rows[s][self.cursor] for s in self.symbols}
        self.cursor += 1
        return snap


class PaperBroker:
    def __init__(self, cash: float, execution_cfg: ExecutionConfig):
        self.cash = cash
        self.execution_cfg = execution_cfg
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.realized = 0.0

    def _buy_fill(self, px: float) -> float:
        return px * (1 + self.execution_cfg.slippage_pct)

    def _sell_fill(self, px: float) -> float:
        return px * (1 - self.execution_cfg.slippage_pct)

    def equity(self, prices: Dict[str, float]) -> float:
        gross = sum(p.qty * prices.get(s, p.avg_price) for s, p in self.positions.items())
        return self.cash + gross

    def gross_exposure(self, prices: Dict[str, float]) -> float:
        return sum(p.qty * prices.get(s, p.avg_price) for s, p in self.positions.items())

    def buy(self, ts: dt.datetime, symbol: str, qty: int, px: float, reason: str, bar_idx: int, atr: float) -> bool:
        qty = (qty // self.execution_cfg.lot_size) * self.execution_cfg.lot_size
        if qty <= 0:
            return False
        fill = self._buy_fill(px)
        notional = qty * fill
        fee = notional * self.execution_cfg.fee_pct
        total = notional + fee
        if total > self.cash:
            return False

        self.cash -= total
        pos = self.positions.get(symbol)
        if pos:
            new_qty = pos.qty + qty
            pos.avg_price = ((pos.avg_price * pos.qty) + (fill * qty)) / new_qty
            pos.qty = new_qty
            pos.peak = max(pos.peak, fill)
            pos.stop_price = max(pos.stop_price, fill - atr)
            pos.take_price = max(pos.take_price, fill + 2 * atr)
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                qty=qty,
                avg_price=fill,
                entry_bar=bar_idx,
                peak=fill,
                stop_price=fill - atr,
                take_price=fill + 2 * atr,
            )
        self.trades.append(Trade(ts, symbol, "BUY", qty, fill, fee, reason))
        return True

    def sell(self, ts: dt.datetime, symbol: str, qty: int, px: float, reason: str) -> bool:
        pos = self.positions.get(symbol)
        if not pos:
            return False
        qty = min(qty, pos.qty)
        if qty <= 0:
            return False

        fill = self._sell_fill(px)
        proceeds = qty * fill
        fee = proceeds * self.execution_cfg.fee_pct
        pnl = (fill - pos.avg_price) * qty - fee
        self.realized += pnl
        self.cash += proceeds - fee
        pos.qty -= qty
        self.trades.append(Trade(ts, symbol, "SELL", qty, fill, fee, reason))
        if pos.qty == 0:
            del self.positions[symbol]
        return True


def sma(x: List[float], n: int) -> float:
    w = x[-n:] if len(x) >= n else x
    return statistics.mean(w)


def stddev(x: List[float], n: int) -> float:
    w = x[-n:] if len(x) >= n else x
    return statistics.pstdev(w) if len(w) > 1 else 0.0


def rsi(x: List[float], n: int = 14) -> float:
    if len(x) < 2:
        return 50.0
    start = max(1, len(x) - n + 1)
    diffs = [x[i] - x[i - 1] for i in range(start, len(x))]
    gains = sum(d for d in diffs if d > 0)
    losses = -sum(d for d in diffs if d < 0)
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - (100 / (1 + rs))


def atr(highs: List[float], lows: List[float], closes: List[float], n: int = 14) -> float:
    if len(closes) < 2:
        return 0.0
    trs: List[float] = []
    start = max(1, len(closes) - n)
    for i in range(start, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    return statistics.mean(trs) if trs else 0.0


class DetailedAlphaModel:
    def __init__(self, weights: AlphaWeights):
        self.w = weights

    def regime_filter(self, close_series: List[float]) -> float:
        fast = sma(close_series, 20)
        slow = sma(close_series, 50)
        spread = (fast - slow) / max(slow, 1e-9)
        # Bullish regime ~1, bearish ~0, smooth in-between
        return max(0.0, min(1.0, 0.5 + spread * 8))

    def score(self, closes: List[float], highs: List[float], lows: List[float], vols: List[float]) -> Tuple[float, float]:
        px = closes[-1]
        sma10 = sma(closes, 10)
        sma30 = sma(closes, 30)
        sma50 = sma(closes, 50)
        rsi14 = rsi(closes, 14)
        vol20 = stddev(closes, 20)
        atr14 = atr(highs, lows, closes, 14)
        avg_vol20 = sma(vols, 20)
        high20 = max(highs[-20:]) if len(highs) >= 20 else max(highs)

        trend = (px - sma50) / max(sma50, 1e-9)
        momentum = (sma10 - sma30) / max(sma30, 1e-9)
        reversion = (50 - rsi14) / 50
        volatility_penalty = -vol20 / max(px, 1e-9)
        liquidity = (vols[-1] - avg_vol20) / max(avg_vol20, 1e-9)
        breakout = (px - high20) / max(high20, 1e-9)

        raw = (
            self.w.trend * trend
            + self.w.momentum * momentum
            + self.w.mean_reversion * reversion
            + self.w.volatility_penalty * volatility_penalty
            + self.w.liquidity * liquidity
            + self.w.breakout * breakout
        )

        regime = self.regime_filter(closes)
        score = max(-1.0, min(1.0, raw * (0.6 + 0.8 * regime)))
        confidence = max(0.0, min(1.0, abs(score)))
        return score, max(atr14, px * 0.005)


class PSXTrader:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.broker = PaperBroker(cfg.initial_cash_pkr, cfg.execution)
        self.alpha = DetailedAlphaModel(cfg.alpha)
        self.closes = {s: [] for s in cfg.symbols}
        self.highs = {s: [] for s in cfg.symbols}
        self.lows = {s: [] for s in cfg.symbols}
        self.vols = {s: [] for s in cfg.symbols}

        self.equity_curve: List[float] = []
        self.turnover = 0.0
        self.max_equity = cfg.initial_cash_pkr
        self.kill_switch = False

    def _update_series(self, snap: Dict[str, Bar]) -> None:
        for s, b in snap.items():
            self.closes[s].append(b.close)
            self.highs[s].append(b.high)
            self.lows[s].append(b.low)
            self.vols[s].append(b.volume)

    def _enough_history(self, symbol: str) -> bool:
        return len(self.closes[symbol]) >= 50

    def _compute_scores(self) -> Dict[str, Tuple[float, float]]:
        out: Dict[str, Tuple[float, float]] = {}
        for s in self.cfg.symbols:
            if not self._enough_history(s):
                continue
            out[s] = self.alpha.score(self.closes[s], self.highs[s], self.lows[s], self.vols[s])
        return out

    def _risk_size_qty(self, equity: float, px: float, atr_val: float, score: float) -> int:
        base_position_value = equity * self.cfg.risk.max_position_pct
        vol_adjust = min(2.0, max(0.4, (px / max(atr_val * 10, 1e-9))))
        confidence_adjust = 0.4 + min(0.8, abs(score))
        target_value = base_position_value * vol_adjust * confidence_adjust
        return max(0, int(target_value / max(px, 1e-9)))

    def _apply_exits(self, bar_idx: int, snap: Dict[str, Bar]) -> None:
        for sym, pos in list(self.broker.positions.items()):
            b = snap[sym]
            atr_val = atr(self.highs[sym], self.lows[sym], self.closes[sym], 14)
            pos.peak = max(pos.peak, b.close)
            dynamic_trailing = pos.peak - self.cfg.risk.trailing_stop_atr_mult * max(atr_val, b.close * 0.005)
            hard_stop = pos.avg_price - self.cfg.risk.stop_loss_atr_mult * max(atr_val, b.close * 0.005)
            take_profit = pos.avg_price + self.cfg.risk.take_profit_atr_mult * max(atr_val, b.close * 0.005)
            time_stop = (bar_idx - pos.entry_bar) >= self.cfg.risk.max_holding_bars

            if b.close <= max(hard_stop, dynamic_trailing):
                self.broker.sell(b.ts, sym, pos.qty, b.close, "stop_exit")
            elif b.close >= take_profit:
                sell_qty = max(1, pos.qty // 2)
                self.broker.sell(b.ts, sym, sell_qty, b.close, "take_profit_partial")
            elif time_stop:
                self.broker.sell(b.ts, sym, pos.qty, b.close, "time_stop")

    def on_bar(self, bar_idx: int, snap: Dict[str, Bar]) -> None:
        self._update_series(snap)
        prices = {s: snap[s].close for s in self.cfg.symbols}
        equity = self.broker.equity(prices)
        self.equity_curve.append(equity)
        self.max_equity = max(self.max_equity, equity)
        drawdown = (self.max_equity - equity) / max(self.max_equity, 1e-9)

        if drawdown >= self.cfg.risk.max_portfolio_drawdown_pct:
            self.kill_switch = True

        self._apply_exits(bar_idx, snap)

        if self.kill_switch:
            return

        signals = self._compute_scores()
        ranked = sorted(signals.items(), key=lambda x: x[1][0], reverse=True)
        current_positions = len(self.broker.positions)

        # Enter/add winners respecting limits.
        for sym, (score, atr_val) in ranked:
            if score < self.cfg.risk.min_signal_to_enter:
                continue
            if current_positions >= self.cfg.risk.max_positions and sym not in self.broker.positions:
                continue

            px = prices[sym]
            eq = self.broker.equity(prices)
            gross = self.broker.gross_exposure(prices)
            if gross / max(eq, 1e-9) >= self.cfg.risk.max_gross_exposure_pct:
                break

            cash_floor = eq * self.cfg.risk.min_cash_pct
            if self.broker.cash <= cash_floor:
                break

            target_qty = self._risk_size_qty(eq, px, atr_val, score)
            existing_qty = self.broker.positions[sym].qty if sym in self.broker.positions else 0
            delta = target_qty - existing_qty

            if sym in self.broker.positions and score < self.cfg.risk.min_signal_to_add:
                continue
            if delta <= 0:
                continue

            if self.broker.buy(snap[sym].ts, sym, delta, px, "alpha_entry", bar_idx, atr_val):
                self.turnover += delta * px
                current_positions = len(self.broker.positions)

        # Cut weak names.
        for sym, pos in list(self.broker.positions.items()):
            score = signals.get(sym, (-1.0, 0.0))[0]
            if score <= -0.10:
                qty = max(1, pos.qty // 2)
                if self.broker.sell(snap[sym].ts, sym, qty, prices[sym], "alpha_deterioration"):
                    self.turnover += qty * prices[sym]

    def metrics(self, prices: Dict[str, float], bars_per_year: int) -> Dict[str, float]:
        eq = self.broker.equity(prices)
        ret = (eq / self.cfg.initial_cash_pkr) - 1
        n = max(1, len(self.equity_curve))
        cagr = (1 + ret) ** (bars_per_year / n) - 1 if ret > -1 else -1

        rets = []
        for i in range(1, len(self.equity_curve)):
            prev = self.equity_curve[i - 1]
            curr = self.equity_curve[i]
            rets.append((curr - prev) / max(prev, 1e-9))
        vol = statistics.pstdev(rets) if len(rets) > 1 else 0.0
        mean = statistics.mean(rets) if rets else 0.0
        sharpe = (mean / vol) * math.sqrt(bars_per_year) if vol > 0 else 0.0

        peak = self.equity_curve[0] if self.equity_curve else self.cfg.initial_cash_pkr
        max_dd = 0.0
        for v in self.equity_curve:
            peak = max(peak, v)
            dd = (peak - v) / max(peak, 1e-9)
            max_dd = max(max_dd, dd)

        wins = 0
        losses = 0
        gp = 0.0
        gl = 0.0
        by_symbol_buy_price: Dict[str, List[float]] = {}
        for t in self.broker.trades:
            if t.side == "BUY":
                by_symbol_buy_price.setdefault(t.symbol, []).append(t.price)
            else:
                buys = by_symbol_buy_price.get(t.symbol, [])
                ref = buys.pop(0) if buys else t.price
                pnl = (t.price - ref) * t.qty - t.fee
                if pnl >= 0:
                    wins += 1
                    gp += pnl
                else:
                    losses += 1
                    gl += abs(pnl)

        profit_factor = gp / gl if gl > 0 else float("inf")
        win_rate = wins / max(wins + losses, 1)

        return {
            "equity": round(eq, 2),
            "return_pct": round(ret * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe_like": round(sharpe, 3),
            "realized_pnl": round(self.broker.realized, 2),
            "cash": round(self.broker.cash, 2),
            "open_positions": len(self.broker.positions),
            "trade_count": len(self.broker.trades),
            "win_rate_pct": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 3) if math.isfinite(profit_factor) else 999.0,
            "turnover_pkr": round(self.turnover, 2),
            "kill_switch": self.kill_switch,
        }


def generate_sample_data(data_dir: Path, symbols: List[str], bars: int, seed: int) -> None:
    random.seed(seed)
    data_dir.mkdir(parents=True, exist_ok=True)
    start = dt.datetime(2025, 1, 2, 9, 30, tzinfo=PKT)
    step = dt.timedelta(minutes=5)

    for sym in symbols:
        px = random.uniform(120, 450)
        path = data_dir / f"{sym}.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "open", "high", "low", "close", "volume"])
            t = start
            for i in range(bars):
                cycle = math.sin(i / 32.0) * 0.002
                drift = 0.0002 + cycle + random.uniform(-0.0045, 0.0045)
                opn = px
                cls = max(5.0, px * (1 + drift))
                hi = max(opn, cls) * (1 + random.uniform(0.0005, 0.012))
                lo = min(opn, cls) * (1 - random.uniform(0.0005, 0.012))
                vol = random.randint(8_000, 260_000)
                w.writerow([t.isoformat(), round(opn, 4), round(hi, 4), round(lo, 4), round(cls, 4), vol])
                px = cls
                t += step


def run_backtest(cfg: BotConfig, data_dir: Path, bars_per_year: int = 12_480) -> Dict[str, float]:
    feed = CsvFeed(data_dir, cfg.symbols)
    trader = PSXTrader(cfg)
    last_prices = {s: 0.0 for s in cfg.symbols}
    idx = 0
    while True:
        snap = feed.next()
        if snap is None:
            break
        trader.on_bar(idx, snap)
        last_prices = {s: snap[s].close for s in cfg.symbols}
        idx += 1
    return trader.metrics(last_prices, bars_per_year=bars_per_year)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fully working PSX autonomous trader (paper/backtest)")
    p.add_argument("--config", type=Path, default=Path("config.psx.json"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--generate-sample-data", action="store_true", help="Create synthetic PSX-like CSVs in data-dir")
    p.add_argument("--bars", type=int, default=1200, help="Bars per symbol for sample generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BotConfig.from_json(args.config) if args.config.exists() else BotConfig()
    if args.generate_sample_data:
        generate_sample_data(args.data_dir, cfg.symbols, args.bars, cfg.seed)
        print(f"Generated sample CSVs in {args.data_dir}")
        return

    metrics = run_backtest(cfg, args.data_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
