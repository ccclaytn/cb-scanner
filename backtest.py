"""
Backtest the structure-detection logic (SFP, MSB, Breaker, S/R Cluster)
against historical Binance candle data.

Reuses the exact detection functions from hyperliquid_scanner.py so the
backtest is testing the same logic that runs live — no logic duplication,
no drift between what's backtested and what's deployed.

Usage:
    python backtest.py --days 90 --assets BTC,ETH,SOL --timeframe H1
    python backtest.py --days 180 --timeframe H12   # all default assets
"""
import argparse
import time
import requests
import pandas as pd
from datetime import datetime, timezone

# Reuse detection logic directly from the live scanner — single source of truth
from hyperliquid_scanner import (
    detect_sfp, detect_msb, detect_breaker, detect_sr_cluster,
    score_setup, get_bias, MIN_CONFLUENCE, _BIN_INTERVAL,
)

BINANCE_API = "https://api.binance.com/api/v3"

DEFAULT_ASSETS = ["BTC", "ETH", "SOL", "DOGE", "XRP", "BNB", "ADA", "LINK"]

# How far forward to check if TP/stop was hit, per timeframe (in candles)
FORWARD_WINDOW = {"H1": 48, "H4": 36, "H12": 20, "D1": 14}


def fetch_historical_candles(asset: str, interval_label: str, days: int) -> pd.DataFrame:
    """
    Pull `days` worth of historical candles from Binance.
    Paginates since a single request caps at 1000 candles.
    """
    interval = _BIN_INTERVAL[interval_label]
    ms_per_candle = {
        "H1": 3600_000, "H4": 14400_000, "H12": 43200_000, "D1": 86400_000,
    }[interval_label]

    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - days * 86400_000
    all_rows = []

    cursor = start_ms
    while cursor < end_ms:
        try:
            r = requests.get(f"{BINANCE_API}/klines", params={
                "symbol":    f"{asset}USDT",
                "interval":  interval,
                "startTime": cursor,
                "limit":     1000,
            }, timeout=10)
            if r.status_code == 400:
                print(f"  {asset}: not listed on Binance, skipping")
                return pd.DataFrame()
            r.raise_for_status()
            rows = r.json()
            if not rows:
                break
            all_rows.extend(rows)
            cursor = rows[-1][0] + ms_per_candle
            time.sleep(0.1)
        except requests.exceptions.RequestException as e:
            print(f"  {asset}: fetch error {e}")
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "t", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "tbbav", "tbqav", "ignore"
    ])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["close"]).reset_index(drop=True)


def simulate_trade(df: pd.DataFrame, signal_idx: int, alert: dict, max_forward: int) -> dict:
    """
    Walk forward from signal_idx and check whether stop, TP1, TP2, or TP3
    was hit first. Returns outcome dict.
    """
    direction = alert["direction"]
    stop, tp1, tp2, tp3 = alert["stop"], alert["tp1"], alert["tp2"], alert["tp3"]

    end_idx = min(signal_idx + max_forward, len(df) - 1)
    outcome = {"result": "no_resolution", "hit_tp": None, "candles_to_resolve": None}

    for i in range(signal_idx + 1, end_idx + 1):
        bar  = df.iloc[i]
        hi, lo = float(bar["high"]), float(bar["low"])

        if direction == "LONG":
            if lo <= stop:
                outcome = {"result": "stopped_out", "hit_tp": None,
                          "candles_to_resolve": i - signal_idx}
                break
            if hi >= tp3:
                outcome = {"result": "tp3_hit", "hit_tp": 3,
                          "candles_to_resolve": i - signal_idx}
                break
            if hi >= tp2:
                outcome = {"result": "tp2_hit", "hit_tp": 2,
                          "candles_to_resolve": i - signal_idx}
                # keep scanning forward for TP3 within the window
                continue
            if hi >= tp1 and outcome["result"] == "no_resolution":
                outcome = {"result": "tp1_hit", "hit_tp": 1,
                          "candles_to_resolve": i - signal_idx}
                # keep scanning forward for TP2/TP3 within the window
        else:  # SHORT
            if hi >= stop:
                outcome = {"result": "stopped_out", "hit_tp": None,
                          "candles_to_resolve": i - signal_idx}
                break
            if lo <= tp3:
                outcome = {"result": "tp3_hit", "hit_tp": 3,
                          "candles_to_resolve": i - signal_idx}
                break
            if lo <= tp2:
                outcome = {"result": "tp2_hit", "hit_tp": 2,
                          "candles_to_resolve": i - signal_idx}
                continue
            if lo <= tp1 and outcome["result"] == "no_resolution":
                outcome = {"result": "tp1_hit", "hit_tp": 1,
                          "candles_to_resolve": i - signal_idx}

    return outcome


def build_alert_from_structures(df_window: pd.DataFrame, direction: str,
                                  sfp, msb, breaker, sr) -> dict | None:
    """Mirror the entry/stop/TP construction logic from analyze_asset() in the scanner."""
    primary = sfp if (sfp and sfp["direction"] == direction) else \
              breaker if (breaker and breaker["direction"] == direction) else \
              sr if (sr and sr["direction"] == direction) else None
    if primary is None:
        return None

    current = float(df_window["close"].iloc[-1])
    if direction == "LONG":
        entry_low  = primary.get("entry_low",  primary.get("zone_low",  primary.get("level", current)))
        entry_high = primary.get("entry_high", primary.get("zone_high", primary.get("level", current)))
        entry_avg  = primary.get("entry_avg",  round((entry_low + entry_high) / 2, 6))
        stop       = primary.get("stop", round(entry_low * 0.985, 6))
        risk       = entry_avg - stop
        if risk <= 0:
            return None
        tp1, tp2, tp3 = (round(entry_avg + risk*m, 6) for m in (2, 3, 5))
    else:
        entry_high = primary.get("entry_high", primary.get("zone_high", primary.get("level", current)))
        entry_low  = primary.get("entry_low",  primary.get("zone_low",  primary.get("level", current)))
        entry_avg  = primary.get("entry_avg",  round((entry_low + entry_high) / 2, 6))
        stop       = primary.get("stop", round(entry_high * 1.015, 6))
        risk       = stop - entry_avg
        if risk <= 0:
            return None
        tp1, tp2, tp3 = (round(entry_avg - risk*m, 6) for m in (2, 3, 5))

    return {"direction": direction, "entry_avg": entry_avg, "stop": stop,
            "tp1": tp1, "tp2": tp2, "tp3": tp3}


def backtest_asset(asset: str, timeframe: str, df: pd.DataFrame,
                    min_confluence: int) -> list[dict]:
    """
    Walk through historical candles one-by-one, re-running the exact same
    detection logic the live scanner uses at each point in time, and record
    what would have happened to each signal.
    """
    records = []
    swing_lookback = 20
    min_history    = swing_lookback + 20   # buffer for bias calc

    if len(df) < min_history + FORWARD_WINDOW.get(timeframe, 30):
        return records

    for i in range(min_history, len(df) - FORWARD_WINDOW.get(timeframe, 30)):
        window = df.iloc[:i+1]   # everything up to and including candle i

        sfp     = detect_sfp(window)
        msb     = detect_msb(window)
        breaker = detect_breaker(window)
        sr      = detect_sr_cluster(window)

        directions = set()
        if sfp:     directions.add(sfp["direction"])
        if msb:     directions.add(msb["direction"])
        if breaker: directions.add(breaker["direction"])
        if sr:      directions.add(sr["direction"])

        for direction in directions:
            conf_score, reasons = score_setup(sfp, msb, breaker, sr, direction)
            if conf_score < min_confluence:
                continue

            alert = build_alert_from_structures(window, direction, sfp, msb, breaker, sr)
            if alert is None:
                continue

            outcome = simulate_trade(df, i, alert, FORWARD_WINDOW.get(timeframe, 30))

            records.append({
                "asset": asset, "timeframe": timeframe,
                "candle_idx": i, "timestamp": int(window.iloc[-1]["t"]),
                "direction": direction, "confluence": conf_score,
                "reasons": " + ".join(reasons),
                "entry": alert["entry_avg"], "stop": alert["stop"],
                "tp1": alert["tp1"], "tp2": alert["tp2"], "tp3": alert["tp3"],
                **outcome,
            })

    return records


def summarize(records: list[dict]) -> None:
    if not records:
        print("\nNo signals generated in this backtest window.")
        return

    df = pd.DataFrame(records)
    total = len(df)

    print(f"\n{'='*60}")
    print(f"BACKTEST SUMMARY — {total} signals")
    print(f"{'='*60}\n")

    # Overall hit rates
    resolved = df[df["result"] != "no_resolution"]
    stopped  = (df["result"] == "stopped_out").sum()
    tp1_plus = df["hit_tp"].notna().sum()
    tp2_plus = (df["hit_tp"] >= 2).sum()
    tp3_hit  = (df["hit_tp"] == 3).sum()
    no_res   = (df["result"] == "no_resolution").sum()

    print(f"Stopped out:        {stopped:4d}  ({stopped/total*100:.1f}%)")
    print(f"Hit TP1 or better:  {tp1_plus:4d}  ({tp1_plus/total*100:.1f}%)")
    print(f"Hit TP2 or better:  {tp2_plus:4d}  ({tp2_plus/total*100:.1f}%)")
    print(f"Hit TP3:            {tp3_hit:4d}  ({tp3_hit/total*100:.1f}%)")
    print(f"No resolution:      {no_res:4d}  ({no_res/total*100:.1f}%)")

    # By confluence score — this tells you if MIN_CONFLUENCE=2 is the right bar
    print(f"\n{'─'*60}")
    print("BY CONFLUENCE SCORE")
    print(f"{'─'*60}")
    for conf in sorted(df["confluence"].unique()):
        sub = df[df["confluence"] == conf]
        n   = len(sub)
        win = sub["hit_tp"].notna().sum()
        print(f"  Confluence {conf}/4 — n={n:4d} — TP1+ hit rate: {win/n*100:.1f}%")

    # By direction
    print(f"\n{'─'*60}")
    print("BY DIRECTION")
    print(f"{'─'*60}")
    for d in df["direction"].unique():
        sub = df[df["direction"] == d]
        n   = len(sub)
        win = sub["hit_tp"].notna().sum()
        print(f"  {d:<6} — n={n:4d} — TP1+ hit rate: {win/n*100:.1f}%")

    # By asset
    print(f"\n{'─'*60}")
    print("BY ASSET")
    print(f"{'─'*60}")
    for asset in sorted(df["asset"].unique()):
        sub = df[df["asset"] == asset]
        n   = len(sub)
        win = sub["hit_tp"].notna().sum()
        print(f"  {asset:<6} — n={n:4d} — TP1+ hit rate: {win/n*100:.1f}%")

    # By timeframe
    print(f"\n{'─'*60}")
    print("BY TIMEFRAME")
    print(f"{'─'*60}")
    for tf in df["timeframe"].unique():
        sub = df[df["timeframe"] == tf]
        n   = len(sub)
        win = sub["hit_tp"].notna().sum()
        print(f"  {tf:<6} — n={n:4d} — TP1+ hit rate: {win/n*100:.1f}%")

    # Average R achieved (rough — counts stop as -1R, TP1=2R, TP2=3R, TP3=5R)
    r_map = {"stopped_out": -1, "tp1_hit": 2, "tp2_hit": 3, "tp3_hit": 5,
              "no_resolution": 0}
    df["r_outcome"] = df["result"].map(r_map)
    avg_r = df["r_outcome"].mean()
    print(f"\n{'─'*60}")
    print(f"AVERAGE R PER SIGNAL: {avg_r:+.2f}R  (n={total})")
    print(f"{'─'*60}")
    print("Note: this is a simplified expectancy estimate. It assumes clean")
    print("fills at calculated entry/stop/TP and ignores slippage, fees, and")
    print("partial fills. Treat as a directional signal, not a precise P&L.")


def main():
    parser = argparse.ArgumentParser(description="Backtest structure-based signals")
    parser.add_argument("--days", type=int, default=90, help="Days of history to test")
    parser.add_argument("--assets", type=str, default=",".join(DEFAULT_ASSETS),
                        help="Comma-separated asset list")
    parser.add_argument("--timeframe", type=str, default="H1",
                        choices=["H1", "H4", "H12", "D1"])
    parser.add_argument("--min-confluence", type=int, default=MIN_CONFLUENCE,
                        help="Minimum confluence score to count as a signal")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path to dump raw records as CSV")
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.assets.split(",")]

    print(f"Backtesting {len(assets)} assets, {args.days} days, "
          f"timeframe={args.timeframe}, min_confluence={args.min_confluence}\n")

    all_records = []
    for asset in assets:
        print(f"Fetching {asset}...")
        df = fetch_historical_candles(asset, args.timeframe, args.days)
        if df.empty or len(df) < 60:
            print(f"  Skipping {asset} — insufficient data ({len(df)} candles)")
            continue
        print(f"  {len(df)} candles fetched. Running detection...")
        records = backtest_asset(asset, args.timeframe, df, args.min_confluence)
        print(f"  {len(records)} signals generated")
        all_records.extend(records)

    summarize(all_records)

    if args.csv and all_records:
        pd.DataFrame(all_records).to_csv(args.csv, index=False)
        print(f"\nRaw records saved to {args.csv}")


if __name__ == "__main__":
    main()
