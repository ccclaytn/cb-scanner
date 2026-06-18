import os
import re
import json
import requests
import time
import pandas as pd
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# Telegram config is read with .get() (not required at import time) so this
# module can be imported by backtest.py without Telegram secrets being set.
# The actual presence check happens in __main__ before any live run.
TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_TOKEN", "")

_DEFAULT_CHAT   = os.environ.get("CHAT_ID", "")
CHAT_LEADERBOARD      = os.environ.get("CHAT_ID_LEADERBOARD", _DEFAULT_CHAT)
CHAT_DAY_TRADE        = os.environ.get("CHAT_ID_DAY_TRADE", _DEFAULT_CHAT)
CHAT_SWING            = os.environ.get("CHAT_ID_SWING", _DEFAULT_CHAT)
CHAT_HIGH_CONVICTION  = os.environ.get("CHAT_ID_HIGH_CONVICTION", _DEFAULT_CHAT)

def _require_telegram_config():
    """Called from __main__ only — keeps module importable without secrets."""
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN is not set in the environment.")
    if not any([CHAT_LEADERBOARD, CHAT_DAY_TRADE, CHAT_SWING, CHAT_HIGH_CONVICTION]):
        raise RuntimeError(
            "No chat IDs configured. Set CHAT_ID (single channel) or "
            "CHAT_ID_LEADERBOARD / CHAT_ID_DAY_TRADE / CHAT_ID_SWING / "
            "CHAT_ID_HIGH_CONVICTION (multi-channel) in your environment."
        )

# Confluence threshold for cross-posting to the high-conviction channel
HIGH_CONVICTION_MIN = 3

HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"
BINANCE_API     = "https://api.binance.com/api/v3"

MIN_VOLUME_USD         = 5_000_000   # HL 24h notional volume floor
MAX_WORKERS            = 30
RATE_LIMIT_RPS         = 18          # Binance: 1200/min; stay under at 15
DEDUP_FILE             = "dedup.json"
DEDUP_COOLDOWN_CANDLES = 3           # suppress re-alert for N × TF duration
LEADERBOARD_TOP_N      = 10          # show top N assets in leaderboard
ALERT_HISTORY_DAYS     = 7           # days to count prior alerts for context

# Structure detection thresholds
SFP_MIN_SWEEP     = 0.002   # min 0.2% wick beyond swing level
SWING_LOOKBACK    = 20      # candles back for swing high/low
MSB_LOOKBACK      = 15      # candles back to find last swing for MSB
SR_CLUSTER_PCT    = 0.005   # levels within 0.5% count as a cluster
MIN_CONFLUENCE    = 2       # minimum score to fire an alert

_hl_semaphore  = Semaphore(4)
_bin_semaphore = Semaphore(RATE_LIMIT_RPS)

# Internal TF labels → Binance interval strings
_BIN_INTERVAL = {"H1": "1h", "H4": "4h", "H12": "12h", "D1": "1d"}

TIMEFRAMES    = {"1h": "H1", "4h": "H4", "12h": "H12", "1d": "D1"}
TF_SECONDS    = {"H1": 3600, "H4": 14400, "H12": 43200, "D1": 86400}

# H1 = day-trade setups, H12 = swing setups (per spec)
ALERT_TIMEFRAMES  = ("H1", "H12")
SFP_TIMEFRAMES    = ("H1", "H4", "H12", "D1")


# ─── DEDUPLICATION ────────────────────────────────────────────────────────────
def load_dedup() -> dict:
    try:
        with open(DEDUP_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_dedup(state: dict) -> None:
    with open(DEDUP_FILE, "w") as f:
        json.dump(state, f, indent=2)

def dedup_key(sig_type: str, asset: str, tf: str, level: float) -> str:
    return f"{sig_type}:{asset}:{tf}:{float(f'{level:.4g}')}"

def is_duplicate(state: dict, key: str, tf: str) -> bool:
    if key not in state:
        return False
    cooldown = TF_SECONDS.get(tf, TF_SECONDS["H4"]) * DEDUP_COOLDOWN_CANDLES
    return (time.time() - state[key]["ts"]) < cooldown

def mark_sent(state: dict, key: str) -> None:
    entry = state.get(key, {"ts": 0, "count_7d": []})
    now   = time.time()
    week  = now - ALERT_HISTORY_DAYS * 86400
    times = [t for t in entry.get("count_7d", []) if t > week]
    times.append(now)
    state[key] = {"ts": now, "count_7d": times}

def alert_count_7d(state: dict, asset: str) -> int:
    """Total alerts sent for this asset in the last 7 days across all signal types."""
    week  = time.time() - ALERT_HISTORY_DAYS * 86400
    total = 0
    for k, v in state.items():
        if f":{asset}:" in k:
            total += sum(1 for t in v.get("count_7d", []) if t > week)
    return total

def prune_dedup(state: dict) -> dict:
    max_age = TF_SECONDS["D1"] * max(DEDUP_COOLDOWN_CANDLES, ALERT_HISTORY_DAYS)
    now     = time.time()
    pruned  = {}
    for k, v in state.items():
        if isinstance(v, dict) and now - v.get("ts", 0) < max_age:
            pruned[k] = v
    return pruned


# ─── HYPERLIQUID API ──────────────────────────────────────────────────────────
def hl_post(payload: dict, retries: int = 3):
    for attempt in range(retries):
        with _hl_semaphore:
            try:
                r = requests.post(HYPERLIQUID_API, json=payload, timeout=10)
                r.raise_for_status()
                time.sleep(0.25)
                return r.json()
            except requests.exceptions.RequestException as e:
                time.sleep(2 ** attempt)
    return None

def get_all_hl_assets() -> list[str]:
    data = hl_post({"type": "meta"})
    if data and "universe" in data:
        assets = [a["name"] for a in data["universe"]]
        print(f"HL: {len(assets)} assets discovered")
        return assets
    return ["BTC", "ETH", "SOL", "DOGE"]

def get_liquid_hl_assets(assets: list[str]) -> list[str]:
    data = hl_post({"type": "metaAndAssetCtxs"})
    if not data or len(data) < 2:
        return assets
    universe, ctxs = data[0].get("universe", []), data[1]
    liquid, skipped = [], 0
    for meta, ctx in zip(universe, ctxs):
        name = meta.get("name")
        if name not in assets:
            continue
        try:
            if float(ctx.get("dayNtlVlm", 0)) >= MIN_VOLUME_USD:
                liquid.append(name)
            else:
                skipped += 1
        except (TypeError, ValueError):
            skipped += 1
    print(f"Volume filter: {len(liquid)} pass (≥${MIN_VOLUME_USD/1e6:.0f}M), {skipped} skipped")
    return liquid

def get_hl_candles(asset: str, display_tf: str) -> pd.DataFrame:
    """Fallback candle source for assets not on Binance."""
    interval_map = {"H1": "1h", "H4": "4h", "H12": "12h", "D1": "1d"}
    interval_ms  = {"H1": 100*3600*1000, "H4": 400*3600*1000,
                    "H12": 1200*3600*1000, "D1": 2400*3600*1000}
    now_ms = int(time.time() * 1000)
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin":      asset,
            "interval":  interval_map[display_tf],
            "startTime": now_ms - interval_ms.get(display_tf, interval_ms["D1"]),
            "endTime":   now_ms,
        },
    }
    data = hl_post(payload)
    if not data or not isinstance(data, list):
        return pd.DataFrame()
    try:
        first = data[0]
        if isinstance(first, dict):
            df = pd.DataFrame(data).rename(
                columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        else:
            df = pd.DataFrame(data, columns=["t","open","high","low","close","volume"])
        for col in ("open","high","low","close","volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        return df.iloc[:-1].reset_index(drop=True)   # drop forming candle
    except Exception:
        return pd.DataFrame()


# ─── BINANCE CANDLES (primary) ────────────────────────────────────────────────
def get_binance_candles(asset: str, display_tf: str) -> pd.DataFrame:
    symbol   = f"{asset}USDT"
    interval = _BIN_INTERVAL[display_tf]
    for attempt in range(3):
        with _bin_semaphore:
            try:
                r = requests.get(f"{BINANCE_API}/klines",
                                 params={"symbol": symbol, "interval": interval, "limit": 110},
                                 timeout=10)
                if r.status_code == 400:
                    return pd.DataFrame()   # not listed on Binance
                r.raise_for_status()
                raw = r.json()
                if not raw:
                    return pd.DataFrame()
                df = pd.DataFrame(raw, columns=[
                    "t","open","high","low","close","volume",
                    "close_time","qav","num_trades","tbbav","tbqav","ignore"])
                for col in ("open","high","low","close","volume"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["close"])
                return df.iloc[:-1].reset_index(drop=True)   # drop forming candle
            except requests.exceptions.RequestException:
                time.sleep(2 ** attempt)
    return pd.DataFrame()

def get_candles(asset: str, display_tf: str) -> tuple[pd.DataFrame, str]:
    """Returns (df, source) where source is 'binance' or 'hyperliquid'."""
    df = get_binance_candles(asset, display_tf)
    if not df.empty:
        return df, "binance"
    df = get_hl_candles(asset, display_tf)
    if not df.empty:
        return df, "hyperliquid"
    return pd.DataFrame(), "none"


# ─── EMA ──────────────────────────────────────────────────────────────────────
def calc_ema(df: pd.DataFrame, period: int) -> pd.Series | None:
    if len(df) < period + 5:
        return None
    return df["close"].ewm(span=period, adjust=False).mean()


# ─── STRUCTURE DETECTION ──────────────────────────────────────────────────────
def detect_sfp(df: pd.DataFrame) -> dict | None:
    """Swept Floor/Ceiling Pattern on last closed candle."""
    if len(df) < SWING_LOOKBACK + 3:
        return None
    candle  = df.iloc[-1]
    history = df.iloc[-(SWING_LOOKBACK+1):-1]
    h, l, c = float(candle["high"]), float(candle["low"]), float(candle["close"])
    swing_low  = float(history["low"].min())
    swing_high = float(history["high"].max())

    # Bullish SFP
    if l < swing_low and c > swing_low:
        sweep = (swing_low - l) / swing_low
        if sweep < SFP_MIN_SWEEP:
            return None
        stop      = round(l * 0.997, 6)
        risk      = c - stop
        e_low     = round(l + (swing_low - l) * 0.5, 6)
        e_high    = round(swing_low, 6)
        return {
            "type": "SFP", "direction": "LONG",
            "sfp_level": round(swing_low, 6),
            "entry_low": e_low, "entry_high": e_high,
            "entry_avg": round((e_low + e_high) / 2, 6),
            "stop": stop,
            "tp1": round(c + risk*2, 6),
            "tp2": round(c + risk*3, 6),
            "tp3": round(c + risk*5, 6),
            "sweep_pct": round(sweep*100, 2),
            "current": round(c, 6),
        }

    # Bearish SFP
    if h > swing_high and c < swing_high:
        sweep = (h - swing_high) / swing_high
        if sweep < SFP_MIN_SWEEP:
            return None
        stop   = round(h * 1.003, 6)
        risk   = stop - c
        e_high = round(h - (h - swing_high) * 0.5, 6)
        e_low  = round(swing_high, 6)
        return {
            "type": "SFP", "direction": "SHORT",
            "sfp_level": round(swing_high, 6),
            "entry_low": e_low, "entry_high": e_high,
            "entry_avg": round((e_low + e_high) / 2, 6),
            "stop": stop,
            "tp1": round(c - risk*2, 6),
            "tp2": round(c - risk*3, 6),
            "tp3": round(c - risk*5, 6),
            "sweep_pct": round(sweep*100, 2),
            "current": round(c, 6),
        }
    return None


def detect_msb(df: pd.DataFrame) -> dict | None:
    """
    Market Structure Break: last closed candle closes beyond the prior
    swing high (bullish MSB) or swing low (bearish MSB).
    """
    if len(df) < MSB_LOOKBACK + 3:
        return None
    candle  = df.iloc[-1]
    history = df.iloc[-(MSB_LOOKBACK+1):-1]
    c = float(candle["close"])
    swing_high = float(history["high"].max())
    swing_low  = float(history["low"].min())

    if c > swing_high:
        return {"type": "MSB", "direction": "LONG",
                "level": round(swing_high, 6), "current": round(c, 6)}
    if c < swing_low:
        return {"type": "MSB", "direction": "SHORT",
                "level": round(swing_low, 6), "current": round(c, 6)}
    return None


def detect_breaker(df: pd.DataFrame) -> dict | None:
    """
    Breaker block: after a market structure break, price pulls back into the
    last opposing candle's range before the break. This is checked across a
    forward window after the break — NOT required to happen on the same
    candle as the break itself, since in practice the retest follows the
    break by several candles.
    """
    if len(df) < MSB_LOOKBACK + 5:
        return None

    current = df.iloc[-1]
    c       = float(current["close"])

    # Look back far enough to find a break that happened recently, then
    # check if the *current* candle is now retesting that break's breaker zone.
    search_window = df.iloc[-(MSB_LOOKBACK + 10):-1]
    if len(search_window) < MSB_LOOKBACK:
        return None

    # Bullish break: find the most recent candle that closed above the
    # swing high established before it.
    for break_pos in range(len(search_window) - 1, MSB_LOOKBACK - 1, -1):
        ref_high = float(search_window["high"].iloc[break_pos - MSB_LOOKBACK:break_pos].max())
        break_close = float(search_window["close"].iloc[break_pos])
        if break_close > ref_high:
            # Found a bullish break — find the last bearish candle before it
            pre_break = search_window.iloc[:break_pos]
            bearish   = pre_break[pre_break["close"] < pre_break["open"]]
            if bearish.empty:
                continue
            bb      = bearish.iloc[-1]
            bb_high = float(bb["high"])
            bb_low  = float(bb["low"])
            if bb_low <= c <= bb_high:
                return {"type": "BREAKER", "direction": "LONG",
                        "zone_low": round(bb_low, 6), "zone_high": round(bb_high, 6),
                        "current": round(c, 6)}
            break   # only consider the most recent break

    # Bearish break: find the most recent candle that closed below the
    # swing low established before it.
    for break_pos in range(len(search_window) - 1, MSB_LOOKBACK - 1, -1):
        ref_low = float(search_window["low"].iloc[break_pos - MSB_LOOKBACK:break_pos].min())
        break_close = float(search_window["close"].iloc[break_pos])
        if break_close < ref_low:
            pre_break = search_window.iloc[:break_pos]
            bullish   = pre_break[pre_break["close"] > pre_break["open"]]
            if bullish.empty:
                continue
            bb      = bullish.iloc[-1]
            bb_high = float(bb["high"])
            bb_low  = float(bb["low"])
            if bb_low <= c <= bb_high:
                return {"type": "BREAKER", "direction": "SHORT",
                        "zone_low": round(bb_low, 6), "zone_high": round(bb_high, 6),
                        "current": round(c, 6)}
            break

    return None


def detect_sr_cluster(df: pd.DataFrame) -> dict | None:
    """
    S/R cluster: two or more swing highs or lows within SR_CLUSTER_PCT of each other.
    Returns the tightest cluster and its direction.
    """
    if len(df) < SWING_LOOKBACK + 3:
        return None
    history = df.iloc[-(SWING_LOOKBACK+1):-1]
    c       = float(df.iloc[-1]["close"])

    # Collect local swing highs and lows (simple: compare to neighbours)
    highs, lows = [], []
    h_arr = history["high"].values
    l_arr = history["low"].values
    for i in range(1, len(h_arr)-1):
        if h_arr[i] > h_arr[i-1] and h_arr[i] > h_arr[i+1]:
            highs.append(h_arr[i])
        if l_arr[i] < l_arr[i-1] and l_arr[i] < l_arr[i+1]:
            lows.append(l_arr[i])

    def find_cluster(levels):
        for i, a in enumerate(levels):
            for b in levels[i+1:]:
                if abs(a - b) / max(a, b) < SR_CLUSTER_PCT:
                    return round((a + b) / 2, 6)
        return None

    # Check if price is at a swing high cluster (resistance → bearish)
    r_cluster = find_cluster(highs)
    if r_cluster and abs(c - r_cluster) / r_cluster < SR_CLUSTER_PCT:
        return {"type": "SR_CLUSTER", "direction": "SHORT",
                "level": r_cluster, "current": round(c, 6)}

    # Check if price is at a swing low cluster (support → bullish)
    s_cluster = find_cluster(lows)
    if s_cluster and abs(c - s_cluster) / s_cluster < SR_CLUSTER_PCT:
        return {"type": "SR_CLUSTER", "direction": "LONG",
                "level": s_cluster, "current": round(c, 6)}

    return None


# ─── BIAS DETECTION ───────────────────────────────────────────────────────────
def get_bias(df_h4: pd.DataFrame, df_d1: pd.DataFrame) -> str:
    """
    Determine HTF bias from D1 structure (primary) + H4 (secondary).
    Returns 'LONG', 'SHORT', or 'NEUTRAL'.
    Uses HH/HL for bullish, LH/LL for bearish — requires 3 swings to confirm.
    """
    def swing_structure(df, lookback=30) -> str:
        if len(df) < lookback:
            return "NEUTRAL"
        data    = df.tail(lookback)
        highs   = []
        lows    = []
        h_arr   = data["high"].values
        l_arr   = data["low"].values
        for i in range(1, len(h_arr)-1):
            if h_arr[i] > h_arr[i-1] and h_arr[i] > h_arr[i+1]:
                highs.append(h_arr[i])
            if l_arr[i] < l_arr[i-1] and l_arr[i] < l_arr[i+1]:
                lows.append(l_arr[i])
        if len(highs) < 2 or len(lows) < 2:
            return "NEUTRAL"
        hh = highs[-1] > highs[-2]
        hl = lows[-1]  > lows[-2]
        lh = highs[-1] < highs[-2]
        ll = lows[-1]  < lows[-2]
        if hh and hl:
            return "LONG"
        if lh and ll:
            return "SHORT"
        return "NEUTRAL"

    d1_bias = swing_structure(df_d1, lookback=40) if not df_d1.empty else "NEUTRAL"
    h4_bias = swing_structure(df_h4, lookback=30) if not df_h4.empty else "NEUTRAL"

    # D1 is primary — if D1 is clear, use it
    if d1_bias != "NEUTRAL":
        return d1_bias
    # Fall back to H4
    return h4_bias


# ─── CONFLUENCE SCORING ───────────────────────────────────────────────────────
def score_setup(sfp, msb, breaker, sr_cluster, direction: str) -> tuple[int, list[str]]:
    """
    Score a setup 0–4 based on how many structure conditions align.
    Only counts conditions that match the intended direction.
    Returns (score, list_of_reasons).
    """
    score   = 0
    reasons = []
    if sfp and sfp["direction"] == direction:
        score += 1
        reasons.append(f"SFP at {sfp['sfp_level']}")
    if msb and msb["direction"] == direction:
        score += 1
        reasons.append(f"MSB {direction.lower()}")
    if breaker and breaker["direction"] == direction:
        score += 1
        reasons.append(f"Breaker block {breaker['zone_low']}–{breaker['zone_high']}")
    if sr_cluster and sr_cluster["direction"] == direction:
        score += 1
        reasons.append(f"S/R cluster at {sr_cluster['level']}")
    return score, reasons


# ─── EMA TREND STATE (for leaderboard) ───────────────────────────────────────
def get_tf_trend(df: pd.DataFrame) -> dict:
    """Returns trend state and EMA values for a single timeframe."""
    if df.empty or len(df) < 25:
        return {"trend": "nodata", "ema13": None, "ema21": None,
                "close": None, "retest": False, "bear_retest": False,
                "retest_dist": None}
    close    = float(df["close"].iloc[-1])
    ema13_s  = calc_ema(df, 13)
    ema21_s  = calc_ema(df, 21)
    if ema13_s is None or ema21_s is None:
        return {"trend": "nodata", "ema13": None, "ema21": None,
                "close": close, "retest": False, "bear_retest": False,
                "retest_dist": None}
    e13 = float(ema13_s.iloc[-1])
    e21 = float(ema21_s.iloc[-1])
    dist = abs(close - e21) / e21

    if close > e13 > e21:
        trend = "trending"
    elif close > e21:
        trend = "reclaiming"
    else:
        trend = "below"

    return {
        "trend":       trend,
        "ema13":       round(e13, 6),
        "ema21":       round(e21, 6),
        "close":       round(close, 6),
        "retest":      dist < 0.015 and trend in ("trending", "reclaiming"),
        "bear_retest": dist < 0.015 and trend == "below",
        "retest_dist": round(dist * 100, 1),
    }


# ─── LEADERBOARD READ LINE ────────────────────────────────────────────────────
def build_read(score: int, tf_states: dict, retest_tf: str | None) -> str:
    if retest_tf:
        return f"Pullback to {retest_tf} EMA21 — continuation entry"

    tfs     = ["D1", "H12", "H4", "H1"]
    green   = [tf for tf in tfs if tf_states.get(tf, {}).get("trend") in ("trending", "reclaiming")]
    lagging = [tf for tf in tfs if tf_states.get(tf, {}).get("trend") == "reclaiming"]
    below   = [tf for tf in tfs if tf_states.get(tf, {}).get("trend") == "below"]

    # Distance above H1 EMA21
    h1 = tf_states.get("H1", {})
    dist_str = ""
    if h1.get("trend") in ("trending", "reclaiming") and h1.get("retest_dist") is not None:
        dist_str = f" · +{h1['retest_dist']}% over H1 EMA21"

    if score == 4:
        return f"Full uptrend — long pullbacks{dist_str}"
    if score == 3:
        lag = lagging[0] if lagging else (below[0] if below else "")
        return f"Strong — {lag} lagging" if lag else "Strong — minor divergence"
    if score == 2:
        g = "/".join(green) if green else "some TFs"
        return f"Mixed — {g} up, wait for alignment"
    if score == 1:
        g = green[0] if green else "H1"
        return f"Weak — only {g} up, countertrend risk"
    return "No trend — avoid"


# ─── SINGLE ASSET ANALYSIS ────────────────────────────────────────────────────
def analyze_asset(asset: str) -> dict:
    # Fetch all timeframes, tracking data source per TF
    dfs     = {}
    sources = {}
    for interval, display in TIMEFRAMES.items():
        df, src = get_candles(asset, display)
        dfs[display]     = df
        sources[display] = src

    # EMA trend states (leaderboard)
    tf_states = {tf: get_tf_trend(dfs[tf]) for tf in ("D1", "H12", "H4", "H1")}
    score     = sum(1 for s in tf_states.values()
                    if s["trend"] in ("trending", "reclaiming"))

    # Retest detection
    retest_tf = None
    for tf in ("D1", "H12", "H4"):
        if tf_states[tf].get("retest"):
            retest_tf = tf
            break

    # HTF bias from D1 + H4
    bias = get_bias(dfs.get("H4", pd.DataFrame()), dfs.get("D1", pd.DataFrame()))

    # Structure detection per alert timeframe
    setup_alerts = []
    for tf in ALERT_TIMEFRAMES:
        df = dfs.get(tf, pd.DataFrame())
        if df.empty or len(df) < SWING_LOOKBACK + 5:
            continue

        sfp      = detect_sfp(df)
        msb      = detect_msb(df)
        breaker  = detect_breaker(df)
        sr       = detect_sr_cluster(df)

        # Determine which direction(s) to evaluate
        directions = []
        if bias == "LONG":
            directions = ["LONG"]
        elif bias == "SHORT":
            directions = ["SHORT"]
        else:
            # Neutral bias: still fire if SFP + MSB both agree
            if sfp:   directions.append(sfp["direction"])
            if msb:   directions.append(msb["direction"])
            directions = list(set(directions))

        for direction in directions:
            conf_score, reasons = score_setup(sfp, msb, breaker, sr, direction)
            if conf_score < MIN_CONFLUENCE:
                continue

            # Build entry / stop / TP from whichever structure fired
            primary = sfp if (sfp and sfp["direction"] == direction) else \
                      breaker if (breaker and breaker["direction"] == direction) else \
                      sr if (sr and sr["direction"] == direction) else None

            if primary is None:
                continue

            current = float(df["close"].iloc[-1])
            if direction == "LONG":
                entry_low  = primary.get("entry_low",  primary.get("zone_low",  primary.get("level", current)))
                entry_high = primary.get("entry_high", primary.get("zone_high", primary.get("level", current)))
                entry_avg  = primary.get("entry_avg",  round((entry_low + entry_high) / 2, 6))
                stop       = primary.get("stop", round(entry_low * 0.985, 6))
                risk       = entry_avg - stop
                tp1        = round(entry_avg + risk * 2, 6)
                tp2        = round(entry_avg + risk * 3, 6)
                tp3        = round(entry_avg + risk * 5, 6)
            else:
                entry_high = primary.get("entry_high", primary.get("zone_high", primary.get("level", current)))
                entry_low  = primary.get("entry_low",  primary.get("zone_low",  primary.get("level", current)))
                entry_avg  = primary.get("entry_avg",  round((entry_low + entry_high) / 2, 6))
                stop       = primary.get("stop", round(entry_high * 1.015, 6))
                risk       = stop - entry_avg
                tp1        = round(entry_avg - risk * 2, 6)
                tp2        = round(entry_avg - risk * 3, 6)
                tp3        = round(entry_avg - risk * 5, 6)

            rr1 = round(abs(tp1 - entry_avg) / max(risk, 1e-10), 1)
            rr2 = round(abs(tp2 - entry_avg) / max(risk, 1e-10), 1)
            rr3 = round(abs(tp3 - entry_avg) / max(risk, 1e-10), 1)

            setup_alerts.append({
                "asset":       asset,
                "timeframe":   tf,
                "trade_type":  "Day Trade" if tf == "H1" else "Swing Trade",
                "direction":   direction,
                "bias":        bias,
                "confluence":  conf_score,
                "reasons":     reasons,
                "entry_low":   round(entry_low, 6),
                "entry_high":  round(entry_high, 6),
                "entry_avg":   round(entry_avg, 6),
                "stop":        round(stop, 6),
                "tp1":         tp1,
                "tp2":         tp2,
                "tp3":         tp3,
                "rr1":         rr1,
                "rr2":         rr2,
                "rr3":         rr3,
                "current":     round(current, 6),
                "source":      sources.get(tf, "binance"),
            })

    return {
        "asset":      asset,
        "score":      score,
        "tf_states":  tf_states,
        "retest_tf":  retest_tf,
        "bias":       bias,
        "read":       build_read(score, tf_states, retest_tf),
        "alerts":     setup_alerts,
        "sources":    sources,
    }


# ─── THREADED SCAN ────────────────────────────────────────────────────────────
def scan_all_assets(assets: list[str]) -> list[dict]:
    results, total, done = [], len(assets), 0
    print(f"\nScanning {total} assets ({MAX_WORKERS} threads)...\n")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(analyze_asset, a): a for a in assets}
        for f in as_completed(futures):
            done += 1
            try:
                results.append(f.result())
                bar = "█"*(done*20//total) + "░"*(20-done*20//total)
                print(f"\r[{bar}] {done}/{total} — {futures[f]}", end="", flush=True)
            except Exception as e:
                print(f"\nError {futures[f]}: {e}")
    print("\n")
    return results


# ─── TRADINGVIEW LINK ─────────────────────────────────────────────────────────
_TF_TV = {"H1": "60", "H4": "240", "H12": "720", "D1": "1D"}

def tv_url(asset: str, tf: str = "H4") -> str:
    interval = _TF_TV.get(tf, "240")
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{asset}USDT&interval={interval}"


# ─── MARKDOWNV2 HELPERS ───────────────────────────────────────────────────────
_ESC_RE = re.compile(r"([_\.\-\+\#\|\{\}\(\)!~`>\\])")

def esc(text) -> str:
    return _ESC_RE.sub(r"\\\1", str(text))

def fmt_price(p) -> str:
    v = float(p)
    if v >= 1000:
        s = f"{v:,.2f}"
    elif v >= 1:
        s = f"{v:.4f}"
    else:
        s = f"{v:.6f}"
    return esc(s)

def dot(trend: str) -> str:
    return {"trending": "🟢", "reclaiming": "🟡", "below": "🔴"}.get(trend, "⚪")


# ─── LEADERBOARD MESSAGE ──────────────────────────────────────────────────────
def format_leaderboard(results: list[dict]) -> str:
    # Filter: only assets with score ≥ 1, sort by score desc
    ranked = sorted(
        [r for r in results if r["score"] >= 1],
        key=lambda x: x["score"],
        reverse=True
    )[:LEADERBOARD_TOP_N]

    ts  = datetime.now(timezone.utc).strftime("%Y\\-%m\\-%d %H:%M UTC")
    msg = f"📊 *TREND LEADERBOARD*\n"
    msg += f"EMA 13/21 · D1 · H12 · H4 · H1 · {ts}\n\n"
    msg += f"*CRYPTO* · Hyperliquid perps · Binance\\+HL\n"
    msg += f"`{'#':<3} {'ASSET':<7} {'D1':^4} {'H12':^4} {'H4':^4} {'H1':^4} {'SCORE':^6}`\n"
    msg += "─────────────────────────────\n"

    for i, r in enumerate(ranked, 1):
        s   = r["tf_states"]
        d1  = dot(s["D1"]["trend"])
        h12 = dot(s["H12"]["trend"])
        h4  = dot(s["H4"]["trend"])
        h1  = dot(s["H1"]["trend"])
        sc  = r["score"]

        link    = f"[{r['asset']}]({tv_url(r['asset'], 'H4')})"
        retest  = " `RETEST`" if r["retest_tf"] else ""
        msg += f"`{i:<3}` *{link}*{retest}\n"
        msg += f"{d1} {h12} {h4} {h1}  *{sc}/4*\n"
        msg += f"_{esc(r['read'])}_\n\n"

    msg += "─────────────────────────────\n"
    msg += "*HOW TO READ IT*\n"
    msg += "🟢 Trending — price \\> EMA13 \\> EMA21\n"
    msg += "🟡 Reclaiming — above EMA21, not yet stacked\n"
    msg += "🔴 Below trend\n"
    msg += "`RETEST` — price pulled back to EMA21 on a trending TF: continuation entry zone\n"
    return msg


# ─── TRADE ALERT MESSAGE ─────────────────────────────────────────────────────
def format_alert(alert: dict, prior_alerts: int) -> str:
    d      = alert["direction"]
    tf     = alert["timeframe"]
    arrow  = "📈" if d == "LONG" else "📉"
    dot_e  = "🟢" if d == "LONG" else "🔴"
    stars  = "⭐" * alert["confluence"] + "☆" * (4 - alert["confluence"])
    link   = f"[{alert['asset']}]({tv_url(alert['asset'], tf)})"
    chart  = f"[Open chart →]({tv_url(alert['asset'], tf)})"

    freq_flag = ""
    if prior_alerts >= 5:
        freq_flag = " ⚠️ _frequent level_"
    elif prior_alerts >= 3:
        freq_flag = " ℹ️ _tested before_"

    msg  = f"{arrow} *{dot_e} {esc(d)} Setup — {link}*\n"
    msg += f"*{esc(tf)} · {esc(alert['trade_type'])}*\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"*Bias*         {esc(alert['bias'])}\n"
    msg += f"*Confluence*   {stars} {alert['confluence']}/4\n\n"

    msg += f"*Entry Zone*\n"
    msg += f"`${fmt_price(alert['entry_low'])} – ${fmt_price(alert['entry_high'])}`\n"
    msg += f"_Scale in \\(avg `${fmt_price(alert['entry_avg'])}`\\)_\n\n"

    msg += f"*Stop*\n"
    msg += f"`${fmt_price(alert['stop'])}` — _invalidation level_\n\n"

    msg += f"*Targets*\n"
    msg += f"TP1 `${fmt_price(alert['tp1'])}` · R:R `{esc(str(alert['rr1']))}R`\n"
    msg += f"TP2 `${fmt_price(alert['tp2'])}` · R:R `{esc(str(alert['rr2']))}R`\n"
    msg += f"TP3 `${fmt_price(alert['tp3'])}` · R:R `{esc(str(alert['rr3']))}R`\n\n"

    msg += f"*Structure*\n"
    msg += esc(" + ".join(alert["reasons"])) + "\n\n"

    msg += f"*Context*\n"
    msg += f"Alerted {prior_alerts}× in last 7 days{freq_flag}\n"
    msg += f"Data: {esc(alert['source'])}\n\n"

    msg += f"*Chart* {chart}"
    return msg


# ─── TELEGRAM ─────────────────────────────────────────────────────────────────
def send_telegram(text: str, chat_id: str) -> bool:
    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    chunks  = [text[i:i+4000] for i in range(0, len(text), 4000)]
    success = True
    for chunk in chunks:
        try:
            r = requests.post(url, json={
                "chat_id":                  chat_id,
                "text":                     chunk,
                "parse_mode":               "MarkdownV2",
                "disable_web_page_preview": True,
            }, timeout=10)
            if not r.ok:
                print(f"Telegram error ({chat_id}): {r.status_code} {r.text}")
                success = False
            time.sleep(0.35)
        except Exception as e:
            print(f"Telegram send error ({chat_id}): {e}")
            success = False
    return success


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    _require_telegram_config()   # fail fast if secrets are missing for a live run

    # Mode passed as CLI arg: "leaderboard" or "alerts" (default: both)
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    start = time.time()
    print(f"Mode: {mode} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")

    # 1. Asset discovery + volume filter
    all_hl    = get_all_hl_assets()
    liquid_hl = get_liquid_hl_assets(all_hl)

    # 2. Scan
    results = scan_all_assets(liquid_hl)

    # 3. Dedup
    dedup = prune_dedup(load_dedup())

    # 4. Leaderboard
    if mode in ("leaderboard", "both"):
        print("Sending leaderboard...")
        send_telegram(format_leaderboard(results), CHAT_LEADERBOARD)

    # 5. Alerts
    if mode in ("alerts", "both"):
        sent, suppressed, cross_posted = 0, 0, 0
        # Highest timeframe first (H12 before H1 — swing before day trade)
        tf_order = {"D1": 0, "H12": 1, "H4": 2, "H1": 3}
        all_alerts = []
        for r in results:
            all_alerts.extend(r["alerts"])
        all_alerts.sort(key=lambda x: tf_order.get(x["timeframe"], 9))

        for alert in all_alerts:
            key = dedup_key("SETUP", alert["asset"], alert["timeframe"],
                            alert["entry_avg"])
            if is_duplicate(dedup, key, alert["timeframe"]):
                suppressed += 1
                print(f"  [dedup] {alert['asset']} {alert['timeframe']} {alert['direction']}")
                continue

            # Route by timeframe: H1 = day-trade channel, H12 = swing channel
            target_chat = CHAT_DAY_TRADE if alert["timeframe"] == "H1" else CHAT_SWING

            prior = alert_count_7d(dedup, alert["asset"])
            msg   = format_alert(alert, prior)
            print(f"Sending: {alert['asset']} {alert['direction']} {alert['timeframe']} "
                  f"(conf {alert['confluence']}/4, prior {prior}) -> "
                  f"{'day-trade' if alert['timeframe']=='H1' else 'swing'}")

            if send_telegram(msg, target_chat):
                mark_sent(dedup, key)
                sent += 1

                # Cross-post to high-conviction channel if it clears the bar
                if alert["confluence"] >= HIGH_CONVICTION_MIN:
                    print(f"  -> cross-posting to high-conviction (conf {alert['confluence']}/4)")
                    send_telegram(msg, CHAT_HIGH_CONVICTION)
                    cross_posted += 1

        print(f"\nAlerts sent: {sent}, suppressed: {suppressed}, "
              f"cross-posted to high-conviction: {cross_posted}")

    # 6. Persist dedup
    save_dedup(dedup)
    print(f"Done in {time.time()-start:.1f}s | dedup: {len(dedup)} entries")
