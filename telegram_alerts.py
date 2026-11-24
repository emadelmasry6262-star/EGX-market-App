"""
EGX SMC / Liquidity Reversal Strategy (v20) - Telegram Live Alerts & Daily Report Worker
Features:
  1) Real-Time Intraday Price Fetching (5-min / 1-min live bars from TradingView merged with Daily bars)
  2) Intraday Live Alerts (Every 15 mins during EGX session 10:00 - 14:45 Cairo time):
     - Instant BUY-STOP Trigger notification with Entry, First Target (+5%), and Initial Stop Loss (-5%)
     - Instant STOP-LOSS Hit notification when live price <= Updated Stop Loss
     - Manual Run Instant Summary (manual_intraday) so manual triggers always send a live session status
  3) End-of-Day Daily Reports (Scheduled at 15:00 & 22:00 Cairo time, independent of manual runs):
     - EGX30 General Index status & % change
     - Newly opened trades today & ongoing open positions with updated trailing stops (5%->+1%, 7.5%->+5%, etc.)
     - Closed trades today (Close price, Exit reason, PnL %)
     - Active Buy-Stop setups & Radar watchlist with Met vs Pending filters
     - Chart image of the best matching stock setup
"""
import os
import sys
import json
import time
import urllib.request
import urllib.parse
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tvDatafeed import TvDatafeed, Interval

# ============================ CONFIG & SECRETS ============================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TV_USER = os.getenv("TV_USER", "").strip()
TV_PASS = os.getenv("TV_PASS", "").strip()

STATE_FILE = "telegram_alert_state.json"
MARKET_TZ = "Africa/Cairo"

DEFAULT_TICKERS = [
    "COMI", "TMGH", "EAST", "ETEL", "HRHO", "ABUK", "FWRY", "PHDC", "EFIH",
    "ISPH", "EMFD", "ALCN", "AMOC", "RMDA", "CCAP", "ORAS", "ORHD",
    "EFID", "BTFH", "JUFO", "GBCO", "RAYA", "SKPC", "CLHO", "VLMR",
    "HELI", "MFPC", "MCQE", "EGAL", "ADIB", "SWDY", "ORWE",
    "MASR", "MTIE", "POUL", "CSAG", "ARCC", "BINV"
]

EXCHANGE = "EGX"
INDEX_SYMBOL = "EGX30"
N_BARS = 600
DATA_START = "2023-01-01"
TEST_START = "2023-01-01"

TREND_EMA = 50
TREND_SLOPE = 10
ENTRY_TIMEOUT = 5
STOP_PCT = 0.05
FIRST_TARGET_PCT = 0.05
MAX_HOLD = 60
COMMISSION = 0.002

USE_CANDLE_FILTER = True
SWEEP_CLOSE_POS = 0.50
USE_EMA200_FILTER = True
LONG_EMA = 200

USE_SWING_LOWS = True
SWING_WINDOW = 10
SWING_MAX_AGE = 250

USE_ORDER_BLOCKS = True
ALLOW_STANDALONE_OB = False
ALLOW_MULTI_CONFLUENCE = False
OB_IMPULSE_PCT = 0.03
OB_MAX_AGE = 180


# ============================ TELEGRAM HELPERS ============================
def send_telegram_msg(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[warn] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID. Message skipped:")
        print(text)
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = []
    if len(text) <= 3900:
        chunks = [text]
    else:
        current = ""
        for line in text.split("\n"):
            if len(current) + len(line) + 1 > 3900:
                chunks.append(current)
                current = line
            else:
                current = (current + "\n" + line) if current else line
        if current:
            chunks.append(current)

    all_ok = True
    for chunk in chunks:
        payload = urllib.parse.urlencode({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true"
        }).encode("utf-8")

        sent = False
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, data=payload, method="POST")
                with urllib.request.urlopen(req, timeout=15) as resp:
                    if resp.status == 200:
                        sent = True
                        break
            except Exception as e:
                print(f"[warn] Telegram send attempt {attempt+1} failed: {e}")
                time.sleep(1.5)
        if not sent:
            all_ok = False
    return all_ok


def send_telegram_photo(image_path: str, caption: str = "") -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or not os.path.exists(image_path):
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    boundary = f"----WebKitFormBoundary{int(time.time() * 1000)}"

    with open(image_path, "rb") as img_f:
        img_bytes = img_f.read()

    parts = []
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{TELEGRAM_CHAT_ID}\r\n".encode("utf-8"))
    if caption:
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption[:1000]}\r\n".encode("utf-8"))
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"parse_mode\"\r\n\r\nHTML\r\n".encode("utf-8"))
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"{os.path.basename(image_path)}\"\r\n"
        f"Content-Type: image/png\r\n\r\n".encode("utf-8")
    )
    parts.append(img_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    body = b"".join(parts)
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body))
    }

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status == 200:
                    return True
        except Exception as e:
            print(f"[warn] Telegram photo attempt {attempt+1} failed: {e}")
            time.sleep(1.5)
    return False


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                data.setdefault("sent_alerts", {})
                data.setdefault("sent_reports", {})
                return data
        except Exception:
            pass
    return {"sent_alerts": {}, "sent_reports": {}}


def save_state(state: dict):
    alerts = state.get("sent_alerts", {})
    if len(alerts) > 400:
        keys = list(alerts.keys())[-400:]
        state["sent_alerts"] = {k: alerts[k] for k in keys}
    reports = state.get("sent_reports", {})
    if len(reports) > 60:
        rkeys = list(reports.keys())[-60:]
        state["sent_reports"] = {k: reports[k] for k in rkeys}
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ============================ DATA & REAL-TIME ENGINE ============================
def get_tv_client():
    if TV_USER and TV_PASS:
        return TvDatafeed(TV_USER, TV_PASS)
    return TvDatafeed()


def clean_df(df):
    cols = ["open", "high", "low", "close"]
    if "volume" in df.columns:
        cols.append("volume")
    df = df[cols].astype(float).dropna()
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_realtime_intraday_snapshot(tv, sym, today_cairo, fallback_bar=None):
    """Fetches the latest 5-min (or 1-min) intraday candles from TradingView for real-time price."""
    for interval in (Interval.in_5_minute, Interval.in_1_minute):
        try:
            idf = tv.get_hist(symbol=sym, exchange=EXCHANGE, interval=interval, n_bars=80)
            if idf is not None and not idf.empty:
                idf = idf.copy()
                idf.columns = [c.lower() for c in idf.columns]
                idx = pd.DatetimeIndex(idf.index)
                if idx.tz is None:
                    # TradingView timestamps are typically UTC; convert to Cairo date
                    idx_cairo = idx.tz_localize("UTC").tz_convert(MARKET_TZ)
                else:
                    idx_cairo = idx.tz_convert(MARKET_TZ)
                idf.index = idx_cairo

                today_bars = idf[idf.index.date == today_cairo.date()]
                if not today_bars.empty:
                    o_val = float(today_bars["open"].iloc[0])
                    h_val = float(today_bars["high"].max())
                    l_val = float(today_bars["low"].min())
                    c_val = float(today_bars["close"].iloc[-1])
                    v_val = float(today_bars["volume"].sum()) if "volume" in today_bars.columns else 0.0
                    last_ts = today_bars.index[-1].strftime("%H:%M")

                    if fallback_bar is not None:
                        o_val = fallback_bar["open"]
                        h_val = max(h_val, fallback_bar["high"])
                        l_val = min(l_val, fallback_bar["low"])
                        v_val = max(v_val, fallback_bar.get("volume", 0.0))

                    return {
                        "date": today_cairo.strftime("%Y-%m-%d"),
                        "open": o_val,
                        "high": h_val,
                        "low": l_val,
                        "close": c_val,
                        "volume": v_val,
                        "last_tick_time": last_ts,
                        "source": "intraday_live"
                    }
        except Exception:
            continue
    return fallback_bar


def download_stock_with_live(tv, sym, today_cairo, fetch_live=True):
    """Returns (completed_daily_df, live_bar_dict_or_None)."""
    for _ in range(3):
        try:
            raw = tv.get_hist(symbol=sym, exchange=EXCHANGE, interval=Interval.in_daily, n_bars=N_BARS)
            if raw is not None and not raw.empty:
                df = raw.copy()
                df.columns = [c.lower() for c in df.columns]
                df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
                df = clean_df(df)

                daily_today_bar = None
                if df.index[-1] == today_cairo:
                    last_row = df.iloc[-1]
                    daily_today_bar = {
                        "date": today_cairo.strftime("%Y-%m-%d"),
                        "open": float(last_row["open"]),
                        "high": float(last_row["high"]),
                        "low": float(last_row["low"]),
                        "close": float(last_row["close"]),
                        "volume": float(last_row.get("volume", 0.0)),
                        "last_tick_time": "Daily",
                        "source": "daily_bar"
                    }
                    completed_df = df.iloc[:-1]
                else:
                    completed_df = df

                completed_df = completed_df[completed_df.index >= pd.Timestamp(DATA_START)]

                live_bar = daily_today_bar
                if fetch_live:
                    rt_bar = fetch_realtime_intraday_snapshot(tv, sym, today_cairo, fallback_bar=daily_today_bar)
                    if rt_bar is not None:
                        live_bar = rt_bar

                return completed_df, live_bar
        except Exception:
            time.sleep(0.5)
    return None, None


def get_egx30_status(tv, today_cairo):
    """Fetches EGX30 index performance for the daily report."""
    comp_df, live_bar = download_stock_with_live(tv, INDEX_SYMBOL, today_cairo, fetch_live=True)
    if comp_df is None or comp_df.empty:
        return None
    if live_bar is not None:
        prev_close = float(comp_df["close"].iloc[-1])
        cur_close = float(live_bar["close"])
    else:
        if len(comp_df) < 2:
            return None
        prev_close = float(comp_df["close"].iloc[-2])
        cur_close = float(comp_df["close"].iloc[-1])

    diff = cur_close - prev_close
    pct = (diff / prev_close) * 100.0 if prev_close > 0 else 0.0
    return {
        "close": round(cur_close, 2),
        "prev_close": round(prev_close, 2),
        "diff": round(diff, 2),
        "pct": round(pct, 2)
    }


# ============================ STRATEGY ENGINE (v20) ============================
def prior_downtrend(close):
    s = pd.Series(close)
    ema = s.ewm(span=TREND_EMA, adjust=False).mean()
    down = (s < ema) & (ema < ema.shift(TREND_SLOPE))
    if USE_EMA200_FILTER:
        ema200 = s.ewm(span=LONG_EMA, adjust=False).mean()
        down = down & (s > ema200)
    return down.shift(1, fill_value=False).values.astype(bool)


def evaluate_current_filters(df):
    """Returns detailed met and pending filters on the latest bar."""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    o = df["open"].values
    v = df["volume"].values

    s = pd.Series(c)
    ema50 = s.ewm(span=TREND_EMA, adjust=False).mean()
    ema200 = s.ewm(span=LONG_EMA, adjust=False).mean()

    below_ema50 = bool(c[-1] < ema50.iloc[-1])
    ema50_falling = bool(ema50.iloc[-1] < ema50.shift(TREND_SLOPE).iloc[-1])
    above_ema200 = bool(c[-1] > ema200.iloc[-1]) if USE_EMA200_FILTER else True

    rng = h[-1] - l[-1]
    close_pos = ((c[-1] - l[-1]) / rng) if rng > 0 else 0.0
    candle_strong = bool(close_pos >= SWEEP_CLOSE_POS)

    vol_series = pd.Series(v)
    vol_ma20 = vol_series.rolling(20, min_periods=1).mean().iloc[-1]
    vol_strong = bool(v[-1] > vol_ma20) if vol_ma20 > 0 else False

    met = []
    pending = []

    if below_ema50:
        met.append("تحت EMA50")
    else:
        pending.append("إغلاق تحت EMA50")

    if ema50_falling:
        met.append("ميل EMA50 هابط")
    else:
        pending.append("انحدار EMA50")

    if USE_EMA200_FILTER:
        if above_ema200:
            met.append("فوق EMA200")
        else:
            pending.append("تداول فوق EMA200")

    if candle_strong:
        met.append(f"شمعة انعكاس قوية ({close_pos*100:.0f}%)")
    else:
        pending.append("إغلاق في النصف العلوي للشمعة")

    if vol_strong:
        met.append("فوليوم مرتفع")
    else:
        pending.append("تأكيد الفوليوم")

    trend_ok = below_ema50 and ema50_falling and above_ema200
    return {
        "trend_ok": trend_ok,
        "all_ok": trend_ok and candle_strong,
        "met": met,
        "pending": pending
    }


def prev_period_lows(df):
    wk = df.index.to_period("W-THU")
    mo = df.index.to_period("M")
    wlow = df["low"].groupby(wk).min()
    mlow = df["low"].groupby(mo).min()
    pwl = wlow.shift(1).reindex(wk).values
    pml = mlow.shift(1).reindex(mo).values
    return pwl, pml


def build_smc_structures(o, h, l, c):
    L = len(c)
    swing_pivots = {}
    if USE_SWING_LOWS:
        w = SWING_WINDOW
        for piv in range(w, L - w):
            window_lows = l[piv - w: piv + w + 1]
            if l[piv] == np.min(window_lows) and np.sum(window_lows == l[piv]) == 1:
                swing_pivots[piv + w] = (piv, float(l[piv]))

    order_blocks = {}
    if USE_ORDER_BLOCKS:
        for k in range(1, L - 1):
            if c[k] < o[k] and c[k + 1] > o[k + 1]:
                impulse = (c[k + 1] - o[k + 1]) / o[k + 1]
                if impulse >= OB_IMPULSE_PCT and c[k + 1] > h[k]:
                    order_blocks[k + 1] = (k, float(l[k]), float(h[k]))

    return swing_pivots, order_blocks


def last_red_run_high(red, h, i, max_len=30):
    r = i
    while r >= 0 and not red[r] and i - r < max_len:
        r -= 1
    if r < 0 or not red[r]:
        return None
    run_high, k = h[r], r
    while k - 1 >= 0 and red[k - 1] and r - k < max_len:
        k -= 1
        run_high = max(run_high, h[k])
    return run_high


def update_trailing_stop_next_session(pos, bar_high, bar_low):
    entry = pos["entry"]
    pos["mfe"] = max(pos["mfe"], bar_high)
    mfe_pct = (pos["mfe"] / entry) - 1.0

    if mfe_pct >= 0.25:
        pos["post_25"] = True
        pos["max_tier"] = "25%+"
        new_stop = max(entry * 1.15, bar_low)
        pos["stop"] = max(pos["stop"], new_stop)
    elif mfe_pct >= 0.15:
        pos["max_tier"] = "15%"
        pos["stop"] = max(pos["stop"], entry * 1.10)
    elif mfe_pct >= 0.10:
        pos["max_tier"] = "10%"
        pos["stop"] = max(pos["stop"], entry * 1.075)
    elif mfe_pct >= 0.075:
        pos["max_tier"] = "7.5%"
        pos["stop"] = max(pos["stop"], entry * 1.05)
    elif mfe_pct >= 0.05:
        pos["max_tier"] = "5%"
        pos["stop"] = max(pos["stop"], entry * 1.01)


def evaluate_stock_state(tk, df):
    """Runs v20 strategy and returns (live_setup_or_pos, closed_today_list, radar_hits)."""
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    dates = df.index
    L = len(df)
    last_idx = L - 1
    red = c < o
    down = prior_downtrend(c)
    pwl, pml = prev_period_lows(df)
    swing_pivots, order_blocks = build_smc_structures(o, h, l, c)
    start_ts = pd.Timestamp(TEST_START)

    active_swings, active_obs = [], []
    used_wm, state, pos, s = set(), "IDLE", None, {}
    closed_today = []

    for i in range(L):
        if i in swing_pivots:
            piv_i, p_val = swing_pivots[i]
            active_swings.append(dict(piv_i=piv_i, price=p_val, used=False))
        if i in order_blocks:
            ob_i, ob_l, ob_h = order_blocks[i]
            active_obs.append(dict(ob_i=ob_i, low=ob_l, high=ob_h, used=False))

        active_swings = [sw for sw in active_swings if (i - sw["piv_i"] <= SWING_MAX_AGE) and not sw["used"]]
        active_obs = [ob for ob in active_obs if (i - ob["ob_i"] <= OB_MAX_AGE) and not ob["used"]]

        if pos is not None:
            if i > pos["i0"]:
                hit_stop = l[i] <= pos["stop"]
                hit_timeout = (not pos["post_25"]) and (i - pos["i0"] >= MAX_HOLD)
                if hit_stop or hit_timeout:
                    if i == last_idx:
                        exit_price = min(o[i], pos["stop"]) if hit_stop else c[i]
                        pnl_pct = (exit_price / pos["entry"] - 1.0) * 100.0
                        if hit_stop:
                            reason = "ضرب وقف الخسارة المتحرك" if pos["stop"] > pos["entry"] else "ضرب وقف الخسارة"
                        else:
                            reason = f"انتهاء مدة الاحتفاظ ({MAX_HOLD} جلسة)"
                        closed_today.append(dict(
                            ticker=tk,
                            entry=round(pos["entry"], 3),
                            exit_price=round(exit_price, 3),
                            pnl_pct=round(pnl_pct, 2),
                            reason=reason,
                            tier=pos["max_tier"]
                        ))
                    pos, state = None, "IDLE"
                else:
                    update_trailing_stop_next_session(pos, h[i], l[i])
            for sw in active_swings:
                if c[i] < sw["price"]:
                    sw["used"] = True
            for ob in active_obs:
                if c[i] < ob["low"]:
                    ob["used"] = True
            continue

        if state == "WAIT":
            if i - s["sweep_i"] > ENTRY_TIMEOUT:
                state = "IDLE"
            else:
                trigger = s["run_high"]
                if h[i] >= trigger:
                    fill = max(o[i], trigger)
                    stop = fill * (1 - STOP_PCT)
                    state = "IDLE"
                    pos = dict(
                        i0=i, entry=fill, stop=stop, mfe=fill,
                        level=s["level"], kind=s["kind"],
                        sweep_low=s["sweep_low"], sweep_i=s["sweep_i"],
                        max_tier="0%", post_25=False
                    )
                    if l[i] <= stop:
                        if i == last_idx:
                            pnl_pct = (stop / fill - 1.0) * 100.0
                            closed_today.append(dict(
                                ticker=tk,
                                entry=round(fill, 3),
                                exit_price=round(stop, 3),
                                pnl_pct=round(pnl_pct, 2),
                                reason="ضرب وقف الخسارة في نفس الجلسة",
                                tier="0%"
                            ))
                        pos = None
                    else:
                        update_trailing_stop_next_session(pos, h[i], l[i])
                    continue
                if c[i] < s["level"]:
                    state = "IDLE"
                    continue
                s["sweep_low"] = min(s["sweep_low"], l[i])
                if red[i]:
                    s["run_high"] = max(s["run_high"], h[i]) if s["last_red"] else h[i]
                s["last_red"] = bool(red[i])

        if state == "IDLE" and dates[i] >= start_ts and down[i]:
            candle_ok = True
            if USE_CANDLE_FILTER:
                rng = h[i] - l[i]
                if rng <= 0 or (c[i] - l[i]) / rng < SWEEP_CLOSE_POS:
                    candle_ok = False

            if candle_ok:
                sweep_hits = []
                if not np.isnan(pwl[i]) and l[i] < pwl[i] and c[i] > pwl[i]:
                    if ("week", round(float(pwl[i]), 4)) not in used_wm:
                        sweep_hits.append(("week", float(pwl[i]), float(pwl[i])))
                if not np.isnan(pml[i]) and l[i] < pml[i] and c[i] > pml[i]:
                    if ("month", round(float(pml[i]), 4)) not in used_wm:
                        sweep_hits.append(("month", float(pml[i]), float(pml[i])))

                matched_swings = []
                for sw in active_swings:
                    if l[i] < sw["price"] and c[i] > sw["price"]:
                        sweep_hits.append(("swing_low", sw["price"], sw["price"]))
                        matched_swings.append(sw)

                ob_hits, matched_obs = [], []
                for ob in active_obs:
                    if l[i] <= ob["high"] and c[i] > ob["low"] and i > ob["ob_i"] + 2:
                        ob_hits.append(("order_block", ob["low"], ob["high"]))
                        matched_obs.append(ob)

                if sweep_hits or (ALLOW_STANDALONE_OB and ob_hits):
                    hits = sweep_hits + ob_hits
                    seen_k = []
                    for k, _, _ in hits:
                        if k not in seen_k:
                            seen_k.append(k)
                    if ALLOW_MULTI_CONFLUENCE or len(seen_k) <= 2:
                        rh = last_red_run_high(red, h, i)
                        if rh is not None:
                            for sw in matched_swings:
                                sw["used"] = True
                            for ob in matched_obs:
                                ob["used"] = True
                            for k, inv_lvl, _ in hits:
                                if k in ("week", "month"):
                                    used_wm.add((k, round(inv_lvl, 4)))
                            kind_lbl = "+".join(seen_k) if len(seen_k) <= 2 else "multi_confluence"
                            invalidation_lvl = min(x[1] for x in hits)
                            s = dict(
                                level=invalidation_lvl, kind=kind_lbl,
                                sweep_low=l[i], sweep_i=i, run_high=rh, last_red=bool(red[i])
                            )
                            state = "WAIT"

        for sw in active_swings:
            if c[i] < sw["price"]:
                sw["used"] = True
        for ob in active_obs:
            if c[i] < ob["low"]:
                ob["used"] = True

    live_info = None
    last_close = float(c[-1])
    if pos is not None:
        pnl_pct = (last_close / pos["entry"] - 1.0) * 100.0
        live_info = dict(
            status="OPEN",
            ticker=tk,
            opened_today=(pos["i0"] == last_idx),
            entry_date=dates[pos["i0"]].strftime("%Y-%m-%d"),
            entry=round(pos["entry"], 3),
            target1=round(pos["entry"] * (1.0 + FIRST_TARGET_PCT), 3),
            last_close=round(last_close, 3),
            pnl_pct=round(pnl_pct, 2),
            stop=round(pos["stop"], 3),
            tier="POST-25% TRAIL" if pos["post_25"] else f"Tier {pos['max_tier']}",
            kind=pos["kind"],
            bars=last_idx - pos["i0"]
        )
    elif state == "WAIT":
        rem_bars = ENTRY_TIMEOUT - (last_idx - s["sweep_i"])
        if rem_bars > 0:
            dist_trigger = (s["run_high"] / last_close - 1.0) * 100.0
            live_info = dict(
                status="WAIT",
                ticker=tk,
                sweep_date=dates[s["sweep_i"]].strftime("%Y-%m-%d"),
                trigger=round(s["run_high"], 3),
                target1=round(s["run_high"] * (1.0 + FIRST_TARGET_PCT), 3),
                last_close=round(last_close, 3),
                dist_pct=round(dist_trigger, 2),
                est_stop=round(s["run_high"] * (1.0 - STOP_PCT), 3),
                invalidation=round(s["level"], 3),
                kind=s["kind"],
                rem_bars=rem_bars
            )

    radar_hits = []
    last_pwl = float(pwl[-1]) if not np.isnan(pwl[-1]) else np.nan
    last_pml = float(pml[-1]) if not np.isnan(pml[-1]) else np.nan
    filt_info = evaluate_current_filters(df)

    if not np.isnan(last_pwl) and last_close >= last_pwl:
        d_w = (last_close - last_pwl) / last_pwl * 100.0
        if 0.0 <= d_w <= 2.5:
            radar_hits.append(dict(
                ticker=tk, close=round(last_close, 3),
                ref_type="قاع أسبوعي", ref_val=round(last_pwl, 3),
                dist_pct=round(d_w, 2),
                trend_ok=filt_info["trend_ok"],
                all_ok=filt_info["all_ok"],
                met=filt_info["met"],
                pending=filt_info["pending"]
            ))
    if not np.isnan(last_pml) and last_close >= last_pml:
        d_m = (last_close - last_pml) / last_pml * 100.0
        if 0.0 <= d_m <= 2.5:
            radar_hits.append(dict(
                ticker=tk, close=round(last_close, 3),
                ref_type="قاع شهري", ref_val=round(last_pml, 3),
                dist_pct=round(d_m, 2),
                trend_ok=filt_info["trend_ok"],
                all_ok=filt_info["all_ok"],
                met=filt_info["met"],
                pending=filt_info["pending"]
            ))

    return live_info, closed_today, radar_hits


def generate_best_setup_chart(tk: str, df: pd.DataFrame, setup_info: dict, out_path: str = "best_setup.png"):
    """Generates a clean technical chart for the top triggered/pending setup."""
    sub = df.tail(65).copy()
    if len(sub) < 10:
        return None

    x = np.arange(len(sub))
    o = sub["open"].values
    h = sub["high"].values
    l = sub["low"].values
    c = sub["close"].values

    ema50 = sub["close"].ewm(span=TREND_EMA, adjust=False).mean().values

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=130)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    for idx_i in range(len(sub)):
        color = "#22c55e" if c[idx_i] >= o[idx_i] else "#ef4444"
        ax.plot([x[idx_i], x[idx_i]], [l[idx_i], h[idx_i]], color=color, linewidth=1.1)
        body_low = min(o[idx_i], c[idx_i])
        body_h = max(abs(c[idx_i] - o[idx_i]), 0.001)
        ax.bar(x[idx_i], body_h, bottom=body_low, width=0.58, color=color, edgecolor=color)

    ax.plot(x, ema50, color="#38bdf8", linewidth=1.4, label="EMA 50")

    if setup_info["status"] == "OPEN":
        ax.axhline(setup_info["entry"], color="#facc15", linestyle="--", linewidth=1.3, label=f"Entry: {setup_info['entry']}")
        ax.axhline(setup_info["target1"], color="#4ade80", linestyle=":", linewidth=1.3, label=f"Target 1 (+5%): {setup_info['target1']}")
        ax.axhline(setup_info["stop"], color="#f87171", linestyle="-.", linewidth=1.3, label=f"Stop: {setup_info['stop']}")
    elif setup_info["status"] == "WAIT":
        ax.axhline(setup_info["trigger"], color="#facc15", linestyle="--", linewidth=1.3, label=f"Buy-Stop: {setup_info['trigger']}")
        ax.axhline(setup_info["target1"], color="#4ade80", linestyle=":", linewidth=1.3, label=f"Target 1 (+5%): {setup_info['target1']}")
        ax.axhline(setup_info["est_stop"], color="#f87171", linestyle="-.", linewidth=1.3, label=f"Init Stop: {setup_info['est_stop']}")

    step = max(1, len(sub) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([d.strftime("%m-%d") for d in sub.index[::step]], color="#cbd5e1", fontsize=8)
    ax.tick_params(axis="y", colors="#cbd5e1", labelsize=8)
    ax.grid(True, color="#1e293b", linestyle="--", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.set_title(f"{tk} - EGX SMC v20 ({setup_info['status']}) | {setup_info['kind']}", color="#f8fafc", fontsize=11, pad=10)
    ax.legend(loc="upper left", facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# ============================ MAIN EXECUTION ============================
def main():
    raw_mode = sys.argv[1].lower().strip() if len(sys.argv) > 1 else "auto"
    now_cairo = datetime.now(ZoneInfo(MARKET_TZ))
    today_norm = pd.Timestamp(now_cairo.date())
    today_str = today_norm.strftime("%Y-%m-%d")

    is_manual_intraday = (raw_mode == "manual_intraday")
    is_manual_daily = (raw_mode == "daily_report")

    if raw_mode == "auto":
        # EGX trades Sun (6) to Thu (3); Fri=4, Sat=5
        if now_cairo.weekday() in (4, 5):
            print(f"[info] Weekend in Cairo ({now_cairo:%A}). Skipping.")
            return
        if now_cairo.hour >= 15:
            mode = "daily_report"
        else:
            mode = "intraday"
    elif is_manual_intraday:
        mode = "intraday"
    else:
        mode = raw_mode

    print(f"[run] RawMode: {raw_mode} -> EffectiveMode: {mode} | Cairo Time: {now_cairo:%Y-%m-%d %H:%M}")
    state = load_state()
    sent_alerts = state.setdefault("sent_alerts", {})
    sent_reports = state.setdefault("sent_reports", {})

    tv = get_tv_client()
    all_live = []
    all_closed_today = []
    all_radar = []
    intraday_bars = {}
    stock_dfs = {}

    for tk in DEFAULT_TICKERS:
        comp_df, live_bar = download_stock_with_live(tv, tk, today_norm, fetch_live=True)
        if comp_df is None or len(comp_df) < 250:
            continue

        if mode == "daily_report" and live_bar is not None:
            new_row = pd.DataFrame([{
                "open": live_bar["open"],
                "high": live_bar["high"],
                "low": live_bar["low"],
                "close": live_bar["close"],
                "volume": live_bar.get("volume", 0.0)
            }], index=[today_norm])
            eval_df = pd.concat([comp_df, new_row])
        else:
            eval_df = comp_df
            if live_bar is not None:
                intraday_bars[tk] = live_bar

        stock_dfs[tk] = eval_df
        live_info, closed_list, rad_hits = evaluate_stock_state(tk, eval_df)
        if live_info is not None:
            all_live.append(live_info)
        all_closed_today.extend(closed_list)
        all_radar.extend(rad_hits)
        time.sleep(0.12)

    # ---------------- 1) INTRADAY REAL-TIME ALERTS ----------------
    if mode == "intraday":
        alerts_triggered = 0
        near_trigger_lines = []
        open_status_lines = []

        for item in all_live:
            tk = item["ticker"]
            bar = intraday_bars.get(tk)
            if not bar:
                continue

            cur_price = bar["close"]
            bar_high = bar["high"]
            bar_low = bar["low"]
            tick_t = bar.get("last_tick_time", now_cairo.strftime("%H:%M"))

            # A) Buy-Stop Trigger Hit
            if item["status"] == "WAIT":
                trig = item["trigger"]
                dist_now = (trig / cur_price - 1.0) * 100.0 if cur_price > 0 else 0.0
                near_trigger_lines.append(
                    f"• <b>#{tk}</b> | الحالي: {cur_price:.3f} | تفعيل: <b>{trig:.3f}</b> (متبقي {dist_now:+.2f}%)"
                )
                if bar_high >= trig or cur_price >= trig:
                    alert_key = f"BUY_{tk}_{item['sweep_date']}_{trig}"
                    if alert_key not in sent_alerts:
                        fill_est = max(bar["open"], trig)
                        target1_est = fill_est * (1.0 + FIRST_TARGET_PCT)
                        stop_est = fill_est * (1.0 - STOP_PCT)
                        msg = (
                            f"🚀 <b>إشارة شراء لحظية محققة (BUY-STOP TRIGGERED)</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📌 <b>السهم:</b> #{tk}\n"
                            f"💰 <b>السعر اللحظي:</b> {cur_price:.3f} ج (أعلى سعر: {bar_high:.3f})\n"
                            f"🎯 <b>سعر الدخول (Trigger):</b> {trig:.3f} ج\n"
                            f"🏆 <b>الهدف الأول (+5%):</b> {target1_est:.3f} ج\n"
                            f"🛑 <b>وقف الخسارة المبدئي (-5%):</b> {stop_est:.3f} ج\n"
                            f"📊 <b>نوع النموذج:</b> {item['kind']}\n"
                            f"⏰ <b>التوقيت:</b> {tick_t} بتوقيت القاهرة"
                        )
                        if send_telegram_msg(msg):
                            sent_alerts[alert_key] = now_cairo.strftime("%Y-%m-%d %H:%M")
                            alerts_triggered += 1

            # B) Stop-Loss Hit on Open Position
            elif item["status"] == "OPEN":
                sl = item["stop"]
                live_pnl = (cur_price / item["entry"] - 1.0) * 100.0
                open_status_lines.append(
                    f"• <b>#{tk}</b> | الحالي: {cur_price:.3f} ({live_pnl:+.2f}%) | الوقف: <b>{sl:.3f}</b>"
                )
                if bar_low <= sl or cur_price <= sl:
                    alert_key = f"STOP_{tk}_{item['entry_date']}_{sl}_{today_str}"
                    if alert_key not in sent_alerts:
                        pnl_at_stop = (sl / item["entry"] - 1.0) * 100.0
                        msg = (
                            f"🚨 <b>تنبيه ضرب وقف الخسارة اللحظي (STOP-LOSS HIT)</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📌 <b>السهم:</b> #{tk}\n"
                            f"📉 <b>السعر اللحظي:</b> {cur_price:.3f} ج (أدنى سعر: {bar_low:.3f})\n"
                            f"🛑 <b>سعر وقف الخسارة:</b> {sl:.3f} ج\n"
                            f"💵 <b>سعر الدخول:</b> {item['entry']:.3f} ج (النتيجة: {pnl_at_stop:+.2f}%)\n"
                            f"🛡️ <b>مستوى الحماية:</b> {item['tier']}\n"
                            f"⏰ <b>التوقيت:</b> {tick_t} بتوقيت القاهرة"
                        )
                        if send_telegram_msg(msg):
                            sent_alerts[alert_key] = now_cairo.strftime("%Y-%m-%d %H:%M")
                            alerts_triggered += 1

        # Send instant live summary when triggered manually
        if is_manual_intraday:
            egx_info = get_egx30_status(tv, today_cairo)
            egx_line = ""
            if egx_info:
                egx_emoji = "🟢" if egx_info["pct"] >= 0 else "🔴"
                egx_line = f"{egx_emoji} <b>مؤشر EGX30:</b> {egx_info['close']} ({egx_info['pct']:+.2f}%)\n"

            summary_lines = [
                f"⚡ <b>فحص لحظي يدوي للجلسة (EGX SMC v20)</b>",
                f"⏰ <b>التوقيت:</b> {now_cairo:%Y-%m-%d %H:%M} بتوقيت القاهرة",
                f"📡 <b>الأسهم المفحوصة لحظياً:</b> {len(intraday_bars)} / {len(stock_dfs)}",
                egx_line.strip() if egx_line else "📊 <b>مؤشر EGX30:</b> قيد التحديث",
                "━━━━━━━━━━━━━━━━━━━━",
                f"🔔 <b>إشارات لحظية جديدة أُرسلت الآن:</b> {alerts_triggered}",
                "",
                f"⏳ <b>أوامر الشراء المعلقة المراقبة لحظياً ({len(near_trigger_lines)}):</b>"
            ]
            summary_lines.extend(near_trigger_lines if near_trigger_lines else ["• لا توجد أوامر شراء معلقة حالياً."])
            summary_lines.append("")
            summary_lines.append(f"🟢 <b>الصفقات المفتوحة المراقبة لحظياً ({len(open_status_lines)}):</b>")
            summary_lines.extend(open_status_lines if open_status_lines else ["• لا توجد صفقات مفتوحة حالياً."])

            send_telegram_msg("\n".join(summary_lines))

        print(f"[intraday] Checked {len(intraday_bars)} live bars. Sent {alerts_triggered} new alerts.")
        save_state(state)

    # ---------------- 2) DAILY END-OF-DAY REPORT (15:00 & 22:00) ----------------
    elif mode == "daily_report":
        # Separate slot keys so manual runs NEVER block scheduled 15:00 or 22:00 reports
        if is_manual_daily:
            slot_key = f"{today_str}_manual_{now_cairo.strftime('%H%M')}"
            report_label = "تقرير يدوي فوري"
        else:
            slot = "22" if now_cairo.hour >= 20 else "15"
            slot_key = f"{today_str}_{slot}"
            report_label = "التقرير المسائي (10:00 م)" if slot == "22" else "تقرير إغلاق الجلسة (3:00 م)"
            if sent_reports.get(slot_key):
                print(f"[daily] Scheduled report for slot {slot_key} already sent.")
                return

        egx_info = get_egx30_status(tv, today_cairo)
        opens = [x for x in all_live if x["status"] == "OPEN"]
        opened_today = [x for x in opens if x.get("opened_today")]
        ongoing_opens = [x for x in opens if not x.get("opened_today")]
        waits = [x for x in all_live if x["status"] == "WAIT"]

        # Deduplicate radar hits per ticker keeping closest reference level
        radar_by_tk = {}
        for r in all_radar:
            tk = r["ticker"]
            if tk not in radar_by_tk or r["dist_pct"] < radar_by_tk[tk]["dist_pct"]:
                radar_by_tk[tk] = r
        sorted_radar = sorted(radar_by_tk.values(), key=lambda x: (not x["all_ok"], not x["trend_ok"], x["dist_pct"]))

        lines = [
            f"📊 <b>{report_label} - رادار البورصة المصرية (EGX SMC v20)</b>",
            f"📅 <b>التاريخ:</b> {today_str} | ⏰ {now_cairo:%H:%M} القاهرة"
        ]

        if egx_info:
            egx_emoji = "🟢" if egx_info["pct"] >= 0 else "🔴"
            lines.append(
                f"{egx_emoji} <b>المؤشر العام EGX30:</b> {egx_info['close']} نقطة "
                f"(<b>{egx_info['pct']:+.2f}%</b> | {egx_info['diff']:+.2f})"
            )

        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"🆕 <b>1) أسهم حققت الاستراتيجية واتفتحت كصفقة اليوم ({len(opened_today)}):</b>")
        if not opened_today:
            lines.append("• لم تتفعل صفقات شراء جديدة في جلسة اليوم.")
        else:
            for o in opened_today:
                lines.append(
                    f"• <b>#{o['ticker']}</b> | دخول: <b>{o['entry']}</b> | إغلاق: {o['last_close']} (<b>{o['pnl_pct']:+.2f}%</b>)\n"
                    f"  └ الهدف الأول: <b>{o['target1']}</b> | الوقف: <b>{o['stop']}</b> | النموذج: {o['kind']}"
                )

        lines.append("")
        lines.append(f"🟢 <b>2) الصفقات المفتوحة المستمرة ({len(ongoing_opens)}):</b>")
        if not ongoing_opens:
            lines.append("• لا توجد صفقات مفتوحة من جلسات سابقة.")
        else:
            for o in ongoing_opens:
                lines.append(
                    f"• <b>#{o['ticker']}</b> | دخول: {o['entry']} | إغلاق: {o['last_close']} (<b>{o['pnl_pct']:+.2f}%</b>)\n"
                    f"  └ الوقف المحدث: <b>{o['stop']}</b> ({o['tier']}) | هدف 1: {o['target1']} | جلسات: {o['bars']}"
                )

        lines.append("")
        lines.append(f"🏁 <b>3) الصفقات المغلقة في جلسة اليوم ({len(all_closed_today)}):</b>")
        if not all_closed_today:
            lines.append("• لا توجد صفقات مغلقة اليوم.")
        else:
            for cl in all_closed_today:
                res_emoji = "✅ ربح" if cl["pnl_pct"] > 0 else "❌ خسارة"
                lines.append(
                    f"• <b>#{cl['ticker']}</b> | دخول: {cl['entry']} ➔ إغلاق: <b>{cl['exit_price']}</b>\n"
                    f"  └ النتيجة: {res_emoji} (<b>{cl['pnl_pct']:+.2f}%</b>) | السبب: {cl['reason']}"
                )

        lines.append("")
        lines.append(f"⏳ <b>4) أوامر Buy-Stop جاهزة للتنفيذ ({len(waits)}):</b>")
        if not waits:
            lines.append("• لا توجد أوامر Buy-Stop معلقة حالياً.")
        else:
            for w in waits:
                lines.append(
                    f"• <b>#{w['ticker']}</b> | سعر التفعيل: <b>{w['trigger']}</b> (+{w['dist_pct']}%)\n"
                    f"  └ هدف 1 (+5%): <b>{w['target1']}</b> | وقف (-5%): {w['est_stop']} | متبقي: {w['rem_bars']} جلسات"
                )

        lines.append("")
        lines.append(f"👀 <b>5) أسهم تقترب من تنفيذ الشروط ({len(sorted_radar)}):</b>")
        if not sorted_radar:
            lines.append("• لا توجد أسهم قريبة من مناطق السيولة حالياً.")
        else:
            for r in sorted_radar[:8]:
                met_str = "، ".join(r["met"]) if r["met"] else "لا يوجد"
                pend_str = "، ".join(r["pending"]) if r["pending"] else "مكتملة - بانتظار كسر القاع"
                lines.append(
                    f"• <b>#{r['ticker']}</b> ({r['close']} ج | فوق {r['ref_type']} {r['ref_val']} بـ +{r['dist_pct']}%)\n"
                    f"  ✅ <b>محقق:</b> {met_str}\n"
                    f"  ⏳ <b>منتظر:</b> {pend_str}"
                )

        report_text = "\n".join(lines)
        if send_telegram_msg(report_text):
            if not is_manual_daily:
                sent_reports[slot_key] = now_cairo.strftime("%Y-%m-%d %H:%M")
                save_state(state)
            print(f"[daily] Report ({slot_key}) sent successfully.")

            # Send chart of the best candidate (prioritize newly opened -> wait setups -> top radar)
            best_candidate = None
            if opened_today:
                best_candidate = opened_today[0]
            elif waits:
                best_candidate = sorted(waits, key=lambda x: x["dist_pct"])[0]
            elif opens:
                best_candidate = opens[0]

            if best_candidate and best_candidate["ticker"] in stock_dfs:
                tk_best = best_candidate["ticker"]
                chart_file = generate_best_setup_chart(tk_best, stock_dfs[tk_best], best_candidate)
                if chart_file:
                    cap = (
                        f"📈 <b>شارت أفضل فرصة ({best_candidate['status']}): #{tk_best}</b>\n"
                        f"النموذج: {best_candidate['kind']} | الهدف الأول: {best_candidate['target1']}"
                    )
                    send_telegram_photo(chart_file, caption=cap)


if __name__ == "__main__":
    main()
