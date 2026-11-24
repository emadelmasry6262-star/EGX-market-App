"""
EGX SMC / Liquidity Reversal Strategy (v20) - Telegram Live Alerts & Daily Report Worker
Features:
  1) Real-Time Intraday Price Fetching:
     - Fetches latest intraday bars (5m / 1m) from TradingView alongside daily bars to capture exact live prices during the session.
  2) Intraday Live Alerts & Manual Run Summary:
     - Instant BUY-STOP Trigger notification when live price >= Trigger price (with Entry, Target 1 (+5%), and Stop-Loss).
     - Instant STOP-LOSS Hit notification when live price <= Updated Stop Loss.
     - Manual run during session sends an instant live session status message even if no new alerts triggered.
  3) End-of-Day Daily Report (15:00 & 22:00 Cairo Time):
     - EGX30 Index status (% change).
     - Open Positions (new & continuing, with updated trailing stops: 5%->+1%, 7.5%->+5%, 10%->+7.5%, 15%->+10%, 25%+->trailing).
     - Closed Positions during the session (exit price, exit reason, PnL%).
     - Ready Buy-Stop Setups & Watchlist Radar with met/pending filter breakdown.
     - Chart image of the best matching stock setup sent with the Daily Report.
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
TARGET1_PCT = 0.05
MAX_HOLD = 60

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
    payload = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true"
    }).encode("utf-8")

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=payload, method="POST")
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    return True
        except Exception as e:
            print(f"[warn] Telegram send attempt {attempt+1} failed: {e}")
            time.sleep(1.5)
    return False


def send_telegram_photo(image_path: str, caption: str = "") -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or not os.path.exists(image_path):
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    boundary = "----EGXBotBoundary7MA4YWxkTrZu0gW"

    with open(image_path, "rb") as f:
        img_data = f.read()

    parts = []
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{TELEGRAM_CHAT_ID}\r\n".encode("utf-8"))
    if caption:
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}\r\n".encode("utf-8"))
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"parse_mode\"\r\n\r\nHTML\r\n".encode("utf-8"))
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"{os.path.basename(image_path)}\"\r\n"
        f"Content-Type: image/png\r\n\r\n".encode("utf-8") + img_data + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
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
                return json.load(f)
        except Exception:
            pass
    return {"sent_alerts": {}, "last_daily_slots": {}}


def save_state(state: dict):
    alerts = state.get("sent_alerts", {})
    if len(alerts) > 400:
        keys = list(alerts.keys())[-400:]
        state["sent_alerts"] = {k: alerts[k] for k in keys}
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ============================ DATA & REAL-TIME FETCHING ============================
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


def fetch_realtime_intraday_snapshot(tv, sym, today_cairo):
    """Fetches latest intraday 5m (or 1m fallback) bars for today to get true real-time price."""
    for interval in (Interval.in_5_minute, Interval.in_1_minute):
        try:
            intra = tv.get_hist(symbol=sym, exchange=EXCHANGE, interval=interval, n_bars=80)
            if intra is not None and not intra.empty:
                intra = intra.copy()
                intra.columns = [c.lower() for c in intra.columns]
                idx = pd.DatetimeIndex(intra.index)
                if idx.tz is not None:
                    idx = idx.tz_convert(MARKET_TZ).tz_localize(None)
                intra.index = idx
                today_bars = intra[intra.index.normalize() == today_cairo]
                if not today_bars.empty:
                    return {
                        "date": today_cairo.strftime("%Y-%m-%d"),
                        "open": float(today_bars["open"].iloc[0]),
                        "high": float(today_bars["high"].max()),
                        "low": float(today_bars["low"].min()),
                        "close": float(today_bars["close"].iloc[-1]),
                        "volume": float(today_bars["volume"].sum()) if "volume" in today_bars.columns else 0.0,
                        "last_bar_time": today_bars.index[-1].strftime("%H:%M"),
                        "source": "intraday_live"
                    }
        except Exception:
            pass
    return None


def download_stock_with_live(tv, sym, today_cairo, fetch_intraday=True):
    """Returns (completed_daily_df, live_bar_dict_or_None)."""
    for _ in range(3):
        try:
            raw = tv.get_hist(symbol=sym, exchange=EXCHANGE, interval=Interval.in_daily, n_bars=N_BARS)
            if raw is not None and not raw.empty:
                df = raw.copy()
                df.columns = [c.lower() for c in df.columns]
                df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
                df = clean_df(df)

                live_bar = None
                if df.index[-1] == today_cairo:
                    last_row = df.iloc[-1]
                    live_bar = {
                        "date": today_cairo.strftime("%Y-%m-%d"),
                        "open": float(last_row["open"]),
                        "high": float(last_row["high"]),
                        "low": float(last_row["low"]),
                        "close": float(last_row["close"]),
                        "volume": float(last_row.get("volume", 0.0)),
                        "last_bar_time": "Daily",
                        "source": "daily_live"
                    }
                    completed_df = df.iloc[:-1]
                else:
                    completed_df = df

                if fetch_intraday:
                    rt_bar = fetch_realtime_intraday_snapshot(tv, sym, today_cairo)
                    if rt_bar is not None:
                        if live_bar is not None:
                            rt_bar["open"] = live_bar["open"]
                            rt_bar["high"] = max(live_bar["high"], rt_bar["high"])
                            rt_bar["low"] = min(live_bar["low"], rt_bar["low"])
                            rt_bar["volume"] = max(live_bar["volume"], rt_bar["volume"])
                        live_bar = rt_bar

                completed_df = completed_df[completed_df.index >= pd.Timestamp(DATA_START)]
                return completed_df, live_bar
        except Exception:
            time.sleep(0.5)
    return None, None


def get_egx30_status(tv, today_cairo):
    comp_df, live_bar = download_stock_with_live(tv, INDEX_SYMBOL, today_cairo, fetch_intraday=True)
    if comp_df is None or len(comp_df) < 2:
        return None
    if live_bar is not None:
        cur_val = live_bar["close"]
        prev_close = float(comp_df["close"].iloc[-1])
    else:
        cur_val = float(comp_df["close"].iloc[-1])
        prev_close = float(comp_df["close"].iloc[-2])
    chg_pts = cur_val - prev_close
    chg_pct = (chg_pts / prev_close) * 100.0 if prev_close > 0 else 0.0
    return {
        "value": round(cur_val, 2),
        "prev_close": round(prev_close, 2),
        "chg_pts": round(chg_pts, 2),
        "chg_pct": round(chg_pct, 2),
        "status_ar": "صاعد (ربح)" if chg_pts >= 0 else "هابط (خسارة)",
        "emoji": "🟢" if chg_pts >= 0 else "🔴"
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


def detailed_filter_check(df):
    """Returns detailed met/pending filters for watchlist stocks."""
    c = df["close"]
    ema50 = c.ewm(span=TREND_EMA, adjust=False).mean()
    ema200 = c.ewm(span=LONG_EMA, adjust=False).mean()

    last_c = float(c.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])
    prev_ema50 = float(ema50.iloc[-1 - TREND_SLOPE]) if len(ema50) > TREND_SLOPE else last_ema50
    last_ema200 = float(ema200.iloc[-1])

    cond_below_ema50 = last_c < last_ema50
    cond_ema50_slope = last_ema50 < prev_ema50
    cond_above_ema200 = (last_c > last_ema200) if USE_EMA200_FILTER else True

    h_last, l_last, o_last = float(df["high"].iloc[-1]), float(df["low"].iloc[-1]), float(df["open"].iloc[-1])
    rng = h_last - l_last
    close_pos = ((last_c - l_last) / rng) if rng > 0 else 0.0
    cond_candle = close_pos >= SWEEP_CLOSE_POS

    vol_ok = True
    if "volume" in df.columns and len(df) >= 20:
        avg_vol = float(df["volume"].iloc[-21:-1].mean())
        last_vol = float(df["volume"].iloc[-1])
        vol_ok = (avg_vol == 0) or (last_vol >= avg_vol * 0.8)

    met = []
    pending = []

    if cond_above_ema200:
        met.append("فوق EMA200")
    else:
        pending.append(f"اختراق EMA200 ({last_ema200:.2f})")

    if cond_below_ema50 and cond_ema50_slope:
        met.append("اتجاه هابط تصحيحي (تحت EMA50)")
    else:
        pending.append("تأكيد ميل EMA50 الهابط")

    if cond_candle:
        met.append("قوة إغلاق الشمعة")
    else:
        pending.append("شمعة انعكاسية قوية")

    if vol_ok:
        met.append("سيولة مناسبة")
    else:
        pending.append("زيادة حجم التداول")

    pending.append("كسر القاع والارتداد فوقه")
    all_trend_ok = cond_below_ema50 and cond_ema50_slope and cond_above_ema200
    return all_trend_ok, met, pending


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
        pos["max_tier"] = "25%+ (تتبع القاع)"
        new_stop = max(entry * 1.15, bar_low)
        pos["stop"] = max(pos["stop"], new_stop)
    elif mfe_pct >= 0.15:
        pos["max_tier"] = "15% -> تأمين +10%"
        pos["stop"] = max(pos["stop"], entry * 1.10)
    elif mfe_pct >= 0.10:
        pos["max_tier"] = "10% -> تأمين +7.5%"
        pos["stop"] = max(pos["stop"], entry * 1.075)
    elif mfe_pct >= 0.075:
        pos["max_tier"] = "7.5% -> تأمين +5%"
        pos["stop"] = max(pos["stop"], entry * 1.05)
    elif mfe_pct >= 0.05:
        pos["max_tier"] = "5% -> تأمين +1%"
        pos["stop"] = max(pos["stop"], entry * 1.01)


def evaluate_stock_state(tk, df, today_str=""):
    """Runs v20 and returns (live_setup_or_pos, closed_today_list, radar_hits)."""
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    dates = df.index
    L = len(df)
    red = c < o
    down = prior_downtrend(c)
    pwl, pml = prev_period_lows(df)
    swing_pivots, order_blocks = build_smc_structures(o, h, l, c)
    start_ts = pd.Timestamp(TEST_START)

    active_swings, active_obs = [], []
    used_wm, state, pos, s = set(), "IDLE", None, {}
    closed_trades = []

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
                    exit_price = min(o[i], pos["stop"]) if hit_stop else c[i]
                    reason = "ضرب وقف الخسارة / جني الأرباح المتحرك" if hit_stop else f"انتهاء مدة الاحتفاظ ({MAX_HOLD} جلسة)"
                    pnl_pct = (exit_price / pos["entry"] - 1.0) * 100.0
                    closed_trades.append(dict(
                        ticker=tk,
                        entry_date=dates[pos["i0"]].strftime("%Y-%m-%d"),
                        exit_date=dates[i].strftime("%Y-%m-%d"),
                        entry=round(pos["entry"], 3),
                        exit_price=round(float(exit_price), 3),
                        pnl_pct=round(float(pnl_pct), 2),
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
                        max_tier="0% (مبدئي -5%)", post_25=False
                    )
                    if l[i] <= stop:
                        pnl_pct = (stop / fill - 1.0) * 100.0
                        closed_trades.append(dict(
                            ticker=tk,
                            entry_date=dates[i].strftime("%Y-%m-%d"),
                            exit_date=dates[i].strftime("%Y-%m-%d"),
                            entry=round(float(fill), 3),
                            exit_price=round(float(stop), 3),
                            pnl_pct=round(float(pnl_pct), 2),
                            reason="ضرب وقف الخسارة في نفس جلسة الدخول",
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
        pnl_pct = (last_close / pos["entry"] - 1.0) * 100
        entry_dt = dates[pos["i0"]].strftime("%Y-%m-%d")
        live_info = dict(
            status="OPEN",
            ticker=tk,
            entry_date=entry_dt,
            is_new_today=(entry_dt == today_str),
            entry=round(pos["entry"], 3),
            target1=round(pos["entry"] * (1 + TARGET1_PCT), 3),
            last_close=round(last_close, 3),
            pnl_pct=round(pnl_pct, 2),
            stop=round(pos["stop"], 3),
            tier=pos["max_tier"],
            kind=pos["kind"],
            bars=(L - 1) - pos["i0"]
        )
    elif state == "WAIT":
        rem_bars = ENTRY_TIMEOUT - ((L - 1) - s["sweep_i"])
        if rem_bars > 0:
            dist_trigger = (s["run_high"] / last_close - 1.0) * 100
            live_info = dict(
                status="WAIT",
                ticker=tk,
                sweep_date=dates[s["sweep_i"]].strftime("%Y-%m-%d"),
                trigger=round(s["run_high"], 3),
                target1=round(s["run_high"] * (1 + TARGET1_PCT), 3),
                last_close=round(last_close, 3),
                dist_pct=round(dist_trigger, 2),
                est_stop=round(s["run_high"] * (1 - STOP_PCT), 3),
                invalidation=round(s["level"], 3),
                kind=s["kind"],
                rem_bars=rem_bars
            )

    radar_hits = []
    last_pwl = float(pwl[-1]) if not np.isnan(pwl[-1]) else np.nan
    last_pml = float(pml[-1]) if not np.isnan(pml[-1]) else np.nan
    filt_ok, met_filters, pending_filters = detailed_filter_check(df)

    if not np.isnan(last_pwl) and last_close >= last_pwl:
        d_w = (last_close - last_pwl) / last_pwl * 100
        if 0.0 <= d_w <= 2.5:
            radar_hits.append(dict(
                ticker=tk, close=round(last_close, 3), ref_type="قاع أسبوعي",
                ref_val=round(last_pwl, 3), dist_pct=round(d_w, 2),
                filter_ok=filt_ok, met=met_filters, pending=pending_filters
            ))
    if not np.isnan(last_pml) and last_close >= last_pml:
        d_m = (last_close - last_pml) / last_pml * 100
        if 0.0 <= d_m <= 2.5:
            radar_hits.append(dict(
                ticker=tk, close=round(last_close, 3), ref_type="قاع شهري",
                ref_val=round(last_pml, 3), dist_pct=round(d_m, 2),
                filter_ok=filt_ok, met=met_filters, pending=pending_filters
            ))

    closed_today = [ct for ct in closed_trades if ct["exit_date"] == today_str]
    return live_info, closed_today, radar_hits


def generate_setup_chart(tk: str, df: pd.DataFrame, item: dict, out_path="best_setup_chart.png"):
    """Generates a clean technical chart for the top candidate stock."""
    sub = df.iloc[-65:].copy()
    closes = sub["close"].values
    ema50 = df["close"].ewm(span=TREND_EMA, adjust=False).mean().iloc[-65:].values
    ema200 = df["close"].ewm(span=LONG_EMA, adjust=False).mean().iloc[-65:].values
    x = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    for idx_i in range(len(sub)):
        o_v, h_v, l_v, c_v = sub["open"].iloc[idx_i], sub["high"].iloc[idx_i], sub["low"].iloc[idx_i], sub["close"].iloc[idx_i]
        color = "#22c55e" if c_v >= o_v else "#ef4444"
        ax.plot([idx_i, idx_i], [l_v, h_v], color=color, linewidth=1.1)
        ax.bar(idx_i, abs(c_v - o_v) or 0.01, bottom=min(o_v, c_v), width=0.6, color=color)

    ax.plot(x, ema50, color="#38bdf8", linewidth=1.3, label="EMA 50")
    ax.plot(x, ema200, color="#f59e0b", linewidth=1.3, label="EMA 200")

    if item["status"] == "OPEN":
        ax.axhline(item["entry"], color="#22c55e", linestyle="--", linewidth=1.3, label=f"Entry: {item['entry']}")
        ax.axhline(item["target1"], color="#a855f7", linestyle=":", linewidth=1.3, label=f"Target 1 (+5%): {item['target1']}")
        ax.axhline(item["stop"], color="#ef4444", linestyle="--", linewidth=1.3, label=f"Stop: {item['stop']}")
    elif item["status"] == "WAIT":
        ax.axhline(item["trigger"], color="#38bdf8", linestyle="--", linewidth=1.3, label=f"Buy Trigger: {item['trigger']}")
        ax.axhline(item["target1"], color="#22c55e", linestyle=":", linewidth=1.3, label=f"Target 1 (+5%): {item['target1']}")
        ax.axhline(item["est_stop"], color="#ef4444", linestyle="--", linewidth=1.3, label=f"Stop (-5%): {item['est_stop']}")

    ax.set_title(f"EGX:{tk} - SMC v20 Setup ({item['status']})", color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#cbd5e1", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.15, color="#94a3b8")
    ax.legend(loc="upper left", facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=8)

    step = max(1, len(sub) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([d.strftime("%m-%d") for d in sub.index[::step]], rotation=0)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


# ============================ MAIN EXECUTION ============================
def main():
    raw_mode = sys.argv[1].lower() if len(sys.argv) > 1 else "auto"
    now_cairo = datetime.now(ZoneInfo(MARKET_TZ))
    today_norm = pd.Timestamp(now_cairo.date())
    today_str = today_norm.strftime("%Y-%m-%d")

    is_manual = raw_mode.startswith("manual")
    report_slot = None

    if raw_mode == "auto":
        if now_cairo.weekday() in (4, 5):
            print(f"[info] Weekend in Cairo ({now_cairo:%A}). Skipping.")
            return
        if now_cairo.hour >= 15:
            mode = "daily_report"
            report_slot = "22:00" if now_cairo.hour >= 21 else "15:00"
        else:
            mode = "intraday"
    elif raw_mode in ("manual_intraday", "intraday"):
        mode = "intraday"
    elif raw_mode == "daily_report":
        mode = "daily_report"
        report_slot = "manual"
    else:
        mode = "intraday"

    print(f"[run] Mode: {mode} (raw={raw_mode}) | Cairo Time: {now_cairo:%Y-%m-%d %H:%M}")
    state = load_state()
    sent_alerts = state.setdefault("sent_alerts", {})
    daily_slots = state.setdefault("last_daily_slots", {})

    tv = get_tv_client()
    all_live = []
    all_closed_today = []
    all_radar = []
    intraday_bars = {}
    stock_dfs = {}

    for tk in DEFAULT_TICKERS:
        comp_df, live_bar = download_stock_with_live(tv, tk, today_norm, fetch_intraday=True)
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
        live_info, closed_today, rad_hits = evaluate_stock_state(tk, eval_df, today_str=today_str)
        if live_info is not None:
            all_live.append(live_info)
        all_closed_today.extend(closed_today)
        all_radar.extend(rad_hits)
        time.sleep(0.12)

    # ---------------- 1) INTRADAY REAL-TIME ALERTS ----------------
    if mode == "intraday":
        alerts_triggered = 0
        closest_watch = []

        for item in all_live:
            tk = item["ticker"]
            bar = intraday_bars.get(tk)
            if not bar:
                continue

            cur_price = bar["close"]
            bar_high = bar["high"]
            bar_low = bar["low"]

            # A) Buy-Stop Trigger Hit
            if item["status"] == "WAIT":
                trig = item["trigger"]
                t1 = item["target1"]
                if bar_high >= trig or cur_price >= trig:
                    alert_key = f"BUY_{tk}_{item['sweep_date']}_{trig}"
                    if alert_key not in sent_alerts:
                        msg = (
                            f"🚀 <b>إشارة شراء لحظية محققة (BUY TRIGGERED)</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📌 <b>السهم:</b> #{tk}\n"
                            f"💰 <b>السعر اللحظي الآن:</b> {cur_price:.3f} ج (أعلى سعر: {bar_high:.3f})\n"
                            f"🎯 <b>سعر الدخول (Trigger):</b> {trig:.3f} ج\n"
                            f"🥇 <b>الهدف الأول (+5%):</b> {t1:.3f} ج\n"
                            f"🛑 <b>وقف الخسارة المبدئي (-5%):</b> {item['est_stop']:.3f} ج\n"
                            f"📊 <b>نوع الإشارة:</b> {item['kind']}\n"
                            f"⏰ <b>التحديث اللحظي:</b> {now_cairo:%H:%M} بتوقيت القاهرة"
                        )
                        if send_telegram_msg(msg):
                            sent_alerts[alert_key] = now_cairo.strftime("%Y-%m-%d %H:%M")
                            alerts_triggered += 1
                else:
                    dist_now = (trig / cur_price - 1.0) * 100.0
                    closest_watch.append((dist_now, tk, cur_price, trig, t1, item["est_stop"]))

            # B) Stop-Loss Hit on Open Position
            elif item["status"] == "OPEN":
                sl = item["stop"]
                if bar_low <= sl or cur_price <= sl:
                    alert_key = f"STOP_{tk}_{item['entry_date']}_{sl}_{today_str}"
                    if alert_key not in sent_alerts:
                        pnl_at_stop = (sl / item["entry"] - 1.0) * 100.0
                        msg = (
                            f"🚨 <b>تنبيه لحظي: وصول السعر لوقف الخسارة (STOP-LOSS)</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📌 <b>السهم:</b> #{tk}\n"
                            f"📉 <b>السعر اللحظي الآن:</b> {cur_price:.3f} ج (أدنى سعر: {bar_low:.3f})\n"
                            f"🛑 <b>سعر وقف الخسارة:</b> {sl:.3f} ج\n"
                            f"💵 <b>سعر الدخول:</b> {item['entry']:.3f} ج (النتيجة عند الوقف: {pnl_at_stop:+.2f}%)\n"
                            f"🛡️ <b>مستوى الحماية:</b> {item['tier']}\n"
                            f"⏰ <b>التحديث اللحظي:</b> {now_cairo:%H:%M} بتوقيت القاهرة"
                        )
                        if send_telegram_msg(msg):
                            sent_alerts[alert_key] = now_cairo.strftime("%Y-%m-%d %H:%M")
                            alerts_triggered += 1

        # If triggered manually, always send a real-time session pulse summary so user gets immediate feedback
        if is_manual:
            opens_now = [x for x in all_live if x["status"] == "OPEN"]
            closest_watch.sort(key=lambda x: x[0])
            pulse_lines = [
                f"⚡ <b>تحديث الفحص اللحظي اليدوي للجلسة (EGX SMC v20)</b>",
                f"⏰ <b>الوقت:</b> {now_cairo:%Y-%m-%d %H:%M} بتوقيت القاهرة",
                f"📡 <b>الأسهم المسحوبة لحظياً:</b> {len(intraday_bars)} سهم | <b>تنبيهات جديدة:</b> {alerts_triggered}",
                "━━━━━━━━━━━━━━━━━━━━",
                f"🟢 <b>الصفقات المفتوحة حالياً ({len(opens_now)}):</b>"
            ]
            if not opens_now:
                pulse_lines.append("• لا توجد صفقات مفتوحة حالياً.")
            else:
                for o in opens_now:
                    live_p = intraday_bars.get(o["ticker"], {}).get("close", o["last_close"])
                    live_pnl = (live_p / o["entry"] - 1.0) * 100.0
                    pulse_lines.append(
                        f"• <b>#{o['ticker']}</b> | السعر اللحظي: <b>{live_p:.3f}</b> ({live_pnl:+.2f}%) | الوقف: {o['stop']:.3f}"
                    )

            pulse_lines.append("")
            pulse_lines.append(f"🎯 <b>أقرب أوامر الشراء الجاهزة للتفعيل ({len(closest_watch)}):</b>")
            if not closest_watch:
                pulse_lines.append("• لا توجد أوامر Buy-Stop معلقة حالياً.")
            else:
                for dist_now, tk, cur_p, trig, t1, sl in closest_watch[:6]:
                    pulse_lines.append(
                        f"• <b>#{tk}</b> | لحظي: {cur_p:.3f} | دخول: <b>{trig:.3f}</b> (باقي {dist_now:+.2f}%)
"
                        f"  └ هدف أول: {t1:.3f} | وقف: {sl:.3f}"
                    )

            send_telegram_msg("\n".join(pulse_lines))

        print(f"[intraday] Checked {len(intraday_bars)} live bars. Sent {alerts_triggered} new alerts.")
        save_state(state)

    # ---------------- 2) DAILY END-OF-DAY REPORT (15:00 & 22:00) ----------------
    elif mode == "daily_report":
        # Scheduled slots ('15:00' and '22:00') don't block each other, and manual runs ('manual') never block scheduled slots
        if report_slot in ("15:00", "22:00"):
            slot_key = f"{today_str}_{report_slot}"
            if daily_slots.get(slot_key):
                print(f"[daily] Scheduled report for {slot_key} already sent.")
                return

        egx30 = get_egx30_status(tv, today_norm)
        opens = [x for x in all_live if x["status"] == "OPEN"]
        new_opens = [x for x in opens if x.get("is_new_today")]
        waits = [x for x in all_live if x["status"] == "WAIT"]
        ready_radar = sorted(all_radar, key=lambda x: (not x["filter_ok"], x["dist_pct"]))

        slot_label = "تقرير إغلاق الجلسة (15:00)" if report_slot == "15:00" else (
            "التقرير المسائي المحدث (22:00)" if report_slot == "22:00" else "تقرير يدوي فوري"
        )

        lines = [
            f"📊 <b>التقرير اليومي لرادار البورصة المصرية (EGX SMC v20)</b>",
            f"🗓️ <b>التاريخ:</b> {today_str} | 🕒 <b>{slot_label}</b>",
            "━━━━━━━━━━━━━━━━━━━━"
        ]

        if egx30:
            lines.append(
                f"{egx30['emoji']} <b>حالة المؤشر العام (EGX30):</b> {egx30['value']:,.2f} نقطة "
                f"(<b>{egx30['chg_pct']:+.2f}%</b> | {egx30['chg_pts']:+.2f} نقطة - {egx30['status_ar']})"
            )
            lines.append("━━━━━━━━━━━━━━━━━━━━")

        # 1. Newly triggered positions today
        lines.append(f"🆕 <b>1) أسهم حققت الاستراتيجية واتفتحت كصفقة اليوم ({len(new_opens)}):</b>")
        if not new_opens:
            lines.append("• لم يتم تفعيل صفقات جديدة خلال جلسة اليوم.")
        else:
            for o in new_opens:
                lines.append(
                    f"• <b>#{o['ticker']}</b> | دخول: <b>{o['entry']}</b> | إغلاق: {o['last_close']} (<b>{o['pnl_pct']:+.2f}%</b>)
"
                    f"  └ هدف أول (+5%): {o['target1']} | وقف مبدئي: {o['stop']}"
                )

        # 2. All Open Positions
        lines.append("")
        lines.append(f"🟢 <b>2) إجمالي الصفقات المفتوحة والمستمرة ({len(opens)}):</b>")
        if not opens:
            lines.append("• لا توجد صفقات مفتوحة حالياً.")
        else:
            for o in opens:
                lines.append(
                    f"• <b>#{o['ticker']}</b> | دخول: {o['entry']} | إغلاق: {o['last_close']} (<b>{o['pnl_pct']:+.2f}%</b>)
"
                    f"  └ الوقف المحدث: <b>{o['stop']}</b> ({o['tier']}) | جلسات: {o['bars']}"
                )

        # 3. Closed Positions Today
        lines.append("")
        lines.append(f"🔒 <b>3) الصفقات المغلقة خلال الجلسة ({len(all_closed_today)}):</b>")
        if not all_closed_today:
            lines.append("• لا توجد صفقات أُغلقت خلال جلسة اليوم.")
        else:
            for ct in all_closed_today:
                res_lbl = "ربح ✅" if ct["pnl_pct"] >= 0 else "خسارة ❌"
                lines.append(
                    f"• <b>#{ct['ticker']}</b> | دخول: {ct['entry']} -> إغلاق: <b>{ct['exit_price']}</b> (<b>{ct['pnl_pct']:+.2f}%</b> - {res_lbl})
"
                    f"  └ سبب الإغلاق: {ct['reason']}"
                )

        # 4. Pending Buy-Stop Setups
        lines.append("")
        lines.append(f"⏳ <b>4) أوامر Buy-Stop جاهزة للتفعيل ({len(waits)}):</b>")
        if not waits:
            lines.append("• لا توجد أوامر Buy-Stop معلقة.")
        else:
            for w in waits:
                lines.append(
                    f"• <b>#{w['ticker']}</b> | تفعيل: <b>{w['trigger']}</b> (+{w['dist_pct']}%)
"
                    f"  └ هدف أول: {w['target1']} | وقف (-5%): {w['est_stop']} | صلاحية: {w['rem_bars']} أيام"
                )

        # 5. Watchlist Radar with Met & Pending Filters
        lines.append("")
        lines.append(f"👀 <b>5) أسهم تقترب من الشروط (حالة الفلاتر المحققة والمنتظرة):</b>")
        if not ready_radar:
            lines.append("• لا توجد أسهم قريبة من القيعان الأسبوعية أو الشهرية حالياً.")
        else:
            for r in ready_radar[:7]:
                met_str = "، ".join(r["met"]) if r["met"] else "لا يوجد"
                pend_str = "، ".join(r["pending"]) if r["pending"] else "مكتملة"
                lines.append(
                    f"• <b>#{r['ticker']}</b> ({r['close']} ج | قرب {r['ref_type']} {r['ref_val']} +{r['dist_pct']}%)
"
                    f"  ✅ محقق: {met_str}
"
                    f"  ⏳ منتظر: {pend_str}"
                )

        report_text = "\n".join(lines)
        if send_telegram_msg(report_text):
            if report_slot in ("15:00", "22:00"):
                daily_slots[f"{today_str}_{report_slot}"] = now_cairo.strftime("%Y-%m-%d %H:%M")
            save_state(state)
            print(f"[daily] Daily report ({report_slot}) for {today_str} sent successfully.")

            # Send chart image of best candidate setup
            best_candidate = waits[0] if waits else (opens[0] if opens else None)
            if best_candidate and best_candidate["ticker"] in stock_dfs:
                chart_tk = best_candidate["ticker"]
                chart_path = generate_setup_chart(chart_tk, stock_dfs[chart_tk], best_candidate)
                cap = f"📈 <b>شارت أفضل فرصة مطابقة لنموذج EGX SMC v20:</b> #{chart_tk}"
                send_telegram_photo(chart_path, caption=cap)


if __name__ == "__main__":
    main()
