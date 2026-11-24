"""
EGX SMC / Liquidity Reversal Strategy (v20) - Telegram Live Alerts & Daily Report Worker
Features:
  1) Intraday Live Alerts (Every 15 mins during EGX session 10:00 - 14:45 Cairo time):
     - Fetches real-time intraday price (5m / 1m candles + daily bar) from TradingView
     - Instant BUY-STOP Trigger notification with Entry Price, Target 1 (+5%), and Initial Stop-Loss (-5%)
     - Instant STOP-LOSS Hit notification when live price <= Updated Stop Loss
  2) Manual Run Support (workflow_dispatch -> manual_intraday):
     - Always scans live intraday prices (never relies on stale yesterday close)
     - Sends an immediate live session summary on Telegram even if no new trigger fired
     - Never marks the scheduled 16:00 daily report as sent, so the 4:00 PM report always sends on time
  3) End-of-Day Daily Report (Once at 16:00 / 4:00 PM Cairo time):
     - EGX30 General Index status (close & % gain/loss)
     - Open Positions (newly triggered today & ongoing, with trailing stops: 5%->+1%, 7.5%->+5%, 10%->+7.5%, 15%->+10%, 25%+->trail)
     - Closed Positions today (exit price, exit reason, and % profit/loss)
     - Ready Buy-Stop Setups for next session (with Entry & Target 1)
     - Approaching Watchlist Radar with exact breakdown of passed vs. pending strategy filters
     - Chart image of the top matching stock setup attached to the daily report
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
CHART_FILE = "best_setup_chart.png"
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


# ============================ TELEGRAM HELPER ============================
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
            print(f"[warn] Telegram send attempt {attempt + 1} failed: {e}")
            time.sleep(1.5)
    return False


def send_telegram_photo(photo_path: str, caption: str = "") -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or not os.path.exists(photo_path):
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    boundary = "----WebKitFormBoundaryEGXBot7MA4YWxkTrZu0gW"

    with open(photo_path, "rb") as f:
        img_bytes = f.read()

    parts = []
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{TELEGRAM_CHAT_ID}\r\n".encode("utf-8"))
    if caption:
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption[:1000]}\r\n".encode("utf-8"))
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"parse_mode\"\r\n\r\nHTML\r\n".encode("utf-8"))
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"{os.path.basename(photo_path)}\"\r\nContent-Type: image/png\r\n\r\n".encode("utf-8")
        + img_bytes
        + f"\r\n--{boundary}--\r\n".encode("utf-8")
    )
    body = b"".join(parts)

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status == 200:
                    return True
        except Exception as e:
            print(f"[warn] Telegram photo attempt {attempt + 1} failed: {e}")
            time.sleep(1.5)
    return False


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                data.setdefault("sent_alerts", {})
                data.setdefault("last_daily_report_date", "")
                return data
        except Exception:
            pass
    return {"sent_alerts": {}, "last_daily_report_date": ""}


def save_state(state: dict):
    alerts = state.get("sent_alerts", {})
    if len(alerts) > 400:
        keys = list(alerts.keys())[-400:]
        state["sent_alerts"] = {k: alerts[k] for k in keys}
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ============================ DATA & REALTIME ENGINE ============================
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


def fetch_realtime_intraday_bar(tv, sym, today_norm):
    """Fetches the freshest intraday price using 5m (or 1m fallback) bars from TradingView."""
    for interval in (Interval.in_5_minute, Interval.in_1_minute):
        try:
            intra = tv.get_hist(symbol=sym, exchange=EXCHANGE, interval=interval, n_bars=80)
            if intra is not None and not intra.empty:
                intra = intra.copy()
                intra.columns = [c.lower() for c in intra.columns]
                intra.index = pd.DatetimeIndex(intra.index).tz_localize(None)
                today_intra = intra[intra.index.normalize() == today_norm]
                if not today_intra.empty:
                    return {
                        "date": today_norm.strftime("%Y-%m-%d"),
                        "open": float(today_intra["open"].iloc[0]),
                        "high": float(today_intra["high"].max()),
                        "low": float(today_intra["low"].min()),
                        "close": float(today_intra["close"].iloc[-1]),
                        "volume": float(today_intra["volume"].sum()) if "volume" in today_intra.columns else 0.0,
                        "source": "5m/1m_live",
                        "bar_time": today_intra.index[-1].strftime("%H:%M")
                    }
        except Exception:
            pass
    return None


def download_stock_with_live(tv, sym, today_norm, fetch_intra=True):
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
                if df.index[-1] == today_norm:
                    last_row = df.iloc[-1]
                    live_bar = {
                        "date": today_norm.strftime("%Y-%m-%d"),
                        "open": float(last_row["open"]),
                        "high": float(last_row["high"]),
                        "low": float(last_row["low"]),
                        "close": float(last_row["close"]),
                        "volume": float(last_row["volume"]),
                        "source": "daily_live",
                        "bar_time": ""
                    }
                    completed_df = df.iloc[:-1]
                else:
                    completed_df = df

                if fetch_intra:
                    intra_bar = fetch_realtime_intraday_bar(tv, sym, today_norm)
                    if intra_bar is not None:
                        if live_bar is not None:
                            intra_bar["open"] = live_bar["open"]
                            intra_bar["high"] = max(live_bar["high"], intra_bar["high"])
                            intra_bar["low"] = min(live_bar["low"], intra_bar["low"])
                            intra_bar["volume"] = max(live_bar["volume"], intra_bar["volume"])
                        live_bar = intra_bar

                completed_df = completed_df[completed_df.index >= pd.Timestamp(DATA_START)]
                return completed_df, live_bar
        except Exception:
            time.sleep(0.6)
    return None, None


def get_egx30_status(tv, today_norm):
    """Fetches EGX30 general index status."""
    try:
        raw = tv.get_hist(symbol=INDEX_SYMBOL, exchange=EXCHANGE, interval=Interval.in_daily, n_bars=10)
        if raw is not None and len(raw) >= 2:
            df = raw.copy()
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
            df = clean_df(df)
            last_c = float(df["close"].iloc[-1])
            prev_c = float(df["close"].iloc[-2])
            chg_pts = last_c - prev_c
            chg_pct = (chg_pts / prev_c) * 100.0 if prev_c > 0 else 0.0
            return {
                "close": last_c,
                "prev_close": prev_c,
                "chg_pts": chg_pts,
                "chg_pct": chg_pct,
                "date": df.index[-1].strftime("%Y-%m-%d")
            }
    except Exception as e:
        print(f"[warn] Could not fetch EGX30: {e}")
    return None


# ============================ STRATEGY ENGINE (v20) ============================
def prior_downtrend(close):
    s = pd.Series(close)
    ema = s.ewm(span=TREND_EMA, adjust=False).mean()
    down = (s < ema) & (ema < ema.shift(TREND_SLOPE))
    if USE_EMA200_FILTER:
        ema200 = s.ewm(span=LONG_EMA, adjust=False).mean()
        down = down & (s > ema200)
    return down.shift(1, fill_value=False).values.astype(bool)


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


def analyze_detailed_filters(df, pwl, pml, active_swings):
    """Evaluates each filter on the latest bar to list passed vs pending conditions."""
    c_series = df["close"]
    ema50 = c_series.ewm(span=TREND_EMA, adjust=False).mean()
    ema200 = c_series.ewm(span=LONG_EMA, adjust=False).mean()

    last_c = float(df["close"].iloc[-1])
    last_l = float(df["low"].iloc[-1])
    last_h = float(df["high"].iloc[-1])
    last_e50 = float(ema50.iloc[-1])
    prev_e50_slope = float(ema50.shift(TREND_SLOPE).iloc[-1])
    last_e200 = float(ema200.iloc[-1])

    passed = []
    pending = []

    # 1. Above EMA200
    if last_c > last_e200:
        passed.append(f"فوق EMA200 ({last_e200:.2f})")
    else:
        pending.append(f"الإغلاق فوق EMA200 ({last_e200:.2f})")

    # 2. Pullback below EMA50 & falling EMA50 slope
    if last_c < last_e50 and last_e50 < prev_e50_slope:
        passed.append(f"تصحيح هابط تحت EMA50 ({last_e50:.2f})")
    else:
        pending.append(f"تصحيح تحت EMA50 ({last_e50:.2f}) بميل هابط")

    # 3. Liquidity level proximity / sweep
    candidates = []
    if not np.isnan(pwl[-1]) and pwl[-1] > 0:
        candidates.append(("قاع الأسبوع السابق", float(pwl[-1])))
    if not np.isnan(pml[-1]) and pml[-1] > 0:
        candidates.append(("قاع الشهر السابق", float(pml[-1])))
    for sw in active_swings:
        candidates.append(("قاع سوينج", float(sw["price"])))

    nearest_lbl, nearest_val, nearest_dist = None, None, 999.0
    swept = False
    for lbl, val in candidates:
        if last_l < val and last_c > val:
            swept = True
            nearest_lbl, nearest_val, nearest_dist = lbl, val, 0.0
            break
        dist = ((last_c - val) / val) * 100.0
        if -1.0 <= dist < nearest_dist:
            nearest_lbl, nearest_val, nearest_dist = lbl, val, dist

    if swept:
        passed.append(f"كسر وهمي وإغلاق أعلى {nearest_lbl} ({nearest_val:.2f})")
    elif nearest_lbl is not None:
        pending.append(f"سحب سيولة {nearest_lbl} ({nearest_val:.2f} | يبعد {nearest_dist:+.2f}%)")
    else:
        pending.append("الاقتراب من قاع سيولة (أسبوعي/شهري/سوينج)")

    # 4. Candle close strength >= 50% of range + Volume check
    rng = last_h - last_l
    close_pos = ((last_c - last_l) / rng) if rng > 0 else 0.0
    if close_pos >= SWEEP_CLOSE_POS:
        passed.append(f"إغلاق شمعة قوي ({close_pos*100:.0f}% من المدى)")
    else:
        pending.append(f"إغلاق الشمعة في النصف العلوي (حالياً {close_pos*100:.0f}%)")

    vol_ok = True
    if "volume" in df.columns and len(df) >= 20:
        avg_vol = float(df["volume"].iloc[-20:-1].mean())
        last_vol = float(df["volume"].iloc[-1])
        if avg_vol > 0 and last_vol > 0:
            if last_vol >= avg_vol:
                passed.append("سيولة/فوليوم أعلى من المتوسط")
            else:
                pending.append("تأكيد زخم الفوليوم")

    return passed, pending, nearest_lbl, nearest_val, nearest_dist


def evaluate_stock_state(tk, df):
    """Runs v20 on df and returns (live_info, closed_today_info, radar_hits)."""
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
    closed_today = None

    for i in range(L):
        is_last_bar = (i == L - 1)
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
                hit_time = (not pos["post_25"]) and (i - pos["i0"] >= MAX_HOLD)
                if hit_stop or hit_time:
                    if is_last_bar:
                        exit_px = pos["stop"] if hit_stop else c[i]
                        reason = (
                            f"ضرب وقف الخسارة / حماية الأرباح ({pos['max_tier']})"
                            if hit_stop else f"انتهاء المدة الزمنية ({MAX_HOLD} جلسة)"
                        )
                        pnl_pct = (exit_px / pos["entry"] - 1.0) * 100.0
                        closed_today = dict(
                            ticker=tk,
                            entry_date=dates[pos["i0"]].strftime("%Y-%m-%d"),
                            entry=round(pos["entry"], 3),
                            exit_price=round(float(exit_px), 3),
                            pnl_pct=round(float(pnl_pct), 2),
                            reason=reason,
                            kind=pos["kind"]
                        )
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
                        if is_last_bar:
                            pnl_pct = (stop / fill - 1.0) * 100.0
                            closed_today = dict(
                                ticker=tk,
                                entry_date=dates[i].strftime("%Y-%m-%d"),
                                entry=round(float(fill), 3),
                                exit_price=round(float(stop), 3),
                                pnl_pct=round(float(pnl_pct), 2),
                                reason="ضرب وقف الخسارة في نفس جلسة الدخول",
                                kind=s["kind"]
                            )
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
        target1 = pos["entry"] * (1.0 + TARGET1_PCT)
        live_info = dict(
            status="OPEN",
            ticker=tk,
            entry_date=dates[pos["i0"]].strftime("%Y-%m-%d"),
            opened_today=(pos["i0"] == L - 1),
            entry=round(pos["entry"], 3),
            target1=round(target1, 3),
            last_close=round(last_close, 3),
            pnl_pct=round(pnl_pct, 2),
            stop=round(pos["stop"], 3),
            tier="POST-25% TRAIL" if pos["post_25"] else f"Tier {pos['max_tier']}",
            kind=pos["kind"],
            bars=(L - 1) - pos["i0"]
        )
    elif state == "WAIT":
        rem_bars = ENTRY_TIMEOUT - ((L - 1) - s["sweep_i"])
        if rem_bars > 0:
            dist_trigger = (s["run_high"] / last_close - 1.0) * 100.0
            target1 = s["run_high"] * (1.0 + TARGET1_PCT)
            live_info = dict(
                status="WAIT",
                ticker=tk,
                sweep_date=dates[s["sweep_i"]].strftime("%Y-%m-%d"),
                trigger=round(s["run_high"], 3),
                target1=round(target1, 3),
                last_close=round(last_close, 3),
                dist_pct=round(dist_trigger, 2),
                est_stop=round(s["run_high"] * (1 - STOP_PCT), 3),
                invalidation=round(s["level"], 3),
                kind=s["kind"],
                rem_bars=rem_bars
            )

    radar_hits = []
    if live_info is None:
        passed_f, pending_f, near_lbl, near_val, near_dist = analyze_detailed_filters(df, pwl, pml, active_swings)
        if len(passed_f) >= 2 and near_lbl is not None and -0.5 <= near_dist <= 3.5:
            radar_hits.append(dict(
                ticker=tk,
                close=round(last_close, 3),
                ref_type=near_lbl,
                ref_val=round(near_val, 3),
                dist_pct=round(near_dist, 2),
                passed=passed_f,
                pending=pending_f,
                score=len(passed_f) * 10 - max(0.0, near_dist)
            ))

    return live_info, closed_today, radar_hits


def generate_best_stock_chart(tk, df, setup_info, out_path=CHART_FILE):
    """Generates a clean price + EMA50/200 + trigger/stop/target chart for the top candidate."""
    try:
        sub = df.iloc[-75:].copy()
        ema50 = df["close"].ewm(span=TREND_EMA, adjust=False).mean().iloc[-75:]
        ema200 = df["close"].ewm(span=LONG_EMA, adjust=False).mean().iloc[-75:]

        fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
        x = np.arange(len(sub))
        for idx in range(len(sub)):
            o_p, h_p, l_p, c_p = sub["open"].iloc[idx], sub["high"].iloc[idx], sub["low"].iloc[idx], sub["close"].iloc[idx]
            color = "#16a34a" if c_p >= o_p else "#dc2626"
            ax.plot([idx, idx], [l_p, h_p], color=color, linewidth=1.1)
            ax.plot([idx, idx], [o_p, c_p], color=color, linewidth=4.2)

        ax.plot(x, ema50.values, label="EMA 50", color="#2563eb", linewidth=1.3)
        ax.plot(x, ema200.values, label="EMA 200", color="#f59e0b", linewidth=1.3)

        if setup_info.get("status") == "OPEN":
            ax.axhline(setup_info["entry"], color="#16a34a", linestyle="--", linewidth=1.2, label=f"Entry: {setup_info['entry']}")
            ax.axhline(setup_info["target1"], color="#0d9488", linestyle=":", linewidth=1.2, label=f"Target 1 (+5%): {setup_info['target1']}")
            ax.axhline(setup_info["stop"], color="#dc2626", linestyle="--", linewidth=1.2, label=f"Stop: {setup_info['stop']}")
        elif setup_info.get("status") == "WAIT":
            ax.axhline(setup_info["trigger"], color="#16a34a", linestyle="--", linewidth=1.2, label=f"Buy-Stop: {setup_info['trigger']}")
            ax.axhline(setup_info["target1"], color="#0d9488", linestyle=":", linewidth=1.2, label=f"Target 1 (+5%): {setup_info['target1']}")
            ax.axhline(setup_info["est_stop"], color="#dc2626", linestyle="--", linewidth=1.2, label=f"Stop (-5%): {setup_info['est_stop']}")

        step = max(1, len(sub) // 8)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([d.strftime("%m-%d") for d in sub.index[::step]], rotation=25, fontsize=8)
        ax.set_title(f"EGX: {tk} - SMC v20 Setup ({setup_info.get('status', 'RADAR')})", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return out_path
    except Exception as e:
        print(f"[warn] Chart generation failed: {e}")
        return None


# ============================ MAIN EXECUTION ============================
def main():
    raw_mode = sys.argv[1].lower() if len(sys.argv) > 1 else "auto"
    now_cairo = datetime.now(ZoneInfo(MARKET_TZ))
    today_norm = pd.Timestamp(now_cairo.date())
    today_str = today_norm.strftime("%Y-%m-%d")

    is_manual = (raw_mode == "manual_intraday")
    mode = raw_mode

    if mode == "auto":
        # EGX trades Sun (6) to Thu (3); Fri=4, Sat=5
        if now_cairo.weekday() in (4, 5):
            print(f"[info] Weekend in Cairo ({now_cairo:%A}). Skipping.")
            return
        if now_cairo.hour >= 16:
            mode = "daily_report"
        elif 10 <= now_cairo.hour < 15:
            mode = "intraday"
        else:
            print(f"[info] Outside EGX session & report window ({now_cairo:%H:%M} Cairo). Skipping.")
            return

    print(f"[run] Mode: {mode} (manual={is_manual}) | Cairo Time: {now_cairo:%Y-%m-%d %H:%M}")
    state = load_state()
    sent_alerts = state.setdefault("sent_alerts", {})

    tv = get_tv_client()
    all_live = []
    all_closed_today = []
    all_radar = []
    intraday_bars = {}
    dfs_by_ticker = {}

    is_intraday_scan = mode in ("intraday", "manual_intraday")

    for tk in DEFAULT_TICKERS:
        comp_df, live_bar = download_stock_with_live(tv, tk, today_norm, fetch_intra=True)
        if comp_df is None or len(comp_df) < 250:
            continue

        if not is_intraday_scan and live_bar is not None:
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

        dfs_by_ticker[tk] = eval_df
        live_info, closed_info, rad_hits = evaluate_stock_state(tk, eval_df)
        if live_info is not None:
            all_live.append(live_info)
        if closed_info is not None:
            all_closed_today.append(closed_info)
        all_radar.extend(rad_hits)
        time.sleep(0.12)

    # ---------------- 1) INTRADAY & MANUAL LIVE SCAN ----------------
    if is_intraday_scan:
        alerts_triggered = 0
        triggered_msgs = []

        for item in all_live:
            tk = item["ticker"]
            bar = intraday_bars.get(tk)
            if not bar:
                continue

            cur_price = bar["close"]
            bar_high = bar["high"]
            bar_low = bar["low"]
            src_tag = f" ({bar['bar_time']})" if bar.get("bar_time") else ""

            # A) Buy-Stop Trigger Hit
            if item["status"] == "WAIT":
                trig = item["trigger"]
                if bar_high >= trig or cur_price >= trig:
                    alert_key = f"BUY_{tk}_{item['sweep_date']}_{trig}"
                    if alert_key not in sent_alerts:
                        entry_est = max(bar["open"], trig)
                        target1 = entry_est * (1.0 + TARGET1_PCT)
                        stop_init = entry_est * (1.0 - STOP_PCT)
                        msg = (
                            f"🚀 <b>إشارة شراء لحظية محققة (BUY-STOP TRIGGERED)</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📌 <b>السهم:</b> #{tk}\n"
                            f"💰 <b>السعر اللحظي الآن:</b> {cur_price:.3f} ج{src_tag} (أعلى سعر اليوم: {bar_high:.3f})\n"
                            f"🎯 <b>سعر الدخول (Trigger):</b> {trig:.3f} ج\n"
                            f"🏆 <b>الهدف الأول (+5%):</b> {target1:.3f} ج\n"
                            f"🛑 <b>وقف الخسارة المبدئي (-5%):</b> {stop_init:.3f} ج\n"
                            f"📊 <b>نوع الإشارة:</b> {item['kind']}\n"
                            f"⏰ <b>التوقيت:</b> {now_cairo:%H:%M} بتوقيت القاهرة"
                        )
                        if send_telegram_msg(msg):
                            sent_alerts[alert_key] = now_cairo.strftime("%Y-%m-%d %H:%M")
                            alerts_triggered += 1
                            triggered_msgs.append(f"🚀 شراء #{tk} @ {trig:.3f}")

            # B) Stop-Loss Hit on Open Position
            elif item["status"] == "OPEN":
                sl = item["stop"]
                if bar_low <= sl or cur_price <= sl:
                    alert_key = f"STOP_{tk}_{item['entry_date']}_{sl}_{today_str}"
                    if alert_key not in sent_alerts:
                        pnl_at_stop = (sl / item["entry"] - 1.0) * 100.0
                        msg = (
                            f"🚨 <b>تنبيه ضرب وقف الخسارة اللحظي (STOP-LOSS HIT)</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📌 <b>السهم:</b> #{tk}\n"
                            f"📉 <b>السعر اللحظي الآن:</b> {cur_price:.3f} ج{src_tag} (أدنى سعر اليوم: {bar_low:.3f})\n"
                            f"🛑 <b>سعر وقف الخسارة:</b> {sl:.3f} ج\n"
                            f"💵 <b>سعر الدخول:</b> {item['entry']:.3f} ج (النتيجة: {pnl_at_stop:+.2f}%)\n"
                            f"🛡️ <b>مستوى الحماية:</b> {item['tier']}\n"
                            f"⏰ <b>التوقيت:</b> {now_cairo:%H:%M} بتوقيت القاهرة"
                        )
                        if send_telegram_msg(msg):
                            sent_alerts[alert_key] = now_cairo.strftime("%Y-%m-%d %H:%M")
                            alerts_triggered += 1
                            triggered_msgs.append(f"🚨 وقف خسارة #{tk} @ {sl:.3f}")

        # If manual run: always send a live intraday status snapshot using real-time prices
        if is_manual:
            waits = [x for x in all_live if x["status"] == "WAIT"]
            opens = [x for x in all_live if x["status"] == "OPEN"]
            lines = [
                f"⚡ <b>فحص لحظي يدوي للجلسة (EGX SMC v20)</b>",
                f"🕒 <b>الوقت:</b> {today_str} | {now_cairo:%H:%M} بتوقيت القاهرة",
                f"📡 <b>الأسهم المسحوبة لحظياً:</b> {len(intraday_bars)} من {len(DEFAULT_TICKERS)}",
                "━━━━━━━━━━━━━━━━━━━━",
                f"⏳ <b>أوامر الشراء المعلقة ({len(waits)}):</b>"
            ]
            if not waits:
                lines.append("• لا توجد أوامر معلقة حالياً.")
            else:
                for w in waits:
                    tk = w["ticker"]
                    b = intraday_bars.get(tk)
                    cur_p = b["close"] if b else w["last_close"]
                    dist_now = ((w["trigger"] / cur_p) - 1.0) * 100.0 if cur_p > 0 else w["dist_pct"]
                    lines.append(
                        f"• <b>#{tk}</b> | اللحظي: <b>{cur_p:.3f}</b> | تفعيل: <b>{w['trigger']:.3f}</b> ({dist_now:+.2f}%) | هدف1: {w['target1']:.3f}"
                    )

            lines.append("")
            lines.append(f"🟢 <b>الصفقات المفتوحة ({len(opens)}):</b>")
            if not opens:
                lines.append("• لا توجد صفقات مفتوحة حالياً.")
            else:
                for o in opens:
                    tk = o["ticker"]
                    b = intraday_bars.get(tk)
                    cur_p = b["close"] if b else o["last_close"]
                    live_pnl = ((cur_p / o["entry"]) - 1.0) * 100.0
                    lines.append(
                        f"• <b>#{tk}</b> | دخول: {o['entry']:.3f} | اللحظي: <b>{cur_p:.3f}</b> (<b>{live_pnl:+.2f}%</b>) | وقف: {o['stop']:.3f}"
                    )

            if triggered_msgs:
                lines.append("")
                lines.append("🔔 <b>تنبيهات تحققت في هذا الفحص:</b> " + " ، ".join(triggered_msgs))

            send_telegram_msg("\n".join(lines))

        print(f"[intraday] Checked {len(intraday_bars)} live bars. Sent {alerts_triggered} new alerts (manual={is_manual}).")
        save_state(state)

    # ---------------- 2) DAILY END-OF-DAY REPORT (16:00 CAIRO) ----------------
    elif mode == "daily_report":
        # Only prevent duplicate automatic report on the same day; manual daily_report always sends
        if raw_mode == "auto" and state.get("last_daily_report_date") == today_str:
            print(f"[daily] Scheduled report for {today_str} already sent.")
            return

        egx30 = get_egx30_status(tv, today_norm)
        opens = [x for x in all_live if x["status"] == "OPEN"]
        new_opens = [x for x in opens if x.get("opened_today")]
        waits = [x for x in all_live if x["status"] == "WAIT"]
        near_radar = sorted(all_radar, key=lambda x: x["score"], reverse=True)

        lines = [
            f"📊 <b>التقرير اليومي الشامل للبورصة المصرية (EGX SMC v20)</b>",
            f"📅 <b>التاريخ:</b> {today_str} | ⏰ <b>التوقيت:</b> {now_cairo:%H:%M} القاهرة",
            "━━━━━━━━━━━━━━━━━━━━"
        ]

        # Section 0: EGX30 Index Status
        if egx30:
            idx_icon = "🟢" if egx30["chg_pts"] >= 0 else "🔴"
            lines.append(
                f"{idx_icon} <b>حالة المؤشر العام (EGX30):</b> {egx30['close']:,.2f} نقطة "
                f"(<b>{egx30['chg_pct']:+.2f}%</b> | {egx30['chg_pts']:+.2f} نقطة)"
            )
        else:
            lines.append("📈 <b>حالة المؤشر العام (EGX30):</b> غير متاح حالياً من المصدر")

        lines.append("━━━━━━━━━━━━━━━━━━━━")

        # Section 1: New & Open Trades
        lines.append(f"🟢 <b>1) أسهم حققت الاستراتيجية ومفتوحة كصفقات ({len(opens)} | جديدة اليوم: {len(new_opens)}):</b>")
        if not opens:
            lines.append("• لا توجد صفقات مفتوحة حالياً.")
        else:
            for o in opens:
                badge = "🆕 [دخول اليوم] " if o.get("opened_today") else ""
                lines.append(
                    f"• {badge}<b>#{o['ticker']}</b> | دخول: {o['entry']:.3f} | إغلاق: {o['last_close']:.3f} (<b>{o['pnl_pct']:+.2f}%</b>)\n"
                    f"  └ الهدف الأول: {o['target1']:.3f} | الوقف المحدث: <b>{o['stop']:.3f}</b> ({o['tier']})"
                )

        lines.append("")

        # Section 2: Closed Trades Today
        lines.append(f"🏁 <b>2) الصفقات المغلقة خلال جلسة اليوم ({len(all_closed_today)}):</b>")
        if not all_closed_today:
            lines.append("• لا توجد صفقات أُغلقت في جلسة اليوم.")
        else:
            for cl in all_closed_today:
                res_icon = "✅ ربح" if cl["pnl_pct"] >= 0 else "❌ خسارة"
                lines.append(
                    f"• <b>#{cl['ticker']}</b> | دخول: {cl['entry']:.3f} ➔ إغلاق الصفقة: <b>{cl['exit_price']:.3f}</b> ({res_icon}: <b>{cl['pnl_pct']:+.2f}%</b>)\n"
                    f"  └ السبب: {cl['reason']}"
                )

        lines.append("")

        # Section 3: Ready Buy-Stop Setups
        lines.append(f"🎯 <b>3) أسهم جاهزة للتنفيذ (أوامر Buy-Stop معلقة: {len(waits)}):</b>")
        if not waits:
            lines.append("• لا توجد أوامر Buy-Stop معلقة للجلسة القادمة.")
        else:
            for w in waits:
                lines.append(
                    f"• <b>#{w['ticker']}</b> | تفعيل: <b>{w['trigger']:.3f}</b> (+{w['dist_pct']:.2f}%) | هدف1: <b>{w['target1']:.3f}</b>\n"
                    f"  └ وقف (-5%): {w['est_stop']:.3f} | صلاحية: {w['rem_bars']} جلسات | {w['kind']}"
                )

        lines.append("")

        # Section 4: Approaching Stocks with Passed & Pending Filters
        lines.append(f"👀 <b>4) أسهم تقترب من تحقيق الشروط وتفصيل الفلاتر ({len(near_radar)}):</b>")
        if not near_radar:
            lines.append("• لا توجد أسهم قريبة من مناطق السيولة حالياً.")
        else:
            for r in near_radar[:6]:
                passed_txt = " + ".join(r["passed"]) if r["passed"] else "لا يوجد"
                pending_txt = " + ".join(r["pending"]) if r["pending"] else "مكتملة"
                lines.append(
                    f"• <b>#{r['ticker']}</b> (إغلاق: {r['close']:.3f} | قرب {r['ref_type']} {r['ref_val']:.2f})\n"
                    f"  ✅ <b>محقق:</b> {passed_txt}\n"
                    f"  ⏳ <b>منتظر:</b> {pending_txt}"
                )

        report_text = "\n".join(lines)
        if send_telegram_msg(report_text):
            if raw_mode == "auto":
                state["last_daily_report_date"] = today_str
            save_state(state)
            print(f"[daily] Daily report for {today_str} sent successfully.")

            # Attach chart for the best candidate setup
            best_item = None
            if new_opens:
                best_item = new_opens[0]
            elif waits:
                best_item = sorted(waits, key=lambda x: x["dist_pct"])[0]
            elif opens:
                best_item = opens[0]
            elif near_radar:
                best_item = {"ticker": near_radar[0]["ticker"], "status": "RADAR"}

            if best_item and best_item["ticker"] in dfs_by_ticker:
                chart_path = generate_best_stock_chart(best_item["ticker"], dfs_by_ticker[best_item["ticker"]], best_item)
                if chart_path:
                    send_telegram_photo(chart_path, caption=f"📈 شارت أفضل فرصة مطابقة للاستراتيجية: <b>#{best_item['ticker']}</b>")


if __name__ == "__main__":
    main()
