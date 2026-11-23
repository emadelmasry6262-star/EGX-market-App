"""
EGX SMC / Liquidity Reversal Strategy (v20) - Telegram Live Alerts & Daily Report Worker
Features:
  1) Intraday Live Alerts (Every 15 mins during EGX session 10:00 - 14:45 Cairo time):
     - Instant BUY-STOP Trigger notification when live price >= Trigger price (includes Entry, First Target +5%, Initial Stop Loss -5%)
     - Instant STOP-LOSS Hit notification when live price <= Updated Stop Loss
  2) Dual Daily Reports (15:00 Cairo time & 22:00 Cairo time, Sun-Thu):
     - EGX30 Index daily performance (close & % change)
     - Open Positions (including newly opened positions today, current PnL, first target, and trailing stop)
     - Closed Positions today (exit price, close reason: Stop-Loss / Trailing Stop / Max Hold, and realized PnL %)
     - Active Buy-Stop Setups ready for next session
     - Stocks approaching setup conditions with detailed filter status (achieved vs. pending filters)
  3) Smart Anti-Spam State Memory (telegram_alert_state.json):
     - Tracks sent intraday alerts and scheduled report slots ('15:00' and '22:00') separately
     - Manual runs ('workflow_dispatch') never block scheduled 15:00 or 22:00 automatic reports!
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
from tvDatafeed import TvDatafeed, Interval

# ============================ CONFIG & SECRETS ============================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TV_USER = os.getenv("TV_USER", "").strip()
TV_PASS = os.getenv("TV_PASS", "").strip()
GITHUB_EVENT_NAME = os.getenv("GITHUB_EVENT_NAME", "").strip()

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
FIRST_TARGET_PCT = 0.05  # First profit lock tier (+5% -> locks +1% profit)
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
            print(f"[warn] Telegram send attempt {attempt+1} failed: {e}")
            time.sleep(1.5)
    return False


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                data.setdefault("sent_alerts", {})
                data.setdefault("sent_daily_slots", {})
                return data
        except Exception:
            pass
    return {"sent_alerts": {}, "sent_daily_slots": {}}


def save_state(state: dict):
    alerts = state.get("sent_alerts", {})
    if len(alerts) > 400:
        keys = list(alerts.keys())[-400:]
        state["sent_alerts"] = {k: alerts[k] for k in keys}

    slots = state.get("sent_daily_slots", {})
    if len(slots) > 60:
        skeys = list(slots.keys())[-60:]
        state["sent_daily_slots"] = {k: slots[k] for k in skeys}

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ============================ DATA & STRATEGY ENGINE ============================
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


def download_stock_with_live(tv, sym, today_cairo, n_bars=N_BARS):
    """Returns (completed_daily_df, live_bar_dict_or_None)."""
    for _ in range(3):
        try:
            raw = tv.get_hist(symbol=sym, exchange=EXCHANGE, interval=Interval.in_daily, n_bars=n_bars)
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
                    }
                    completed_df = df.iloc[:-1]
                else:
                    completed_df = df

                completed_df = completed_df[completed_df.index >= pd.Timestamp(DATA_START)]
                return completed_df, live_bar
        except Exception:
            time.sleep(0.7)
    return None, None


def get_egx30_status(tv, today_cairo):
    """Fetches EGX30 index performance for the daily report."""
    comp_df, live_bar = download_stock_with_live(tv, INDEX_SYMBOL, today_cairo, n_bars=30)
    if comp_df is None or len(comp_df) < 2:
        return None

    if live_bar is not None:
        cur_close = live_bar["close"]
        prev_close = float(comp_df["close"].iloc[-1])
        bar_date = live_bar["date"]
        bar_high = live_bar["high"]
        bar_low = live_bar["low"]
    else:
        cur_close = float(comp_df["close"].iloc[-1])
        prev_close = float(comp_df["close"].iloc[-2])
        bar_date = comp_df.index[-1].strftime("%Y-%m-%d")
        bar_high = float(comp_df["high"].iloc[-1])
        bar_low = float(comp_df["low"].iloc[-1])

    chg_pts = cur_close - prev_close
    chg_pct = (chg_pts / prev_close) * 100.0 if prev_close > 0 else 0.0
    return {
        "symbol": INDEX_SYMBOL,
        "date": bar_date,
        "close": round(cur_close, 2),
        "prev_close": round(prev_close, 2),
        "chg_pts": round(chg_pts, 2),
        "chg_pct": round(chg_pct, 2),
        "high": round(bar_high, 2),
        "low": round(bar_low, 2),
    }


def prior_downtrend(close):
    s = pd.Series(close)
    ema = s.ewm(span=TREND_EMA, adjust=False).mean()
    down = (s < ema) & (ema < ema.shift(TREND_SLOPE))
    if USE_EMA200_FILTER:
        ema200 = s.ewm(span=LONG_EMA, adjust=False).mean()
        down = down & (s > ema200)
    return down.shift(1, fill_value=False).values.astype(bool)


def detailed_filter_diagnostics(df, last_pwl, last_pml):
    """
    Evaluates all strategy filters on the latest bar to show which filters
    are achieved (✅) and which are still pending (⏳).
    """
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    s = pd.Series(c)

    ema50 = s.ewm(span=TREND_EMA, adjust=False).mean()
    ema200 = s.ewm(span=LONG_EMA, adjust=False).mean()

    last_close = float(c[-1])
    last_ema50 = float(ema50.iloc[-1])
    prev_ema50 = float(ema50.iloc[-1 - TREND_SLOPE]) if len(ema50) > TREND_SLOPE else last_ema50
    last_ema200 = float(ema200.iloc[-1])

    cond_above_ema200 = last_close > last_ema200
    cond_below_ema50 = last_close < last_ema50
    cond_ema50_slope = last_ema50 < prev_ema50

    rng = float(h[-1] - l[-1])
    close_pos = ((last_close - float(l[-1])) / rng) if rng > 0 else 0.0
    cond_candle = close_pos >= SWEEP_CLOSE_POS

    # Liquidity distance check (Week Low / Month Low)
    near_levels = []
    if not np.isnan(last_pwl) and last_pwl > 0:
        d_w = (last_close - last_pwl) / last_pwl * 100.0
        if -1.5 <= d_w <= 2.5:
            near_levels.append(("قاع الأسبوع السابق", last_pwl, d_w))
    if not np.isnan(last_pml) and last_pml > 0:
        d_m = (last_close - last_pml) / last_pml * 100.0
        if -1.5 <= d_m <= 2.5:
            near_levels.append(("قاع الشهر السابق", last_pml, d_m))

    achieved = []
    pending = []

    if cond_above_ema200:
        achieved.append(f"فوق EMA200 ({last_ema200:.2f})")
    else:
        pending.append(f"الإغلاق تحت EMA200 ({last_ema200:.2f})")

    if cond_below_ema50 and cond_ema50_slope:
        achieved.append(f"تصحيح هابط تحت EMA50 ({last_ema50:.2f})")
    else:
        if not cond_below_ema50:
            pending.append(f"السعر فوق EMA50 ({last_ema50:.2f})")
        if not cond_ema50_slope:
            pending.append("ميل EMA50 غير هابط")

    if cond_candle:
        achieved.append(f"قوة إغلاق الشمعة ({close_pos*100:.0f}%)")
    else:
        pending.append(f"إغلاق الشمعة في النصف العلوي (حالياً {close_pos*100:.0f}%)")

    # Sweep status
    swept_today = False
    for lvl_name, lvl_val, dist in near_levels:
        if float(l[-1]) < lvl_val and last_close > lvl_val:
            achieved.append(f"كسر كاذب (Sweep) لـ {lvl_name} ({lvl_val:.2f})")
            swept_today = True
        elif dist >= 0:
            pending.append(f"كسر سيولة {lvl_name} ({lvl_val:.2f} | يبعد +{dist:.2f}%)")
        else:
            pending.append(f"استعادة الإغلاق فوق {lvl_name} ({lvl_val:.2f})")

    return {
        "near_levels": near_levels,
        "achieved": achieved,
        "pending": pending,
        "all_trend_ok": cond_above_ema200 and cond_below_ema50 and cond_ema50_slope,
        "swept_today": swept_today,
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
    """
    Runs v20 strategy across bars and returns:
      - live_info: active OPEN position or WAIT buy-stop setup
      - closed_trades: list of closed trades (with exit_date, exit_price, reason, pnl_pct)
      - radar_hits: stocks approaching liquidity zones with full filter breakdown
    """
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
                hit_sl = l[i] <= pos["stop"]
                hit_max_hold = (not pos["post_25"]) and (i - pos["i0"] >= MAX_HOLD)
                if hit_sl or hit_max_hold:
                    if hit_sl:
                        exit_px = min(o[i], pos["stop"]) if o[i] < pos["stop"] else pos["stop"]
                        if pos["max_tier"] == "0%":
                            reason = "ضرب وقف الخسارة المبدئي (-5%)"
                        else:
                            reason = f"ضرب الوقف المتحرك (حماية {pos['max_tier']})"
                    else:
                        exit_px = c[i]
                        reason = f"انتهاء مدة الاحتفاظ ({MAX_HOLD} جلسة)"

                    pnl_pct = ((exit_px / pos["entry"]) - 1.0 - COMMISSION) * 100.0
                    closed_trades.append(dict(
                        ticker=tk,
                        entry_date=dates[pos["i0"]].strftime("%Y-%m-%d"),
                        exit_date=dates[i].strftime("%Y-%m-%d"),
                        entry=round(pos["entry"], 3),
                        exit_price=round(float(exit_px), 3),
                        reason=reason,
                        pnl_pct=round(float(pnl_pct), 2),
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
                        pnl_pct = ((stop / fill) - 1.0 - COMMISSION) * 100.0
                        closed_trades.append(dict(
                            ticker=tk,
                            entry_date=dates[i].strftime("%Y-%m-%d"),
                            exit_date=dates[i].strftime("%Y-%m-%d"),
                            entry=round(float(fill), 3),
                            exit_price=round(float(stop), 3),
                            reason="ضرب وقف الخسارة في نفس جلسة الدخول (-5%)",
                            pnl_pct=round(float(pnl_pct), 2),
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
        first_target = pos["entry"] * (1.0 + FIRST_TARGET_PCT)
        live_info = dict(
            status="OPEN",
            ticker=tk,
            entry_date=dates[pos["i0"]].strftime("%Y-%m-%d"),
            entry=round(pos["entry"], 3),
            first_target=round(first_target, 3),
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
            trig_val = float(s["run_high"])
            live_info = dict(
                status="WAIT",
                ticker=tk,
                sweep_date=dates[s["sweep_i"]].strftime("%Y-%m-%d"),
                trigger=round(trig_val, 3),
                first_target=round(trig_val * (1.0 + FIRST_TARGET_PCT), 3),
                last_close=round(last_close, 3),
                dist_pct=round(dist_trigger, 2),
                est_stop=round(trig_val * (1.0 - STOP_PCT), 3),
                invalidation=round(s["level"], 3),
                kind=s["kind"],
                rem_bars=rem_bars
            )

    radar_hits = []
    if live_info is None:
        last_pwl = float(pwl[-1]) if not np.isnan(pwl[-1]) else np.nan
        last_pml = float(pml[-1]) if not np.isnan(pml[-1]) else np.nan
        diag = detailed_filter_diagnostics(df, last_pwl, last_pml)

        if diag["near_levels"]:
            best_lvl_name, best_lvl_val, best_dist = sorted(diag["near_levels"], key=lambda x: abs(x[2]))[0]
            radar_hits.append(dict(
                ticker=tk,
                close=round(last_close, 3),
                ref_type=best_lvl_name,
                ref_val=round(best_lvl_val, 3),
                dist_pct=round(best_dist, 2),
                all_trend_ok=diag["all_trend_ok"],
                achieved=diag["achieved"],
                pending=diag["pending"]
            ))

    return live_info, closed_trades, radar_hits


# ============================ REPORT BUILDER ============================
def build_daily_report_text(today_str, slot_label, egx30_info, all_live, all_closed, all_radar):
    opens = [x for x in all_live if x["status"] == "OPEN"]
    opened_today = [x for x in opens if x["entry_date"] == today_str]
    older_opens = [x for x in opens if x["entry_date"] != today_str]
    waits = sorted([x for x in all_live if x["status"] == "WAIT"], key=lambda x: x["dist_pct"])
    closed_today = [x for x in all_closed if x["exit_date"] == today_str]

    # Sort radar: trend-qualified stocks closest to support first
    sorted_radar = sorted(
        all_radar,
        key=lambda x: (not x["all_trend_ok"], abs(x["dist_pct"]))
    )

    lines = [
        f"📊 <b>التقرير اليومي لرادار البورصة المصرية (EGX SMC v20)</b>",
        f"📅 <b>التاريخ:</b> {today_str} | 🕒 <b>توقيت التقرير:</b> {slot_label}",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    # 0) EGX30 Index Performance
    if egx30_info:
        idx_emoji = "🟢" if egx30_info["chg_pct"] >= 0 else "🔴"
        lines.append(
            f"{idx_emoji} <b>حالة المؤشر العام (EGX30):</b> {egx30_info['close']:,.2f} نقطة "
            f"(<b>{egx30_info['chg_pct']:+.2f}%</b> | {egx30_info['chg_pts']:+.2f} نقطة)"
        )
        lines.append("━━━━━━━━━━━━━━━━━━━━")

    # 1) Newly Opened Positions Today
    lines.append(f"🚀 <b>1) أسهم حققت الاستراتيجية وتم فتحها كصفقة اليوم ({len(opened_today)}):</b>")
    if not opened_today:
        lines.append("• لم يتم تفعيل صفقات جديدة خلال جلسة اليوم.")
    else:
        for o in opened_today:
            lines.append(
                f"• <b>#{o['ticker']}</b> | دخول: <b>{o['entry']}</b> | إغلاق: {o['last_close']} (<b>{o['pnl_pct']:+.2f}%</b>)
"
                f"  └ 🎯 الهدف الأول (+5%): <b>{o['first_target']}</b> | 🛑 الوقف المبدئي: <b>{o['stop']}</b> | {o['kind']}"
            )

    # 2) All Active Open Positions (Previous + Today)
    lines.append("")
    lines.append(f"🟢 <b>2) إجمالي الصفقات المفتوحة والمستمرة ({len(opens)}):</b>")
    if not older_opens and not opened_today:
        lines.append("• لا توجد صفقات مفتوحة حالياً.")
    elif not older_opens and opened_today:
        lines.append("• جميع الصفقات المفتوحة هي صفقات اليوم المذكورة أعلاه.")
    else:
        for o in older_opens:
            lines.append(
                f"• <b>#{o['ticker']}</b> (دخول {o['entry_date']} بسعر {o['entry']}) | إغلاق: {o['last_close']} (<b>{o['pnl_pct']:+.2f}%</b>)
"
                f"  └ 🎯 الهدف الأول: {o['first_target']} | 🛡️ الوقف المحدث: <b>{o['stop']}</b> ({o['tier']}) | جلسات: {o['bars']}"
            )

    # 3) Closed Trades Today
    lines.append("")
    lines.append(f"🏁 <b>3) الصفقات المغلقة خلال جلسة اليوم ({len(closed_today)}):</b>")
    if not closed_today:
        lines.append("• لا توجد صفقات أُغلقت خلال جلسة اليوم.")
    else:
        for ct in closed_today:
            res_icon = "✅ ربح" if ct["pnl_pct"] > 0 else "❌ خسارة"
            lines.append(
                f"• <b>#{ct['ticker']}</b> ({res_icon}: <b>{ct['pnl_pct']:+.2f}%</b>)
"
                f"  └ دخول: {ct['entry']} ⬅️ إغلاق: <b>{ct['exit_price']}</b> | السبب: {ct['reason']}"
            )

    # 4) Ready Buy-Stop Setups for Next Session
    lines.append("")
    lines.append(f"⏳ <b>4) أسهم جاهزة للتنفيذ بأمر Buy-Stop للجلسة القادمة ({len(waits)}):</b>")
    if not waits:
        lines.append("• لا توجد أوامر Buy-Stop معلقة حالياً.")
    else:
        for w in waits:
            lines.append(
                f"• <b>#{w['ticker']}</b> | سعر الدخول (Trigger): <b>{w['trigger']}</b> (يبعد +{w['dist_pct']}%)
"
                f"  └ 🎯 الهدف الأول (+5%): <b>{w['first_target']}</b> | 🛑 الوقف (-5%): {w['est_stop']} | صلاحية: {w['rem_bars']} جلسات"
            )

    # 5) Stocks Approaching Strategy Conditions (Achieved vs Pending Filters)
    lines.append("")
    lines.append(f"👀 <b>5) أسهم تقترب من تنفيذ الشروط وحالة الفلاتر ({len(sorted_radar)}):</b>")
    if not sorted_radar:
        lines.append("• لا توجد أسهم قريبة من مناطق السيولة حالياً.")
    else:
        for r in sorted_radar[:10]:
            status_badge = "جاهز للـ Sweep" if r["all_trend_ok"] else "قيد اكتمال الاتجاه"
            ach_txt = "، ".join(r["achieved"]) if r["achieved"] else "لا يوجد"
            pen_txt = "، ".join(r["pending"]) if r["pending"] else "مكتمل"
            lines.append(
                f"• <b>#{r['ticker']}</b> ({r['close']} ج | {status_badge})
"
                f"  ├ ✅ <b>المحقق:</b> {ach_txt}
"
                f"  └ ⏳ <b>المنتظر:</b> {pen_txt}"
            )

    return "
".join(lines)


# ============================ MAIN EXECUTION ============================
def main():
    raw_mode = sys.argv[1].lower() if len(sys.argv) > 1 else "auto"
    now_cairo = datetime.now(ZoneInfo(MARKET_TZ))
    today_norm = pd.Timestamp(now_cairo.date())
    today_str = today_norm.strftime("%Y-%m-%d")

    is_manual = (GITHUB_EVENT_NAME == "workflow_dispatch") or (raw_mode in ("manual", "force_report"))
    mode = raw_mode

    # Decide mode automatically if 'auto'
    if mode == "auto":
        # EGX trades Sun (6) to Thu (3); Fri=4, Sat=5
        if now_cairo.weekday() in (4, 5):
            print(f"[info] Weekend in Cairo ({now_cairo:%A}). Skipping.")
            return
        if now_cairo.hour >= 15:
            mode = "daily_report"
        else:
            mode = "intraday"

    print(f"[run] Mode: {mode} | Manual: {is_manual} | Cairo Time: {now_cairo:%Y-%m-%d %H:%M}")
    state = load_state()
    sent_alerts = state.setdefault("sent_alerts", {})
    sent_daily_slots = state.setdefault("sent_daily_slots", {})

    tv = get_tv_client()
    all_live = []
    all_closed = []
    all_radar = []
    intraday_bars = {}

    # Determine whether today's bar should be treated as completed for strategy evaluation
    # After 14:45 Cairo time, the EGX session (10:00 - 14:30) is closed
    session_closed = (now_cairo.hour > 14) or (now_cairo.hour == 14 and now_cairo.minute >= 45)

    for tk in DEFAULT_TICKERS:
        comp_df, live_bar = download_stock_with_live(tv, tk, today_norm)
        if comp_df is None or len(comp_df) < 250:
            continue

        if (mode == "daily_report" or session_closed) and live_bar is not None:
            new_row = pd.DataFrame([{
                "open": live_bar["open"],
                "high": live_bar["high"],
                "low": live_bar["low"],
                "close": live_bar["close"],
                "volume": 0.0
            }], index=[today_norm])
            eval_df = pd.concat([comp_df, new_row])
        else:
            eval_df = comp_df
            if live_bar is not None:
                intraday_bars[tk] = live_bar

        live_info, closed_trades, rad_hits = evaluate_stock_state(tk, eval_df)
        if live_info is not None:
            all_live.append(live_info)
        all_closed.extend(closed_trades)
        all_radar.extend(rad_hits)
        time.sleep(0.15)

    # ---------------- 1) INTRADAY REAL-TIME ALERTS ----------------
    if mode == "intraday":
        alerts_triggered = 0
        for item in all_live:
            tk = item["ticker"]
            bar = intraday_bars.get(tk)
            if not bar:
                continue

            cur_price = bar["close"]
            bar_high = bar["high"]
            bar_low = bar["low"]

            # A) Buy-Stop Trigger Hit -> Send Instant Entry + First Target + Stop Loss Alert
            if item["status"] == "WAIT":
                trig = item["trigger"]
                if bar_high >= trig or cur_price >= trig:
                    alert_key = f"BUY_{tk}_{item['sweep_date']}_{trig}"
                    if alert_key not in sent_alerts:
                        msg = (
                            f"🚀 <b>إشارة شراء لحظية (BUY-STOP TRIGGERED)</b>
"
                            f"━━━━━━━━━━━━━━
"
                            f"📌 <b>السهم:</b> #{tk}
"
                            f"💰 <b>السعر الحالي:</b> {cur_price:.3f} ج (أعلى سعر اليوم: {bar_high:.3f})
"
                            f"🎯 <b>سعر الدخول (Trigger):</b> {trig:.3f} ج
"
                            f"🥇 <b>الهدف الأول (+5%):</b> {item['first_target']:.3f} ج
"
                            f"🛑 <b>وقف الخسارة المبدئي (-5%):</b> {item['est_stop']:.3f} ج
"
                            f"📊 <b>نوع الإشارة:</b> {item['kind']}
"
                            f"⏰ <b>الوقت:</b> {now_cairo:%H:%M} بتوقيت القاهرة"
                        )
                        if send_telegram_msg(msg):
                            sent_alerts[alert_key] = now_cairo.strftime("%Y-%m-%d %H:%M")
                            alerts_triggered += 1

            # B) Stop-Loss Hit on Open Position -> Send Instant Stop-Loss Alert
            elif item["status"] == "OPEN":
                sl = item["stop"]
                if bar_low <= sl or cur_price <= sl:
                    alert_key = f"STOP_{tk}_{item['entry_date']}_{sl}_{today_str}"
                    if alert_key not in sent_alerts:
                        pnl_at_stop = (sl / item["entry"] - 1.0) * 100.0
                        msg = (
                            f"🚨 <b>تنبيه ضرب وقف الخسارة (STOP-LOSS HIT)</b>
"
                            f"━━━━━━━━━━━━━━
"
                            f"📌 <b>السهم:</b> #{tk}
"
                            f"📉 <b>السعر الحالي:</b> {cur_price:.3f} ج (أدنى سعر اليوم: {bar_low:.3f})
"
                            f"🛑 <b>سعر وقف الخسارة:</b> {sl:.3f} ج
"
                            f"💵 <b>سعر الدخول:</b> {item['entry']:.3f} ج (النتيجة عند الوقف: {pnl_at_stop:+.2f}%)
"
                            f"🛡️ <b>مستوى الحماية:</b> {item['tier']}
"
                            f"⏰ <b>الوقت:</b> {now_cairo:%H:%M} بتوقيت القاهرة"
                        )
                        if send_telegram_msg(msg):
                            sent_alerts[alert_key] = now_cairo.strftime("%Y-%m-%d %H:%M")
                            alerts_triggered += 1

        print(f"[intraday] Checked {len(intraday_bars)} live bars. Sent {alerts_triggered} new alerts.")
        save_state(state)

    # ---------------- 2) DAILY END-OF-DAY & EVENING REPORTS ----------------
    elif mode == "daily_report":
        # Determine report slot:
        # - Manual runs use a 'MANUAL' slot key so they NEVER consume or block
        #   the scheduled 15:00 ('AFTER_CLOSE_15') or 22:00 ('EVENING_22') reports!
        if is_manual:
            slot_key = f"{today_str}_MANUAL"
            slot_label = f"تقرير يدوي ({now_cairo:%H:%M} القاهرة)"
        elif now_cairo.hour >= 21:
            slot_key = f"{today_str}_EVENING_22"
            slot_label = "الملخص المسائي (22:00 القاهرة)"
        else:
            slot_key = f"{today_str}_AFTER_CLOSE_15"
            slot_label = "تقرير ما بعد الإغلاق (15:00 القاهرة)"

        if not is_manual and slot_key in sent_daily_slots:
            print(f"[daily] Scheduled report slot {slot_key} already sent.")
            return

        egx30_info = get_egx30_status(tv, today_norm)
        report_text = build_daily_report_text(
            today_str=today_str,
            slot_label=slot_label,
            egx30_info=egx30_info,
            all_live=all_live,
            all_closed=all_closed,
            all_radar=all_radar
        )

        if send_telegram_msg(report_text):
            if not is_manual:
                sent_daily_slots[slot_key] = now_cairo.strftime("%Y-%m-%d %H:%M")
            save_state(state)
            print(f"[daily] Report ({slot_key}) sent successfully.")


if __name__ == "__main__":
    main()
