"""
EGX SMC / Liquidity Reversal Strategy (v20) - Telegram Live Alerts & Daily Report Worker
Features:
  1) Intraday Live Alerts (Every 5 mins during EGX session 10:00 - 14:30 Cairo time):
     - Instant BUY-STOP Trigger notification when live price >= Trigger price
     - Instant STOP-LOSS Hit notification when live price <= Updated Stop Loss
  2) End-of-Day Daily Report (After 15:10 Cairo time):
     - Full summary of Open Positions (with updated trailing stops: 5%->+1%, 7.5%->+5%, etc.)
     - Active Buy-Stop Setups ready for next session
     - Watchlist Radar (0% - 2% above Prev Week / Month Low)
  3) Anti-Spam State Memory (telegram_alert_state.json) committed back to GitHub repo so no duplicate alerts are sent.
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
N_BARS = 600
DATA_START = "2023-01-01"
TEST_START = "2023-01-01"

TREND_EMA = 50
TREND_SLOPE = 10
ENTRY_TIMEOUT = 5
STOP_PCT = 0.05
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
                return json.load(f)
        except Exception:
            pass
    return {"sent_alerts": {}, "last_daily_report_date": ""}


def save_state(state: dict):
    # Keep only recent 300 alert keys to prevent file bloat
    alerts = state.get("sent_alerts", {})
    if len(alerts) > 300:
        keys = list(alerts.keys())[-300:]
        state["sent_alerts"] = {k: alerts[k] for k in keys}
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


def download_stock_with_live(tv, sym, today_cairo):
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
                    }
                    completed_df = df.iloc[:-1]
                else:
                    completed_df = df

                completed_df = completed_df[completed_df.index >= pd.Timestamp(DATA_START)]
                return completed_df, live_bar
        except Exception:
            time.sleep(0.7)
    return None, None


def prior_downtrend(close):
    s = pd.Series(close)
    ema = s.ewm(span=TREND_EMA, adjust=False).mean()
    down = (s < ema) & (ema < ema.shift(TREND_SLOPE))
    if USE_EMA200_FILTER:
        ema200 = s.ewm(span=LONG_EMA, adjust=False).mean()
        down = down & (s > ema200)
    return down.shift(1, fill_value=False).values.astype(bool)


def current_filter_status(close):
    s = pd.Series(close)
    ema = s.ewm(span=TREND_EMA, adjust=False).mean()
    down = (s < ema) & (ema < ema.shift(TREND_SLOPE))
    if USE_EMA200_FILTER:
        ema200 = s.ewm(span=LONG_EMA, adjust=False).mean()
        down = down & (s > ema200)
    return bool(down.iloc[-1])


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
        pos["stop"] = max(pos["stop"], entry * 1.01)  # Updated rule: +1% profit lock at +5%


def evaluate_stock_state(tk, df):
    """Runs v20 on completed bars and returns (live_setup_or_pos, radar_hits)."""
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
                if l[i] <= pos["stop"] or ((not pos["post_25"]) and (i - pos["i0"] >= MAX_HOLD)):
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
        live_info = dict(
            status="OPEN",
            ticker=tk,
            entry_date=dates[pos["i0"]].strftime("%Y-%m-%d"),
            entry=round(pos["entry"], 3),
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
            dist_trigger = (s["run_high"] / last_close - 1.0) * 100
            live_info = dict(
                status="WAIT",
                ticker=tk,
                sweep_date=dates[s["sweep_i"]].strftime("%Y-%m-%d"),
                trigger=round(s["run_high"], 3),
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
    filt_ok = current_filter_status(c)

    if not np.isnan(last_pwl) and last_close >= last_pwl:
        d_w = (last_close - last_pwl) / last_pwl * 100
        if 0.0 <= d_w <= 2.0:
            radar_hits.append(dict(ticker=tk, close=round(last_close, 3), ref_type="Week Low", ref_val=round(last_pwl, 3), dist_pct=round(d_w, 2), filter_ok=filt_ok))
    if not np.isnan(last_pml) and last_close >= last_pml:
        d_m = (last_close - last_pml) / last_pml * 100
        if 0.0 <= d_m <= 2.0:
            radar_hits.append(dict(ticker=tk, close=round(last_close, 3), ref_type="Month Low", ref_val=round(last_pml, 3), dist_pct=round(d_m, 2), filter_ok=filt_ok))

    return live_info, radar_hits


# ============================ MAIN EXECUTION ============================
def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "auto"
    now_cairo = datetime.now(ZoneInfo(MARKET_TZ))
    today_norm = pd.Timestamp(now_cairo.date())
    today_str = today_norm.strftime("%Y-%m-%d")

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

    print(f"[run] Mode: {mode} | Cairo Time: {now_cairo:%Y-%m-%d %H:%M}")
    state = load_state()
    sent_alerts = state.setdefault("sent_alerts", {})

    tv = get_tv_client()
    all_live = []
    all_radar = []
    intraday_bars = {}

    for tk in DEFAULT_TICKERS:
        comp_df, live_bar = download_stock_with_live(tv, tk, today_norm)
        if comp_df is None or len(comp_df) < 250:
            continue

        # In daily_report mode after 15:00, today's bar is completed -> include it!
        if mode == "daily_report" and live_bar is not None:
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

        live_info, rad_hits = evaluate_stock_state(tk, eval_df)
        if live_info is not None:
            all_live.append(live_info)
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

            # A) Buy-Stop Trigger Hit
            if item["status"] == "WAIT":
                trig = item["trigger"]
                if bar_high >= trig or cur_price >= trig:
                    alert_key = f"BUY_{tk}_{item['sweep_date']}_{trig}"
                    if alert_key not in sent_alerts:
                        msg = (
                            f"🚀 <b>إشارة شراء لحظية (BUY-STOP TRIGGERED)</b>\n"
                            f"━━━━━━━━━━━━━━\n"
                            f"📌 <b>السهم:</b> #{tk}\n"
                            f"💰 <b>السعر الحالي:</b> {cur_price:.3f} ج (أعلى سعر اليوم: {bar_high:.3f})\n"
                            f"🎯 <b>سعر التفعيل (Trigger):</b> {trig:.3f} ج\n"
                            f"🛑 <b>وقف الخسارة المبدئي (-5%):</b> {item['est_stop']:.3f} ج\n"
                            f"📊 <b>نوع الإشارة:</b> {item['kind']}\n"
                            f"⏰ <b>الوقت:</b> {now_cairo:%H:%M} بتوقيت القاهرة"
                        )
                        if send_telegram_msg(msg):
                            sent_alerts[alert_key] = now_cairo.strftime("%Y-%m-%d %H:%M")
                            alerts_triggered += 1

            # B) Stop-Loss Hit on Open Position
            elif item["status"] == "OPEN":
                sl = item["stop"]
                if bar_low <= sl or cur_price <= sl:
                    alert_key = f"STOP_{tk}_{item['entry_date']}_{sl}_{today_str}"
                    if alert_key not in sent_alerts:
                        pnl_at_stop = (sl / item["entry"] - 1.0) * 100.0
                        msg = (
                            f"🚨 <b>تنبيه ضرب وقف الخسارة (STOP-LOSS HIT)</b>\n"
                            f"━━━━━━━━━━━━━━\n"
                            f"📌 <b>السهم:</b> #{tk}\n"
                            f"📉 <b>السعر الحالي:</b> {cur_price:.3f} ج (أدنى سعر اليوم: {bar_low:.3f})\n"
                            f"🛑 <b>سعر وقف الخسارة:</b> {sl:.3f} ج\n"
                            f"💵 <b>سعر الدخول:</b> {item['entry']:.3f} ج (النتيجة عند الوقف: {pnl_at_stop:+.2f}%)\n"
                            f"🛡️ <b>مستوى الحماية:</b> {item['tier']}\n"
                            f"⏰ <b>الوقت:</b> {now_cairo:%H:%M} بتوقيت القاهرة"
                        )
                        if send_telegram_msg(msg):
                            sent_alerts[alert_key] = now_cairo.strftime("%Y-%m-%d %H:%M")
                            alerts_triggered += 1

        print(f"[intraday] Checked {len(intraday_bars)} live bars. Sent {alerts_triggered} new alerts.")
        save_state(state)

    # ---------------- 2) DAILY END-OF-DAY REPORT ----------------
    elif mode == "daily_report":
        if state.get("last_daily_report_date") == today_str:
            print(f"[daily] Report for {today_str} already sent.")
            return

        opens = [x for x in all_live if x["status"] == "OPEN"]
        waits = [x for x in all_live if x["status"] == "WAIT"]
        ready_radar = sorted([r for r in all_radar if r["filter_ok"]], key=lambda x: x["dist_pct"])

        lines = [
            f"📊 <b>التقرير اليومي لرادار البورصة المصرية (EGX SMC v20)</b>",
            f"📅 <b>التاريخ:</b> {today_str}",
            "━━━━━━━━━━━━━━━━━━━━",
            f"🟢 <b>1) الصفقات المفتوحة حالياً ({len(opens)}):</b>"
        ]
        if not opens:
            lines.append("• لا توجد صفقات مفتوحة حالياً.")
        else:
            for o in opens:
                lines.append(
                    f"• <b>#{o['ticker']}</b> | دخول: {o['entry']} | إغلاق: {o['last_close']} (<b>{o['pnl_pct']:+.2f}%</b>)\n"
                    f"  └ الوقف المحدث: <b>{o['stop']}</b> ({o['tier']}) | جلسات: {o['bars']}"
                )

        lines.append("")
        lines.append(f"⏳ <b>2) أوامر Buy-Stop المعلقة للجلسة القادمة ({len(waits)}):</b>")
        if not waits:
            lines.append("• لا توجد أوامر Buy-Stop معلقة.")
        else:
            for w in waits:
                lines.append(
                    f"• <b>#{w['ticker']}</b> | تفعيل: <b>{w['trigger']}</b> (+{w['dist_pct']}%)\n"
                    f"  └ وقف (-5%): {w['est_stop']} | صلاحية: {w['rem_bars']} أيام | {w['kind']}"
                )

        lines.append("")
        lines.append(f"👀 <b>3) أسهم تحت المراقبة ومستوفية للفلتر ({len(ready_radar)}):</b>")
        if not ready_radar:
            lines.append("• لا توجد أسهم مستوفية قريبة من القيعان حالياً.")
        else:
            for r in ready_radar[:8]:
                lines.append(f"• <b>#{r['ticker']}</b>: {r['close']} (قرب {r['ref_type']} {r['ref_val']} | +{r['dist_pct']}%)")

        report_text = "\n".join(lines)
        if send_telegram_msg(report_text):
            state["last_daily_report_date"] = today_str
            save_state(state)
            print(f"[daily] Daily report for {today_str} sent successfully.")


if __name__ == "__main__":
    main()
