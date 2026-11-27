# -*- coding: utf-8 -*-
"""
Gold Agent – data builder (v1.5.0)
Fetches GC=F from Yahoo Finance, computes basic technicals + optional news,
writes a single data.json with clear freshness fields.

v1.3.0: adds 15-minute and 30-minute timeframes (analysis, candle bias,
short-term confirmation for the decision, and chart candles).

Designed for GitHub Actions (every 10–30 min) + static GitHub Pages.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = "data.json"
SYMBOL = "GC=F"
MAX_RETRIES = 3
BASE_DELAY = 5

# all timeframes shown in the UI (order = tab order)
TF_LIST = (
    ("15m", "۱۵ دقیقه"),
    ("30m", "۳۰ دقیقه"),
    ("1h", "ساعتی"),
    ("1d", "روزانه"),
    ("1wk", "هفتگی"),
)
# timeframes that feed the overall "tech" score (same as before, so the
# long-term score does not change because of the new short timeframes)
MAIN_TFS = ("1h", "1d", "1wk")


# ── helpers ──────────────────────────────────────────────────────────────

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt) -> str | None:
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    if isinstance(o, (pd.Timestamp, datetime)):
        return iso(o)
    return o


def load_old() -> dict:
    if not os.path.exists(OUT):
        return {}
    try:
        with open(OUT, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_json(data: dict) -> None:
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(clean(data), f, ensure_ascii=False, separators=(",", ":"))


# ── market data ──────────────────────────────────────────────────────────

def fetch_yahoo(symbol: str = SYMBOL) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    t = yf.Ticker(symbol)
    out: dict[str, pd.DataFrame] = {}

    # 15m / 30m – Yahoo allows up to ~60 days of intraday data.
    # If one of them fails we simply skip it (the rest still works).
    for key in ("15m", "30m"):
        try:
            m = t.history(period="30d", interval=key, auto_adjust=True)
            if m is not None and len(m) > 60:
                out[key] = m
        except Exception as e:
            print(f"{key} fetch failed:", e)

    # 1h – last ~60 days is enough for indicators
    h = t.history(period="60d", interval="1h", auto_adjust=True)
    if h is not None and len(h) > 20:
        out["1h"] = h

    d = t.history(period="2y", interval="1d", auto_adjust=True)
    if d is not None and len(d) > 50:
        out["1d"] = d

    w = t.history(period="5y", interval="1wk", auto_adjust=True)
    if w is not None and len(w) > 20:
        out["1wk"] = w

    if not out:
        raise RuntimeError("Yahoo Finance returned empty history for " + symbol)
    return out


def rsi(series: pd.Series, period: int = 14) -> float | None:
    if series is None or len(series) < period + 2:
        return None
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    val = 100 - (100 / (1 + rs))
    v = float(val.iloc[-1])
    return None if math.isnan(v) else round(v, 1)


def atr(df: pd.DataFrame, period: int = 14) -> float | None:
    if df is None or len(df) < period + 2:
        return None
    high, low, close = df["High"], df["Low"], df["Close"]
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    v = float(tr.rolling(period).mean().iloc[-1])
    return None if math.isnan(v) else round(v, 2)


def support_resistance(df: pd.DataFrame, lookback: int = 40) -> tuple[float | None, float | None]:
    if df is None or len(df) < 10:
        return None, None
    window = df.tail(lookback)
    return float(window["Low"].min()), float(window["High"].max())


def score_tf(df: pd.DataFrame) -> dict:
    """Simple rule-based score −100 … +100."""
    if df is None or len(df) < 50:
        return {"score": 0, "price": None, "rsi": None, "atr": None, "sup": None, "res": None, "sma50": None, "sma200": None}

    close = df["Close"]
    price = float(close.iloc[-1])
    r = rsi(close)
    a = atr(df)
    sup, res = support_resistance(df)
    sma50 = float(close.rolling(50, min_periods=20).mean().iloc[-1])
    sma200 = float(close.rolling(200, min_periods=50).mean().iloc[-1]) if len(close) >= 50 else None

    s = 0
    # trend vs SMA
    if sma50 and price > sma50:
        s += 15
    elif sma50 and price < sma50:
        s -= 15
    if sma200 and price > sma200:
        s += 20
    elif sma200 and price < sma200:
        s -= 20
    if sma50 and sma200:
        if sma50 > sma200:
            s += 10
        else:
            s -= 10
    # RSI
    if r is not None:
        if r < 30:
            s += 15
        elif r < 40:
            s += 8
        elif r > 70:
            s -= 15
        elif r > 60:
            s -= 8
    # position in range
    if sup and res and res > sup:
        pos = (price - sup) / (res - sup)
        if pos > 0.85:
            s -= 10
        elif pos < 0.15:
            s += 10

    s = max(-100, min(100, int(s)))
    return {
        "score": s,
        "price": round(price, 2),
        "rsi": r,
        "atr": a,
        "sup": round(sup, 2) if sup else None,
        "res": round(res, 2) if res else None,
        "sma50": round(sma50, 2) if sma50 else None,
        "sma200": round(sma200, 2) if sma200 else None,
    }


def bias_label(score: int) -> tuple[str, str]:
    if score >= 40:
        return "صعودی قوی", "pos"
    if score >= 15:
        return "متمایل به صعود", "pos"
    if score > -15:
        return "خنثی / بدون جهت مشخص", "neu"
    if score > -40:
        return "متمایل به نزول", "neg"
    return "نزولی قوی", "neg"


# ── news (optional, graceful) ────────────────────────────────────────────

def fetch_news(hours: int = 36) -> dict | None:
    """Lightweight Google News RSS for gold. Fails soft."""
    try:
        import feedparser
        import urllib.parse

        q = urllib.parse.quote("gold OR XAU OR GC=F OR طلا")
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        if not feed.entries:
            return None

        now = utcnow().timestamp()
        top = []
        for e in feed.entries[:40]:
            title = getattr(e, "title", "") or ""
            link = getattr(e, "link", "") or ""
            source = ""
            if hasattr(e, "source") and getattr(e.source, "title", None):
                source = e.source.title
            published = getattr(e, "published_parsed", None)
            age_h = None
            if published:
                try:
                    age_h = round((now - time.mktime(published)) / 3600, 2)
                except Exception:
                    age_h = None
            if age_h is not None and age_h > hours:
                continue

            # very crude keyword bias
            tlow = title.lower()
            score = 0
            for w in ("rate cut", "fed pivot", "geopolit", "war", "etf inflow", "central bank buy"):
                if w in tlow:
                    score += 2
            for w in ("rate hike", "hawkish", "strong dollar", "cpi hot", "etf outflow"):
                if w in tlow:
                    score -= 2

            top.append({
                "title": title[:180],
                "link": link,
                "source": source or "News",
                "age_h": age_h,
                "score": score,
            })
            if len(top) >= 8:
                break

        if not top:
            return None
        bias = int(sum(x["score"] for x in top) * 8)
        bias = max(-100, min(100, bias))
        return {
            "bias": bias,
            "conf": "متوسط" if abs(bias) < 40 else "بالا",
            "n": len(feed.entries),
            "top": top,
            "note": "حدس خبری فقط از تیترهاست و می‌تواند با واکنش واقعی بازار فرق کند.",
        }
    except Exception as e:
        print("news failed:", e)
        return None



# ── candle direction bias per timeframe ─────────────────────────────────

def last_candle_pattern(df: pd.DataFrame) -> str | None:
    """Very light pattern tag for the last closed-ish bar."""
    if df is None or len(df) < 3:
        return None
    o, h, l, c = (float(df[x].iloc[-1]) for x in ("Open", "High", "Low", "Close"))
    prev_o, prev_c = float(df["Open"].iloc[-2]), float(df["Close"].iloc[-2])
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    upper = h - max(o, c)
    lower = min(o, c) - l
    if body / rng < 0.15:
        return "دوجی/رنج"
    # bullish engulfing
    if c > o and prev_c < prev_o and c >= prev_o and o <= prev_c:
        return "اینگالف صعودی"
    # bearish engulfing
    if c < o and prev_c > prev_o and c <= prev_o and o >= prev_c:
        return "اینگالف نزولی"
    if lower > 2 * body and upper < body:
        return "پین بار صعودی"
    if upper > 2 * body and lower < body:
        return "پین بار نزولی"
    if c > o:
        return "کندل صعودی"
    if c < o:
        return "کندل نزولی"
    return None


def candle_bias_tf(df: pd.DataFrame, tf: str, label: str) -> dict:
    info = score_tf(df)
    score = info.get("score") or 0
    reasons = []
    side = "range"
    conf = "پایین"

    price, sma50, sma200 = info.get("price"), info.get("sma50"), info.get("sma200")
    r = info.get("rsi")
    pat = last_candle_pattern(df)

    pts = 0
    if sma50 and price:
        if price > sma50:
            pts += 1
            reasons.append("بالای SMA50")
        else:
            pts -= 1
            reasons.append("زیر SMA50")
    if sma200 and price:
        if price > sma200:
            pts += 1
            reasons.append("بالای SMA200")
        else:
            pts -= 1
            reasons.append("زیر SMA200")
    if r is not None:
        if r < 35:
            pts += 1
            reasons.append(f"RSI={r} (نزدیک اشباع فروش)")
        elif r > 65:
            pts -= 1
            reasons.append(f"RSI={r} (نزدیک اشباع خرید)")
        else:
            reasons.append(f"RSI={r}")
    if pat:
        reasons.append(pat)
        if "صعودی" in pat:
            pts += 1
        elif "نزولی" in pat:
            pts -= 1

    if pts >= 2:
        side, conf = "up", "بالا" if pts >= 3 else "متوسط"
    elif pts <= -2:
        side, conf = "down", "بالا" if pts <= -3 else "متوسط"
    else:
        side, conf = "range", "پایین" if abs(pts) == 0 else "متوسط"

    side_fa = {"up": "صعودی", "down": "نزولی", "range": "رنج/خنثی"}[side]
    return {
        "tf": tf,
        "label": label,
        "side": side,
        "side_fa": side_fa,
        "conf": conf,
        "score": score,
        "why": " · ".join(reasons[:4]) if reasons else "داده کافی نیست",
        "price": info.get("price"),
        "rsi": info.get("rsi"),
        "atr": info.get("atr"),
        "sup": info.get("sup"),
        "res": info.get("res"),
    }



# ── session (Tehran-aware labels, UTC logic) ─────────────────────────────

def market_session(now: datetime | None = None) -> dict:
    """Gold futures are nearly 24h; label major FX overlap windows in UTC."""
    now = now or utcnow()
    h = now.hour  # UTC
    # Rough windows:
    # Asia: 00-07 UTC, London: 07-12, London+NY: 12-16, NY: 16-21, thin: 21-24
    if 7 <= h < 12:
        key, fa, good = "london", "لندن", True
    elif 12 <= h < 16:
        key, fa, good = "london_ny", "لندن + نیویورک", True
    elif 16 <= h < 21:
        key, fa, good = "ny", "نیویورک", True
    elif 0 <= h < 7:
        key, fa, good = "asia", "آسیا", False
    else:
        key, fa, good = "thin", "کم‌نقد (پایان نیویورک)", False
    return {
        "key": key,
        "label_fa": fa,
        "trade_friendly": good,
        "utc_hour": h,
        "note": "در سشن لندن/نیویورک نقدشوندگی معمولاً بهتر است." if good else "خارج سشن اصلی؛ قدرت سیگنال یک پله کم می‌شود.",
    }


# ── economic calendar with weekday heuristics ────────────────────────────

def build_calendar(now: datetime | None = None) -> list[dict]:
    """
    Heuristic high-impact calendar for gold.
    Not a substitute for ForexFactory; marks likely windows and lock status.
    """
    now = now or utcnow()
    # weekday: Mon=0 ... Sun=6
    wd = now.weekday()
    hour = now.hour

    events = []

    # NFP: first Friday of month ~12:30 UTC
    if wd == 4 and now.day <= 7:
        dist = 12.5 - (hour + now.minute / 60)
        if abs(dist) <= 3:
            status = "imminent"
        elif -1 <= dist <= 6:
            status = "today"
        else:
            status = "watch"
        events.append({
            "title": "NFP / گزارش اشتغال آمریکا",
            "impact": "high",
            "gold_bias": "mixed",
            "gold_note": "نوسان اول خبر شدید است. اشتغال قوی معمولاً فشار روی طلا.",
            "when_fa": "اولین جمعه ماه · حدود ۱۶:۰۰ تهران",
            "status": status,
            "lock": status in ("imminent", "today") and abs(dist) <= 2,
        })
    else:
        events.append({
            "title": "NFP / گزارش اشتغال آمریکا",
            "impact": "high",
            "gold_bias": "mixed",
            "gold_note": "اشتغال قوی → فشار روی طلا؛ ضعیف → حمایت نسبی.",
            "when_fa": "معمولاً اولین جمعه ماه",
            "status": "watch",
            "lock": False,
        })

    # CPI often mid-month around 12:30 UTC
    if 10 <= now.day <= 16 and wd < 5:
        dist = 12.5 - (hour + now.minute / 60)
        status = "imminent" if abs(dist) <= 2 else ("today" if wd < 5 and abs(dist) < 8 else "watch")
        events.append({
            "title": "CPI / تورم آمریکا",
            "impact": "high",
            "gold_bias": "neg",
            "gold_note": "CPI بالاتر از انتظار معمولاً به نفع دلار و فشار روی طلاست.",
            "when_fa": "معمولاً میانه ماه · حدود ۱۶:۰۰ تهران",
            "status": status,
            "lock": status == "imminent",
        })
    else:
        events.append({
            "title": "CPI / تورم آمریکا",
            "impact": "high",
            "gold_bias": "neg",
            "gold_note": "CPI بالاتر از انتظار معمولاً فشار روی طلا؛ پایین‌تر حمایت نسبی.",
            "when_fa": "رویداد دوره‌ای — زمان دقیق را از تقویم فارکس چک کن",
            "status": "watch",
            "lock": False,
        })

    # FOMC / Fed: roughly every 6 weeks; mark Wednesdays mid-month-ish as caution
    if wd == 2 and 12 <= now.day <= 22:
        events.append({
            "title": "احتمال تصمیم/کنفرانس فدرال رزرو (FOMC)",
            "impact": "high",
            "gold_bias": "neg",
            "gold_note": "روز FOMC نوسان بالاست. قبل و حین اعلام معمولاً صبر منطقی است.",
            "when_fa": "چهارشنبه‌های دوره‌ای · عصر به وقت تهران",
            "status": "today",
            "lock": hour >= 16 or hour < 2,
        })
    else:
        events.append({
            "title": "تصمیم نرخ بهره / فدرال رزرو",
            "impact": "high",
            "gold_bias": "neg",
            "gold_note": "لحن hawkish معمولاً برای طلا منفی؛ dovish مثبت.",
            "when_fa": "رویداد دوره‌ای — زمان دقیق را از تقویم فارکس چک کن",
            "status": "watch",
            "lock": False,
        })

    return events


def calendar_lock(calendar: list[dict]) -> tuple[bool, str | None]:
    for ev in calendar:
        if ev.get("lock"):
            return True, ev.get("title")
        if ev.get("status") == "imminent":
            return True, ev.get("title")
    return False, None


# ── signal history (last 5) ──────────────────────────────────────────────

HIST_KEY = "signal_history"
HIST_MAX = 5


def load_history(old: dict) -> list[dict]:
    h = old.get(HIST_KEY) or []
    if not isinstance(h, list):
        return []
    return h[-HIST_MAX:]


def update_history(old_hist: list[dict], decision: dict, price: float | None, now_iso: str) -> list[dict]:
    side = decision.get("side") or "wait"
    # only record when actionable or explicit wait after a previous signal change
    entry = {
        "at": now_iso,
        "side": side,
        "title": decision.get("title"),
        "strength": decision.get("strength"),
        "price": price,
        "head": (decision.get("head") or "")[:120],
    }
    hist = list(old_hist)
    if hist and hist[-1].get("side") == side and hist[-1].get("title") == entry["title"]:
        # same decision — refresh price/time on last row only if > 6h apart skip
        try:
            prev_t = hist[-1].get("at")
            if prev_t:
                age_h = (datetime.fromisoformat(now_iso.replace("Z", "+00:00")) -
                         datetime.fromisoformat(prev_t.replace("Z", "+00:00"))).total_seconds() / 3600
                if age_h < 6:
                    hist[-1] = entry
                    return hist[-HIST_MAX:]
        except Exception:
            pass
    hist.append(entry)
    return hist[-HIST_MAX:]


# ── final decision: buy / sell / wait ───────────────────────────────────

def decide(reps: list[dict], candle_bias: list[dict], news: dict | None,
           calendar: list[dict], session: dict) -> dict:
    daily = next((r for r in reps if r.get("tf") == "1d"), reps[0] if reps else {})
    d_bias = next((b for b in candle_bias if b["tf"] == "1d"), None)
    h_bias = next((b for b in candle_bias if b["tf"] == "1h"), None)
    w_bias = next((b for b in candle_bias if b["tf"] == "1wk"), None)
    m30_bias = next((b for b in candle_bias if b["tf"] == "30m"), None)
    m15_bias = next((b for b in candle_bias if b["tf"] == "15m"), None)
    m30_rep = next((r for r in reps if r.get("tf") == "30m"), None)

    news_bias = (news or {}).get("bias") or 0
    reasons = []
    blockers = []

    d_side = (d_bias or {}).get("side") or "range"
    h_side = (h_bias or {}).get("side") or "range"
    w_side = (w_bias or {}).get("side") or "range"
    m30_side = (m30_bias or {}).get("side") or "range"
    m15_side = (m15_bias or {}).get("side") or "range"

    buy_pts = 0
    sell_pts = 0

    if d_side == "up":
        buy_pts += 2
        reasons.append("تمایل روزانه صعودی")
    elif d_side == "down":
        sell_pts += 2
        reasons.append("تمایل روزانه نزولی")
    else:
        blockers.append("روزانه رنج/خنثی است")

    if h_side == "up":
        buy_pts += 1
        reasons.append("تمایل ساعتی صعودی")
    elif h_side == "down":
        sell_pts += 1
        reasons.append("تمایل ساعتی نزولی")

    # weekly soft filter
    if w_side == "down" and buy_pts > 0:
        buy_pts -= 1
        blockers.append("هفتگی نزولی است — بای سخت‌گیرانه‌تر")
    if w_side == "up" and sell_pts > 0:
        sell_pts -= 1
        blockers.append("هفتگی صعودی است — سل سخت‌گیرانه‌تر")

    if news_bias >= 25:
        buy_pts += 1
        reasons.append("تیتر اخبار نسبتاً حمایتی")
    elif news_bias <= -25:
        sell_pts += 1
        reasons.append("تیتر اخبار نسبتاً فشارنده")

    price, atr_v, sup, res = daily.get("price"), daily.get("atr"), daily.get("sup"), daily.get("res")
    if price and res and atr_v and (res - price) < 0.4 * atr_v:
        blockers.append(f"نزدیک مقاومت ({res}) — برای بای نامناسب")
        buy_pts -= 1
    if price and sup and atr_v and (price - sup) < 0.4 * atr_v:
        blockers.append(f"نزدیک حمایت ({sup}) — برای سل نامناسب")
        sell_pts -= 1

    rsi_d = daily.get("rsi")
    if rsi_d is not None:
        if rsi_d > 68:
            blockers.append(f"RSI روزانه بالاست ({rsi_d})")
            buy_pts -= 1
        if rsi_d < 32:
            blockers.append(f"RSI روزانه پایین است ({rsi_d})")
            sell_pts -= 1

    # short-term timing check (15m + 30m): they do NOT add points,
    # they only confirm or warn about the entry timing.
    if m15_side == "up" and m30_side == "up":
        reasons.append("۱۵ و ۳۰ دقیقه‌ای هر دو صعودی‌اند")
        if sell_pts >= 3:
            blockers.append("۱۵ و ۳۰ دقیقه‌ای هر دو صعودی‌اند — زمان‌بندی ورود سل مناسب نیست")
            sell_pts -= 1
    elif m15_side == "down" and m30_side == "down":
        reasons.append("۱۵ و ۳۰ دقیقه‌ای هر دو نزولی‌اند")
        if buy_pts >= 3:
            blockers.append("۱۵ و ۳۰ دقیقه‌ای هر دو نزولی‌اند — زمان‌بندی ورود بای مناسب نیست")
            buy_pts -= 1

    # session penalty
    if not session.get("trade_friendly"):
        blockers.append(f"سشن: {session.get('label_fa')} — نقدشوندگی کمتر")
        buy_pts -= 1
        sell_pts -= 1

    # hard calendar lock
    locked, lock_title = calendar_lock(calendar)
    if locked:
        blockers.append(f"قفل خبر: {lock_title}")
        buy_pts = 0
        sell_pts = 0

    # soft calendar caution
    if any(ev.get("status") in ("today", "imminent") for ev in calendar) and not locked:
        blockers.append("امروز/نزدیک رویداد مهم تقویم — احتیاط")

    side = "wait"
    strength = "—"
    title = "صبر"
    head = "شرایط برای ورود تازه به اندازه کافی هم‌راستا نیست."

    if session.get("closed"):
        side, title = "wait", "بازار بسته"
        head = session.get("closed_note") or "بازار بسته است؛ ورود جدید پیشنهاد نمی‌شود."
    elif locked:
        side, title = "wait", "صبر (قفل خبر)"
        head = f"نزدیک رویداد مهم ({lock_title}). تا آرام شدن نوسان بعد خبر، ورود جدید پیشنهاد نمی‌شود."
    elif buy_pts >= 3 and buy_pts > sell_pts and not any("مقاومت" in b for b in blockers):
        side = "buy"
        strength = "متوسط" if buy_pts == 3 else "قوی"
        if not session.get("trade_friendly"):
            strength = "ضعیف"
        title = "بای"
        head = "تمایل کلی به سمت صعود است؛ فقط با حد ضرر و حجم مشخص وارد شو."
    elif sell_pts >= 3 and sell_pts > buy_pts and not any("حمایت" in b for b in blockers):
        side = "sell"
        strength = "متوسط" if sell_pts == 3 else "قوی"
        if not session.get("trade_friendly"):
            strength = "ضعیف"
        title = "سل"
        head = "تمایل کلی به سمت نزول است؛ فقط با حد ضرر و حجم مشخص وارد شو."
    else:
        side, title = "wait", "صبر"
        if buy_pts > sell_pts:
            head = "کمی متمایل به صعود است ولی فیلترها برای بای کافی نیستند."
        elif sell_pts > buy_pts:
            head = "کمی متمایل به نزول است ولی فیلترها برای سل کافی نیستند."
        else:
            head = "جهت‌ها هم‌راستا نیستند یا نزدیک سطح حساس/خبر/سشن نامناسب هستیم."

    entry = sl = tp1 = tp2 = None
    inv = None
    rr = None
    if price and atr_v and side in ("buy", "sell"):
        if side == "buy":
            entry = price
            sl = round(price - 1.5 * atr_v, 2)
            tp1 = round(price + 1.5 * atr_v, 2)
            tp2 = round(price + 3.0 * atr_v, 2)
            inv = f"بسته شدن پایدار زیر {sl}"
            rr = "1 : 1 تا 1 : 2"
        else:
            entry = price
            sl = round(price + 1.5 * atr_v, 2)
            tp1 = round(price - 1.5 * atr_v, 2)
            tp2 = round(price - 3.0 * atr_v, 2)
            inv = f"بسته شدن پایدار بالای {sl}"
            rr = "1 : 1 تا 1 : 2"

    # short-term levels based on the 30-minute ATR (closer, for intraday traders)
    short = None
    atr30 = (m30_rep or {}).get("atr")
    p30 = (m30_rep or {}).get("price")
    if atr30 and p30 and side in ("buy", "sell"):
        k = 1 if side == "buy" else -1
        short = {
            "basis": "بر پایه نوسان ۳۰ دقیقه‌ای (ATR)",
            "entry": round(p30, 2),
            "sl": round(p30 - k * 1.5 * atr30, 2),
            "tp1": round(p30 + k * 1.5 * atr30, 2),
            "tp2": round(p30 + k * 3.0 * atr30, 2),
        }

    all_reasons = reasons + [f"⚠ {b}" for b in blockers]
    return {
        "side": side,
        "strength": strength,
        "title": title,
        "head": head,
        "reasons": all_reasons[:8],
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "invalidation": inv,
        "short": short,
        "locked": locked,
        "session": session.get("label_fa"),
        "note": "این تصمیم قاعده‌محور و آموزشی است؛ سیگنال تضمینی یا سفارش خودکار نیست.",
    }


# ── pending-order planner (sell/buy limit at range edges, 30m) ──────────

def pending_plans(reps: list[dict], candle_bias: list[dict]) -> dict | None:
    """
    Builds sell-limit / buy-limit plans at the edges of the recent 30m range.
    SL is placed beyond the edge by a buffer based on 30m ATR (not glued to
    the extreme). A plan is shown only if reward/risk to target 1 is >= 1.5.
    """
    m30 = next((r for r in reps if r.get("tf") == "30m"), None)
    if not m30:
        return None
    sup, res, a, price = m30.get("sup"), m30.get("res"), m30.get("atr"), m30.get("price")
    if not (sup and res and a and price) or res <= sup:
        return None

    d_bias = next((b for b in candle_bias if b["tf"] == "1d"), None)
    d_side = (d_bias or {}).get("side") or "range"

    mid = (res + sup) / 2
    buf_in = 0.2 * a
    buf_sl = 0.5 * a
    plans = []

    def tag(plan_side: str) -> str:
        if d_side == "range":
            return "روزانه خنثی است"
        if (plan_side == "sell" and d_side == "down") or (plan_side == "buy" and d_side == "up"):
            return "هم‌جهت با روند روزانه"
        return "خلاف روند روزانه — با احتیاط"

    # sell limit near the top
    e = res - buf_in
    sl = res + buf_sl
    risk = sl - e
    tp1, tp2 = mid, sup + buf_in
    if risk > 0 and (e - tp1) / risk >= 1.5:
        plans.append({
            "side": "sell",
            "title": "سل‌لیمیت (نزدیک سقف رنج)",
            "trend": tag("sell"),
            "entry": round(e, 2), "sl": round(sl, 2),
            "tp1": round(tp1, 2), "tp2": round(tp2, 2),
            "rr": round((e - tp1) / risk, 1),
            "away": round(e - price, 2),
            "cancel": f"اگر کندل ۳۰ دقیقه‌ای بالای {round(sl, 2)} بسته شد، سفارش را بردار.",
        })

    # buy limit near the bottom
    e = sup + buf_in
    sl = sup - buf_sl
    risk = e - sl
    tp1, tp2 = mid, res - buf_in
    if risk > 0 and (tp1 - e) / risk >= 1.5:
        plans.append({
            "side": "buy",
            "title": "بای‌لیمیت (نزدیک کف رنج)",
            "trend": tag("buy"),
            "entry": round(e, 2), "sl": round(sl, 2),
            "tp1": round(tp1, 2), "tp2": round(tp2, 2),
            "rr": round((tp1 - e) / risk, 1),
            "away": round(price - e, 2),
            "cancel": f"اگر کندل ۳۰ دقیقه‌ای زیر {round(sl, 2)} بسته شد، سفارش را بردار.",
        })

    return {
        "basis": "رنج اخیر چارت ۳۰ دقیقه (حدود ۲۰ ساعت)",
        "price": round(price, 2),
        "sup": round(sup, 2), "res": round(res, 2), "mid": round(mid, 2),
        "atr": round(a, 2),
        "plans": plans,
        "note": "اگر قیمت از نقطه‌ی ورود گذشته، دنبالش نرو؛ منتظر برگشت بمان. اعداد GC=F با XAUUSD کارگزاری چند دلار فرق دارند.",
    }


# ── dollar index quick read (display only, not part of the score) ───────

def fetch_dxy() -> dict | None:
    """Dollar index from Yahoo (DX-Y.NYB). Fails soft."""
    try:
        import yfinance as yf

        h = yf.Ticker("DX-Y.NYB").history(period="3mo", interval="1d", auto_adjust=True)
        if h is None or len(h) < 25:
            return None
        c = h["Close"].dropna()
        last = float(c.iloc[-1])
        sma20 = float(c.tail(20).mean())
        prev5 = float(c.iloc[-6])
        chg5 = (last / prev5 - 1) * 100
        if last > sma20 and chg5 > 0.3:
            side, fa, note = "up", "صعودی", "دلار قوی است؛ معمولاً فشار روی طلا."
        elif last < sma20 and chg5 < -0.3:
            side, fa, note = "down", "نزولی", "دلار ضعیف است؛ معمولاً حمایت برای طلا."
        else:
            side, fa, note = "flat", "خنثی", "دلار جهت مشخصی ندارد."
        return {
            "price": round(last, 2),
            "chg5": round(chg5, 2),
            "side": side,
            "side_fa": fa,
            "note": note + " فقط برای اطلاع؛ در امتیاز و سیگنال لحاظ نشده.",
        }
    except Exception as e:
        print("dxy failed:", e)
        return None


# ── market closed (weekend) detection ───────────────────────────────────

def market_closed_info(now: datetime) -> dict:
    """Gold futures: closed from Friday 17:00 ET to Sunday 18:00 ET."""
    try:
        et = now.astimezone(ZoneInfo("America/New_York"))
        wd = et.weekday()  # Mon=0 … Sun=6
        h = et.hour + et.minute / 60
        closed = (wd == 4 and h >= 17) or wd == 5 or (wd == 6 and h < 18)
        if not closed:
            return {"closed": False}
        days = (6 - wd) % 7
        nxt = (et + timedelta(days=days)).replace(hour=18, minute=0, second=0, microsecond=0)
        t = nxt.astimezone(ZoneInfo("Asia/Tehran")).strftime("%Y-%m-%d %H:%M")
        return {
            "closed": True,
            "reopen_tehran": t,
            "closed_note": f"بازار طلای آتی بسته است و داده‌ها قدیمی‌اند. بازگشایی حدود {t} به وقت تهران.",
        }
    except Exception as e:
        print("market_closed_info failed:", e)
        return {"closed": False}


# ── outcome tracking (did the signals / plans hit target or stop?) ──────

TRACK_MAX = 60
COOLDOWN_H = 2
PLAN_EXPIRE_H = 24
OPEN_EXPIRE_H = 48
ACTIVE = ("pending", "open")


def evaluate_item(item: dict, df: pd.DataFrame | None, now: datetime) -> dict:
    """Walk candles after creation. Conservative: if one candle touches both
    stop and target, it counts as a stop."""
    if item.get("status") not in ACTIVE:
        return item
    try:
        created = pd.Timestamp(item["created_at"])
        if created.tzinfo is None:
            created = created.tz_localize("UTC")
        status = item["status"]
        side, e, sl, tp = item["side"], item["entry"], item["sl"], item["tp1"]

        if df is not None and len(df):
            idx = df.index
            if getattr(idx, "tz", None) is None:
                idx = idx.tz_localize("UTC")
            bars = df[idx > created]
            for ts, r in bars.iterrows():
                hi, lo = float(r["High"]), float(r["Low"])
                if status == "pending":
                    filled = hi >= e if side == "sell" else lo <= e
                    if not filled:
                        continue
                    status = "open"
                    item["filled_at"] = iso(ts)
                hit_stop = hi >= sl if side == "sell" else lo <= sl
                hit_tp = lo <= tp if side == "sell" else hi >= tp
                if hit_stop:
                    status = "stop"
                    item["closed_at"] = iso(ts)
                    break
                if hit_tp:
                    status = "target"
                    item["closed_at"] = iso(ts)
                    break

        if status in ACTIVE:
            age_h = (now - created.to_pydatetime()).total_seconds() / 3600
            if status == "pending" and age_h > PLAN_EXPIRE_H:
                status = "expired"
            elif status == "open" and age_h > OPEN_EXPIRE_H:
                status = "expired"
        item["status"] = status
    except Exception as ex:
        print("evaluate_item failed:", ex)
    return item


def _recent_or_active(items: list[dict], kind: str, side: str, now: datetime) -> bool:
    for it in items:
        if it.get("kind") != kind or it.get("side") != side:
            continue
        if it.get("status") in ACTIVE:
            return True
        try:
            t = datetime.fromisoformat(str(it.get("created_at")).replace("Z", "+00:00"))
            if (now - t).total_seconds() / 3600 < COOLDOWN_H:
                return True
        except Exception:
            pass
    return False


def update_tracking(old_tracking: dict | None, decision: dict | None, pending: dict | None,
                    dfs: dict, now: datetime, closed: bool) -> dict:
    items = [dict(x) for x in ((old_tracking or {}).get("items") or []) if isinstance(x, dict)]
    df = dfs.get("15m")
    if df is None:
        df = dfs.get("30m")
    items = [evaluate_item(it, df, now) for it in items]

    if not closed:
        now_iso = iso(now)
        stamp = int(now.timestamp())

        sh = (decision or {}).get("short")
        side = (decision or {}).get("side")
        if side in ("buy", "sell") and sh and not _recent_or_active(items, "signal", side, now):
            items.append({
                "id": f"signal-{side}-{stamp}", "kind": "signal", "side": side,
                "entry": sh["entry"], "sl": sh["sl"], "tp1": sh["tp1"],
                "created_at": now_iso, "status": "open",
            })

        for pl in (pending or {}).get("plans") or []:
            pside = pl.get("side")
            if pl.get("away") is None or pl["away"] < 0:
                continue  # price already passed the entry — do not chase / track
            if _recent_or_active(items, "plan", pside, now):
                continue
            items.append({
                "id": f"plan-{pside}-{stamp}", "kind": "plan", "side": pside,
                "entry": pl["entry"], "sl": pl["sl"], "tp1": pl["tp1"],
                "created_at": now_iso, "status": "pending",
            })

    items = items[-TRACK_MAX:]
    n = lambda s: sum(1 for it in items if it.get("status") == s)
    target, stop = n("target"), n("stop")
    closed_n = target + stop
    stats = {
        "target": target,
        "stop": stop,
        "expired": n("expired"),
        "active": n("pending") + n("open"),
        "closed": closed_n,
        "win_rate": round(100 * target / closed_n) if closed_n else None,
    }
    return {"items": items, "stats": stats, "since": items[0]["created_at"] if items else None}


# ── candles for chart ────────────────────────────────────────────────────

def _label(ts, intraday: bool) -> str:
    if not intraday:
        return ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
    try:
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("Asia/Tehran").strftime("%m-%d %H:%M")
    except Exception:
        return str(ts)[:16]


def build_candles(df: pd.DataFrame, n: int = 120, intraday: bool = False) -> list[dict]:
    if df is None or len(df) < 10:
        return []
    t = df.tail(max(n + 50, 200)).copy()
    t["s50"] = t["Close"].rolling(50, min_periods=1).mean()
    t["s200"] = t["Close"].rolling(200, min_periods=1).mean()
    rows = []
    for ts, r in t.tail(n).iterrows():
        rows.append({
            "t": _label(ts, intraday),
            "o": float(r["Open"]),
            "h": float(r["High"]),
            "l": float(r["Low"]),
            "c": float(r["Close"]),
            "s50": float(r["s50"]) if not math.isnan(r["s50"]) else None,
            "s200": float(r["s200"]) if not math.isnan(r["s200"]) else None,
        })
    return rows


# ── main build ───────────────────────────────────────────────────────────

def build() -> dict:
    last_err = None
    dfs = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Yahoo fetch attempt {attempt}/{MAX_RETRIES}")
            dfs = fetch_yahoo()
            break
        except Exception as e:
            last_err = e
            print(f"fetch failed: {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES:
                delay = BASE_DELAY * (2 ** (attempt - 1))
                print(f"retry in {delay}s...")
                time.sleep(delay)
            else:
                raise last_err

    assert dfs is not None
    now = utcnow()

    reps = []
    for tf, label in TF_LIST:
        if tf not in dfs:
            continue
        info = score_tf(dfs[tf])
        info["tf"] = tf
        info["label"] = label
        reps.append(info)

    if not reps:
        raise RuntimeError("No usable timeframes after fetch")

    # long-term tech score: only 1h / 1d / 1wk (unchanged behaviour)
    main_scores = [r["score"] for r in reps if r["tf"] in MAIN_TFS]
    if not main_scores:
        main_scores = [r["score"] for r in reps]
    tech = int(round(sum(main_scores) / len(main_scores)))

    news = fetch_news()
    news_bias = news["bias"] if news else 0
    # overall: 75% tech + 25% news
    overall = int(round(0.75 * tech + 0.25 * news_bias))
    overall = max(-100, min(100, overall))

    # last bar time from the finest available series
    last_bar_at = None
    bar_tz = "نامشخص"
    for key in ("15m", "30m", "1h", "1d", "1wk"):
        if key in dfs and len(dfs[key]):
            idx = dfs[key].index[-1]
            last_bar_at = iso(idx)
            bar_tz = str(getattr(idx, "tz", None) or "UTC/unknown")
            break

    label_txt, label_cls = bias_label(overall)

    # per-TF candle bias
    candle_bias = []
    for tf, label in TF_LIST:
        if tf in dfs:
            candle_bias.append(candle_bias_tf(dfs[tf], tf, label))

    session = market_session(now)
    session.update(market_closed_info(now))
    calendar = build_calendar(now)
    decision = decide(reps, candle_bias, news, calendar, session)
    pending = None if session.get("closed") else pending_plans(reps, candle_bias)
    tracking = update_tracking(load_old().get("tracking"), decision, pending, dfs, now, bool(session.get("closed")))
    dxy = fetch_dxy()
    old_hist = load_history(load_old())
    price_now = next((r.get("price") for r in reps if r.get("tf") == "1d"), None)

    simple = {
        "key": decision["side"] if decision["side"] != "buy" else "buy",
        "title": decision["title"],
        "head": decision["head"],
        "confidence": decision.get("strength") or "—",
        "why": [],
    }
    # map reasons for old UI card
    for r in decision.get("reasons") or []:
        if r.startswith("⚠"):
            simple["why"].append(["bad", r.replace("⚠ ", "")])
        elif any(x in r for x in ("صعودی", "حمایتی")):
            simple["why"].append(["good", r])
        elif any(x in r for x in ("نزولی", "فشارنده")):
            simple["why"].append(["bad", r])
        else:
            simple["why"].append(["info", r])
    if not simple["why"]:
        simple["why"].append(["info", "دادهٔ کافی برای جزئیات بیشتر نبود."])

    # chart candles per timeframe (daily kept as "candles" for compatibility)
    candles_by_tf = {}
    for tf in ("15m", "30m"):
        if tf in dfs:
            rows = build_candles(dfs[tf], 100, intraday=True)
            if rows:
                candles_by_tf[tf] = rows
    daily_rows = build_candles(dfs.get("1d"))
    if daily_rows:
        candles_by_tf["1d"] = daily_rows

    data = {
        "updated": iso(now),
        "status": "ok",
        "error": None,
        "error_at": None,
        "symbol": SYMBOL,
        "demo": False,
        "overall": overall,
        "tech": tech,
        "label": label_txt,
        "label_cls": label_cls,
        "market_data": {
            "provider": "Yahoo Finance",
            "symbol": SYMBOL,
            "instrument": "نماد پیوستهٔ قرارداد آتی طلا (GC=F)؛ جانشین تقریبی قیمت، نه XAU/USD نقدی و نه نرخ کارگزاری",
            "retrieved_at": iso(now),
            "last_bar_at": last_bar_at,
            "bar_timezone": bar_tz,
            "delay_note": "دادهٔ عمومی ممکن است تأخیری باشد و برای اجرای معاملهٔ زنده قابل اتکا نیست.",
        },
        "reps": reps,
        "news": news,
        "simple": simple,
        "candle_bias": candle_bias,
        "calendar": calendar,
        "decision": decision,
        "pending": pending,
        "tracking": tracking,
        "dxy": dxy,
        "candles": daily_rows,
        "candles_by_tf": candles_by_tf,
        "session": session,
        "signal_history": update_history(old_hist, decision, price_now, iso(now)),
        "meta": {
            "builder": "gold-agent-clean",
            "version": "1.5.0",
        },
    }
    return data


def main() -> None:
    old = load_old()
    try:
        data = build()
    except Exception as e:
        traceback.print_exc()
        err_at = iso(utcnow())
        # Do NOT advance "updated" on total failure — UI will correctly show stale + error
        if old:
            old["error"] = f"{type(e).__name__}: {e}"
            old["error_at"] = err_at
            old["status"] = "error"
            write_json(old)
        else:
            write_json({
                "updated": None,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "error_at": err_at,
                "demo": False,
                "symbol": SYMBOL,
            })
        sys.exit(1)

    write_json(data)
    print(
        "ok", data["updated"],
        "overall", data["overall"],
        "last_bar", (data.get("market_data") or {}).get("last_bar_at"),
    )


if __name__ == "__main__":
    main()
