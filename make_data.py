# -*- coding: utf-8 -*-
"""
Gold Agent – data builder
Fetches GC=F from Yahoo Finance, computes basic technicals + optional news,
writes a single data.json with clear freshness fields.

Designed for GitHub Actions (every 20–30 min) + static GitHub Pages.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

OUT = "data.json"
SYMBOL = "GC=F"
MAX_RETRIES = 3
BASE_DELAY = 5


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


# ── candles for chart ────────────────────────────────────────────────────

def build_candles(df: pd.DataFrame, n: int = 120) -> list[dict]:
    if df is None or len(df) < 10:
        return []
    t = df.tail(max(n + 50, 200)).copy()
    t["s50"] = t["Close"].rolling(50, min_periods=1).mean()
    t["s200"] = t["Close"].rolling(200, min_periods=1).mean()
    rows = []
    for ts, r in t.tail(n).iterrows():
        rows.append({
            "t": ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10],
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
    for tf, label in (("1h", "ساعتی"), ("1d", "روزانه"), ("1wk", "هفتگی")):
        if tf not in dfs:
            continue
        info = score_tf(dfs[tf])
        info["tf"] = tf
        info["label"] = label
        reps.append(info)

    if not reps:
        raise RuntimeError("No usable timeframes after fetch")

    scores = [r["score"] for r in reps]
    tech = int(round(sum(scores) / len(scores)))
    daily = next((r for r in reps if r["tf"] == "1d"), reps[0])

    news = fetch_news()
    news_bias = news["bias"] if news else 0
    # overall: 75% tech + 25% news
    overall = int(round(0.75 * tech + 0.25 * news_bias))
    overall = max(-100, min(100, overall))

    # last bar time from the finest available series
    last_bar_at = None
    bar_tz = "نامشخص"
    for key in ("1h", "1d", "1wk"):
        if key in dfs and len(dfs[key]):
            idx = dfs[key].index[-1]
            last_bar_at = iso(idx)
            bar_tz = str(getattr(idx, "tz", None) or "UTC/unknown")
            break

    label_txt, label_cls = bias_label(overall)

    simple = {
        "key": "wait" if abs(overall) < 20 else ("buy" if overall > 0 else "sell"),
        "title": label_txt,
        "head": (
            "شرایط برای ورود تازه خیلی روشن نیست؛ صبر منطقی‌تر است."
            if abs(overall) < 20
            else (
                "نشانه‌ها بیشتر به سمت صعود متمایل‌اند؛ تأیید شکست مقاومت و مدیریت ریسک لازم است."
                if overall > 0
                else "نشانه‌ها بیشتر به سمت نزول متمایل‌اند؛ تأیید شکست حمایت و مدیریت ریسک لازم است."
            )
        ),
        "confidence": "بالا" if abs(overall) >= 40 else "متوسط" if abs(overall) >= 20 else "پایین",
        "why": [],
    }
    if daily.get("rsi") is not None:
        simple["why"].append([
            "info" if 40 <= daily["rsi"] <= 60 else ("good" if daily["rsi"] < 40 else "bad"),
            f"RSI روزانه ≈ {daily['rsi']}"
        ])
    if daily.get("sma50") and daily.get("price"):
        side = "بالای" if daily["price"] > daily["sma50"] else "زیر"
        simple["why"].append([
            "good" if daily["price"] > daily["sma50"] else "bad",
            f"قیمت {side} میانگین ۵۰ روزه ({daily['sma50']})"
        ])
    if news:
        simple["why"].append([
            "good" if news_bias > 10 else "bad" if news_bias < -10 else "info",
            f"جهت‌گیری تیتر اخبار: {news_bias:+d}"
        ])
    if not simple["why"]:
        simple["why"].append(["info", "دادهٔ کافی برای جزئیات بیشتر نبود."])

    data = {
        "updated": iso(now),                 # script run time (UTC)
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
            "last_bar_at": last_bar_at,       # actual last candle time
            "bar_timezone": bar_tz,
            "delay_note": "دادهٔ عمومی ممکن است تأخیری باشد و برای اجرای معاملهٔ زنده قابل اتکا نیست.",
        },
        "reps": reps,
        "news": news,
        "simple": simple,
        "candles": build_candles(dfs.get("1d")),
        "meta": {
            "builder": "gold-agent-clean",
            "version": "1.0.0",
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
