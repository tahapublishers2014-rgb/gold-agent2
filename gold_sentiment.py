# -*- coding: utf-8 -*-
"""نشانه‌های غیرمستقیم احساس بازار. ربات روانشناسی جمع را نمی‌بیند؛ این‌ها فقط نشانه‌اند، نه پیش‌بینی.
هر منبعی که در دسترس نباشد بی‌صدا حذف می‌شود."""
import json
import urllib.request
import urllib.parse

import numpy as np
import pandas as pd

COT_URL = ("https://publicreporting.cftc.gov/resource/72hh-3qpy.json?"
           "cftc_contract_market_code=088691&$order=report_date_as_yyyy_mm_dd%20DESC&$limit=80")


def heat(daily):
    """«گرمای قیمت»: آیا حرکت اخیر بیش‌ازحد هیجانی بوده؟ فقط از خود قیمت محاسبه می‌شود."""
    c = daily["Close"]
    if len(c) < 60:
        return None
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = float(100 - 100 / (1 + up.iloc[-1] / max(dn.iloc[-1], 1e-9)))
    tr = pd.concat([daily["High"] - daily["Low"], (daily["High"] - c.shift()).abs(), (daily["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    ext = float((c.iloc[-1] - c.rolling(50).mean().iloc[-1]) / atr)
    streak = 0
    for v in d.iloc[::-1].values:
        if v > 0 and streak >= 0:
            streak += 1
        elif v < 0 and streak <= 0:
            streak -= 1
        else:
            break
    hi, lo = c.tail(60).max(), c.tail(60).min()
    pos = float((c.iloc[-1] - lo) / max(hi - lo, 1e-9))
    pts = 0
    pts += 2 if rsi >= 70 else (1 if rsi >= 62 else (-2 if rsi <= 30 else (-1 if rsi <= 38 else 0)))
    pts += 2 if ext >= 3 else (1 if ext >= 2 else (-2 if ext <= -3 else (-1 if ext <= -2 else 0)))
    pts += 1 if streak >= 5 else (-1 if streak <= -5 else 0)
    if pts >= 3:
        lab, txt, tone = "هیجان خرید زیاد", "قیمت در چند روز اخیر خیلی سریع بالا رفته و خریدارها هیجان‌زده‌اند. دنبال‌کردن حرکت در این نقطه پرریسک است؛ اصلاح یا استراحت محتمل‌تر از حالت عادی است.", "warn"
    elif pts <= -3:
        lab, txt, tone = "ترس و فروش هیجانی", "قیمت در چند روز اخیر خیلی سریع ریخته و بازار ترسیده. فروش در کف‌های هیجانی پرریسک است؛ برگشت کوتاه‌مدت محتمل‌تر از حالت عادی است.", "warn"
    elif pts >= 1:
        lab, txt, tone = "کمی گرم", "خریدارها فعال‌ترند ولی هنوز به حد افراط نرسیده است.", "info"
    elif pts <= -1:
        lab, txt, tone = "کمی سرد", "فروشنده‌ها فعال‌ترند ولی هنوز به حد افراط نرسیده است.", "info"
    else:
        lab, txt, tone = "متعادل", "نه هیجان خرید دیده می‌شود و نه ترس فروش.", "info"
    return dict(label=lab, text=txt, tone=tone, rsi=round(rsi, 1), ext_atr=round(ext, 1), streak=int(streak), pos60=round(pos, 2))


def vix(fetch):
    """VIX = «شاخص ترس» بازار سهام آمریکا."""
    try:
        v = fetch("^VIX", "1d")["Close"].dropna()
        last = float(v.iloc[-1])
        pct = float((v.tail(250) < last).mean())
        chg = float(last / v.iloc[-6] - 1) if len(v) > 6 else 0.0
        if last >= 30:
            txt = "ترس در بازار سهام بالاست. طلا معمولاً حمایت می‌شود، ولی در وحشت شدید گاهی همه‌چیز فروخته می‌شود تا نقد شود."
        elif last >= 20:
            txt = "ترس بالاتر از حالت عادی است؛ فضای مناسب‌تر برای طلا."
        elif last <= 14:
            txt = "بازار آرام و بی‌ترس است؛ پناهگاه امن کمتر موردتوجه است."
        else:
            txt = "ترس در حد معمولی است."
        return dict(value=round(last, 1), pct1y=round(pct, 2), chg5d=round(chg, 3), text=txt)
    except Exception:
        return None


def cot(timeout=20):
    """موقعیت صندوق‌های بزرگ (Managed Money) در قرارداد طلای COMEX، گزارش هفتگی CFTC."""
    try:
        req = urllib.request.Request(COT_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            rows = json.loads(r.read().decode())
        if not rows:
            return None
        net = []
        for x in rows:
            k = {a.lower(): b for a, b in x.items()}
            lg, sh = float(k["m_money_positions_long_all"]), float(k["m_money_positions_short_all"])
            net.append((k.get("report_date_as_yyyy_mm_dd", "")[:10], lg - sh, lg, sh))
        cur = net[0]
        hist = [n[1] for n in net[:52]]
        pct = float(np.mean([h < cur[1] for h in hist]))
        prev = net[1][1] if len(net) > 1 else cur[1]
        if pct >= 0.85:
            txt = "صندوق‌های بزرگ خیلی روی خرید طلا شلوغ شده‌اند (نسبت به یک سال اخیر در بالاترین محدوده). هرچه بازار یک‌طرفه‌تر باشد، اگر خبر بدی بیاید فروش دسته‌جمعی شدیدتر می‌شود."
            tone = "warn"
        elif pct <= 0.15:
            txt = "صندوق‌های بزرگ خیلی کم روی طلا خرید دارند (پایین‌ترین محدوده‌ی یک سال اخیر). اگر خبر خوب بیاید، جا برای خرید جدید زیاد است."
            tone = "info"
        else:
            txt = "موقعیت صندوق‌های بزرگ در محدوده‌ی عادی است و فشار خاصی از این بابت نیست."
            tone = "info"
        return dict(date=cur[0], net=int(cur[1]), change=int(cur[1] - prev), pct1y=round(pct, 2), text=txt, tone=tone)
    except Exception:
        return None


def analyze(dfs, fetch):
    out = dict(heat=None, vix=None, cot=None)
    d = dfs.get("1d")
    if d is not None:
        try:
            out["heat"] = heat(d)
        except Exception:
            pass
    out["vix"] = vix(fetch)
    out["cot"] = cot()
    out["note"] = ("ربات نمی‌تواند احساسات و روانشناسی جمع را بخواند. این‌ها فقط نشانه‌های غیرمستقیم‌اند و برای هشدار «افراط» به کار می‌روند، نه برای پیش‌بینی جهت.")
    return out
