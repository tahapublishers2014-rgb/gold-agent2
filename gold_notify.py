# -*- coding: utf-8 -*-
"""هشدار تلگرام (اختیاری). فقط وقتی TELEGRAM_TOKEN و TELEGRAM_CHAT_ID تنظیم شده باشند کار می‌کند.
فقط در دو حالت پیام می‌دهد: تغییر حکم «الان چه کار کنم؟» و نزدیک‌شدن رویداد مهم (یک بار برای هر رویداد)."""
import json
import os
import urllib.parse
import urllib.request


def _send(text):
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        return False
    body = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendMessage", data=body), timeout=20) as r:
            return r.status == 200
    except Exception as e:
        print("telegram failed:", e)
        return False


def run(data, prev):
    """data: داده‌ی جدید؛ prev: داده‌ی قبلی (برای تشخیص تغییر). وضعیت را در data['_notify'] نگه می‌دارد."""
    st = dict((prev or {}).get("_notify") or {})
    data["_notify"] = st
    if data.get("demo") or not data.get("simple"):
        return
    S = data["simple"]
    price = next((r["price"] for r in data["reps"] if r["tf"] == "1d"), None)
    ptxt = f" · قیمت {price:,.0f}" if price else ""
    last = st.get("verdict")
    if last is not None and last != S["key"]:
        if _send(f"ایجنت طلا{ptxt}\nحکم عوض شد: {S['title']}\n{S['head']}\n(این توصیه‌ی قطعی نیست؛ حد ضرر فراموش نشود.)"):
            st["verdict"] = S["key"]
    elif last is None:
        st["verdict"] = S["key"]
    sent = set(st.get("events") or [])
    for e in (data.get("calendar") or []):
        k = e["title"] + e["time_utc"]
        if 0 < e["hours"] <= 1.0 and k not in sent:
            if _send(f"ایجنت طلا\nکمتر از یک ساعت دیگر: {e['name']} ({e['time_tehran'][5:]} تهران)\nقیمت ناگهانی حرکت می‌کند؛ قبل و بعدش وارد معامله نشو.\n{e['rule']}"):
                sent.add(k)
    st["events"] = sorted(sent)[-40:]
