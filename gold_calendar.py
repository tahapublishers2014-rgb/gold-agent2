# -*- coding: utf-8 -*-
"""تقویم اقتصادی (رویدادهای دلاری با اثر زیاد) از فید رایگان ForexFactory-compatible؛ اگر در دسترس نبود حذف می‌شود."""
import json
import re
import urllib.request
from datetime import datetime, timezone, timedelta

FEEDS = ["https://nfs.faireconomy.media/ff_calendar_thisweek.json",
         "https://nfs.faireconomy.media/ff_calendar_nextweek.json"]
TEHRAN = timezone(timedelta(hours=3, minutes=30))

# (الگو، نام ساده، توضیح ساده، hot) ؛ hot = اثر «عدد بالاتر از پیش‌بینی» روی طلا: -1 یعنی طلا پایین، +1 یعنی طلا بالا، 0 مبهم
INFO = [
    (r"non-?farm|nfp", "اشتغال آمریکا (NFP)", "تعداد شغل‌های جدید ماه گذشته در آمریکا.", -1),
    (r"unemployment rate", "نرخ بیکاری آمریکا", "درصد بیکاران.", 1),
    (r"unemployment claims|jobless", "بیمه‌ی بیکاری هفتگی", "تعداد تازه‌بیکارها در هفته.", 1),
    (r"adp", "اشتغال بخش خصوصی (ADP)", "پیش‌نمای گزارش NFP.", -1),
    (r"core cpi|cpi", "تورم مصرف‌کننده (CPI)", "سرعت گران‌شدن کالاها و خدمات.", -1),
    (r"ppi", "تورم تولیدکننده (PPI)", "پیش‌نمای CPI.", -1),
    (r"pce", "تورم PCE", "شاخص تورم موردعلاقه‌ی Fed.", -1),
    (r"fomc statement|federal funds rate|interest rate", "تصمیم نرخ بهره‌ی Fed", "مهم‌ترین رویداد برای طلا؛ لحن بیانیه از خود عدد مهم‌تر است.", -1),
    (r"fomc press|powell|fed chair", "سخنرانی یا کنفرانس رئیس Fed", "هر جمله می‌تواند بازار را تکان دهد.", 0),
    (r"fomc.*minutes|meeting minutes", "صورت‌جلسه‌ی Fed", "نشان می‌دهد اعضا درباره‌ی نرخ چه فکر می‌کردند.", 0),
    (r"ism|pmi", "شاخص مدیران خرید (PMI/ISM)", "بالای ۵۰ یعنی رشد اقتصاد و زیر ۵۰ یعنی انقباض.", -1),
    (r"gdp", "رشد اقتصادی (GDP)", "اندازه‌ی رشد اقتصاد آمریکا.", -1),
    (r"retail sales", "فروش خرده‌فروشی", "میزان خرج مصرف‌کننده‌ها.", -1),
    (r"jolts|job openings", "فرصت‌های شغلی (JOLTS)", "تعداد شغل‌های باز.", -1),
    (r"treasury|auction|bond", "حراج اوراق خزانه", "تقاضای ضعیف یعنی بازده بالا می‌رود و به طلا فشار می‌آید.", 0),
]


def _info(title):
    for pat, name, expl, hot in INFO:
        if re.search(pat, title, re.I):
            return name, expl, hot
    return None


def _hot_text(hot):
    if hot < 0:
        return "اگر عدد واقعی بالاتر از پیش‌بینی بیاید معمولاً دلار بالا می‌رود و طلا پایین می‌آید؛ پایین‌تر از پیش‌بینی برعکس."
    if hot > 0:
        return "اگر عدد واقعی بالاتر از پیش‌بینی بیاید معمولاً به نفع طلا است؛ پایین‌تر از پیش‌بینی علیه طلا."
    return "جهت اثرش به محتوای خبر بستگی دارد؛ لحن نرم به نفع طلا و لحن سخت علیه آن است."


def fetch_events(now=None, hours_ahead=96, hours_back=4, timeout=20):
    now = now or datetime.now(timezone.utc)
    events = []
    got = False
    for u in FEEDS:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                rows = json.loads(r.read().decode())
            got = True
        except Exception:
            continue
        for x in rows:
            try:
                if x.get("country") != "USD" or x.get("impact") not in ("High",):
                    continue
                t = datetime.fromisoformat(x["date"]).astimezone(timezone.utc)
                dh = (t - now).total_seconds() / 3600
                if not (-hours_back <= dh <= hours_ahead):
                    continue
                inf = _info(x.get("title", ""))
                name, expl, hot = inf if inf else (x.get("title", ""), "رویداد مهم اقتصادی آمریکا.", 0)
                events.append(dict(title=x.get("title", ""), name=name, explain=expl, rule=_hot_text(hot),
                                   time_utc=t.isoformat(), time_tehran=t.astimezone(TEHRAN).strftime("%Y-%m-%d %H:%M"),
                                   hours=round(dh, 2), forecast=x.get("forecast") or "", previous=x.get("previous") or ""))
            except Exception:
                continue
    if not got:
        return None
    uniq = {}
    for e in events:
        uniq[(e["title"], e["time_utc"])] = e
    events = sorted(uniq.values(), key=lambda e: e["hours"])
    return events[:10]
