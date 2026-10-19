# -*- coding: utf-8 -*-
"""تشخیص الگوهای کندلی با توضیح ساده و در نظر گرفتن «کجا» و «بعد از چه روندی» ظاهر شده‌اند.
الگوی کندلی بدون محل مناسب (کنار حمایت/مقاومت و بعد از یک حرکت) ارزش کمی دارد؛ این ماژول همین را اعمال می‌کند."""
import numpy as np
import pandas as pd

TF_NAME = {"1h": "۱ ساعته", "4h": "۴ ساعته", "1d": "روزانه", "1wk": "هفتگی"}


def _atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    v = tr.rolling(n, min_periods=5).mean().iloc[-1]
    return float(v) if v == v else None


def resample4h(df1h):
    o = df1h.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    return o


def _parts(o, h, l, c):
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    return body, rng, min(o, c) - l, h - max(o, c)


def _trend_before(cl, i, atr):
    """جهت حرکت پنج کندل قبل از کندل i : -1 نزولی، +1 صعودی، 0 بی‌جهت."""
    if i < 6:
        return 0
    d = cl[i - 1] - cl[i - 6]
    if d <= -0.8 * atr:
        return -1
    if d >= 0.8 * atr:
        return 1
    return 0


def _scan(df, i, atr, sup, res):
    """الگوهایی که آخرین کندلشان اندیس i است."""
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    out = []
    n = len(c)
    if i < 3 or i >= n:
        return out
    body, rng, lw, uw = _parts(o[i], h[i], l[i], c[i])
    tr = _trend_before(c, i, atr)
    near_sup = sup is not None and abs(l[i] - sup) <= 0.6 * atr
    near_res = res is not None and abs(h[i] - res) <= 0.6 * atr
    bull = c[i] > o[i]
    bear = c[i] < o[i]

    def add(key, name, bias, strength, meaning, ctx_ok=True, at_level=False):
        if not ctx_ok:
            return
        s = strength + (1 if at_level else 0)
        out.append(dict(key=key, name=name, bias=bias, strength=int(min(3, s)), meaning=meaning,
                        at_level=bool(at_level)))

    # ---- تک‌کندلی
    if rng >= 0.5 * atr:
        if lw >= 2 * body and uw <= 0.2 * rng and body >= 0.04 * rng:
            add("hammer", "چکش (Hammer / پین‌بار صعودی)", 1, 1,
                "فروشنده‌ها قیمت را خیلی پایین بردند ولی خریدارها همه‌اش را برگرداندند. اگر بعد از ریزش و کنار حمایت بیاید، نشانه‌ی احتمالی ته‌گیری است.",
                ctx_ok=(tr <= 0), at_level=near_sup)
        if uw >= 2 * body and lw <= 0.2 * rng and body >= 0.04 * rng:
            add("star", "ستاره‌ی دنباله‌دار (Shooting Star / پین‌بار نزولی)", -1, 1,
                "خریدارها قیمت را بالا بردند ولی فروشنده‌ها همه‌اش را پایین کشیدند. اگر بعد از صعود و کنار مقاومت بیاید، نشانه‌ی احتمالی سقف است.",
                ctx_ok=(tr >= 0), at_level=near_res)
        if body <= 0.1 * rng:
            add("doji", "دوجی (Doji)", 0, 1,
                "قیمت باز و بسته شد تقریباً در یک نقطه؛ یعنی خریدار و فروشنده مساوی‌اند و بازار مردد است. به‌تنهایی جهت نمی‌دهد؛ کندل بعدی مهم است.")
    if body >= 0.85 * rng and rng >= 1.0 * atr:
        add("marubozu", "کندل پرقدرت بدون سایه (Marubozu)", 1 if bull else -1, 2,
            ("خریدارها از اول تا آخر کندل مسلط بودند." if bull else "فروشنده‌ها از اول تا آخر کندل مسلط بودند.") +
            " نشانه‌ی شتاب یک‌طرفه است، ولی بعد از چند کندل مشابه احتمال استراحت یا اصلاح بالا می‌رود.")

    # ---- دوکندلی
    if i >= 1:
        pb, pr, _, _ = _parts(o[i - 1], h[i - 1], l[i - 1], c[i - 1])
        pbull, pbear = c[i - 1] > o[i - 1], c[i - 1] < o[i - 1]
        big_prev = pb >= 0.25 * atr
        if bull and pbear and big_prev and o[i] <= c[i - 1] and c[i] >= o[i - 1] and body >= pb:
            add("bull_engulf", "پوشای صعودی (Bullish Engulfing)", 1, 2,
                "کندل صعودی امروز کل کندل نزولی دیروز را پوشاند؛ یعنی خریدارها کنترل را از فروشنده‌ها گرفتند.",
                ctx_ok=(tr <= 0), at_level=near_sup)
        if bear and pbull and big_prev and o[i] >= c[i - 1] and c[i] <= o[i - 1] and body >= pb:
            add("bear_engulf", "پوشای نزولی (Bearish Engulfing)", -1, 2,
                "کندل نزولی امروز کل کندل صعودی دیروز را پوشاند؛ یعنی فروشنده‌ها کنترل را از خریدارها گرفتند.",
                ctx_ok=(tr >= 0), at_level=near_res)
        if bull and pbear and big_prev and o[i] < c[i - 1] and o[i - 1] > c[i] > (o[i - 1] + c[i - 1]) / 2:
            add("piercing", "نفوذی صعودی (Piercing Line)", 1, 1,
                "بعد از یک کندل نزولی، قیمت پایین‌تر باز شد ولی بیش از نصف کندل قبل را پس گرفت؛ خریدار دارد برمی‌گردد.",
                ctx_ok=(tr <= 0), at_level=near_sup)
        if bear and pbull and big_prev and o[i] > c[i - 1] and o[i - 1] < c[i] < (o[i - 1] + c[i - 1]) / 2:
            add("dark_cloud", "ابر تیره (Dark Cloud Cover)", -1, 1,
                "بعد از یک کندل صعودی، قیمت بالاتر باز شد ولی بیش از نصف کندل قبل را از دست داد؛ فروشنده دارد وارد می‌شود.",
                ctx_ok=(tr >= 0), at_level=near_res)
        if abs(l[i] - l[i - 1]) <= 0.1 * atr and pbear and bull:
            add("tweezer_bot", "کف دوقلو (Tweezer Bottom)", 1, 1,
                "دو کندل پشت‌سرهم کف تقریباً یکسانی ساختند و قیمت نتوانست پایین‌تر برود؛ یعنی آنجا خریدار هست.",
                ctx_ok=(tr <= 0), at_level=near_sup)
        if abs(h[i] - h[i - 1]) <= 0.1 * atr and pbull and bear:
            add("tweezer_top", "سقف دوقلو (Tweezer Top)", -1, 1,
                "دو کندل پشت‌سرهم سقف تقریباً یکسانی ساختند و قیمت نتوانست بالاتر برود؛ یعنی آنجا فروشنده هست.",
                ctx_ok=(tr >= 0), at_level=near_res)
        if h[i] < h[i - 1] and l[i] > l[i - 1] and pr >= 0.8 * atr:
            add("inside", "کندل داخلی (Inside Bar)", 0, 1,
                "کندل کاملاً داخل کندل قبلی ماند؛ بازار فشرده شده. معمولاً بعد از آن شکست به یک طرف رخ می‌دهد. جهت را همان شکست مشخص می‌کند.")

    # ---- سه‌کندلی
    if i >= 2:
        b0, r0, _, _ = _parts(o[i - 2], h[i - 2], l[i - 2], c[i - 2])
        b1, r1, _, _ = _parts(o[i - 1], h[i - 1], l[i - 1], c[i - 1])
        big0 = b0 >= 0.5 * atr
        if big0 and c[i - 2] < o[i - 2] and b1 <= 0.35 * b0 and bull and c[i] > (o[i - 2] + c[i - 2]) / 2:
            add("morning", "ستاره‌ی صبحگاهی (Morning Star)", 1, 2,
                "یک کندل نزولی بزرگ، بعد یک کندل کوچک مردد، بعد یک کندل صعودی قوی. این سه‌تایی بعد از ریزش یکی از معروف‌ترین نشانه‌های ته‌گیری است.",
                ctx_ok=(tr <= 0), at_level=near_sup)
        if big0 and c[i - 2] > o[i - 2] and b1 <= 0.35 * b0 and bear and c[i] < (o[i - 2] + c[i - 2]) / 2:
            add("evening", "ستاره‌ی شامگاهی (Evening Star)", -1, 2,
                "یک کندل صعودی بزرگ، بعد یک کندل کوچک مردد، بعد یک کندل نزولی قوی. این سه‌تایی بعد از صعود یکی از معروف‌ترین نشانه‌های سقف است.",
                ctx_ok=(tr >= 0), at_level=near_res)
        mn = 0.4 * atr
        if all(c[k] > o[k] and (c[k] - o[k]) >= mn for k in (i - 2, i - 1, i)) and c[i] > c[i - 1] > c[i - 2]:
            add("soldiers", "سه سرباز سفید (Three White Soldiers)", 1, 2,
                "سه کندل صعودی قوی پشت‌سرهم و هرکدام بالاتر از قبلی بسته شد؛ نشانه‌ی ورود جدی خریدارها.")
        if all(c[k] < o[k] and (o[k] - c[k]) >= mn for k in (i - 2, i - 1, i)) and c[i] < c[i - 1] < c[i - 2]:
            add("crows", "سه کلاغ سیاه (Three Black Crows)", -1, 2,
                "سه کندل نزولی قوی پشت‌سرهم و هرکدام پایین‌تر از قبلی بسته شد؛ نشانه‌ی فروش جدی.")
    return out


def describe(df, atr):
    """توصیف ساده‌ی کندل آخرِ بسته‌شده."""
    r = df.iloc[-2] if len(df) > 2 else df.iloc[-1]
    o, h, l, c = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"])
    body, rng, lw, uw = _parts(o, h, l, c)
    size = "بزرگ" if rng >= 1.2 * atr else ("کوچک" if rng < 0.5 * atr else "معمولی")
    col = "صعودی" if c > o else ("نزولی" if c < o else "خنثی")
    pos = (c - l) / rng
    if pos >= 0.7:
        close = "نزدیک سقف بسته شد (خریدار تا آخر مسلط بود)"
    elif pos <= 0.3:
        close = "نزدیک کف بسته شد (فروشنده تا آخر مسلط بود)"
    else:
        close = "وسط محدوده بسته شد (بدون برتری روشن)"
    wick = ""
    if lw >= 0.4 * rng:
        wick = "؛ سایه‌ی پایینی بلند دارد (پایین‌ها رد شدند)"
    elif uw >= 0.4 * rng:
        wick = "؛ سایه‌ی بالایی بلند دارد (بالاها رد شدند)"
    return f"آخرین کندل بسته‌شده: {col} و {size}، {close}{wick}."


def analyze(dfs, reps):
    """dfs: {tf: DataFrame}  reps: فهرست گزارش هر تایم‌فریم (برای ATR و حمایت/مقاومت)."""
    by_tf = {r["tf"]: r for r in reps}
    work = dict(dfs)
    if "1h" in dfs and "4h" not in work:
        try:
            work["4h"] = resample4h(dfs["1h"])
        except Exception:
            pass
    res = []
    score = 0.0
    wt = {"1wk": 2.0, "1d": 1.5, "4h": 1.0, "1h": 0.6}
    for tf in ("1wk", "1d", "4h", "1h"):
        df = work.get(tf)
        if df is None or len(df) < 30:
            continue
        rep = by_tf.get(tf) or by_tf.get("1d") or {}
        atr = rep.get("atr") if tf != "4h" else None
        atr = atr or _atr(df)
        if not atr:
            continue
        sup, res_ = rep.get("sup"), rep.get("res")
        pats = []
        for i, when in ((len(df) - 2, "کندل بسته‌شده"), (len(df) - 1, "کندل جاری (هنوز کامل نیست)")):
            for p in _scan(df, i, atr, sup, res_):
                p = dict(p)
                p["when"] = when
                p["closed"] = when.startswith("کندل بسته")
                pats.append(p)
        pats.sort(key=lambda p: (-p["strength"], not p["closed"]))
        pats = pats[:4]
        for p in pats:
            if p["closed"]:
                score += p["bias"] * p["strength"] * wt[tf]
        res.append(dict(tf=tf, label=TF_NAME[tf], read=describe(df, atr), patterns=pats))
    bias = int(max(-100, min(100, round(score * 12))))
    if bias >= 25:
        summ = "کندل‌های بسته‌شده بیشتر نشانه‌ی برگشت یا ادامه‌ی صعود می‌دهند."
    elif bias <= -25:
        summ = "کندل‌های بسته‌شده بیشتر نشانه‌ی برگشت یا ادامه‌ی نزول می‌دهند."
    else:
        summ = "الگوی کندلی قوی و مشخصی دیده نمی‌شود؛ کندل‌ها تصمیم‌ساز نیستند."
    return dict(bias=bias, summary=summ, tfs=res)
