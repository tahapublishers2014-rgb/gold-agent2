# -*- coding: utf-8 -*-
"""خلاصه‌ی ساده برای مبتدی: حکم روشن، دلیل‌ها، سه سناریو، برنامه‌ی معامله به زبان ساده.
هیچ‌چیز اینجا «پیش‌بینی قطعی» نیست؛ ربات فقط می‌گوید شرایط برای کدام کار مناسب‌تر یا خطرناک‌تر است."""
import math


def _sign(x, thr=0):
    return 1 if x > thr else (-1 if x < -thr else 0)


def _fmt(x):
    return f"{x:,.0f}" if x is not None else "—"


APP_TERMS = [
    ("حمایت و مقاومت", "حمایت یعنی کفی که قیمت چند بار از آن برگشته و مقاومت یعنی سقفی که چند بار از آن پایین آمده. شکست هرکدام معمولاً حرکت جدید می‌سازد."),
    ("حد ضرر (Stop Loss)", "قیمتی که اگر به آن برسی، معامله را می‌بندی تا ضرر از آن بیشتر نشود. بدون حد ضرر وارد معامله نشو."),
    ("حد سود (Take Profit)", "قیمتی که اگر به آن برسی، معامله را می‌بندی و سود را برمی‌داری."),
    ("ریسک (R)", "فاصله‌ی ورود تا حد ضرر. «سود ۲R» یعنی دو برابر آنچه ریسک کرده‌ای سود کنی. اگر ۱۰۰ دلار ریسک کنی، ۱R یعنی ۱۰۰ دلار."),
    ("ATR (نوسان معمول)", "میانگین حرکت یک کندل در این بازه. مثلاً اگر ATR روزانه ۸۰ دلار باشد، حرکت‌های ۸۰ دلاری عادی است و حد ضرر ۱۰ دلاری با نوسان معمول می‌خورد."),
    ("لات (Lot)", "واحد حجم معامله. هر ۱ لات طلا = ۱۰۰ اونس. ۰٫۰۱ لات = ۱ اونس. هر ۱ دلار حرکت قیمت روی ۰٫۰۱ لات یعنی ۱ دلار سود یا ضرر."),
    ("RSI", "عددی بین ۰ تا ۱۰۰ که سرعت حرکت را نشان می‌دهد. بالای ۷۰ یعنی خریدِ زیاد (احتمال استراحت) و زیر ۳۰ یعنی فروشِ زیاد."),
    ("روند و رنج", "روند یعنی قیمت مسیر مشخصی (بالا یا پایین) دارد. رنج یعنی بین یک سقف و یک کف می‌چرخد. روش معامله در این دو حالت فرق می‌کند."),
    ("امتیاز", "عدد بین منفی ۱۰۰ تا مثبت ۱۰۰: صفر یعنی بی‌طرف، مثبت یعنی تمایل به صعود، منفی یعنی تمایل به نزول. عدد بزرگ‌تر یعنی نشانه‌ها هم‌جهت‌ترند، نه اینکه سود قطعی است."),
    ("تایم‌فریم", "مدت هر کندل: ۱ ساعته برای کوتاه‌مدت، روزانه برای میان‌مدت و هفتگی برای دید بلندمدت. جهت تایم‌فریم بزرگ‌تر معمولاً قوی‌تر است."),
]


def _blocker_plain(b, plan, daily):
    if "ضعیف" in b or "آستانه" in b:
        return "نشانه‌های تکنیکال قوی نیستند."
    if "رنج" in b:
        return "بازار بین یک سقف و یک کف می‌چرخد و شکست‌ها زیاد کاذب‌اند."
    if "اخبار" in b:
        return "اخبار خلاف جهت تکنیکال‌اند."
    if "سطح مقابل" in b:
        rr = plan.get("rr_level")
        tgt = daily.get("res") if plan.get("side") == "long" else daily.get("sup")
        if rr is not None and tgt:
            return (f"جلوی راه به‌زودی مانعی هست (سطح {_fmt(tgt)}). تا آنجا فقط حدود {rr:.1f} برابر ریسک سود داری؛ "
                    "یعنی به ازای هر ۱ دلار ریسک، سود احتمالی کمتر از ۱٫۵ دلار است و ارزش ریسک ندارد.")
    return b


def build(reps, pro, news_bias, candles, sentiment, calendar, equity=10000, risk_pct=1.0):
    daily = next((r for r in reps if r["tf"] == "1d"), reps[0])
    plan = (pro or {}).get("plan") or {}
    adj = int((pro or {}).get("adj", daily["score"]))
    reg = (pro or {}).get("regime") or {}
    nb = int(news_bias or 0)
    cb = int((candles or {}).get("bias", 0))
    tfs = {r["tf"]: r["score"] for r in reps}
    d = _sign(adj, 24)
    word = {1: "خرید", -1: "فروش", 0: ""}
    agree = sum(1 for s in tfs.values() if d and _sign(s, 14) == d)
    why, wait_for = [], []

    # ---- دلایل به زبان ساده
    if d == 0:
        why.append(("bad", f"نشانه‌های تکنیکال (امتیاز {adj:+d}) جهت روشنی نمی‌دهند."))
    else:
        side_fa = "صعودی" if d > 0 else "نزولی"
        why.append(("good", f"تکنیکال تمایل {side_fa} دارد (امتیاز {adj:+d} از ۱۰۰)."))
        if agree >= 2:
            why.append(("good", f"{agree} تایم‌فریم از ۳ تا هم‌جهت‌اند؛ هرچه تایم‌فریم‌ها بیشتر هم‌نظر باشند، سیگنال مطمئن‌تر است."))
        else:
            why.append(("bad", "فقط یکی از سه تایم‌فریم با این جهت هم‌نظر است؛ تایم‌فریم‌ها با هم تناقض دارند."))
    if nb >= 25:
        why.append(("good" if d >= 0 else "bad", f"اخبار به نفع طلاست (امتیاز {nb:+d})."))
    elif nb <= -25:
        why.append(("good" if d <= 0 else "bad", f"اخبار علیه طلاست (امتیاز {nb:+d})."))
    else:
        why.append(("info", "اخبار جهت روشنی نمی‌دهد."))
    if d and nb * d <= -25:
        why.append(("bad", "تکنیکال و خبر خلاف هم‌اند. وقتی این دو مخالف باشند، معمولاً بازار بی‌ثبات و احتمال موفقیت پایین‌تر است."))
    if candles and abs(cb) >= 25:
        c_dir = 1 if cb > 0 else -1
        why.append(("good" if (d == 0 or c_dir == d) else "bad", "الگوی کندل‌ها " + ("صعودی" if cb > 0 else "نزولی") + " است."))
    if reg.get("kind") == "range":
        why.append(("info", "بازار در حالت رنج است؛ ورود از وسط رنج بی‌دلیل ریسک دارد."))
    elif reg.get("kind") == "trend":
        why.append(("info", "بازار در حالت روند است؛ هم‌جهت‌شدن با روند ساده‌تر از مقابله با آن است."))
    heat = (sentiment or {}).get("heat")
    chase = False
    if heat and heat["tone"] == "warn":
        if (d > 0 and "خرید" in heat["label"]) or (d < 0 and "ترس" in heat["label"]):
            chase = True
            why.append(("bad", f"قیمت از قبل خیلی هیجانی حرکت کرده ({heat['label']}). دنبال‌کردن حرکت در این نقطه خطرناک است."))

    # ---- رویداد نزدیک
    ev_soon = None
    for e in (calendar or []):
        if -0.5 <= e["hours"] <= 1.0:
            ev_soon = e
            break
    next_ev = next((e for e in (calendar or []) if e["hours"] > 1.0), None)

    blockers = [_blocker_plain(b, plan, daily) for b in (plan.get("blockers") or [])]
    pts = (agree >= 2) + (nb * d >= 25 if d else 0) + (bool(d) and cb * d >= 25) + (reg.get("kind") == "trend")

    if ev_soon:
        key = "wait"
        title = "الان صبر کن"
        head = f"رویداد مهم «{ev_soon['name']}» نزدیک است (یا همین الان منتشر شده). قیمت در این دقایق ناگهانی و بی‌منطق حرکت می‌کند."
        why.insert(0, ("bad", "نوسان ناگهانی خبر مهم؛ حتی تحلیل خوب هم در این دقایق قابل‌اعتماد نیست."))
        wait_for.append("حدود ۳۰ دقیقه بعد از انتشار خبر، تا قیمت آرام بگیرد و جهت واقعی مشخص شود.")
    elif d == 0:
        key, title = "wait", "الان صبر کن"
        head = "نشانه‌ها جهت روشنی ندارند. بدون برتری مشخص، بهترین معامله گاهی همین «معامله‌نکردن» است."
    elif blockers or (nb * d <= -25):
        key, title = "wait", "الان صبر کن"
        head = "جهت تکنیکال " + ("صعودی" if d > 0 else "نزولی") + " است ولی چند مانع جدی برای ورود وجود دارد."
        for b in blockers:
            why.append(("bad", "مانع ورود: " + b))
    elif agree < 2 or chase:
        key, title = "wait", "صبر کن؛ هنوز تأیید کامل نیست"
        head = "جهت تکنیکال " + ("صعودی" if d > 0 else "نزولی") + " است ولی تأیید کافی ندارد."
    else:
        key = "buy" if d > 0 else "sell"
        title = f"{word[d]} فقط با احتیاط و حد ضرر"
        head = f"شرایط برای {word[d]} از حالت عادی مناسب‌تر است، ولی تضمینی در کار نیست. اگر وارد شدی، حتماً با حد ضرر و حجم کم."

    conf = "پایین" if pts <= 1 else ("متوسط" if pts == 2 else "بالا")
    if key == "wait":
        conf = "—"

    # ---- منتظر چه باشم
    sup, res = daily.get("sup"), daily.get("res")
    if key == "wait":
        if res:
            wait_for.append(f"بسته‌شدن یک کندل روزانه بالای {_fmt(res)}: نشانه‌ی شروع حرکت صعودی.")
        if sup:
            wait_for.append(f"بسته‌شدن یک کندل روزانه زیر {_fmt(sup)}: نشانه‌ی شروع حرکت نزولی.")
        if d and nb * d <= -25:
            wait_for.append("هم‌جهت‌شدن خبرها با تکنیکال.")
        if next_ev:
            wait_for.append(f"بعد از رویداد «{next_ev['name']}» ({next_ev['time_tehran'][5:]} به وقت تهران) جهت روشن‌تر می‌شود.")
    else:
        stop = plan.get("sl")
        if stop:
            wait_for.append(f"اگر قیمت به {_fmt(stop)} رسید، تحلیل شکست خورده؛ ضرر را بپذیر و بیرون بیا.")
        if next_ev:
            wait_for.append(f"قبل از «{next_ev['name']}» ({next_ev['time_tehran'][5:]} به وقت تهران) معامله باز نگه نداری یا حجم را کم کن.")

    # ---- سناریوها
    er = reg.get("er") or 0.2
    c = 0.6 * adj + 0.25 * nb + 0.15 * cb
    p_range = 0.20 + 0.30 * (1 - min(er / 0.5, 1))
    share = 0.5 + 0.18 * math.tanh(c / 45)
    p_up, p_dn = (1 - p_range) * share, (1 - p_range) * (1 - share)

    def r5(p):
        return int(round(p * 100 / 5) * 5)
    pu, pd_, pr = r5(p_up), r5(p_dn), r5(p_range)
    diff = 100 - (pu + pd_ + pr)
    pr += diff
    atr = daily.get("atr") or 0
    scen = [
        dict(key="up", title="صعود", prob=pu,
             text=(f"قیمت با کندل روزانه از {_fmt(res)} بالاتر می‌ماند و راه به سمت حدود {_fmt(res + 1.5 * atr)} باز می‌شود." if res else "قیمت ادامه‌ی صعود می‌دهد."),
             trigger=(f"نشانه‌ی شروع: بسته‌شدن روزانه بالای {_fmt(res)}" if res else "")),
        dict(key="range", title="چرخش بی‌جهت", prob=pr,
             text=(f"قیمت بین {_fmt(sup)} و {_fmt(res)} می‌ماند و بالا و پایین می‌رود. معامله‌گر تازه‌کار در این حالت بیشترین ضرر را با حد ضررهای پیاپی می‌خورد." if sup and res else "قیمت بی‌جهت می‌چرخد."),
             trigger="نشانه‌اش: کندل‌های کوچک و برگشت از سقف و کف"),
        dict(key="down", title="نزول", prob=pd_,
             text=(f"قیمت با کندل روزانه زیر {_fmt(sup)} می‌رود و راه به سمت حدود {_fmt(sup - 1.5 * atr)} باز می‌شود." if sup else "قیمت ادامه‌ی نزول می‌دهد."),
             trigger=(f"نشانه‌ی شروع: بسته‌شدن روزانه زیر {_fmt(sup)}" if sup else "")),
    ]
    scen.sort(key=lambda s: -s["prob"])

    # ---- برنامه‌ی ساده
    plain_plan = None
    if plan.get("side"):
        long_ = plan["side"] == "long"
        active = key in ("buy", "sell")
        rk, ent, sl = plan["risk"], plan["entry"], plan["sl"]
        plain_plan = dict(
            active=active, side="خرید" if long_ else "فروش",
            entry=ent, sl=sl, tp1=plan["tp1"], tp2=plan["tp2"], risk=rk,
            equity=equity, risk_pct=risk_pct,
            lots=plan["lots"],
            note=("این برنامه فقط زمانی کاربرد دارد که شرایط ورود فراهم باشد. الان فقط برای آشنایی با طرز فکر است و توصیه‌ی ورود نیست." if not active
                  else "اگر تصمیم گرفتی وارد شوی، این اعداد را بدون تغییر بگذار، مخصوصاً حد ضرر را."),
        )
    return dict(key=key, title=title, head=head, confidence=conf, why=why, wait_for=wait_for,
                scenarios=scen, plan=plain_plan, terms=[dict(t=a, d=b) for a, b in APP_TERMS],
                parts=dict(tech=adj, news=nb, candles=cb, agree=agree))
