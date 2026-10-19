# -*- coding: utf-8 -*-
"""تحلیل کامل را اجرا و در data.json می‌نویسد (GitHub Actions هر ۲۰ دقیقه این را اجرا می‌کند)."""
import argparse
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import gold_feeds
import gold_agent as ga
from gold_pro import plan_lines

gold_feeds.install()
OUT = "data.json"


def clean(o):
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
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return o


def alerts(lst):
    return [{"lvl": a[0], "key": a[1], "text": a[2]} for a in lst]


def build(demo):
    ns = argparse.Namespace(symbol="GC=F", tf="1h,1d,1wk", demo=demo, no_macro=False,
                            no_news=False, no_llm=True, news_hours=36)
    reps, extra, news, overall = ga.run_once(ns)
    dfs = reps[0].pop("_dfs")
    pro = reps[0].pop("pro", None)
    scores = [r["score"] for r in reps]
    data = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "symbol": "XAU/USD (GC=F)",
        "demo": bool(demo),
        "overall": int(overall),
        "tech": int(np.mean(scores)),
        "reps": [{"tf": r["tf"], "label": ga.TF_FA.get(r["tf"], r["tf"]), "price": r["price"], "score": r["score"],
                  "classic": r["classic"], "atr": r["atr"], "sup": r["sup"], "res": r["res"], "rsi": r["rsi"],
                  "votes": [{"name": k, "v": v} for k, v in r["votes"].items()],
                  "alerts": alerts(r["alerts"])} for r in reps],
        "extra": alerts(extra),
        "news": None,
        "pro": None,
        "candles": [],
    }
    if news:
        data["news"] = {"bias": news["bias"], "conf": news["conf"], "n": news["n"], "n_imp": news["n_imp"],
                        "mode": news["mode"], "note": news.get("note"),
                        "top": [{"title": x["title"], "link": x.get("link", ""), "source": x["source"],
                                 "age_h": x["age_h"], "score": x["score"], "why": (x["why"] or [""])[0]}
                                for x in news["top"]],
                        "events": [x["title"] for x in news["events"]]}
        try:
            import gold_fundamental as gf
            data["news"]["fund"] = gf.analyze(news.get("scored") or news["top"], news["bias"])
        except Exception as e:  # تحلیل فاندامنتال نباید کل به‌روزرسانی را خراب کند
            print("fundamental failed:", e)
    if pro:
        data["pro"] = {"lines": plan_lines(pro), "adj": pro["adj"], "regime": pro["regime"], "tf": pro["tf"],
                       "plan": {k: v for k, v in pro["plan"].items() if k != "blockers"}}
    cand = sent = cal = None
    try:
        import gold_candles
        cand = gold_candles.analyze(dfs, reps)
        data["candles_pat"] = cand
    except Exception as e:
        print("candles failed:", e)
    try:
        import gold_sentiment
        if demo:
            sent = dict(heat=gold_sentiment.heat(dfs["1d"]) if "1d" in dfs else None, vix=None, cot=None,
                        note="ربات نمی‌تواند احساسات و روانشناسی جمع را بخواند. این‌ها فقط نشانه‌های غیرمستقیم‌اند.")
        else:
            sent = gold_sentiment.analyze(dfs, ga.fetch)
        data["sentiment"] = sent
    except Exception as e:
        print("sentiment failed:", e)
    try:
        import gold_calendar
        cal = None if demo else gold_calendar.fetch_events()
        data["calendar"] = cal
    except Exception as e:
        print("calendar failed:", e)
    try:
        import gold_simple
        data["simple"] = gold_simple.build(reps, pro, news["bias"] if news else 0, cand, sent, cal)
    except Exception as e:
        print("simple failed:", e)
        traceback.print_exc()
    d = dfs.get("1d")
    if d is not None:
        t = d.tail(250).copy()
        t["s50"] = t["Close"].rolling(50, min_periods=1).mean()
        t["s200"] = d["Close"].rolling(200, min_periods=1).mean().reindex(t.index)
        for ts, r in t.tail(120).iterrows():
            data["candles"].append({"t": ts.strftime("%Y-%m-%d"), "o": r["Open"], "h": r["High"], "l": r["Low"],
                                    "c": r["Close"], "s50": r["s50"], "s200": r["s200"]})
    return clean(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    try:
        data = build(a.demo)
    except Exception as e:
        traceback.print_exc()
        old = {}
        if os.path.exists(OUT):
            try:
                old = json.load(open(OUT, encoding="utf-8"))
            except Exception:
                pass
        old["error"] = f"{type(e).__name__}: {e}"
        old["error_at"] = datetime.now(timezone.utc).isoformat()
        json.dump(old, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
        sys.exit(0)  # صفحه با داده‌ی قبلی و هشدار خطا نمایش داده می‌شود
    try:
        import gold_notify
        prev = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
        gold_notify.run(data, prev)
    except Exception as e:
        print("notify failed:", e)
    json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print("ok", data["updated"], "overall", data["overall"])


if __name__ == "__main__":
    main()
