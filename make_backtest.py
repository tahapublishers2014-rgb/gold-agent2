# -*- coding: utf-8 -*-
"""بک‌تست هفتگی روی ۱۰ سال داده‌ی روزانه‌ی طلا؛ خروجی backtest.json برای نمایش در اپ."""
import json
import math
import sys
import traceback
import urllib.request
from datetime import datetime, timezone

import numpy as np

import gold_agent as ga
import gold_backtest as bt


def get_df():
    url = "https://query1.finance.yahoo.com/v8/finance/chart/GC%3DF?range=10y&interval=1d"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            df = ga._parse_chart(json.loads(r.read().decode()))
        if len(df) > 600:
            return df
    except Exception as e:
        print("direct failed:", e)
    import yfinance as yf
    df = yf.download("GC=F", period="10y", interval="1d", progress=False, auto_adjust=True)
    if hasattr(df.columns, "droplevel") and getattr(df.columns, "nlevels", 1) > 1:
        df.columns = df.columns.droplevel(1)
    return df.rename(columns=str.title).dropna(subset=["Close"])


def st(r):
    s = bt._stats(r)
    return {k: (None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else (float(v) if not isinstance(v, int) else v)) for k, v in s.items()}


def main():
    df = get_df()
    res = bt.run(df, step=3, verbose=False)
    half = len(res) // 2
    out = dict(updated=datetime.now(timezone.utc).isoformat(), bars=len(df), points=len(res),
               start=str(df.index[0])[:10], end=str(df.index[-1])[:10],
               base_up=float((res["fwd"] > 0).mean()),
               adj=dict(all=st(res["r_adj"]), first=st(res["r_adj"].iloc[:half]), second=st(res["r_adj"].iloc[half:])),
               raw=dict(all=st(res["r_raw"])))
    names = sorted({n for v in res["votes"] for n in v})
    rows = []
    for n in names:
        v = np.array([x.get(n, 0) for x in res["votes"]])
        m = v != 0
        if m.sum() < 30:
            continue
        hit = float((np.sign(res["fwd"].values[m]) == v[m]).mean())
        z = (hit - 0.5) / math.sqrt(0.25 / m.sum())
        rows.append(dict(name=n, n=int(m.sum()), acc=hit, z=float(z)))
    out["styles"] = sorted(rows, key=lambda x: -x["z"])
    a = out["adj"]["all"]
    f, s2 = out["adj"]["first"], out["adj"]["second"]
    if a["n"] and a["exp"] is not None and a["exp"] > 0.05 and (a["t"] or 0) >= 2 and (f["exp"] or 0) > 0 and (s2["exp"] or 0) > 0:
        verdict = "در گذشته، سیگنال‌های قوی برتری آماری کوچکی داشته‌اند و در هر دو نیمه‌ی دوره مثبت بوده‌اند؛ ولی این تضمین آینده نیست."
    elif a["n"] and a["exp"] is not None and a["exp"] > 0:
        verdict = "نتیجه کمی مثبت است ولی از شانس قابل‌تشخیص نیست. برتری آماری ثابت نشده؛ فقط مثل یک فیلتر کمکی نگاهش کن."
    else:
        verdict = "در گذشته، سیگنال‌های ربات سود پایدار نداشته‌اند. به سیگنال تنها اعتماد نکن و از آن فقط برای ساختاردهی تحلیل خودت استفاده کن."
    out["verdict"] = verdict
    json.dump(out, open("backtest.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("ok", out["points"], verdict)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(0)
