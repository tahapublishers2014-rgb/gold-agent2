# ایجنت طلا (نسخهٔ تمیز)

داشبورد استاتیک فارسی برای نماد پیوستهٔ آتی طلا `GC=F`.

- داده از **Yahoo Finance** (عمومی، ممکن است تأخیری باشد)
- اجرای خودکار با **GitHub Actions** هر ~۳۰ دقیقه
- UI فقط JSON منتشرشده را می‌خواند (با cache-bust قوی)
- جداسازی واضح: `updated` (زمان اجرای اسکریپت) و `last_bar_at` (زمان آخرین کندل)

> این ابزار تحلیل و سناریو است، نه مشاورهٔ مالی و نه سیگنال معاملاتی.

## ساختار

```
.
├── index.html                 # رابط کاربری
├── make_data.py               # ساخت data.json
├── data.json                  # خروجی (بعد از اولین اجرا)
├── requirements.txt
└── .github/workflows/update.yml
```

## راه‌اندازی روی GitHub Pages

1. یک ریپوی جدید بساز و این فایل‌ها را push کن.
2. **Settings → Actions → General → Workflow permissions** را روی **Read and write** بگذار.
3. **Settings → Pages** را روی branch `main` / root تنظیم کن.
4. از تب **Actions** یک‌بار workflow با نام `update-gold` را دستی Run کن.
5. بعد از commit شدن `data.json`، سایت روی  
   `https://<user>.github.io/<repo>/` در دسترس است.

### اجرای دستی داده

Actions → **update-gold** → **Run workflow**

دکمهٔ «تازه‌سازی صفحه» در UI فقط `data.json` منتشرشده را دوباره می‌خواند و workflow را اجرا نمی‌کند (امنیت: توکن در مرورگر گذاشته نمی‌شود).

## اجرای محلی

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python make_data.py
# سپس یک سرور استاتیک ساده:
python -m http.server 8080
# باز کن: http://localhost:8080
```

## فیلدهای مهم در `data.json`

| فیلد | معنی |
|------|------|
| `updated` | زمان اجرای موفق اسکریپت (UTC) |
| `market_data.last_bar_at` | زمان آخرین کندل دریافتی از Yahoo |
| `market_data.retrieved_at` | زمان بازیابی در همان اجرا |
| `status` | `ok` یا `error` |
| `error` / `error_at` | در صورت شکست جزئی/کامل |

در شکست کامل، `updated` جلو نمی‌رود تا UI درست «قدیمی» نشان دهد.

## محدودیت‌ها

- `GC=F` نماد پیوستهٔ آتی است، نه قیمت نقدی XAU/USD و نه نرخ اجرایی کارگزاری.
- دادهٔ Yahoo ممکن است تأخیری یا محدود شود.
- امتیازها قاعده‌محور و ساده هستند (SMA / RSI / موقعیت در رنج + تیتر خبر).
- هیچ سفارشی ثبت نمی‌شود و به بروکر وصل نیست.
