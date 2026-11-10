# Arduino Firmware (ESP32 CSI)

هذا المجلد يحتوي سكتشين Arduino:

| المجلد | الوظيفة |
|---|---|
| `csi_scope_esp32/` | ESP32 #1 — اللينك الأفقي (X) |
| `csi_scope_esp32_2/` | ESP32 #2 — اللينك الرأسي (Y) |

## إعداد بيانات الواي فاي (مهم)

بيانات الشبكة **لم تعد مكتوبة داخل الكود**. كل سكتش بيقرأ من ملف `secrets.h`
الموجود بجواره، والملف ده مستثنى من Git.

قبل أول Build:

1. انسخ `secrets.h.example` إلى `secrets.h` داخل مجلد السكتش.
2. عدّل القيم:

```c
const char* PRIMARY_SSID = "YOUR_SSID";
const char* PRIMARY_PASS = "YOUR_PASSWORD";
```

3. لازم تكون الشبكة **2.4 GHz** (ESP32 لا يدعم 5 GHz).

> ملف `secrets.h` الحقيقي بيبقى محلي على جهازك فقط ومش هيتعمل له commit.

## صيغة البيانات على السيريال

السكتشين بيطبعوا نفس الصيغة (JSON array بفواصل):

```text
CSI_DATA,<seq>,<mac>,<rssi>,...,<channel>,...,<len>,<first_word>,"[v0,v1,v2,...]"
```

الـ parser في `src/csi_reader.py` بيقبل الصيغة دي وكمان الصيغة القديمة المفصولة
بمسافات، فمش هتحتاج تعديل لو عندك جهاز متبرمج بنسخة قديمة.
