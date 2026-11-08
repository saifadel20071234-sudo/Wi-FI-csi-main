# الفيرموير (Firmware) — حرق الكود على الـ ESP32

ده الفولدر اللي فيه كود الـ ESP32 اللي بيبعت الـ CSI للسيريال. الكود ده منقول من
مشروع [ESP-CSI](https://github.com/espressif/esp-csi) بتاع Espressif (ترخيص Apache-2.0)
— مثال `csi_recv_router` — وهو **بالظبط** السيناريو بتاعك: الـ ESP32 بيتصل بالراوتر،
ويبعت ping للراوتر، ويستقبل الـ CSI من رد الراوتر.

> المهم جدًا: **نفس الفيرموير بيتحرق على الاثنين ESP32** (مفيش فرق بين الشمال واليمين
> في الكود نفسه — الفرق بيتحدد من جهاز الكمبيوتر حسب الـ USB port).

---

## 0) المتطلبات قبل ما تبدأ

1. **ESP-IDF v5.x** (يفضل v5.3 أو أحدث) مثبت على جهازك. أسهل طريقة على ويندوز:
   - نزّل **Visual Studio Code** + إضافة **Espressif IDF**، أو
   - استخدم **ESP-IDF Windows Installer** من [docs.espressif.com](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/windows-setup.html).
   - بعد التثبيت، لازم تشغّل الأوامر من **"ESP-IDF Terminal"** (اللي بيجهّز `idf.py` في الـ PATH).

2. **درايفر الـ USB-to-UART** بتاع كابل الـ ESP32 (غالبًا CP2102 أو CH340). لو الجهاز
   مش ظاهر في Device Manager، نزّل الدرايفر المناسب.

---

## 1) ضبط اسم الشبكة والباسوورد

من الـ ESP-IDF Terminal:

```bat
cd C:\Users\Adel\Desktop\CSI\firmware\csi_recv_router
idf.py set-target esp32
idf.py menuconfig
```

في القايمة افتح:
```
Example Connection Configuration
```
وسجّل:
- `WiFi SSID`  = اسم شبكة الراوتر بتاعك
- `WiFi Password` = باسوورد الراوتر

> لو راوترك مفتوح من غير باسوورد، سيب الباسوورد فاضي.

> لو عندك ESP32-S3 مش ESP32 العادي، استخدم `idf.py set-target esp32s3`.

بعدها `S` للحفظ ثم `Q` للخروج.

---

## 2) البناء والحرق (Build & Flash)

وصل **أول ESP32** بالكابل، وحدد الـ COM port بتاعه (اعرفه من Device Manager، أو من
أمر `python src\main.py --list-ports`):

```bat
idf.py build
idf.py -p COM3 flash
```

> أول مرة `build` هينزّل مكوّن `esp_csi_gain_ctrl` من الإنترنت تلقائيًا (Component Manager)،
> فخليك متصل بالنت.

بعد ما يخلص، **افصل ووصّل التاني ESP32** واعمل نفس الحاجة (نفس الفيرموير):

```bat
idf.py -p COM4 flash
```

---

## 3) التأكد إنه شغال (اختياري)

اعمل monitor لواحد منهم:

```bat
idf.py -p COM3 monitor
```

المفروض تشوف سطور زي:

```
I (3331) example_connect: sta ip: 192.168.0.107 ...
I (4346) csi_recv_router: ================ CSI RECV ================
CSI_DATA,0,94:d9:b3:80:8c:81,-30,11,1,6,1,0,1,0,1,0,0,-93,0,13,2,2751923,0,67,0,128,1,"[67,48,4,...]"
```

يعني الـ ESP32 اتصل بالراوتر وبيطبع بيانات CSI كل شوية. اخرج بـ `Ctrl+]`.

> **مهم:** لازم تكون الراوتر على تردد **2.4 GHz** (الـ ESP32 العادي بيدعم 2.4 بس).
> لو راوترك dual-band، اسم الشبكة بتاع 2.4 GHz هو اللي تحطه في الـ SSID.

---

## 4) لو فيه مشكلة: "الـ serial بيعلّق" أو بيانات ناقصة

المعدل الافتراضي 100 ping في الثانية، وده ممكن يكون تقيل على السيريال في بعض الراوترات.
الحل: في `main\app_main.c` غيّر السطر:

```c
#define CONFIG_SEND_FREQUENCY      100
```

إلى:

```c
#define CONFIG_SEND_FREQUENCY      50
```

وبعدين أعد `idf.py build` و `idf.py -p COMx flash`.

---

## ملخص سريع

| الخطوة | الأمر |
|---|---|
| ضبط الهدف | `idf.py set-target esp32` |
| ضبط الواي فاي | `idf.py menuconfig` (Example Connection Configuration) |
| البناء | `idf.py build` |
| الحرق | `idf.py -p COM3 flash` (والتاني `idf.py -p COM4 flash`) |
| مراقبة | `idf.py -p COM3 monitor` |
