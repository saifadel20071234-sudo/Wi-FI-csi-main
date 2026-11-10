# 📡 CSI — دليل المجلد

المجلد ده فيه كل كود النظام: المعالجة، الواجهات، الفيرموير، والتدريب.

> 📘 **الدليل الرئيسي:** [`../README.md`](../README.md) (English) ·
> [`../README.ar.md`](../README.ar.md) (العربية) ·
> [`../PRESENTATION_DOCUMENTATION.md`](../PRESENTATION_DOCUMENTATION.md) (شرح تقني مفصّل).

---

## ⚡ تشغيل سريع

```bat
:: من مجلد CSI
.venv\Scripts\activate
pip install -r requirements.txt

:: محاكاة المتتبع 2D (بدون هاردوير)
python src\dual_link_tracker.py --demo

:: محاكاة شاشة الرصد والعدّاد
python src\scope_monitor.py --demo

:: تشغيل حقيقي
python src\dual_link_tracker.py --port1 COM8 --port2 COM9
```

---

## 🗂️ محتويات المجلد

| المسار | الوظيفة |
|---|---|
| `config.json` | الغرفة، الأجهزة، معالجة الإشارة، وإعدادات `tracker` |
| `requirements.txt` | المكتبات المطلوبة |
| `src/dual_link_tracker.py` | متتبع 2D بالخريطة الحرارية (لينكين متعامدين) |
| `src/main.py` | خط المعالجة الكامل (جهاز واحد / عدة أجهزة + RTI) |
| `src/scope_monitor.py` | شاشة رصد الإشارة وعدّاد الأشخاص |
| `src/csi_reader.py` | قراءة السيريال، فك CSI، وإعادة الاتصال التلقائية |
| `src/processor.py` | الحضور والحركة والاتجاه والتقدير المبدئي للعدد |
| `src/occupancy.py` | الخريطة الحرارية وكشف الـ blobs بـ OpenCV |
| `src/rti.py` | Radio Tomographic Imaging (3 أجهزة أو أكثر) |
| `src/kalman.py` | فلتر Kalman لتنعيم المسار |
| `src/count_model.py` | استخراج الخصائص وتوقّع العدد (Random Forest) |
| `src/simulator.py` | مصدر محاكاة لوضع الديمو |
| `src/visualizer.py` | واجهة OpenCV |
| `src/config.py` | تحميل الإعدادات واكتشاف المنافذ |
| `firmware/` | فيرموير ESP-IDF + دليل الحرق |
| `firmware_arduino/` | سكتشات Arduino + `secrets.h.example` |
| `assets/tracker_preview.png` | صورة معاينة الواجهة |
| `train_count_model.py` | تدريب موديل عدّ الأشخاص |
| `collect_count_data.py` | جمع بيانات تدريب موسومة |
| `count_data.csv` | بيانات نموذجية |
| `extracted_research/` | سكربتات ونتائج التقييم والتقارير |

---

## 🔐 الفيرموير وبيانات الشبكة

بيانات الواي فاي مش في الكود. لكل سكتش:

```bat
cd firmware_arduino\csi_scope_esp32
copy secrets.h.example secrets.h
```

وبعدين عدّل `PRIMARY_SSID` و`PRIMARY_PASS`. التفاصيل في
[`firmware_arduino/README.md`](firmware_arduino/README.md).

---

## 🎛️ إعدادات سريعة

| الإعداد (في `config.json`) | الافتراضي | المعنى |
|---|---|---|
| `tracker.threshold` | `50.0` | عتبة الحركة |
| `tracker.stationary_hold_sec` | `10.0` | مدة بقاء الحضور بعد آخر حركة |
| `tracker.static_shadow_db` | `1.5` | ظل الجسم الثابت (يضاعف المدة) |
| `tracker.max_step` | `0.065` | منع القفزات المفاجئة |
| `tracker.deadband` | `0.016` | منع الاهتزاز الصغير |
| `processing.count_ml_override` | `false` | الـ ML يستبدل عدّ الـ blobs؟ |

---

## 🧪 التحقق

```bat
python -m py_compile src\*.py
python src\dual_link_tracker.py --demo --no-gui --seconds 50
python src\main.py --demo --no-gui --seconds 10
```

**المتوقع:** الحضور يظهر `YES` أثناء الحركة، ويرجع `NO` بعد ما الغرفة تفضى
ويثبت.
