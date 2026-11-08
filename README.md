# 📡 WiFi CSI Sensing & Real-Time Tracking System
## مشروع تتبع وكشف الأشخاص باستخدام إشارات الواي فاي (CSI)

نظام متكامل لمعالجة بيانات **Channel State Information (CSI)** في الوقت الفعلي عبر موجات الواي فاي باستخدام متحكمات **ESP32**، لتوفير:
1. **تتبع الحركة بالخريطة الحرارية (2D Heatmap Tracker):** خريطة تفاعلية بنظام إحداثيات ثنائي الأبعاد وتنعيم بفلتر كالمان (Kalman Filter).
2. **عدّاد ورصد الأشخاص (Scope & People Counter):** رصد لحظي لمستوى تشتت الإشارة وعدد الأشخاص بدقة.
3. **دعم تشغيل محاكاة (Demo Mode):** تجربة النظام بالكامل على الكمبيوتر بدون الحاجة لأي هاردوير متصل.

---

### 🚀 تشغيل البرامج السريعة (Quick Start)

بمجرد الضغط المزدوج (Double-Click) على أي من الملفات التالية:

* **`run_tracker.bat`**: تشغيل متتبع الحركة المباشر بالخريطة الحرارية (2D Heatmap Tracker).
* **`run_tracker_demo.bat`**: تشغيل محاكاة تجريبية للخريطة الحرارية بدون أجهزة.
* **`run_monitor.bat`**: تشغيل شاشة رصد الإشارة وعدّاد الأشخاص المباشر (Scope Monitor).
* **`run_monitor_demo.bat`**: تشغيل محاكاة عدّاد ورصد الأشخاص.

---

### 📁 هيكل المشروع (Project Structure)

```
csii/
├── CSI/
│   ├── src/                 # الأكواد البرمجية وخوارزميات المعالجة
│   │   ├── dual_link_tracker.py  # محرك الخريطة الحرارية وفلتر كالمان
│   │   ├── scope_monitor.py      # راسم الإشارة وعدّاد الأشخاص
│   │   ├── csi_reader.py         # قارئ بيانات السيريال للـ ESP32
│   │   ├── kalman.py             # خوارزمية التنعيم الرياضي
│   │   └── config.py             # إعدادات النظام واكتشاف المنافذ
│   ├── firmware/            # فيرموير ESP-IDF الأصلي
│   ├── firmware_arduino/    # فيرموير Arduino C++ للـ ESP32
│   ├── requirements.txt     # المكتبات المطلوبة (OpenCV, NumPy, PySerial)
│   └── config.json          # إعدادات المنافذ والشبكة
├── run_tracker.bat          # اختصار تشغيل الخريطة الحرارية
├── run_monitor.bat          # اختصار تشغيل عداد الإشارة
└── README.md                # دليل المشروع
```

---

### ⚙️ المتطلبات (Requirements)
* Python 3.10 - 3.12
* OpenCV & NumPy & PySerial (`pip install -r CSI/requirements.txt`)
* جهاز أو جهازين ESP32 (للربط الحي)
