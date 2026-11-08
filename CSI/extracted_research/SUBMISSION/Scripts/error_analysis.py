import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("results/esp32_advanced_features.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["dataset_file", "timestamp"])

advanced = [
    c for c in df.columns
    if c.startswith(("variance_lag_", "variance_ema_", "variance_q",
                     "variance_range_", "variance_mean_", "variance_std_"))
    or c in ["ema_difference", "variance_rate_change"]
]

combined = list(dict.fromkeys(
    ["csi_variance", "variance_mean_3", "variance_mean_5"] + advanced
))

all_reports = []

for i, file in enumerate(df["dataset_file"].unique()):
    part = df[df["dataset_file"] == file].copy().reset_index(drop=True)

    cut = int(len(part) * 0.70)
    train = part.iloc[:cut].copy()
    test = part.iloc[cut + 5:].copy()

    features = combined if i == 0 else ["csi_variance"]
    model_name = "Combined" if i == 0 else "Raw"

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42
        )
    )

    model.fit(train[features], train["people_count"])

    test["prediction"] = model.predict(test[features])
    test["rounded_prediction"] = np.clip(
        np.rint(test["prediction"]), 0, train["people_count"].max()
    )
    test["baseline_prediction"] = train["people_count"].median()

    print("\n" + "=" * 60)
    print("DATASET:", file)
    print("MODEL:", model_name)
    print("Baseline MAE:", round(mean_absolute_error(
        test["people_count"], test["baseline_prediction"]
    ), 3))
    print("Model MAE:", round(mean_absolute_error(
        test["people_count"], test["prediction"]
    ), 3))

    rows = []

    for count, group in test.groupby("people_count"):
        rows.append({
            "people_count": count,
            "Samples": len(group),
            "Average_Prediction": group["prediction"].mean(),
            "Average_Rounded": group["rounded_prediction"].mean(),
            "Model_MAE": mean_absolute_error(
                group["people_count"], group["prediction"]
            ),
            "Baseline_MAE": mean_absolute_error(
                group["people_count"], group["baseline_prediction"]
            )
        })

    report = pd.DataFrame(rows).round(3)
    print("\nERROR BY ACTUAL PEOPLE COUNT:")
    print(report.to_string(index=False))

    report["dataset_file"] = file
    report["model"] = model_name
    all_reports.append(report)

final = pd.concat(all_reports, ignore_index=True)
final.to_csv("results/error_analysis_by_count.csv", index=False)

print("\nSaved: results/error_analysis_by_count.csv")
