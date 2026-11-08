import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("results/esp32_advanced_features.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["dataset_file", "timestamp"]).reset_index(drop=True)

target = "people_count"

raw = ["csi_variance"]

smoothed = [
    "variance_mean_3",
    "variance_mean_5"
]

advanced = [
    c for c in df.columns
    if c.startswith("variance_lag_")
    or c.startswith("variance_ema_")
    or c.startswith("variance_q")
    or c.startswith("variance_range_")
    or c.startswith("variance_mean_")
    or c.startswith("variance_std_")
    or c in ["ema_difference", "variance_rate_change"]
]

feature_sets = {
    "Raw": raw,
    "Smoothed": smoothed,
    "Advanced": advanced,
    "Combined": raw + smoothed + advanced
}

models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=200, max_depth=8,
        min_samples_leaf=5, random_state=42
    ),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=200, max_depth=8,
        min_samples_leaf=5, random_state=42
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100, max_depth=2,
        learning_rate=0.05, random_state=42
    )
}

results = []

def evaluate(train, test, split_name):
    y_train = train[target]
    y_test = test[target]
    baseline = np.median(y_train)

    results.append({
        "Split": split_name,
        "Features": "Baseline",
        "Model": "Median",
        "MAE": mean_absolute_error(
            y_test, np.full(len(y_test), baseline)
        )
    })

    for feature_name, cols in feature_sets.items():
        cols = list(dict.fromkeys(c for c in cols if c in df.columns))

        for model_name, model in models.items():
            pipeline = make_pipeline(
                SimpleImputer(strategy="median"),
                model
            )

            pipeline.fit(train[cols], y_train)
            predictions = pipeline.predict(test[cols])

            results.append({
                "Split": split_name,
                "Features": feature_name,
                "Model": model_name,
                "MAE": mean_absolute_error(y_test, predictions)
            })

recordings = list(df["dataset_file"].unique())

for train_file, test_file in [
    (recordings[0], recordings[1]),
    (recordings[1], recordings[0])
]:
    train = df[df.dataset_file == train_file].copy()
    test = df[df.dataset_file == test_file].copy()

    evaluate(train, test, f"Cross: {train_file} -> {test_file}")

for file in recordings:
    part = df[df.dataset_file == file].copy()
    cut = int(len(part) * 0.70)

    train = part.iloc[:cut].copy()
    test = part.iloc[cut + 5:].copy()

    evaluate(train, test, f"Within: {file}")

out = pd.DataFrame(results).sort_values(["Split", "MAE"])
out.to_csv("results/advanced_model_comparison.csv", index=False)

print("\nADVANCED MODEL COMPARISON")
print(out.round(3).to_string(index=False))
print("\nSaved: results/advanced_model_comparison.csv")
