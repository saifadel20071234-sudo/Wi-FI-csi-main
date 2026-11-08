import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

DATA = "results/esp32_features_v1.csv"
df = pd.read_csv(DATA)
df = df.sort_values(["dataset_file", "timestamp"]).reset_index(drop=True)

target = "people_count"

feature_sets = {
    "Raw": ["csi_variance"],
    "Smoothed": [
        "variance_mean_3", "variance_median_3",
        "variance_mean_5", "variance_median_5"
    ],
    "Dynamic": [
        "variance_std_3", "variance_std_5",
        "variance_delta", "variance_abs_delta"
    ],
    "Combined": [
        "csi_variance", "csi_variance_log1p",
        "variance_mean_3", "variance_std_3",
        "variance_median_3", "variance_mean_5",
        "variance_std_5", "variance_median_5",
        "variance_delta", "variance_abs_delta"
    ]
}

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=8,
        min_samples_leaf=5, random_state=42
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300, max_depth=8,
        min_samples_leaf=5, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, max_depth=2,
        learning_rate=0.03, random_state=42
    ),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=100, max_leaf_nodes=7,
        learning_rate=0.05, random_state=42
    )
}

def evaluate(train, test, description):
    print("\n" + "=" * 65)
    print(description)
    print("=" * 65)

    y_train = train[target]
    y_test = test[target]

    baseline = np.median(y_train)
    baseline_mae = mean_absolute_error(
        y_test, np.full(len(y_test), baseline)
    )
    print(f"Baseline MAE: {baseline_mae:.3f}")

    results = []

    for set_name, features in feature_sets.items():
        features = [c for c in features if c in df.columns]

        if not features:
            continue

        X_train = train[features]
        X_test = test[features]

        for model_name, model in models.items():
            pipeline = make_pipeline(
                SimpleImputer(strategy="median"),
                model
            )
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)

            results.append({
                "Features": set_name,
                "Model": model_name,
                "MAE": round(mae, 3),
                "Baseline_MAE": round(baseline_mae, 3),
                "Improvement": round(baseline_mae - mae, 3)
            })

    result_df = pd.DataFrame(results).sort_values("MAE")
    print(result_df.to_string(index=False))
    return result_df

all_results = []

files = df["dataset_file"].dropna().unique()

for train_file, test_file in [
    (files[0], files[1]),
    (files[1], files[0])
]:
    train = df[df["dataset_file"] == train_file].copy()
    test = df[df["dataset_file"] == test_file].copy()

    all_results.append(
        evaluate(train, test, f"TRAIN: {train_file} | TEST: {test_file}")
    )

for file in files:
    part = df[df["dataset_file"] == file].copy()
    split = int(len(part) * 0.70)

    train = part.iloc[:split].copy()
    test = part.iloc[split + 5:].copy()

    all_results.append(
        evaluate(train, test, f"CHRONOLOGICAL TEST: {file}")
    )

final = pd.concat(all_results, ignore_index=True)
final.to_csv("results/model_comparison.csv", index=False)

print("\nSaved: results/model_comparison.csv")
print("\nBEST RESULTS:")
print(final.sort_values("MAE").head(10).to_string(index=False))
