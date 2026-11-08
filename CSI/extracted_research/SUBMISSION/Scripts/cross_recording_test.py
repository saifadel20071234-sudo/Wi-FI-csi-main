import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score

df = pd.read_csv("results/esp32_features_v1.csv")

features = [
    "csi_variance",
    "csi_variance_log1p",
    "variance_mean_3",
    "variance_std_3",
    "variance_median_3",
    "variance_mean_5",
    "variance_std_5",
    "variance_median_5",
    "variance_delta",
    "variance_abs_delta"
]

df[features] = df[features].replace([np.inf, -np.inf], np.nan)
df[features] = df[features].fillna(0)

files = df["dataset_file"].unique()

for train_file, test_file in [(files[0], files[1]), (files[1], files[0])]:
    train = df[df["dataset_file"] == train_file]
    test = df[df["dataset_file"] == test_file]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(train[features], train["people_count"])
    predictions = model.predict(test[features])
    rounded = np.clip(np.rint(predictions), 0, 8).astype(int)

    print("\nTRAIN:", train_file)
    print("TEST:", test_file)
    print("MAE:", round(mean_absolute_error(test["people_count"], predictions), 3))
    print("Rounded Accuracy:", round(accuracy_score(test["people_count"], rounded) * 100, 2), "%")

    importance = pd.Series(model.feature_importances_, index=features)
    print("\nTop Features:")
    print(importance.sort_values(ascending=False).head(5).round(4).to_string())

