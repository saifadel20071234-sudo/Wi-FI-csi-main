import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("results/esp32_features_v1.csv")
features = [
    "csi_variance",
    "variance_mean_3",
    "variance_std_3",
    "variance_median_3",
    "variance_mean_5",
    "variance_std_5",
    "variance_median_5",
    "variance_delta",
    "variance_abs_delta"
]

for name, g in df.groupby("dataset_file"):
    g = g.sort_values("timestamp").reset_index(drop=True)
    split = int(len(g) * 0.70)

    train = g.iloc[:split]
    test = g.iloc[split + 5:]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(train[features].fillna(0), train["people_count"])
    pred = model.predict(test[features].fillna(0))

    baseline = np.full(len(test), train["people_count"].median())

    print("\nFILE:", name)
    print("Train rows:", len(train), "| Test rows:", len(test))
    print("Model MAE:", round(mean_absolute_error(test["people_count"], pred), 3))
    print("Baseline MAE:", round(mean_absolute_error(test["people_count"], baseline), 3))
