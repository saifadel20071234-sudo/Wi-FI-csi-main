import pandas as pd
import numpy as np

df = pd.read_csv("Data/cleand_esp32_csi.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["dataset_file", "timestamp"]).copy()

def make_features(g):
    g = g.copy()
    x = g["csi_variance"]

    for lag in [1, 2, 3, 5]:
        g[f"variance_lag_{lag}"] = x.shift(lag)

    for span in [3, 5, 10]:
        g[f"variance_ema_{span}"] = x.ewm(
            span=span, adjust=False
        ).mean()

    for window in [3, 5, 10]:
        r = x.rolling(window, min_periods=1)
        g[f"variance_q25_{window}"] = r.quantile(0.25)
        g[f"variance_q75_{window}"] = r.quantile(0.75)
        g[f"variance_range_{window}"] = r.max() - r.min()
        g[f"variance_mean_{window}"] = r.mean()
        g[f"variance_std_{window}"] = r.std()

    g["ema_difference"] = g["variance_ema_3"] - g["variance_ema_10"]
    g["variance_rate_change"] = x.diff()
    return g

parts = []
for _, group in df.groupby("dataset_file", sort=False):
    parts.append(make_features(group))

result = pd.concat(parts, ignore_index=True)
result = result.replace([np.inf, -np.inf], np.nan)

result.to_csv("results/esp32_advanced_features.csv", index=False)

print("FIXED FEATURE FILE")
print("Rows:", len(result))
print("Columns:", len(result.columns))
print("dataset_file exists:", "dataset_file" in result.columns)
print("Saved successfully.")
