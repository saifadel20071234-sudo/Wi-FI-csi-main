import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("Data/cleand_esp32_csi.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["dataset_file", "timestamp"]).reset_index(drop=True)

# Create time-based features separately for each recording
groups = df.groupby("dataset_file", sort=False)

for window in [3, 5]:
    df[f"variance_mean_{window}"] = groups["csi_variance"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df[f"variance_std_{window}"] = groups["csi_variance"].transform(
        lambda x: x.rolling(window, min_periods=2).std()
    )
    df[f"variance_median_{window}"] = groups["csi_variance"].transform(
        lambda x: x.rolling(window, min_periods=1).median()
    )

df["variance_delta"] = groups["csi_variance"].diff()
df["variance_abs_delta"] = df["variance_delta"].abs()

# Save processed data
df.to_csv(OUT / "esp32_features_v1.csv", index=False)

# Summary report
summary = df.groupby("dataset_file").agg(
    rows=("csi_variance", "size"),
    people_mean=("people_count", "mean"),
    variance_mean=("csi_variance", "mean"),
    variance_std=("csi_variance", "std"),
    variance_min=("csi_variance", "min"),
    variance_max=("csi_variance", "max")
).round(3)

summary.to_csv(OUT / "dataset_summary.csv")

# Plot variance over time, separately per recording
for name, group in df.groupby("dataset_file"):
    plt.figure(figsize=(12, 4))
    plt.plot(group["timestamp"], group["csi_variance"], label="Raw CSI variance")
    plt.plot(group["timestamp"], group["variance_mean_5"], label="Rolling mean (5)")
    plt.xlabel("Time")
    plt.ylabel("CSI variance")
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    filename = name.replace(" ", "_").replace(".csv", "")
    plt.savefig(OUT / f"{filename}_signal.png", dpi=150)
    plt.close()

print("PROCESSING COMPLETED")
print("Processed shape:", df.shape)
print("\nNEW FEATURES:")
print([c for c in df.columns if "variance_" in c])
print("\nSUMMARY:")
print(summary.to_string())
print("\nSaved files in results/")
