import pandas as pd
import numpy as np

df = pd.read_csv("Data/cleand_esp32_csi.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["dataset_file", "timestamp"])

for name, g in df.groupby("dataset_file"):
    g = g.copy()
    g["people_change"] = g["people_count"].diff()
    g["variance_change"] = g["csi_variance"].diff()

    changed = g[g["people_change"] != 0].copy()

    print("\n" + "=" * 60)
    print("DATASET:", name)
    print("Total rows:", len(g))
    print("People count transitions:", len(changed))

    print("\nPEOPLE COUNT TRANSITIONS:")
    print(changed[[
        "timestamp",
        "people_count",
        "people_change",
        "csi_variance",
        "variance_change"
    ]].head(20).round(3).to_string(index=False))

    print("\nCorrelation between changes:",
          round(g["people_change"].corr(g["variance_change"]), 3))

