import pandas as pd

df = pd.read_csv("Data/cleand_esp32_csi.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

for name, g in df.groupby("dataset_file"):
    g = g.sort_values("timestamp").copy()
    g["minute"] = g["timestamp"].dt.floor("min")

    result = g.groupby("minute").agg(
        people_mean=("people_count", "mean"),
        people_std=("people_count", "std"),
        variance_mean=("csi_variance", "mean"),
        variance_std=("csi_variance", "std"),
        readings=("csi_variance", "size")
    )

    print("\nFILE:", name)
    print("Minute groups:", len(result))
    print("Correlation:")
    print(result[["people_mean", "variance_mean"]].corr().round(3))
    print("\nFirst 10 minutes:")
    print(result.head(10).round(2).to_string())
    print("\nLast 5 minutes:")
    print(result.tail(5).round(2).to_string())
