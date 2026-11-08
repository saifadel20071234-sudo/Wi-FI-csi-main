import pandas as pd

df = pd.read_csv("Data/cleand_esp32_csi.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["dataset_file", "timestamp"])

for name, part in df.groupby("dataset_file"):
    print("\n", name)
    print("Duration:", part["timestamp"].max() - part["timestamp"].min())
    print("Median sampling interval:",
          part["timestamp"].diff().dt.total_seconds().median())
    print("Duplicate timestamps:", part["timestamp"].duplicated().sum())
    print("Unique people counts:", sorted(part["people_count"].unique()))
    print("Rows:", len(part))

print("\nAVAILABLE COLUMNS:")
print(df.columns.tolist())
