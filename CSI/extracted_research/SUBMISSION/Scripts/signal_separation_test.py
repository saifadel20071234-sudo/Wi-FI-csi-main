import pandas as pd
import numpy as np

df = pd.read_csv("Data/cleand_esp32_csi.csv")

for name, part in df.groupby("dataset_file"):
    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    report = part.groupby("people_count").agg(
        samples=("csi_variance", "size"),
        variance_mean=("csi_variance", "mean"),
        variance_median=("csi_variance", "median"),
        variance_std=("csi_variance", "std"),
        variance_min=("csi_variance", "min"),
        variance_max=("csi_variance", "max")
    ).round(3)

    print(report.to_string())

    print("\nCORRELATIONS:")
    print(part[["people_count", "csi_variance"]].corr().round(3))

    print("\nVARIANCE SEPARATION:")
    for count in sorted(part["people_count"].unique()):
        values = part.loc[
            part["people_count"] == count, "csi_variance"
        ]
        print(
            f"People={count}: "
            f"Median={values.median():.3f}, "
            f"IQR={values.quantile(.75)-values.quantile(.25):.3f}"
        )
