import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score

df = pd.read_csv("results/esp32_features_v1.csv")
files = df["dataset_file"].unique()

for train_file, test_file in [(files[0], files[1]), (files[1], files[0])]:
    train = df[df["dataset_file"] == train_file]
    test = df[df["dataset_file"] == test_file]

    baseline_value = train["people_count"].median()
    predictions = np.full(len(test), baseline_value)
    rounded = np.clip(np.rint(predictions), 0, 8).astype(int)

    print("\nTRAIN:", train_file)
    print("TEST:", test_file)
    print("Baseline prediction:", round(baseline_value, 2))
    print("Baseline MAE:", round(mean_absolute_error(test["people_count"], predictions), 3))
    print("Baseline Rounded Accuracy:",
          round(accuracy_score(test["people_count"], rounded) * 100, 2), "%")
