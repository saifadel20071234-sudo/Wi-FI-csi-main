"""Train the people-counting model (RandomForest).

Usage:
    python train_count_model.py --synthetic          # demo: generate data + train
    python train_count_model.py --csv count_data.csv # train on your labeled data

The labeled CSV header must be:  count,<feature...>  where the feature columns are
exactly those produced by count_model.feature_names(devices) for the SAME device
list (use collect_count_data.py to build it).

Legacy CSVs whose feature columns were recorded under different device names
(e.g. left_*/right_* instead of esp32_h_*/esp32_v_*) are accepted as long as the
column count matches: the columns are then mapped by position with a warning.

Outputs (next to this script): count_model.pkl, count_scaler.pkl,
count_feature_names.pkl.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

from config import load_config
from count_model import feature_names


def synthetic_rows(devices, n_per=600, seed=42):
    """Generate plausible feature vectors for counts 0..3."""
    names = feature_names(devices)
    rng = np.random.default_rng(seed)
    act_mean_by_class = [0.04, 0.40, 0.62, 0.80]
    act_std_by_class = [0.02, 0.10, 0.16, 0.22]
    act_max_by_class = [0.08, 0.55, 0.80, 0.95]
    drop_by_class = [0.5, 3.0, 6.5, 9.5]
    pres_by_class = [0.02, 0.95, 0.97, 0.98]

    rows = []
    for c in range(4):
        for _ in range(n_per):
            vals = []
            acts_mean, acts_max = [], []
            for _dev in devices:
                am = float(np.clip(rng.normal(act_mean_by_class[c], 0.08), 0.0, 1.0))
                ast = float(max(0.0, rng.normal(act_std_by_class[c], 0.03)))
                ax = float(np.clip(rng.normal(act_max_by_class[c], 0.06), 0.0, 1.0))
                dm = float(max(0.0, rng.normal(drop_by_class[c], 0.7)))
                vals += [am, ast, ax, dm]
                acts_mean.append(am)
                acts_max.append(ax)
            pres = float(np.clip(rng.normal(pres_by_class[c], 0.03), 0.0, 1.0))
            vals += [float(sum(acts_mean)), float(sum(acts_max)), pres]
            rows.append([c] + vals)
    return names, rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Train people-counting model")
    p.add_argument("--config", default="config.json")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--csv", default=None)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--force",
        action="store_true",
        help="save the model even if the data contains a single label (not recommended)",
    )
    p.add_argument(
        "--out-dir",
        default=".",
        help="directory to write the model files into (default: current directory)",
    )
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    devices = cfg["devices"]
    names = feature_names(devices)
    print(f"Devices: {[d['id'] for d in devices]}")
    print(f"Feature count: {len(names)}")

    if args.csv:
        df = pd.read_csv(args.csv)
        if "count" not in df.columns:
            print("ERROR: CSV must have a 'count' label column.")
            return 1
        missing = [c for c in names if c not in df.columns]
        if missing:
            csv_feats = [c for c in df.columns if c != "count"]
            if len(csv_feats) == len(names):
                print("WARNING: CSV feature names differ from config; mapping by column order:")
                for old, new in zip(csv_feats, names):
                    print(f"  {old} -> {new}")
                df = df.rename(columns=dict(zip(csv_feats, names)))
            else:
                print(f"ERROR: CSV missing feature columns: {missing}")
                print(
                    f"       CSV has {len(csv_feats)} feature columns, "
                    f"expected {len(names)} for devices {[d['id'] for d in devices]}."
                )
                return 1
    else:
        names, rows = synthetic_rows(devices)
        df = pd.DataFrame(rows, columns=["count"] + names)
        print("Using synthetic demo data.")

    X = df[names].astype(float)
    y = df["count"].astype(int)
    print(f"Rows: {len(df)} | label distribution:\n{df['count'].value_counts().sort_index()}")

    # A single-class dataset can only ever predict that one class, so refuse to
    # silently overwrite a working model with it.
    if y.nunique() < 2 and not args.force:
        print(
            f"\nERROR: all {len(y)} rows have the same label ({int(y.iloc[0])}). "
            "The trained model would always predict that class and be useless."
        )
        print(
            "       Collect data for several counts (e.g. --count 0, 1, 2) and retrain, "
            "or pass --force to save anyway."
        )
        print("       Existing model files were NOT modified.")
        return 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    # n_jobs=1 is single-threaded and works everywhere (no multiprocessing pools).
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, random_state=args.seed, n_jobs=1
    )
    model.fit(scaler.transform(X_train), y_train)

    y_pred = model.predict(scaler.transform(X_test))
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc*100:.2f}%")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.out_dir, "count_model.pkl"))
    joblib.dump(scaler, os.path.join(args.out_dir, "count_scaler.pkl"))
    joblib.dump(names, os.path.join(args.out_dir, "count_feature_names.pkl"))
    print(
        f"\nSaved to {os.path.abspath(args.out_dir)}: "
        "count_model.pkl, count_scaler.pkl, count_feature_names.pkl"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
