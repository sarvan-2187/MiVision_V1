"""
Filters the raw Military Aircraft Detection Dataset CSV down to MiVision's
17 selected classes and writes a filtered CSV.

See dataset/DATASET.md for where the raw dataset (labels_with_split.csv)
comes from and how to download it via kagglehub.

Usage:
    python scripts/filter_dataset.py \
        --csv /path/to/labels_with_split.csv \
        --out dataset/labels_filtered.csv
"""

import argparse

import pandas as pd

CLASSES = [
    "F16", "F18", "C130", "F35", "F15", "F22", "B2", "B52", "AH64",
    "V22", "J20", "C17", "F14", "Su57", "Mig31", "CH47", "Tejas",
]


def filter_dataset(csv_path: str, out_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {df['class'].nunique()} classes")

    df_filtered = df[df["class"].isin(CLASSES)].copy()
    print(f"Filtered shape: {df_filtered.shape}")

    print("\nPer-class instance counts:")
    print(df_filtered["class"].value_counts())

    print("\nSplit distribution after filtering:")
    print(df_filtered["split"].value_counts())

    missing = set(CLASSES) - set(df_filtered["class"].unique())
    if missing:
        print(f"\nWARNING - these classes had zero matches: {missing}")
    else:
        print(f"\nAll {len(CLASSES)} classes present.")

    df_filtered.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({df_filtered.shape[0]} rows)")
    return df_filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to labels_with_split.csv")
    parser.add_argument("--out", default="dataset/labels_filtered.csv", help="Output CSV path")
    args = parser.parse_args()
    filter_dataset(args.csv, args.out)
