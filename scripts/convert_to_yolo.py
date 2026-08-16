"""
Converts the filtered CSV (pixel-space xmin/ymin/xmax/ymax boxes) into a
YOLO-format dataset: one label .txt per image plus train/validation/test
image/label folders and a data.yaml, using the 'split' column already
present in the CSV.

Usage:
    python scripts/convert_to_yolo.py \
        --csv dataset/labels_filtered.csv \
        --source-dir /path/to/kagglehub/download/dataset \
        --out-dir dataset
"""

import argparse
import os
import shutil

import pandas as pd

CLASSES = [
    "F16", "F18", "C130", "F35", "F15", "F22", "B2", "B52", "AH64",
    "V22", "J20", "C17", "F14", "Su57", "Mig31", "CH47", "Tejas",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}
SPLITS = ["train", "validation", "test"]


def find_source_image(source_dir: str, filename: str) -> str | None:
    candidate = os.path.join(source_dir, filename)
    if os.path.exists(candidate):
        return candidate
    for ext in [".jpg", ".jpeg", ".png"]:
        alt = candidate if candidate.endswith(ext) else candidate + ext
        if os.path.exists(alt):
            return alt
    return None


def convert(csv_path: str, source_dir: str, out_dir: str) -> None:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {df['filename'].nunique()} unique images")

    for split in SPLITS:
        os.makedirs(os.path.join(out_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, split, "labels"), exist_ok=True)

    written = {s: 0 for s in SPLITS}
    missing_images = []

    for i, (filename, group) in enumerate(df.groupby("filename")):
        split = group.iloc[0]["split"]
        if split not in SPLITS:
            continue

        src_img_path = find_source_image(source_dir, filename)
        if src_img_path is None:
            missing_images.append(filename)
            continue

        img_w = group.iloc[0]["width"]
        img_h = group.iloc[0]["height"]

        label_lines = []
        for _, row in group.iterrows():
            cls_id = CLASS_TO_ID[row["class"]]
            xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            box_w = (xmax - xmin) / img_w
            box_h = (ymax - ymin) / img_h
            label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        base_name = os.path.splitext(os.path.basename(filename))[0]
        label_path = os.path.join(out_dir, split, "labels", base_name + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

        img_ext = os.path.splitext(src_img_path)[1]
        dst_img_path = os.path.join(out_dir, split, "images", base_name + img_ext)
        shutil.copyfile(src_img_path, dst_img_path)

        written[split] += 1
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1} images...")

    yaml_content = f"""train: {out_dir}/train/images
val: {out_dir}/validation/images
test: {out_dir}/test/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print("\n=== DONE ===")
    print("Files written per split:", written)
    print(f"Missing images: {len(missing_images)}")
    if missing_images[:10]:
        print("Sample missing:", missing_images[:10])
    print(f"\ndata.yaml written to {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="dataset/labels_filtered.csv", help="Filtered CSV path")
    parser.add_argument("--source-dir", required=True, help="Directory containing the raw source images")
    parser.add_argument("--out-dir", default="dataset", help="Output directory for the YOLO dataset")
    args = parser.parse_args()
    convert(args.csv, args.source_dir, args.out_dir)
