"""
Runs standalone validation for a trained checkpoint against a data.yaml,
printing overall and per-class precision/recall/mAP50/mAP50-95 and saving
a confusion matrix, without re-running training.

Usage:
    python scripts/evaluate.py --weights model/best.pt --data dataset/data.yaml
"""

import argparse

from ultralytics import YOLO


def evaluate(weights: str, data: str) -> None:
    model = YOLO(weights)
    metrics = model.val(data=data, plots=True)

    print("\n=== Overall ===")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")
    print(f"mAP50:     {metrics.box.map50:.3f}")
    print(f"mAP50-95:  {metrics.box.map:.3f}")

    print("\n=== Per-class ===")
    names = metrics.names
    for i, cls_id in enumerate(metrics.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        print(f"{names[cls_id]:>8}: P={p:.3f} R={r:.3f} mAP50={ap50:.3f} mAP50-95={ap:.3f}")

    print(f"\nConfusion matrix and plots saved to {metrics.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="model/best.pt", help="Path to trained .pt checkpoint")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml")
    args = parser.parse_args()
    evaluate(args.weights, args.data)
