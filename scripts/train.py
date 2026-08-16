"""
Trains a YOLOv8 model on the MiVision dataset. Reusable across versions
(V1 used yolov8n; V2 is planned to use yolov8s).

Usage:
    python scripts/train.py --model yolov8n.pt --epochs 50 --data dataset/data.yaml --name mivision_v1
    python scripts/train.py --model yolov8s.pt --epochs 50 --data dataset/data.yaml --name mivision_v2
"""

import argparse

from ultralytics import YOLO


def train(model_name: str, data: str, epochs: int, imgsz: int, batch: int, name: str) -> None:
    model = YOLO(model_name)
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        save_period=10,
        patience=15,
        name=name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt", help="Base model, e.g. yolov8n.pt / yolov8s.pt / yolov8m.pt")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="mivision_run", help="Run name / output folder")
    args = parser.parse_args()
    train(args.model, args.data, args.epochs, args.imgsz, args.batch, args.name)
