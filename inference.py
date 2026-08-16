"""
Shared inference function for MiVision. Loads the trained YOLOv8 model once
and runs detection on a single image, returning structured results.

Usage:
    from inference import load_model, detect

    model = load_model("model/best.pt")
    detections = detect(model, "some_image.jpg", conf=0.4)
    for d in detections:
        print(d["class_name"], d["confidence"], d["bbox"])
"""

from pathlib import Path
from typing import Union

from PIL import Image
from ultralytics import YOLO

DEFAULT_MODEL_PATH = "model/best.pt"
DEFAULT_CONF = 0.4  # peak of the F1-confidence curve (~0.39), not YOLO's default 0.25


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> YOLO:
    return YOLO(str(model_path))


def detect(
    model: YOLO,
    image: Union[str, Path, Image.Image],
    conf: float = DEFAULT_CONF,
) -> list[dict]:
    """Run inference on one image. Returns a list of detections:
    {"class_name": str, "confidence": float, "bbox": (x1, y1, x2, y2)}
    """
    results = model.predict(source=image, conf=conf, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        detections.append(
            {
                "class_name": result.names[cls_id],
                "confidence": float(box.conf.item()),
                "bbox": tuple(box.xyxy[0].tolist()),
            }
        )
    return detections


def demo():
    """Self-check: fails loudly if the model or a Path/PIL image can't be run."""
    model = load_model()
    img = Image.new("RGB", (640, 640), color=(128, 128, 128))
    detections = detect(model, img)
    assert isinstance(detections, list)
    print(f"OK - model loaded, blank-image inference returned {len(detections)} detections")


if __name__ == "__main__":
    demo()
