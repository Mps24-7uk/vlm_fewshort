import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "yolo11m-obb-1024.pt"
SOURCE = "images/"            # image | folder | video | 0
CONF = 0.25
IMG_SIZE = 1024               # 🔥 FORCE 1024
PROJECT = "predict"
NAME = "exp1"
DEVICE = 0                   # 0 = GPU | "cpu"
# ---------------------------------------

# Output directories
BASE_DIR = Path(PROJECT) / NAME
IMG_DIR = BASE_DIR / "images"
CROP_DIR = BASE_DIR / "crops"

IMG_DIR.mkdir(parents=True, exist_ok=True)
CROP_DIR.mkdir(parents=True, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)
names = model.names


def draw_obb(img, obb):
    cx, cy, w, h, angle = obb
    rect = ((cx, cy), (w, h), np.degrees(angle))
    box = cv2.boxPoints(rect).astype(int)
    cv2.polylines(img, [box], True, (0, 255, 0), 2)
    return box


def crop_obb(img, obb):
    """
    Rotation-aware OBB crop (accurate for 1024 resolution)
    """
    cx, cy, w, h, angle = obb

    # Rotate entire image
    M = cv2.getRotationMatrix2D((cx, cy), np.degrees(angle), 1.0)
    rotated = cv2.warpAffine(
        img,
        M,
        (img.shape[1], img.shape[0]),
        flags=cv2.INTER_CUBIC
    )

    # Axis-aligned crop after rotation
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(rotated.shape[1], x2), min(rotated.shape[0], y2)

    return rotated[y1:y2, x1:x2]


# ---------------- INFERENCE ----------------
results = model(
    SOURCE,
    conf=CONF,
    imgsz=IMG_SIZE,     # 🔥 KEY CHANGE
    device=DEVICE,
    stream=True
)

for frame_idx, r in enumerate(results):
    img = r.orig_img.copy()
    img_name = Path(r.path).stem

    if r.obb is not None:
        obbs = r.obb.xywhra.cpu().numpy()
        clss = r.obb.cls.cpu().numpy()
        confs = r.obb.conf.cpu().numpy()

        for i, (obb, cls_id, conf) in enumerate(zip(obbs, clss, confs)):
            cls_name = names[int(cls_id)]

            # Draw rotated box
            draw_obb(img, obb)

            # Crop
            crop = crop_obb(r.orig_img, obb)
            if crop.size == 0:
                continue

            class_dir = CROP_DIR / cls_name
            class_dir.mkdir(exist_ok=True)

            crop_path = class_dir / f"{img_name}_crop_{i}.jpg"
            cv2.imwrite(str(crop_path), crop)

    # Save prediction image
    out_img = IMG_DIR / f"{img_name}.jpg"
    cv2.imwrite(str(out_img), img)

    print(f"Saved → {out_img}")
