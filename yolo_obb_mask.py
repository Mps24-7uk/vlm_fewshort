import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ---------------- CONFIG ----------------
MODEL_PATH = "yolo11m-obb-1024.pt"
SOURCE = "images/"            # image | folder | video | 0
CONF = 0.25
IMG_SIZE = 1024
PROJECT = "predict"
NAME = "exp1"
DEVICE = 0                   # 0 = GPU | "cpu"
# ---------------------------------------

# Output directory for masks
BASE_DIR = Path(PROJECT) / NAME
MASK_DIR = BASE_DIR / "masks"
MASK_DIR.mkdir(parents=True, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)


def obb_to_polygon(obb):
    """
    Convert (cx, cy, w, h, angle) -> 4 corner polygon
    """
    cx, cy, w, h, angle = obb
    rect = ((cx, cy), (w, h), np.degrees(angle))
    box = cv2.boxPoints(rect).astype(np.int32)
    return box


# ---------------- INFERENCE ----------------
results = model(
    SOURCE,
    conf=CONF,
    imgsz=IMG_SIZE,
    device=DEVICE,
    stream=True
)

for frame_idx, r in enumerate(results):
    orig = r.orig_img
    h, w = orig.shape[:2]
    img_name = Path(r.path).stem

    # Create black mask
    mask = np.zeros((h, w), dtype=np.uint8)

    if r.obb is not None:
        obbs = r.obb.xywhra.cpu().numpy()

        for obb in obbs:
            poly = obb_to_polygon(obb)

            # Fill OBB area with white
            cv2.fillPoly(mask, [poly], 255)

    # Save binary mask
    mask_path = MASK_DIR / f"{img_name}_mask.png"
    cv2.imwrite(str(mask_path), mask)

    print(f"Saved mask → {mask_path}")
