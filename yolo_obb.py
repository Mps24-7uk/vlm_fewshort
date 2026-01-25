import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ---------------- CONFIG ----------------
MODEL_PATH = "yolo11n-obb.pt"
SOURCE = "images/"          # image | folder | video | 0
IMG_SIZE = 1024             # 🔥 1024 inference
CONF = 0.25
PROJECT = "predict"
NAME = "exp1"
DEVICE = 0                 # 0 = GPU | "cpu"
# ---------------------------------------

# Output dirs
BASE_DIR = Path(PROJECT) / NAME
IMG_DIR = BASE_DIR / "images"
CROP_DIR = BASE_DIR / "crops"

IMG_DIR.mkdir(parents=True, exist_ok=True)
CROP_DIR.mkdir(parents=True, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# ---------------- INFERENCE ----------------
results = model(
    SOURCE,
    imgsz=IMG_SIZE,
    conf=CONF,
    device=DEVICE,
    stream=True
)

for idx, result in enumerate(results):
    orig = result.orig_img.copy()
    img_name = Path(result.path).stem

    if result.obb is not None:
        # Official Ultralytics OBB outputs
        polygons = result.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
        classes = result.obb.cls.cpu().numpy()
        confs = result.obb.conf.cpu().numpy()
        names = result.names

        for i, (poly, cls_id, conf) in enumerate(zip(polygons, classes, confs)):
            cls_name = names[int(cls_id)]

            # ---------------- DRAW OBB ----------------
            poly_int = poly.astype(int)
            cv2.polylines(orig, [poly_int], True, (0, 255, 0), 2)

            label = f"{cls_name} {conf:.2f}"
            cv2.putText(
                orig,
                label,
                tuple(poly_int[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # ---------------- POLYGON CROP (BEST) ----------------
            mask = np.zeros(orig.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [poly_int], 255)

            masked = cv2.bitwise_and(result.orig_img, result.orig_img, mask=mask)

            x, y, w, h = cv2.boundingRect(poly_int)
            crop = masked[y:y+h, x:x+w]

            if crop.size == 0:
                continue

            class_dir = CROP_DIR / cls_name
            class_dir.mkdir(exist_ok=True)

            crop_path = class_dir / f"{img_name}_crop_{i}.jpg"
            cv2.imwrite(str(crop_path), crop)

    # Save predicted image
    out_img = IMG_DIR / f"{img_name}.jpg"
    cv2.imwrite(str(out_img), orig)

    print(f"Saved → {out_img}")
