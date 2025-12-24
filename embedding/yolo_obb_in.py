import cv2
import os
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# =========================
# CONFIG
# =========================
MODEL_PATH = "yolov11-obb.pt"
INPUT_PATH = "data/input"   # image / video / folder
OUTPUT_DIR = "outputs"
CONF_THRES = 0.3

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/annotated", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/crops", exist_ok=True)

model = YOLO(MODEL_PATH)

# =========================
# UTILS
# =========================
def is_image(path):
    return path.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))

def is_video(path):
    return path.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

def crop_rotated(img, rect):
    """Crop rotated rectangle from image"""
    center, size, angle = rect
    size = tuple(map(int, size))

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    x, y = int(center[0] - size[0] / 2), int(center[1] - size[1] / 2)
    return rotated[y:y+size[1], x:x+size[0]]

# =========================
# IMAGE PROCESSING
# =========================
def process_image(img_path):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)

    crop_dir = f"{OUTPUT_DIR}/crops/{img_name}"
    os.makedirs(crop_dir, exist_ok=True)

    results = model(img, conf=CONF_THRES)

    for r in results:
        if r.obb is None:
            continue

        for i, (box, angle, cls, score) in enumerate(
            zip(r.obb.xywh, r.obb.angle, r.obb.cls, r.obb.conf)
        ):
            cx, cy, w, h = map(float, box)
            rect = ((cx, cy), (w, h), float(angle))

            pts = cv2.boxPoints(rect).astype(int)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)

            label = f"{model.names[int(cls)]} {score:.2f}"
            cv2.putText(img, label, (pts[0][0], pts[0][1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            crop = crop_rotated(img, rect)
            if crop.size > 0:
                cv2.imwrite(
                    f"{crop_dir}/{model.names[int(cls)]}_{i}_{score:.2f}.jpg",
                    crop
                )

    cv2.imwrite(f"{OUTPUT_DIR}/annotated/{img_name}.jpg", img)

# =========================
# VIDEO PROCESSING
# =========================
def process_video(video_path):
    name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        f"{OUTPUT_DIR}/annotated/{name}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    crop_dir = f"{OUTPUT_DIR}/crops/{name}"
    os.makedirs(crop_dir, exist_ok=True)

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRES)

        for r in results:
            if r.obb is None:
                continue

            for i, (box, angle, cls, score) in enumerate(
                zip(r.obb.xywh, r.obb.angle, r.obb.cls, r.obb.conf)
            ):
                cx, cy, bw, bh = map(float, box)
                rect = ((cx, cy), (bw, bh), float(angle))

                pts = cv2.boxPoints(rect).astype(int)
                cv2.polylines(frame, [pts], True, (255, 0, 0), 2)

                crop = crop_rotated(frame, rect)
                if crop.size > 0:
                    cv2.imwrite(
                        f"{crop_dir}/frame_{frame_id:06d}_obj{i}.jpg",
                        crop
                    )

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

# =========================
# MAIN
# =========================
if os.path.isdir(INPUT_PATH):
    for file in tqdm(os.listdir(INPUT_PATH)):
        full_path = os.path.join(INPUT_PATH, file)
        if is_image(full_path):
            process_image(full_path)
        elif is_video(full_path):
            process_video(full_path)

elif is_image(INPUT_PATH):
    process_image(INPUT_PATH)

elif is_video(INPUT_PATH):
    process_video(INPUT_PATH)

print("✅ Inference & cropping completed.")
