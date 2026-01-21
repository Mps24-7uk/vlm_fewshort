import os
import csv
import faiss
import numpy as np
import torch
from tqdm import tqdm

from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# ==============================
# CONFIG
# ==============================
QUERY_IMAGE_DIR = "images/inference"     # 8000 images
FAISS_INDEX_PATH = "chip.index"
PATHS_SAVE_PATH = "chip_paths.npy"
CSV_OUTPUT_PATH = "inference_results.csv"

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
TOP_K = 1
REJECTION_THRESHOLD = 0.8   # tune this

# ==============================
# LOAD MODEL
# ==============================
model = Qwen3VLEmbedder(
    model_name_or_path=MODEL_NAME
)

# ==============================
# LOAD FAISS INDEX
# ==============================
index = faiss.read_index(FAISS_INDEX_PATH)
image_paths_db = np.load(PATHS_SAVE_PATH)

print(f"[INFO] FAISS index loaded with {index.ntotal} vectors")

# ==============================
# LOAD QUERY IMAGES
# ==============================
query_images = [
    os.path.join(QUERY_IMAGE_DIR, f)
    for f in os.listdir(QUERY_IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
]

query_images.sort()
print(f"[INFO] Found {len(query_images)} images for inference")

# ==============================
# CSV SETUP
# ==============================
with open(CSV_OUTPUT_PATH, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_name", "decision", "best_distance"])

    # ==============================
    # INFERENCE LOOP (ONE BY ONE)
    # ==============================
    for img_path in tqdm(query_images):
        try:
            with torch.no_grad():
                embedding = model.process({"image": img_path})

            embedding = np.asarray(embedding, dtype="float32")

            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)

            distances, indices = index.search(embedding, TOP_K)
            best_distance = float(distances[0][0])

            if best_distance <= REJECTION_THRESHOLD:
                decision = "NORMAL_CHIP"
            else:
                decision = "ANOMALY_DEFECT"

            writer.writerow([
                os.path.basename(img_path),
                decision,
                round(best_distance, 6)
            ])

        except Exception as e:
            writer.writerow([
                os.path.basename(img_path),
                "ERROR",
                -1
            ])
            print(f"[ERROR] {img_path} → {e}")

print("===================================")
print("[SUCCESS] Bulk inferencing complete")
print(f"CSV saved at → {CSV_OUTPUT_PATH}")
print("===================================")
