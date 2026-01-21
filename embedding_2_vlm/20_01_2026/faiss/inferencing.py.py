import os
import faiss
import numpy as np
import torch
from tqdm import tqdm
import csv

from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# ==============================
# CONFIG (EDIT ONLY THIS)
# ==============================
INPUT_IMAGE_DIR = "images/inference"   # folder with input images
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
db_image_paths = np.load(PATHS_SAVE_PATH)

print(f"[INFO] FAISS index loaded with {index.ntotal} vectors")

# ==============================
# LOAD INPUT IMAGES
# ==============================
input_images = [
    os.path.join(INPUT_IMAGE_DIR, f)
    for f in os.listdir(INPUT_IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
]

input_images.sort()
print(f"[INFO] Found {len(input_images)} images for inference")

# ==============================
# CSV SETUP
# ==============================
with open(CSV_OUTPUT_PATH, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_name", "decision", "best_distance", "nearest_image"])

    # ==============================
    # INFERENCE LOOP (ONE BY ONE)
    # ==============================
    for img_path in tqdm(input_images):

        # MUST be list of dicts for Qwen
        inputs = [{"image": img_path}]

        with torch.no_grad():
            embedding = model.process(inputs)

        # GPU → CPU → numpy
        if torch.is_tensor(embedding):
            embedding = embedding.cpu().numpy()
        else:
            embedding = np.asarray(embedding)

        embedding = embedding.astype("float32")

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # ==============================
        # FAISS SEARCH
        # ==============================
        distances, indices = index.search(embedding, TOP_K)
        best_distance = float(distances[0][0])
        best_match_path = db_image_paths[indices[0][0]]

        # ==============================
        # DECISION
        # ==============================
        if best_distance <= REJECTION_THRESHOLD:
            decision = "NORMAL_CHIP"
        else:
            decision = "ANOMALY_DEFECT"

        # ==============================
        # WRITE CSV
        # ==============================
        writer.writerow([
            os.path.basename(img_path),
            decision,
            round(best_distance, 6),
            os.path.basename(best_match_path)
        ])

print("===================================")
print("[SUCCESS] Inference complete")
print(f"CSV saved at → {CSV_OUTPUT_PATH}")
print("===================================")
