import os
import faiss
import numpy as np
import torch
from tqdm import tqdm

from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# ==============================
# CONFIG
# ==============================
IMAGE_DIR = "images/chip"
FAISS_INDEX_PATH = "chip.index"
PATHS_SAVE_PATH = "chip_paths.npy"

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
EMBED_DIM = 1024   # Qwen3-VL embedding dimension

# ==============================
# LOAD MODEL
# ==============================
model = Qwen3VLEmbedder(
    model_name_or_path=MODEL_NAME
)

# ==============================
# LOAD IMAGE PATHS
# ==============================
image_paths = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
]

image_paths.sort()
print(f"[INFO] Found {len(image_paths)} chip images")

# ==============================
# CREATE FAISS INDEX
# ==============================
index = faiss.IndexFlatL2(EMBED_DIM)

# ==============================
# GENERATE EMBEDDINGS (ONE BY ONE)
# ==============================
for img_path in tqdm(image_paths):
    inputs = {"image": img_path}

    with torch.no_grad():
        embedding = model.process(inputs)

    embedding = np.asarray(embedding, dtype="float32")

    # shape safety: (1, D)
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    index.add(embedding)

# ==============================
# SAVE INDEX + PATHS
# ==============================
faiss.write_index(index, FAISS_INDEX_PATH)
np.save(PATHS_SAVE_PATH, np.array(image_paths))

print("===================================")
print("[SUCCESS] Embedding generation done")
print(f"FAISS index saved  → {FAISS_INDEX_PATH}")
print(f"Paths file saved  → {PATHS_SAVE_PATH}")
print(f"Total vectors     → {index.ntotal}")
print("===================================")
