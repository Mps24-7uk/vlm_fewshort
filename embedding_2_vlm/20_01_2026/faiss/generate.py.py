import os
import faiss
import numpy as np
from tqdm import tqdm

from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# ==============================
# CONFIG
# ==============================
IMAGE_DIR = "images/chip"
FAISS_INDEX_PATH = "chip.index"
PATHS_SAVE_PATH = "chip_paths.npy"

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
BATCH_SIZE = 8              # adjust based on GPU memory
EMBED_DIM = 1024            # Qwen3-VL embedding dimension

# ==============================
# LOAD MODEL
# ==============================
model = Qwen3VLEmbedder(
    model_name_or_path=MODEL_NAME
)

# ==============================
# COLLECT IMAGE PATHS
# ==============================
image_paths = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
]

image_paths.sort()
print(f"[INFO] Found {len(image_paths)} chip images")

# ==============================
# FAISS INDEX (L2 / Cosine ready)
# ==============================
index = faiss.IndexFlatL2(EMBED_DIM)

all_embeddings = []

# ==============================
# EMBEDDING GENERATION
# ==============================
for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[i:i + BATCH_SIZE]

    inputs = [
        {"image": img_path}
        for img_path in batch_paths
    ]

    with torch.no_grad():
        embeddings = model.process(inputs)

    # ensure numpy float32
    embeddings = np.asarray(embeddings, dtype="float32")

    index.add(embeddings)
    all_embeddings.append(embeddings)

# ==============================
# SAVE ARTIFACTS
# ==============================
faiss.write_index(index, FAISS_INDEX_PATH)
np.save(PATHS_SAVE_PATH, np.array(image_paths))

print("===================================")
print("[SUCCESS] Embeddings generated")
print(f"FAISS index saved → {FAISS_INDEX_PATH}")
print(f"Image paths saved → {PATHS_SAVE_PATH}")
print("Total vectors      →", index.ntotal)
print("===================================")
