# embedding.py
import os
import json
import numpy as np
import faiss
import torch
from tqdm import tqdm

from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# ---------------- CONFIG ----------------
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
JSONL_PATH = "output.jsonl"
IMAGE_DIR = "data/chip_images"
ARTIFACT_DIR = "artifacts"
BATCH_SIZE = 2        # 🔥 set to 1 if VRAM < 24GB
CHIP_THRESHOLD = 0.75

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
embedder = Qwen3VLEmbedder(
    model_name_or_path=MODEL_NAME,
    device="cuda"
)

# ---------------- LOAD DATA ----------------
with open(JSONL_PATH, "r") as f:
    records = [json.loads(l) for l in f]

# ---------------- BUILD INPUTS ----------------
def build_text(rec):
    label_text = (
        "electronic chip"
        if rec["label"].lower() == "chip"
        else "not an electronic chip"
    )
    return f"Label: {label_text}. Visual description: {rec['description']}"

inputs = [
    {
        "image": os.path.join(IMAGE_DIR, r["image_name"]),
        "text": build_text(r)
    }
    for r in records
]

# ---------------- BATCH EMBEDDING ----------------
all_embeddings = []

for i in tqdm(range(0, len(inputs), BATCH_SIZE), desc="Embedding batches"):
    batch = inputs[i:i + BATCH_SIZE]

    with torch.no_grad():
        batch_emb = embedder.process(batch)

    all_embeddings.extend(batch_emb)

    # 🔥 important for long runs
    torch.cuda.empty_cache()

# ---------------- SAVE FAISS INDEX ----------------
embeddings = np.array(all_embeddings).astype("float32")
dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, f"{ARTIFACT_DIR}/faiss.index")

# ---------------- SAVE METADATA ----------------
with open(f"{ARTIFACT_DIR}/metadata.json", "w") as f:
    json.dump(records, f, indent=2)

with open(f"{ARTIFACT_DIR}/config.json", "w") as f:
    json.dump({"chip_threshold": CHIP_THRESHOLD}, f, indent=2)

print("✅ Embeddings saved without OOM")
