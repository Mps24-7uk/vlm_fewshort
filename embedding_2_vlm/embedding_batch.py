# embedding.py
import os
import json
import torch
import faiss
import numpy as np
from tqdm import tqdm

from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# ---------------- CONFIG ----------------
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
JSONL_PATH = "output.jsonl"
IMAGE_DIR = "data/chip_images"
ARTIFACT_DIR = "artifacts"

BATCH_SIZE = 2        # 🔥 set to 1 if VRAM < 16GB
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

# ---------------- TEXT BUILDER ----------------
def build_text(rec):
    label_text = (
        "electronic chip"
        if rec["label"].lower() == "chip"
        else "not an electronic chip"
    )
    return f"Label: {label_text}. Visual description: {rec['description']}"

# ---------------- BUILD INPUT LIST ----------------
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
    batch = inputs[i : i + BATCH_SIZE]

    with torch.no_grad(), torch.cuda.amp.autocast():
        batch_embs = embedder.process(batch)

    for emb in batch_embs:
        # 🔥 CRITICAL: move CUDA tensor → CPU numpy
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        all_embeddings.append(emb)

    # 🔥 prevent fragmentation
    torch.cuda.empty_cache()

# ---------------- BUILD FAISS INDEX ----------------
embeddings = np.array(all_embeddings, dtype="float32")
dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, os.path.join(ARTIFACT_DIR, "faiss.index"))

# ---------------- SAVE METADATA ----------------
with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w") as f:
    json.dump(records, f, indent=2)

with open(os.path.join(ARTIFACT_DIR, "config.json"), "w") as f:
    json.dump({"chip_threshold": CHIP_THRESHOLD}, f, indent=2)

print("✅ Embeddings saved successfully (batched)")
