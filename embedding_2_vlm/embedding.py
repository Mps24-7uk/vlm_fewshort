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

# ---------------- EMBEDDING LOOP (SINGLE ITEM) ----------------
all_embeddings = []

for rec in tqdm(records, desc="Embedding (no batch)"):
    inp = {
        "image": os.path.join(IMAGE_DIR, rec["image_name"]),
        "text": build_text(rec)
    }

    with torch.no_grad(), torch.cuda.amp.autocast():
        emb = embedder.process([inp])[0]

    # 🔥 CRITICAL: move off GPU immediately
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()

    all_embeddings.append(emb)

    # 🔥 aggressive cleanup
    torch.cuda.empty_cache()

# ---------------- FAISS INDEX ----------------
embeddings = np.array(all_embeddings, dtype="float32")
dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, f"{ARTIFACT_DIR}/faiss.index")

# ---------------- SAVE METADATA ----------------
with open(f"{ARTIFACT_DIR}/metadata.json", "w") as f:
    json.dump(records, f, indent=2)

with open(f"{ARTIFACT_DIR}/config.json", "w") as f:
    json.dump({"chip_threshold": CHIP_THRESHOLD}, f, indent=2)

print("✅ Embeddings saved (no batch, optimized)")
