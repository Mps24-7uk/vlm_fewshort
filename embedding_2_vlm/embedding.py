# embedding.py
import os
import json
import numpy as np
import faiss
from tqdm import tqdm

from scripts.qwen3_vl_embedding import Qwen3VLEmbedder  # huggingface helper
# Note: The module is provided in the model repo and loaded via trust_remote_code=True

# ---------------- CONFIG ----------------
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
JSONL_PATH = "output.jsonl"
IMAGE_DIR = "data/chip_images"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
embedder = Qwen3VLEmbedder(model_name_or_path=MODEL_NAME)

# ---------------- READ DATA ----------------
with open(JSONL_PATH, "r") as f:
    records = [json.loads(line) for line in f]

# ---------------- BUILD INPUTS ----------------
inputs = []
for rec in records:
    # format expected by the embedder
    item = {
        "text": rec["description"],
        "image": os.path.join(IMAGE_DIR, rec["image_name"])
    }
    inputs.append(item)

# ---------------- COMPUTE EMBEDDINGS ----------------
embeddings = embedder.process(inputs)  # NxD array

# ---------------- SAVE INDEX ----------------
embeddings = np.array(embeddings).astype("float32")
dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, os.path.join(ARTIFACT_DIR, "faiss.index"))

# ---------------- SAVE METADATA ----------------
metadata = records
with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f)

print("✅ Saved embeddings + FAISS index")
