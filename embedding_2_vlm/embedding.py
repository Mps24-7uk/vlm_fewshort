# embedding.py
import os
import json
import torch
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

# ---------------- CONFIG ----------------
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
IMAGE_DIR = "data/chip_images"
JSONL_PATH = "output.jsonl"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ---------------- LOAD DATA ----------------
with open(JSONL_PATH, "r") as f:
    records = [json.loads(line) for line in f]

embeddings = []
metadata = []

# ---------------- BUILD EMBEDDINGS ----------------
for rec in tqdm(records, desc="Embedding images"):
    image_path = os.path.join(IMAGE_DIR, rec["image_name"])
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        text=rec["description"],
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        emb = model.get_image_text_features(**inputs)
        emb = torch.nn.functional.normalize(emb, dim=-1)

    embeddings.append(emb.cpu().numpy()[0])
    metadata.append({
        "image_name": rec["image_name"],
        "label": rec["label"],
        "description": rec["description"]
    })

# ---------------- SAVE FAISS INDEX ----------------
embeddings = np.array(embeddings).astype("float32")
dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, os.path.join(ARTIFACT_DIR, "faiss.index"))

# ---------------- SAVE METADATA ----------------
with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# ---------------- SAVE CONFIG ----------------
with open(os.path.join(ARTIFACT_DIR, "config.json"), "w") as f:
    json.dump({
        "chip_threshold": 0.75
    }, f, indent=2)

print("✅ Embeddings, index, and metadata saved successfully")
