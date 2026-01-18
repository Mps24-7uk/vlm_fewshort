# embedding.py
import os
import json
import torch
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# ---------------- CONFIG ----------------
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
IMAGE_DIR = "data/chip_images"
JSONL_PATH = "output.jsonl"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto"
)
model.eval()

# ---------------- LOAD DATA ----------------
with open(JSONL_PATH, "r") as f:
    records = [json.loads(line) for line in f]

embeddings = []
metadata = []

# ---------------- EMBEDDING FUNCTION ----------------
def compute_embedding(image: Image.Image, text: str):
    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            output_hidden_states=True,
            return_dict=True
        )

    # Last hidden state: [batch, seq_len, hidden_dim]
    hidden = outputs.hidden_states[-1]

    # Mean pool over sequence
    emb = hidden.mean(dim=1)

    # Normalize for cosine similarity
    emb = torch.nn.functional.normalize(emb, dim=-1)

    return emb.cpu().numpy()[0]


# ---------------- BUILD EMBEDDINGS ----------------
for rec in tqdm(records, desc="Building embeddings"):
    image_path = os.path.join(IMAGE_DIR, rec["image_name"])
    image = Image.open(image_path).convert("RGB")

    emb = compute_embedding(image, rec["description"])

    embeddings.append(emb)
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

print("✅ Embeddings and index saved successfully")
