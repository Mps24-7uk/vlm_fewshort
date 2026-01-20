import os
import faiss
import numpy as np
from tqdm import tqdm
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# ---------------- CONFIG ----------------
IMAGE_DIR = "data/chip"
INDEX_DIR = "index"
INDEX_PATH = os.path.join(INDEX_DIR, "chip.index")
META_PATH = os.path.join(INDEX_DIR, "metadata.npy")

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
BATCH_SIZE = 8
# ----------------------------------------

os.makedirs(INDEX_DIR, exist_ok=True)

# Load model
model = Qwen3VLEmbedder(
    model_name_or_path=MODEL_NAME
    # torch_dtype=torch.float16,
    # attn_implementation="flash_attention_2"
)

image_paths = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

assert len(image_paths) > 0, "No images found!"

all_embeddings = []

print(f"Processing {len(image_paths)} chip images...")

for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[i : i + BATCH_SIZE]

    inputs = [
        {"image": img_path}
        for img_path in batch_paths
    ]

    embeddings = model.process(inputs)  # (B, D)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    all_embeddings.append(embeddings)

all_embeddings = np.vstack(all_embeddings).astype("float32")
dim = all_embeddings.shape[1]

# ---------------- FAISS INDEX ----------------
index = faiss.IndexFlatIP(dim)  # cosine similarity (normalized vectors)
index.add(all_embeddings)

faiss.write_index(index, INDEX_PATH)
np.save(META_PATH, np.array(image_paths))

print("✅ FAISS index saved:", INDEX_PATH)
print("✅ Metadata saved:", META_PATH)
print("Embedding shape:", all_embeddings.shape)
