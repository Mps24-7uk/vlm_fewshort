import os
import numpy as np
import torch
import faiss
from PIL import Image
from tqdm import tqdm
import open_clip

# =========================
# CONFIG
# =========================
IMAGE_DIR = "chip_images"
FAISS_INDEX_PATH = "chip.index"
PATHS_SAVE_PATH = "chip_paths.npy"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL
# =========================
print("Loading CLIP model...")
model, preprocess, _ = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED
)
model = model.to(DEVICE)
model.eval()

# =========================
# LOAD IMAGES
# =========================
image_paths = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Found {len(image_paths)} chip images")

# =========================
# EMBEDDING LOOP
# =========================
all_embeddings = []
valid_paths = []

with torch.no_grad():
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        images = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                img = preprocess(img)
                images.append(img)
                valid_paths.append(p)
            except Exception as e:
                print("Skipping:", p, e)

        if len(images) == 0:
            continue

        images = torch.stack(images).to(DEVICE)
        embeddings = model.encode_image(images)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        all_embeddings.append(embeddings.cpu().numpy())

# =========================
# STACK EMBEDDINGS
# =========================
embeddings_np = np.vstack(all_embeddings).astype("float32")
dim = embeddings_np.shape[1]

print("Embedding shape:", embeddings_np.shape)

# =========================
# BUILD FAISS INDEX
# =========================
index = faiss.IndexFlatIP(dim)  # cosine similarity
index.add(embeddings_np)

faiss.write_index(index, FAISS_INDEX_PATH)
np.save(PATHS_SAVE_PATH, np.array(valid_paths))

print("Saved:")
print(" FAISS index ->", FAISS_INDEX_PATH)
print(" Paths meta ->", PATHS_SAVE_PATH)
