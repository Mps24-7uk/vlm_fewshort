import os
import csv
import faiss
import numpy as np
import torch
from PIL import Image
import open_clip
from tqdm import tqdm

# =========================
# CONFIG
# =========================
QUERY_DIR = "query_images"          # folder to test
FAISS_INDEX_PATH = "chip.index"
PATHS_SAVE_PATH = "chip_paths.npy"
OUTPUT_CSV = "inference_results.csv"

THRESHOLD = 0.85
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD FAISS + META
# =========================
print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)
ref_paths = np.load(PATHS_SAVE_PATH)

# =========================
# LOAD CLIP
# =========================
print("Loading CLIP model...")
model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model = model.to(DEVICE)
model.eval()

# =========================
# INFERENCE LOOP
# =========================
results = []

query_images = [
    f for f in os.listdir(QUERY_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Running inference on {len(query_images)} images")

with torch.no_grad():
    for img_name in tqdm(query_images):
        img_path = os.path.join(QUERY_DIR, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

            emb = model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=1, keepdim=True)
            emb = emb.cpu().numpy().astype("float32")

            scores, ids = index.search(emb, 1)

            confidence = float(scores[0][0])
            nearest_img = ref_paths[ids[0][0]]

            prediction = "chip" if confidence >= THRESHOLD else "no_chip"

            results.append([
                img_name,
                os.path.basename(nearest_img),
                round(confidence, 4),
                prediction
            ])

        except Exception as e:
            print("Failed:", img_name, e)

# =========================
# SAVE CSV
# =========================
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "nearest_image", "confidence", "prediction"])
    writer.writerows(results)

print("Saved results to:", OUTPUT_CSV)
