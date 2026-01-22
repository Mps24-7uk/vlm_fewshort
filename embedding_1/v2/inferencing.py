import os
import faiss
import torch
import numpy as np
import csv
from PIL import Image
from torchvision import models, transforms

# ---------------- CONFIG ----------------
QUERY_DIR = "data/query_images"
FAISS_INDEX_PATH = "chip.index"
PATHS_SAVE_PATH = "chip_paths.npy"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 1
OUTPUT_CSV = "chip_inference_results.csv"

CONFIDENCE_THRESHOLD = 0.75   # tune this!
# ---------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load ResNet50
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()
model = model.to(DEVICE)
model.eval()

# Load FAISS + paths
index = faiss.read_index(FAISS_INDEX_PATH)
paths = np.load(PATHS_SAVE_PATH, allow_pickle=True)

def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(img)
    emb = emb.cpu().numpy().astype("float32")

    # L2 normalize for cosine
    faiss.normalize_L2(emb)
    return emb

# ----------- INFERENCE LOOP ------------

query_images = sorted([
    os.path.join(QUERY_DIR, f)
    for f in os.listdir(QUERY_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

results = []

for qpath in query_images:
    query_emb = get_embedding(qpath)
    D, I = index.search(query_emb, TOP_K)

    nearest_idx = I[0][0]
    confidence = float(D[0][0])
    query_name = os.path.basename(qpath)
    nearest_name = os.path.basename(paths[nearest_idx])

    # ---- Final prediction ----
    if confidence >= CONFIDENCE_THRESHOLD:
        prediction = "chip"
    else:
        prediction = "no_chip"

    results.append([query_name, confidence, nearest_name, prediction])

    print(f"{query_name} -> {nearest_name} | conf={confidence:.3f} | pred={prediction}")

# -------- SAVE CSV --------

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["query_image", "confidence", "nearest_image", "prediction"])
    writer.writerows(results)

print("\nSaved results to:", OUTPUT_CSV)
