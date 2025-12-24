import os
import csv
import torch
import faiss
import numpy as np
from PIL import Image
from torchvision import transforms

from model import ResNetEmbedding


FAISS_INDEX_PATH = "chip_resnet.index"
TOP_K = 5
USE_GPU = True
CSV_OUTPUT = "folder_inference_results.csv"

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# ------------------------
# Image Transform
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------------
# FAISS Loader
# ------------------------
def load_faiss_index(index_path, use_gpu=True):
    index = faiss.read_index(index_path)

    if use_gpu and faiss.get_num_gpus() > 0:
        print("🚀 Using FAISS GPU")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        print("⚠️ Using FAISS CPU")

    return index


# ------------------------
# Load Model
# ------------------------
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNetEmbedding().to(device).eval()
    return model, device


# ------------------------
# Infer One Image
# ------------------------
def infer_single_image(image_path, model, index, device, top_k):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img_tensor).cpu().numpy().astype("float32")

    scores, ids = index.search(emb, top_k)
    scores = scores[0]
    ids = ids[0]  # FAISS IDs = labels

    defect_scores = scores[ids == 1]
    no_defect_scores = scores[ids == 0]

    defect_sim = float(defect_scores.mean()) if len(defect_scores) else 0.0
    no_defect_sim = float(no_defect_scores.mean()) if len(no_defect_scores) else 0.0

    decision = "DEFECT" if defect_sim > no_defect_sim else "NO_DEFECT"

    return decision, defect_sim, no_defect_sim, ids, scores


# ------------------------
# Infer Folder + Save CSV
# ------------------------
def infer_folder(folder_path):
    index = load_faiss_index(FAISS_INDEX_PATH, USE_GPU)
    model, device = load_model()

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(SUPPORTED_EXT)
    ]

    print(f"📂 Found {len(images)} images")

    # Prepare CSV header
    header = [
        "image_name",
        "decision",
        "defect_similarity",
        "no_defect_similarity"
    ]

    for i in range(TOP_K):
        header.append(f"top{i+1}_label")
        header.append(f"top{i+1}_similarity")

    rows = []

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)

        try:
            decision, defect_sim, no_defect_sim, ids, scores = infer_single_image(
                img_path, model, index, device, TOP_K
            )

            row = [
                img_name,
                decision,
                round(defect_sim, 6),
                round(no_defect_sim, 6)
            ]

            for i in range(TOP_K):
                row.append(int(ids[i]))
                row.append(round(float(scores[i]), 6))

            rows.append(row)

            print(f"✅ {img_name} → {decision}")

        except Exception as e:
            print(f"❌ Failed {img_name}: {e}")

    # Write CSV
    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n💾 CSV saved: {CSV_OUTPUT}")


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    FOLDER_PATH = "test_images"
    infer_folder(FOLDER_PATH)
