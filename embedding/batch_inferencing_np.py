import os
import torch
import faiss
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import ResNetEmbedding

FAISS_INDEX_PATH = "chip_resnet.index"
BATCH_SIZE = 32
TOP_K = 5
USE_GPU = True

# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# FAISS loader
# -----------------------------
def load_faiss_index(index_path, use_gpu=True):
    index = faiss.read_index(index_path)

    if use_gpu and faiss.get_num_gpus() > 0:
        print("🚀 FAISS GPU enabled")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        print("⚠️ FAISS CPU mode")

    return index

# -----------------------------
# Batch image loader
# -----------------------------
def load_image_batch(image_paths):
    images = []
    valid_paths = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(transform(img))
            valid_paths.append(path)
        except Exception as e:
            print(f"❌ Skipped {path}: {e}")

    if not images:
        return None, None

    return torch.stack(images), valid_paths

# -----------------------------
# Batch inference
# -----------------------------
def batch_infer(
    image_dir,
    index,
    batch_size=BATCH_SIZE,
    top_k=TOP_K,
    similarity_threshold=0.75
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNetEmbedding().to(device).eval()

    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    results = []

    for i in tqdm(range(0, len(image_files), batch_size), desc="Batch inferencing"):
        batch_paths = image_files[i:i + batch_size]
        images, valid_paths = load_image_batch(batch_paths)

        if images is None:
            continue

        images = images.to(device)

        with torch.no_grad():
            embeddings = model(images)
            embeddings = embeddings.cpu().numpy().astype("float32")

        scores, indices = index.search(embeddings, top_k)

        for j, img_path in enumerate(valid_paths):
            top_score = float(scores[j][0])

            decision = "CHIP" if top_score >= similarity_threshold else "NOT_A_CHIP"

            results.append({
                "image": img_path,
                "decision": decision,
                "top_similarity": top_score
            })

    return results

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    index = load_faiss_index(
        FAISS_INDEX_PATH,
        use_gpu=USE_GPU
    )

    results = batch_infer(
        image_dir="batch_test_images",
        index=index
    )

    for r in results:
        print(
            f"{r['image']} | "
            f"{r['decision']} | "
            f"similarity={r['top_similarity']:.3f}"
        )
