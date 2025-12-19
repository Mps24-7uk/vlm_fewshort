import os
import torch
import numpy as np
import faiss
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import ResNetEmbedding

EMBEDDING_DB = "chip_resnet_embeddings.npz"
FAISS_INDEX_PATH = "chip_resnet.index"
BATCH_SIZE = 32
TOP_K = 5
USE_GPU = True

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_faiss_index(index_path, use_gpu=True):
    index = faiss.read_index(index_path)

    if use_gpu and faiss.get_num_gpus() > 0:
        print("🚀 FAISS GPU enabled")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        print("⚠️ FAISS CPU mode")

    return index

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


def batch_infer(
    image_dir,
    index,
    batch_size=BATCH_SIZE,
    top_k=TOP_K
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = ResNetEmbedding().to(device).eval()

    # Load metadata
    db = np.load(EMBEDDING_DB, allow_pickle=True)
    labels = db["labels"]
    paths = db["paths"]

    # Collect images
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

        # Generate embeddings
        with torch.no_grad():
            batch_emb = model(images).cpu().numpy().astype("float32")

        # FAISS search
        scores, indices = index.search(batch_emb, top_k)

        for j, img_path in enumerate(valid_paths): # type: ignore
            s = scores[j]
            idx = indices[j]

            top_labels = labels[idx]

            defect_scores = s[top_labels == 1]
            no_defect_scores = s[top_labels == 0]

            defect_sim = defect_scores.mean() if len(defect_scores) else 0.0
            no_defect_sim = no_defect_scores.mean() if len(no_defect_scores) else 0.0

            decision = "DEFECT" if defect_sim > no_defect_sim else "NO_DEFECT"

            results.append({
                "image": img_path,
                "decision": decision,
                "defect_similarity": float(defect_sim),
                "no_defect_similarity": float(no_defect_sim),
                "top_similarity": float(s[0])
            })

    return results


if __name__ == "__main__":
    index = load_faiss_index(
        FAISS_INDEX_PATH,
        use_gpu=USE_GPU
    )

    results = batch_infer(image_dir="batch_test_images", index=index)

    for r in results:
        print(
            f"{r['image']} | "
            f"{r['decision']} | "
            f"defect={r['defect_similarity']:.3f} | "
            f"no_defect={r['no_defect_similarity']:.3f}"
        )
