import os
import faiss
import clip
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv

# ================= CONFIG =================
TEST_IMAGE_DIR = "data/test_images"
INDEX_PATH = "chip.index"
PATHS_SAVE_PATH = "chip_paths.npy"
MODEL_NAME = "ViT-L/14"
THRESHOLD = 0.85   # tune this
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_CSV = "inference_results.csv"
# =========================================


def load_image(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        return image
    except:
        print(f"[WARN] Corrupted image: {img_path}")
        return None


def main():
    print("Loading CLIP model...")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()

    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_PATH)

    print("Loading image paths...")
    chip_paths = np.load(PATHS_SAVE_PATH)

    results = []

    test_images = [
        os.path.join(TEST_IMAGE_DIR, f)
        for f in os.listdir(TEST_IMAGE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    print("Total test images:", len(test_images))

    with torch.no_grad():
        for img_path in tqdm(test_images):
            img = load_image(img_path)
            if img is None:
                continue

            img_input = preprocess(img).unsqueeze(0).to(DEVICE)
            emb = model.encode_image(img_input)

            # Normalize
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.cpu().numpy().astype("float32")

            # Search
            D, I = index.search(emb, k=1)

            similarity = float(D[0][0])   # cosine similarity
            nearest_idx = int(I[0][0])
            nearest_image = chip_paths[nearest_idx]

            prediction = "chip" if similarity >= THRESHOLD else "no_chip"

            results.append([
                os.path.basename(img_path),
                prediction,
                os.path.basename(nearest_image),
                round(similarity, 4)
            ])

    # Save CSV (industrial logging)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "prediction", "nearest_image", "confidence"])
        writer.writerows(results)

    print("\n=== INFERENCE COMPLETE ===")
    print("Results saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
