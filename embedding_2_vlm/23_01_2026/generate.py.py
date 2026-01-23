import os
import sys
import faiss
import clip
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# ================= CONFIG =================
IMAGE_DIR = "data/chip_images"     # folder with 1500 chip images
INDEX_SAVE_PATH = "chip.index"
PATHS_SAVE_PATH = "chip_paths.npy"

MODEL_NAME = "ViT-L/14"
BATCH_SIZE = 1                     # streaming (industrial safe)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================


def load_image(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        return image
    except Exception as e:
        print(f"[WARN] Skipping corrupted image: {img_path}")
        return None


def main():
    print("Loading CLIP model:", MODEL_NAME)
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()

    image_paths = []
    embeddings = []

    all_images = [
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    print("Total images found:", len(all_images))

    with torch.no_grad():
        for img_path in tqdm(all_images):
            img = load_image(img_path)
            if img is None:
                continue

            img_input = preprocess(img).unsqueeze(0).to(DEVICE)
            emb = model.encode_image(img_input)

            # Normalize → cosine similarity ready
            emb = emb / emb.norm(dim=-1, keepdim=True)

            emb = emb.cpu().numpy().astype("float32")

            embeddings.append(emb[0])
            image_paths.append(img_path)

    embeddings = np.vstack(embeddings)
    print("Final embedding shape:", embeddings.shape)

    dim = embeddings.shape[1]

    print("Creating FAISS index...")
    index = faiss.IndexFlatIP(dim)   # Inner Product = Cosine (since normalized)
    index.add(embeddings)

    print("Saving FAISS index...")
    faiss.write_index(index, INDEX_SAVE_PATH)

    print("Saving image paths metadata...")
    np.save(PATHS_SAVE_PATH, np.array(image_paths))

    print("\n=== DONE ===")
    print("Index:", INDEX_SAVE_PATH)
    print("Paths:", PATHS_SAVE_PATH)
    print("Total vectors:", index.ntotal)


if __name__ == "__main__":
    main()
