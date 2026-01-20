import os
import json
import numpy as np
from tqdm import tqdm
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# ---------------- CONFIG ----------------
IMAGE_DIR = "data/chip"
SAVE_DIR = "embeddings"
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
# ---------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    # Initialize model
    model = Qwen3VLEmbedder(
        model_name_or_path=MODEL_NAME
        # torch_dtype=torch.float16,
        # attn_implementation="flash_attention_2"
    )

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    all_embeddings = []
    metadata = []

    for img_name in tqdm(image_files, desc="Generating embeddings"):
        img_path = os.path.join(IMAGE_DIR, img_name)

        inputs = {
            "image": img_path
        }

        embedding = model.process(inputs)
        embedding = embedding.squeeze()  # (D,)

        all_embeddings.append(embedding)
        metadata.append({
            "image_name": img_name,
            "label": "chip"
        })

    # Convert to numpy array
    all_embeddings = np.stack(all_embeddings)

    # Save embeddings & metadata
    np.save(os.path.join(SAVE_DIR, "chip_embeddings.npy"), all_embeddings)

    with open(os.path.join(SAVE_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(all_embeddings)} embeddings")

if __name__ == "__main__":
    main()
