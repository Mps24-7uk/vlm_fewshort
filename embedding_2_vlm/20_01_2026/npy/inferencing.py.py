import os
import json
import csv
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# ---------------- CONFIG ----------------
EMBEDDING_PATH = "embeddings/chip_embeddings.npy"
METADATA_PATH = "embeddings/metadata.json"
TEST_IMAGE_DIR = "test_images"
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"

SIMILARITY_THRESHOLD = 0.78
OUTPUT_CSV = "batch_inference_results.csv"
# ---------------------------------------


def l2_normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cosine_similarity(query, database):
    return np.dot(database, query)


def main():
    # Load model once
    model = Qwen3VLEmbedder(model_name_or_path=MODEL_NAME)

    # Load embeddings
    db_embeddings = np.load(EMBEDDING_PATH)
    db_embeddings = l2_normalize(db_embeddings)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    image_files = [
        f for f in os.listdir(TEST_IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        raise ValueError(f"No images found in {TEST_IMAGE_DIR}")

    results = []

    for img_name in tqdm(image_files, desc="Inferencing images"):
        img_path = os.path.join(TEST_IMAGE_DIR, img_name)

        try:
            inputs = [
                {
                    "image": img_path
                }
            ]

            query_embedding = model.process(inputs)
            query_embedding = query_embedding[0].squeeze()
            query_embedding = query_embedding.detach().cpu().numpy()
            query_embedding = l2_normalize(query_embedding)

            similarities = cosine_similarity(query_embedding, db_embeddings)

            best_idx = int(np.argmax(similarities))
            best_similarity = float(similarities[best_idx])

            confidences = softmax(similarities)
            confidence = float(confidences[best_idx])

            if best_similarity < SIMILARITY_THRESHOLD:
                result = {
                    "image": img_name,
                    "predicted_label": "unknown",
                    "matched_image": "",
                    "similarity_score": round(best_similarity, 4),
                    "confidence": round(1 - confidence, 4)
                }
            else:
                result = {
                    "image": img_name,
                    "predicted_label": "chip",
                    "matched_image": metadata[best_idx]["image_name"],
                    "similarity_score": round(best_similarity, 4),
                    "confidence": round(confidence, 4)
                }

            results.append(result)
            print(result)

        except Exception as e:
            print(f"[ERROR] {img_name}: {e}")

    # -------- Save to CSV --------
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        fieldnames = [
            "image",
            "predicted_label",
            "matched_image",
            "similarity_score",
            "confidence"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Batch inference completed")
    print(f"📄 Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
