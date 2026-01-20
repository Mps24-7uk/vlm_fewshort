import json
import numpy as np
from scipy.special import softmax
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# ---------------- CONFIG ----------------
EMBEDDING_PATH = "embeddings/chip_embeddings.npy"
METADATA_PATH = "embeddings/metadata.json"
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
# ---------------------------------------

def l2_normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def cosine_similarity(query, database):
    return np.dot(database, query)

def main(input_image_path):
    # Load model
    model = Qwen3VLEmbedder(model_name_or_path=MODEL_NAME)

    # Load embeddings
    db_embeddings = np.load(EMBEDDING_PATH)
    db_embeddings = l2_normalize(db_embeddings)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    # Generate query embedding
    query_embedding = model.process({"image": input_image_path})
    query_embedding = l2_normalize(query_embedding.squeeze())

    # Similarity
    similarities = cosine_similarity(query_embedding, db_embeddings)

    # Best match
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    # Confidence (softmax over similarities)
    confidences = softmax(similarities)
    confidence = float(confidences[best_idx])

    result = {
        "predicted_label": "chip",
        "matched_image": metadata[best_idx]["image_name"],
        "similarity_score": round(best_score, 4),
        "confidence": round(confidence, 4)
    }

    print(result)
    return result

if __name__ == "__main__":
    # Example usage
    test_image = "test_chip.jpg"
    main(test_image)
