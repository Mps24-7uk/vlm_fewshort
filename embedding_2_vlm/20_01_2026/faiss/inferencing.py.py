import faiss
import numpy as np
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# ---------------- CONFIG ----------------
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
INDEX_PATH = "index/chip.index"
META_PATH = "index/metadata.npy"
TOP_K = 5
CONFIDENCE_THRESHOLD = 0.6
# ----------------------------------------

# Load model
model = Qwen3VLEmbedder(
    model_name_or_path=MODEL_NAME
)

# Load FAISS index & metadata
index = faiss.read_index(INDEX_PATH)
image_paths = np.load(META_PATH)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def infer(image_path):
    # Generate embedding
    emb = model.process({"image": image_path})
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb.astype("float32")

    # FAISS search
    scores, indices = index.search(emb, TOP_K)

    scores = scores[0]      # cosine similarities
    indices = indices[0]

    confidences = softmax(scores)
    final_confidence = float(confidences[0])

    # ---------------- DECISION LOGIC ----------------
    if final_confidence > CONFIDENCE_THRESHOLD:
        predicted_label = "chip"
    else:
        predicted_label = "unknown"
    # ------------------------------------------------

    results = []
    for i in range(TOP_K):
        results.append({
            "matched_image": image_paths[indices[i]],
            "similarity": float(scores[i]),
            "confidence": float(confidences[i])
        })

    return {
        "predicted_label": predicted_label,
        "final_confidence": final_confidence,
        "results": results
    }

# ---------------- TEST ----------------
if __name__ == "__main__":
    test_image = "data/chip/img_001.jpg"
    output = infer(test_image)

    print("\nPrediction:", output["predicted_label"])
    print("Final Confidence:", output["final_confidence"])
    print("Top Matches:")
    for r in output["results"]:
        print(r)
