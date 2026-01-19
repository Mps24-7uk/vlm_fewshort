# similarity.py
import os
import json
import numpy as np
import faiss

from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

ARTIFACT_DIR = "artifacts"
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"

# ---------------- LOAD MODEL ----------------
embedder = Qwen3VLEmbedder(model_name_or_path=MODEL_NAME)

# ---------------- LOAD INDEX ----------------
index = faiss.read_index(os.path.join(ARTIFACT_DIR, "faiss.index"))

with open(os.path.join(ARTIFACT_DIR, "metadata.json")) as f:
    metadata = json.load(f)

def embed_query(image_path, description=""):
    """
    Compute embedding for a query chip image + optional text.
    """
    inp = {"image": image_path, "text": description}
    emb = embedder.process([inp])[0]
    return np.array(emb).astype("float32")

def classify_image(image_path, top_k=5):
    q_emb = embed_query(image_path)

    scores, indices = index.search(np.expand_dims(q_emb, 0), top_k)

    sims = scores[0].tolist()
    matches = [metadata[i] for i in indices[0]]

    # average similarity as confidence
    conf = float(np.mean(sims))
    decision = "true chip" if conf > 0.75 else "not"

    return {"decision": decision, "confidence": conf, "sims": sims, "matches": matches}

# ---------------- TEST ----------------
if __name__ == "__main__":
    res = classify_image("test.jpg")
    print(res)
