# similarity.py
import os
import json
import torch
import faiss
import numpy as np
from tqdm import tqdm

from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# ---------------- CONFIG ----------------
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
ARTIFACT_DIR = "artifacts"

# ---------------- LOAD MODEL ----------------
embedder = Qwen3VLEmbedder(
    model_name_or_path=MODEL_NAME,
    device="cuda"
)

# ---------------- LOAD INDEX & METADATA ----------------
index = faiss.read_index(os.path.join(ARTIFACT_DIR, "faiss.index"))

with open(os.path.join(ARTIFACT_DIR, "metadata.json")) as f:
    metadata = json.load(f)

with open(os.path.join(ARTIFACT_DIR, "config.json")) as f:
    config = json.load(f)

CHIP_THRESHOLD = config.get("chip_threshold", 0.75)

# ---------------- QUERY TEXT ----------------
QUERY_TEXT = """
Task: Determine whether the object is an electronic chip.
Analyze shape, orientation, texture, material, and context.
"""

# ---------------- EMBEDDING FUNCTION (NO BATCH) ----------------
def embed_query(image_path: str) -> np.ndarray:
    inp = {
        "image": image_path,
        "text": QUERY_TEXT
    }

    with torch.no_grad(), torch.cuda.amp.autocast():
        emb = embedder.process([inp])[0]

    # 🔥 critical: move to CPU immediately
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()

    return emb.astype("float32")

# ---------------- CLASSIFICATION ----------------
def classify_image(image_path: str, top_k: int = 5) -> dict:
    query_emb = embed_query(image_path)
    query_emb = np.expand_dims(query_emb, axis=0)

    scores, indices = index.search(query_emb, top_k)

    similarities = scores[0].tolist()
    matches = [metadata[i] for i in indices[0]]

    confidence = float(np.mean(similarities))
    decision = "true chip" if confidence >= CHIP_THRESHOLD else "not"

    return {
        "decision": decision,
        "confidence": round(confidence, 3),
        "similarities": [round(s, 3) for s in similarities],
        "matches": matches
    }

# ---------------- OPTIONAL EXPLANATION (TEXT ONLY) ----------------
def explain_decision(result: dict) -> str:
    """
    Lightweight explanation using retrieved metadata.
    No generation call (keeps GPU free).
    """
    lines = [
        f"Decision: {result['decision']}",
        f"Confidence: {result['confidence']}",
        "Top matches:"
    ]

    for m, s in zip(result["matches"], result["similarities"]):
        lines.append(
            f"- {m['image_name']} ({m['label']}): similarity={s}"
        )

    return "\n".join(lines)

# ---------------- TEST ----------------
if __name__ == "__main__":
    test_image = "test.jpg"  # change path

    result = classify_image(test_image)
    print(result)

    print("\nExplanation:\n")
    print(explain_decision(result))
