# similarity.py
import json
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

# ---------------- CONFIG ----------------
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
ARTIFACT_DIR = "artifacts"

# ---------------- LOAD MODEL ----------------
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ---------------- LOAD INDEX & METADATA ----------------
index = faiss.read_index(f"{ARTIFACT_DIR}/faiss.index")

with open(f"{ARTIFACT_DIR}/metadata.json") as f:
    metadata = json.load(f)

with open(f"{ARTIFACT_DIR}/config.json") as f:
    config = json.load(f)

CHIP_THRESHOLD = config["chip_threshold"]

# ---------------- QUERY EMBEDDING ----------------
def embed_query(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")

    prompt = (
        "Describe the object focusing on shape, orientation, texture, "
        "material, context, and whether it resembles an electronic chip."
    )

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        emb = model.get_image_text_features(**inputs)
        emb = torch.nn.functional.normalize(emb, dim=-1)

    return emb.cpu().numpy().astype("float32")

# ---------------- SIMILARITY + CONFIDENCE ----------------
def classify_image(image_path: str, top_k: int = 5) -> dict:
    query_emb = embed_query(image_path)

    scores, indices = index.search(query_emb, top_k)

    similarities = scores[0]
    matches = [metadata[i] for i in indices[0]]

    confidence = float(np.mean(similarities))
    decision = "true chip" if confidence >= CHIP_THRESHOLD else "not"

    return {
        "decision": decision,
        "confidence": round(confidence, 3),
        "similarities": [round(float(s), 3) for s in similarities],
        "matches": matches
    }

# ---------------- OPTIONAL FINAL REASONING ----------------
def explain_decision(image_path: str) -> str:
    result = classify_image(image_path)

    context = "\n".join(
        f"- {m['description']} (label: {m['label']})"
        for m in result["matches"]
    )

    prompt = f"""
You are an expert in identifying electronic chips.

Reference examples:
{context}

Similarity decision: {result['decision']}
Confidence score: {result['confidence']}

Analyze the image and provide final confirmation.

Answer format:
Decision:
Explanation:
"""

    inputs = processor(
        images=Image.open(image_path).convert("RGB"),
        text=prompt,
        return_tensors="pt"
    ).to("cuda")

    output = model.generate(**inputs, max_new_tokens=200)
    return processor.decode(output[0], skip_special_tokens=True)

# ---------------- TEST ----------------
if __name__ == "__main__":
    result = classify_image("test.jpg")
    print(result)

    explanation = explain_decision("test.jpg")
    print(explanation)
