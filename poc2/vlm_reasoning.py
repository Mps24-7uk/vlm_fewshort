# vlm_reasoning.py
import os
from PIL import Image
from models import qwen_model, qwen_processor
from config import REFERENCE_DIR, QWEN_CONF_TH
import torch
def run_vlm_reasoning(results):
    resolved = 0
    unresolved = 0

    for idx, result in enumerate(results):
        if result["status"] != "review":
            continue

        query_image = Image.fromarray(result["roi"])

        ref_images = []
        for k in result["top_k"]:
            ref_path = os.path.join(REFERENCE_DIR, k["match"])
            ref_images.append(Image.open(ref_path).convert("RGB"))

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": query_image},
                {"type": "image", "image": ref_images[0]},
                {"type": "image", "image": ref_images[1]},
                {"type": "image", "image": ref_images[2]},
                {
                    "type": "text",
                    "text": "Return similarity score (0–100). Only number."
                }
            ]
        }]

        inputs = qwen_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(qwen_model.device)

        with torch.no_grad():
            out = qwen_model.generate(**inputs, max_new_tokens=10)

        decoded = qwen_processor.batch_decode(out, skip_special_tokens=True)[0]

        try:
            score = float("".join(filter(str.isdigit, decoded))) / 100
        except:
            score = 0.0

        if score >= QWEN_CONF_TH:
            result["status"] = "accepted"
            result["confidence"] = score
            resolved += 1
        else:
            unresolved += 1

    stats = {
        "total": resolved + unresolved,
        "resolved": resolved,
        "unresolved": unresolved
    }

    return results, stats
s