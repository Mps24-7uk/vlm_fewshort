import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os
from PIL import Image
from config import REFERENCE_DIR, QWEN_CONF_TH

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-32B-Instruct",
    dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")

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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": query_image},
                    {"type": "image", "image": ref_images[0]},
                    {"type": "image", "image": ref_images[1]},
                    {"type": "image", "image": ref_images[2]},
                    {
                        "type": "text",
                        "text": """
The first image is the QUERY.
The next three images are REFERENCE.

Task:
Compare query with each reference.
Return similarity score (0–100).
"""
                    }
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]

        output = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True
        )[0]


        if int(output)*0.01>= QWEN_CONF_TH:
            results[idx]["status"]="accepted"
            results[idx]["confidence"]=int(output)*0.01
            resolved += 1
        else:   
            unresolved += 1
            
        vlm_stats = {
            "total": resolved + unresolved,
            "resolved": resolved,
            "unresolved": unresolved,
#       "resolution_ratio": resolved / max(1, (resolved + unresolved))
        }

    return results,vlm_stats
