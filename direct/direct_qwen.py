import os
import json
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from tqdm import tqdm

# =====================
# CONFIG
# =====================
IMAGE_DIR = "images"
OUTPUT_JSONL = "output.jsonl"
BATCH_SIZE = 4        # increase gradually (4, 8) based on GPU VRAM
MAX_NEW_TOKENS = 256

PROMPT = (
    "Given an image of a semiconductor chip, produce a concise, factual, "
    "and structured description of its visible physical characteristics. "
    "Describe only what is directly observable in the image, including chip type, "
    "viewing angle, orientation, materials, geometry, and surface features. "
    "Do not infer missing information, do not add assumptions or opinions, "
    "and avoid any speculative language. Use consistent field-style formatting "
    "and neutral technical wording."
)

# =====================
# LOAD MODEL
# =====================
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# =====================
# UTILS
# =====================
def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# =====================
# LOAD IMAGE LIST
# =====================
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
])

# =====================
# BATCH PROCESSING
# =====================
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
    for batch_files in tqdm(
        list(chunk_list(image_files, BATCH_SIZE)),
        desc="Processing batches"
    ):
        messages_batch = []
        valid_image_names = []

        for image_name in batch_files:
            image_path = os.path.join(IMAGE_DIR, image_name)
            try:
                image = Image.open(image_path).convert("RGB")

                messages_batch.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ])
                valid_image_names.append(image_name)

            except Exception as e:
                print(f"❌ Failed to load {image_name}: {e}")

        if not messages_batch:
            continue

        # Prepare batch inputs
        inputs = processor.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True
        ).to(model.device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        # Trim prompts
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        descriptions = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Write JSONL
        for image_name, description in zip(valid_image_names, descriptions):
            record = {
                "image_name": image_name,
                "description": description.strip()
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
