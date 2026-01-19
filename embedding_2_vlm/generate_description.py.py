from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from PIL import Image
import torch

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto"
)
model.eval()

def generate_chip_description(image_path, matches):
    reference_text = "\n".join(
        f"- {m['description']}" for m in matches
    )

    prompt = f"""
You are analyzing an image of a semiconductor chip.

Reference examples:
{reference_text}

Task:
Produce a concise, factual, and structured description of the chip.

Rules:
- Describe only what is directly observable.
- Do not infer missing information.
- Avoid assumptions or opinions.
- Use neutral, technical language.

Fields:
- Shape
- Orientation
- Body
- End caps / pins
- Surface
- Context
"""

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=False
        )

    return processor.decode(output[0], skip_special_tokens=True)
