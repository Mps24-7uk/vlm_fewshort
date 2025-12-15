import io
from typing import List
from fastapi import File, Form, UploadFile, HTTPException
from PIL import Image

@app.post("/vision-fewshot-chip")
async def vision_fewshot_chip(
    reference_images: List[UploadFile] = File(...),
    reference_labels: List[str] = Form(...),
    query_image: UploadFile = File(...),
    instruction_prompt: str = Form(...),
    max_new_tokens: int = Form(256),
):
    """
    TRUE labeled few-shot chip classification.
    reference_labels must align with reference_images.
    """

    if len(reference_images) != len(reference_labels):
        raise HTTPException(
            status_code=400,
            detail="reference_images and reference_labels length mismatch",
        )

    if len(reference_images) < 4:
        raise HTTPException(
            status_code=400,
            detail="At least 4 reference images recommended",
        )

    try:
        content = []

        # -------- Reference images with labels --------
        for img_file, label in zip(reference_images, reference_labels):
            img_bytes = await img_file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Add image
            content.append({"type": "image", "image": img})

            # Add label text (VERY IMPORTANT)
            content.append({
                "type": "text",
                "text": f"This image is labeled as: {label.upper()}"
            })

        # -------- Query image --------
        query_bytes = await query_image.read()
        query_img = Image.open(io.BytesIO(query_bytes)).convert("RGB")

        content.append({"type": "image", "image": query_img})

        # -------- Instruction from client --------
        content.append({
            "type": "text",
            "text": instruction_prompt
        })

        messages = [{"role": "user", "content": content}]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        trimmed_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]

        response_text = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return {
            "result": response_text.strip()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



################################################



import requests

URL = "http://107.99.100.53:8000/vision-fewshot-chip"

# -------- Reference set (LABELED) --------
reference_data = [
    # CHIP examples
    ("D:/chips/chip_1.jpg", "chip"),
    ("D:/chips/chip_2.jpg", "chip"),
    ("D:/chips/chip_3.jpg", "chip"),

    # NON-CHIP examples
    ("D:/chips/noise_1.jpg", "non_chip"),
    ("D:/chips/noise_2.jpg", "non_chip"),
]

query_image_path = "D:/chips/query.jpg"

# -------- Instruction prompt (CLIP-style) --------
instruction_prompt = """
You are a strict industrial vision classifier.

You have seen labeled examples of CHIP and NON_CHIP.
The last image is the QUERY image.

Compare the query image visually against BOTH classes.

Respond STRICTLY in this format:

RESULT: CHIP or NOT_A_CHIP
CONFIDENCE: 0-100
REASON: one short sentence

Rules:
- Visual similarity only
- If closer to NON_CHIP → NOT_A_CHIP
- If ambiguous → NOT_A_CHIP
"""

files = []
labels = []

for path, label in reference_data:
    files.append(("reference_images", open(path, "rb")))
    labels.append(label)

files.append(("query_image", open(query_image_path, "rb")))

response = requests.post(
    URL,
    files=files,
    data={
        "reference_labels": labels,      # 🔑 labels aligned with images
        "instruction_prompt": instruction_prompt,
        "max_new_tokens": "200",
    },
    timeout=120,
)

print(response.json()["result"])

