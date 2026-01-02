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


@app.post("/vision-fire-smoke")
async def vision_fire_smoke(
    image_file: UploadFile = File(...),
    instruction_prompt: str = Form(...),
    max_new_tokens: int = Form(256),
):
    """
    Generic vision detection endpoint.
    Supports MULTIPLE detections via client-defined instruction.
    """

    # -------- Load image --------
    try:
        content = await image_file.read()
        if not content:
            raise ValueError("Empty image file")

        image = Image.open(io.BytesIO(content)).convert("RGB")
        image = image.resize((224, 224))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # -------- Build messages --------
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction_prompt},
            ],
        }
    ]

    # -------- Prepare inputs --------
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processor error: {e}")

    # -------- Generate --------
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )

        trimmed_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]

        result = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # -------- Light validation --------
    lines = [l for l in result.splitlines() if l.strip()]
    for line in lines:
        if line.count(",") != 4:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid output line: {line}"
            )

    return {
        "result": result
    }



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


######################################


import os
import csv
import requests

URL = "http://127.0.0.1:8000/vision-fire-smoke"

IMAGE_DIR = r"D:/fire_images"        # 🔁 CHANGE THIS
OUTPUT_CSV = "fire_smoke_results.csv"

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp")

instruction_prompt = """
You are an industrial fire & smoke detection vision system.

Task:
1. Analyze the image.
2. Detect ALL visible regions of:
   - FIRE
   - SMOKE

For EACH detected region:
- Draw ONE bounding box.

Image resolution is 224x224.

Respond STRICTLY in this format:
class,x1,y1,x2,y2

Rules:
- One line per detected region
- Bounding box values must be integers
- If no fire or smoke detected:
  NO_FIRE_SMOKE,0,0,0,0
- Do not add explanations
"""

# -------- Collect images --------
image_paths = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(SUPPORTED_EXT)
]

if not image_paths:
    raise RuntimeError("No images found in directory")

print(f"📂 Found {len(image_paths)} images")

# -------- CSV setup --------
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image", "class", "x1", "y1", "x2", "y2"])

    # -------- Process images --------
    for img_path in image_paths:
        try:
            with open(img_path, "rb") as img_file:
                response = requests.post(
                    URL,
                    files={"image_file": img_file},
                    data={
                        "instruction_prompt": instruction_prompt,
                        "max_new_tokens": "256",
                    },
                    timeout=60,
                )

            result = response.json()["result"]
            print(f"\n🖼️ {os.path.basename(img_path)}")
            print(result)

            lines = [l for l in result.splitlines() if l.strip()]

            for line in lines:
                cls, x1, y1, x2, y2 = line.split(",")
                writer.writerow([
                    os.path.basename(img_path),
                    cls.strip(),
                    x1.strip(),
                    y1.strip(),
                    x2.strip(),
                    y2.strip(),
                ])

        except Exception as e:
            print(f"❌ Failed {img_path}: {e}")



