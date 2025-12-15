import io
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ================= CONFIG =================
MODEL_DIR = Path(r"D:\hf_models\Qwen3-VL-8B-Instruct")

# ================= LOAD MODEL =================
torch.set_grad_enabled(False)

print("🔄 Loading Qwen3-VL model...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    dtype="auto",
    device_map="auto",
    local_files_only=True,
)
model.eval()

processor = AutoProcessor.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

print("✅ Model loaded successfully")

# ================= FASTAPI APP =================
app = FastAPI(title="Qwen3-VL Few-Shot Chip Inspector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "service": "Qwen3-VL Few-Shot Chip Inspector"}

# ================= TRUE FEW-SHOT ENDPOINT =================
@app.post("/vision-fewshot-chip")
async def vision_fewshot_chip(
    reference_images: List[UploadFile] = File(
        ..., description="5 reference chip images"
    ),
    query_image: UploadFile = File(
        ..., description="Query image to classify"
    ),
    max_new_tokens: int = Form(256),
):
    """
    TRUE few-shot chip classification using multiple images
    """

    if len(reference_images) < 3:
        raise HTTPException(
            status_code=400,
            detail="At least 3 reference chip images are required",
        )

    try:
        # ---------- Build multimodal message ----------
        content = []

        # Reference chips
        for idx, ref in enumerate(reference_images):
            img_bytes = await ref.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            content.append({"type": "image", "image": img})

        # Query image (LAST image is the one to classify)
        query_bytes = await query_image.read()
        query_img = Image.open(io.BytesIO(query_bytes)).convert("RGB")
        content.append({"type": "image", "image": query_img})

        # Instruction
        content.append({
            "type": "text",
            "text": """
You are an industrial vision inspector.

The first images are VALID semiconductor chips.
The LAST image is the QUERY.

TASK:
Compare the query image with the reference chips.

Respond STRICTLY in this format:

RESULT: CHIP or NOT_A_CHIP
CONFIDENCE: 0-100
REASON: one short sentence

Rules:
- Be very strict
- If unsure → NOT_A_CHIP
- No extra text
"""
        })

        messages = [{"role": "user", "content": content}]

        # ---------- Prepare inputs ----------
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = inputs.to(model.device)

        # ---------- Generate ----------
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


#uvicorn server:app --host 0.0.0.0 --port 8000


import requests

URL = "http://107.99.100.53:8000/vision-fewshot-chip"

reference_paths = [
    r"D:\chips\chip1.jpg",
    r"D:\chips\chip2.jpg",
    r"D:\chips\chip3.jpg",
    r"D:\chips\chip4.jpg",
    r"D:\chips\chip5.jpg",
]

query_path = r"D:\chips\test.jpg"

files = []

# Reference images
for p in reference_paths:
    files.append(("reference_images", open(p, "rb")))

# Query image
files.append(("query_image", open(query_path, "rb")))

response = requests.post(
    URL,
    files=files,
    data={"max_new_tokens": "200"},
    timeout=120,
)

print(response.json()["result"])









