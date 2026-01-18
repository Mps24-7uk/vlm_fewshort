# server.py
import io
import torch
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ================= CONFIG =================
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MAX_BATCH_SIZE = 8   # 🔧 tune based on GPU memory

# ================= LOAD MODEL =================
print("🔄 Loading Qwen3-VL-8B-Instruct...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
).eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)

print("✅ Model loaded")

# ================= FASTAPI =================
app = FastAPI(
    title="Qwen3-VL Batch Generation API",
    version="1.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= BATCH ENDPOINT =================
@app.post("/generate-batch")
async def generate_batch(
    image_files: List[UploadFile] = File(...),
    prompt: str = Form(...),
    max_new_tokens: int = Form(256),
):
    if len(image_files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(image_files)} exceeds MAX_BATCH_SIZE={MAX_BATCH_SIZE}"
        )

    images = []
    filenames = []

    # -------- Load images --------
    for file in image_files:
        try:
            content = await file.read()
            if not content:
                raise ValueError("Empty image")

            img = Image.open(io.BytesIO(content)).convert("RGB")
            images.append(img)
            filenames.append(file.filename)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image {file.filename}: {e}"
            )

    # -------- Build messages (one per image) --------
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
        for img in images
    ]

    # -------- Prepare batched inputs --------
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True
        ).to(model.device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processor error: {e}")

    # -------- Batched Generation --------
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        # Trim prompt tokens per sample
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]

        texts = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {
        "results": [
            {
                "image_name": name,
                "description": text.strip()
            }
            for name, text in zip(filenames, texts)
        ]
    }
