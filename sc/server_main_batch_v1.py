import io
import torch
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ================= PERFORMANCE FLAGS =================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ================= CONFIG =================
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MAX_BATCH_SIZE = 16          # ⬆️ increase until OOM
MAX_NEW_TOKENS = 256

# ================= LOAD MODEL =================
print("🔄 Loading model...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,     # best for Ampere+
    device_map="cuda",
    attn_implementation="flash_attention_2",
).eval()

# 🔥 torch.compile for kernel fusion
model = torch.compile(model, mode="max-autotune")

processor = AutoProcessor.from_pretrained(MODEL_ID)

# ================= WARMUP =================
print("🔥 Warming up GPU...")
dummy_img = Image.new("RGB", (224, 224))
dummy_msg = [{
    "role": "user",
    "content": [
        {"type": "image", "image": dummy_img},
        {"type": "text", "text": "warmup"},
    ],
}]
with torch.inference_mode():
    inputs = processor.apply_chat_template(
        dummy_msg,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding="max_length",
        max_length=1024
    ).to("cuda")
    model.generate(**inputs, max_new_tokens=32)

print("✅ Model ready")

# ================= FASTAPI =================
app = FastAPI(title="Qwen3-VL High-Throughput API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= ENDPOINT =================
@app.post("/generate-batch")
async def generate_batch(
    image_files: List[UploadFile] = File(...),
    prompt: str = Form(...),
    max_new_tokens: int = Form(MAX_NEW_TOKENS),
):
    if len(image_files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch exceeds MAX_BATCH_SIZE={MAX_BATCH_SIZE}"
        )

    images, filenames = [], []

    for file in image_files:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        images.append(img)
        filenames.append(file.filename)

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

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",        # 🔥 static shapes
        max_length=1024
    ).to("cuda", non_blocking=True)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            use_cache=True,
        )

    generated = output_ids[:, inputs.input_ids.shape[1]:]

    texts = processor.batch_decode(
        generated,
        skip_special_tokens=True,
    )

    return {
        "results": [
            {"image_name": n, "description": t.strip()}
            for n, t in zip(filenames, texts)
        ]
    }
