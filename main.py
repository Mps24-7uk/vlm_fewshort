import io
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ========== CONFIG ==========
MODEL_DIR = Path("Qwen3-VL-8B-Instruct")

# ========== LOAD MODEL ON STARTUP ==========

print("🔄 Loading Qwen3-VL-8B-Instruct from local path...")

# Disable gradients globally
torch.set_grad_enabled(False)

# Load model from local disk only
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    dtype="auto",           # ✅ use dtype instead of deprecated torch_dtype
    device_map="auto",
    local_files_only=True,
)
model.eval()

# Load processor
processor = AutoProcessor.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

print("✅ Model & processor loaded.")

# ========== FASTAPI APP SETUP ==========

app = FastAPI(title="Qwen3-VL FastAPI Server", version="1.0")

# Optional: enable CORS if you’ll call from browser/frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok", "model": "Qwen3-VL-8B-Instruct"}


# ========== VISION CHAT ENDPOINT ==========

@app.post("/vision-chat")
async def vision_chat(
    prompt: str = Form(..., description="User prompt about the image"),
    image_file: UploadFile = File(..., description="Image file"),
    max_new_tokens: int = Form(128, description="Max new tokens to generate"),
):
    """
    Send an image + text prompt, get back model-generated text.
    """
    # 1. Read image from uploaded file
    try:
        content = await image_file.read()
        if not content:
            raise ValueError("Empty image file.")
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # 2. Build messages in the same format you used earlier
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 3. Prepare inputs using the processor
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Move tensors to model device
        inputs = inputs.to(model.device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing inputs: {e}")

    # 4. Generate
    try:
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        # Trim input tokens from the output (same as your script)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        answer = output_texts[0] if output_texts else ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return {
        "prompt": prompt,
        "response": answer,
    }


# ========== OPTIONAL: TEXT-ONLY CHAT ENDPOINT ==========

@app.post("/text-chat")
async def text_chat(
    prompt: str = Form(..., description="Text-only prompt"),
    max_new_tokens: int = Form(128),
):
    """
    Simple text-only chat with Qwen3-VL.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        answer = output_texts[0] if output_texts else ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {e}")

    return {
        "prompt": prompt,
        "response": answer,
    }
