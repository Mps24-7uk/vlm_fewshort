import io
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ================= CONFIG =================
MODEL_ID = "./Qwen3-VL-32B-Instruct"

# ================= LOAD MODEL =================
print("🔄 Loading Qwen3-VL model...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)

print("✅ Model and processor loaded")

# ================= FASTAPI =================
app = FastAPI(
    title="Qwen3-VL Image Description API",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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



@app.post("/vision-fire-smoke-check")
async def vision_fire_smoke_check(
    image_file: UploadFile = File(...),
    instruction_prompt: str = Form(...),
    max_new_tokens: int = Form(64),
):
    """
    Vision classification endpoint:
    Checks whether image contains FIRE, SMOKE, or NO_FIRE_SMOKE.
    Instruction is fully controlled by client.
    """

    # -------- Load image --------
    try:
        content = await image_file.read()
        if not content:
            raise ValueError("Empty image file")

        image = Image.open(io.BytesIO(content)).convert("RGB")
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

    return {
        "result": result
    }




# ================= ENDPOINT =================
@app.post("/generate")
async def generate(
    image_file: UploadFile = File(...),
    prompt: str = Form(...),
    max_new_tokens: int = Form(512),
):
    # -------- Load image --------
    try:
        image_bytes = await image_file.read()
        if not image_bytes:
            raise ValueError("Empty image")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # -------- Build messages --------
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
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
                top_p=1.0,
                do_sample=False,
            )

        trimmed_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]

        text = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {
        "image_name": image_file.filename,
        "result": text
    }










