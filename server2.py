# -*- coding: utf-8 -*-
import os
import io
import torch
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# ================== ENV ==================
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# ================== CONFIG ==================
CHECKPOINT_PATH = "Qwen/Qwen3-VL-32B-Instruct-FP8"
GPU_UTIL = 0.70

# ================== LOAD MODEL ==================
print("🔄 Loading processor...")
processor = AutoProcessor.from_pretrained(CHECKPOINT_PATH)

print("🔄 Loading vLLM engine...")
llm = LLM(
    model=CHECKPOINT_PATH,
    trust_remote_code=True,
    gpu_memory_utilization=GPU_UTIL,
    tensor_parallel_size=torch.cuda.device_count(),
    enforce_eager=False,
    seed=0,
)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=1024,
    top_k=-1,
    stop_token_ids=[],
)

print("✅ Qwen3-VL vLLM ready")

# ================== FASTAPI ==================
app = FastAPI(title="Qwen3-VL vLLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "ok", "model": CHECKPOINT_PATH}


# ================== HELPER ==================
def prepare_inputs_for_vllm(messages):
    """
    Converts Qwen-style messages to vLLM multimodal input
    """
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


# ================== API ==================
@app.post("/vision-generate")
async def vision_generate(
    prompt: str = Form(...),
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    max_tokens: int = Form(512),
):
    """
    Vision + Language inference using Qwen3-VL via vLLM
    """

    if image is None and image_url is None:
        raise HTTPException(
            status_code=400,
            detail="Provide either image file or image_url",
        )

    # ---------- Build Qwen messages ----------
    content = []

    if image is not None:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        content.append({"type": "image", "image": img})

    if image_url is not None:
        content.append({"type": "image", "image": image_url})

    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    # ---------- Prepare vLLM input ----------
    vllm_input = prepare_inputs_for_vllm(messages)

    params = sampling_params.clone()
    params.max_tokens = max_tokens

    # ---------- Generate ----------
    outputs = llm.generate([vllm_input], sampling_params=params)

    generated_text = outputs[0].outputs[0].text.strip()

    return {
        "response": generated_text
    }
############################################################################


# -*- coding: utf-8 -*-
import io
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= CONFIG =================
MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"

# ================= LOAD MODEL =================
print("🔄 Loading Qwen2.5-VL model...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)

print("✅ Model and processor loaded")

# ================= FASTAPI =================
app = FastAPI(
    title="Qwen2.5-VL Image Description API",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": str(model.device),
    }

# ================= INFERENCE ENDPOINT =================
@app.post("/vision-generate")
async def vision_generate(
    prompt: str = Form("Describe this image."),
    image_file: UploadFile = File(...),
    max_new_tokens: int = Form(128),
):
    """
    Image + text inference using Qwen2.5-VL
    """

    try:
        # ---------- Read image ----------
        image_bytes = await image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ---------- Build messages ----------
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # ---------- Prepare inputs ----------
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(model.device)

        # ---------- Generate ----------
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return {"response": output_text.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

