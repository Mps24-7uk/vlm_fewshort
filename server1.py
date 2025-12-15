@app.post("/vision-fewshot-chip")
async def vision_fewshot_chip(
    reference_images: List[UploadFile] = File(
        ..., description="Reference chip images"
    ),
    query_image: UploadFile = File(
        ..., description="Query image"
    ),
    instruction_prompt: str = Form(
        ..., description="Few-shot / CLIP-style instruction prompt from client"
    ),
    max_new_tokens: int = Form(256),
):
    """
    TRUE few-shot chip classification.
    Instruction prompt is fully controlled by the client.
    """

    if len(reference_images) < 3:
        raise HTTPException(
            status_code=400,
            detail="At least 3 reference images required",
        )

    try:
        content = []

        # ---- Reference images (positive examples) ----
        for ref in reference_images:
            img_bytes = await ref.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            content.append({"type": "image", "image": img})

        # ---- Query image (last image) ----
        query_bytes = await query_image.read()
        query_img = Image.open(io.BytesIO(query_bytes)).convert("RGB")
        content.append({"type": "image", "image": query_img})

        # ---- Instruction prompt from client ----
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

        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]

        response_text = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return {"result": response_text.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





#####################

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

# 🔥 CLIP-style instruction prompt (YOU control this)
instruction_prompt = """
You are a strict semiconductor inspection model.

The FIRST images are verified GOOD chips.
The LAST image is a QUERY.

Evaluate purely on visual similarity:
- Shape
- Size
- Edges
- Material
- Surface texture

Output STRICTLY in this format:

RESULT: CHIP or NOT_A_CHIP
CONFIDENCE: 0-100
REASON: one short sentence

If similarity is low or ambiguous → NOT_A_CHIP
"""

files = []

for p in reference_paths:
    files.append(("reference_images", open(p, "rb")))

files.append(("query_image", open(query_path, "rb")))

response = requests.post(
    URL,
    files=files,
    data={
        "instruction_prompt": instruction_prompt,
        "max_new_tokens": "200",
    },
    timeout=120,
)

print(response.json()["result"])











