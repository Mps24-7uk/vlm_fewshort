import torch
from pathlib import Path
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ====== 1. CONFIG ======
# Change this to wherever you stored the downloaded model
MODEL_DIR = Path(r"D:\hf_models\Qwen3-VL-8B-Instruct")

# Example few-shot images (you provide your own paths)
FEW_SHOT_EXAMPLES = [
    # (image_path, label)
    (r"D:\chips_dataset\chip_001.jpg", "chip"),
    (r"D:\chips_dataset\chip_002.jpg", "chip"),
    (r"D:\chips_dataset\nonchip_001.jpg", "non-chip"),
    (r"D:\chips_dataset\nonchip_002.jpg", "non-chip"),
]

# Image you actually want to classify now
QUERY_IMAGE_PATH = r"D:\chips_dataset\test_roi_001.jpg"


# ====== 2. LOAD MODEL & PROCESSOR LOCALLY ======
print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype="auto",       # or torch.bfloat16 / torch.float16 if GPU supports
    device_map="auto",
    local_files_only=True     # 🔒 strictly local, no internet
)

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)


# ====== 3. BUILD FEW-SHOT MESSAGE ======
def load_image(path):
    img = Image.open(path).convert("RGB")
    return img

def build_few_shot_messages(few_shot_examples, query_image_path):
    """
    Build a single chat turn with:
    - System: instructions
    - User: multiple (image, label) examples + query image
    """
    content = []

    # Instruction text (in plain language)
    instruction_text = (
        "You are a strict industrial visual-inspection expert.\n"
        "Task: Determine whether each image region contains a real semiconductor chip or not.\n\n"
        "Evaluation Criteria:\n"
        "- Look for chip-like geometric structure: rectangular/square outline, sharp edges, metallic or PCB-like surfaces.\n"
        "- Observe surface texture: smooth, uniform, patterned with micro-circuits, solder pads, or connector pins.\n"
        "- Check color cues: typical chip tones include black/grey packages, metallic pads, green/blue PCB, or matte textures.\n"
        "- Identify patterns: pin grids, traces, pads, labels, laser markings.\n"
        "- Reject regions with: random background patterns, shadows, fixtures, machine parts, cables, clamps, metal brackets, screws.\n"
        "- Consider context: chips usually appear mounted on boards or trays, aligned, and not randomly shaped.\n"
        "- Reject ambiguous or incomplete shapes that lack chip-like structural consistency.\n\n"
        "Few-shot examples are provided below. Use them to understand the visual concept of 'chip' and 'non-chip'.\n\n"
        "Your Output:\n"
        "- For the final image ONLY, respond with exactly ONE WORD:\n"
        "    - 'chip' (if the ROI clearly contains a semiconductor chip)\n"
        "    - 'non-chip' (if it does NOT)\n\n"
        "Be extremely strict: if uncertain, classify as 'non-chip'."
    )


    # Add instruction as text first
    content.append({
        "type": "text",
        "text": instruction_text
    })

    # Add few-shot example blocks
    for idx, (img_path, label) in enumerate(few_shot_examples, start=1):
        img = load_image(img_path)
        example_header = f"\n\nExample {idx}:\nLabel: {label}\n"
        content.append({
            "type": "text",
            "text": example_header
        })
        content.append({
            "type": "image",
            "image": img
        })

    # Now add the query image
    query_image = load_image(query_image_path)
    query_text = (
        "\n\nNow classify this image.\n"
        "Reply only with 'chip' or 'non-chip'."
    )

    content.append({
        "type": "text",
        "text": query_text
    })
    content.append({
        "type": "image",
        "image": query_image
    })

    # Build messages in Qwen chat format (system + user)
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful vision assistant for industrial chip inspection."
                }
            ]
        },
        {
            "role": "user",
            "content": content
        }
    ]
    return messages


# ====== 4. RUN INFERENCE (FEW-SHOT) ======
def classify_with_few_shot(model, processor, few_shot_examples, query_image_path, max_new_tokens=32):
    messages = build_few_shot_messages(few_shot_examples, query_image_path)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    # Trim the input tokens from the generated sequence
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Usually batch size = 1
    return output_text[0] if isinstance(output_text, list) else output_text


if __name__ == "__main__":
    result = classify_with_few_shot(
        model=model,
        processor=processor,
        few_shot_examples=FEW_SHOT_EXAMPLES,
        query_image_path=QUERY_IMAGE_PATH,
        max_new_tokens=16
    )

    print("Model raw output:")
    print(result)

    # Optional: simple post-processing (force lowercase, strip)
    normalized = result.strip().lower()
    if "chip" in normalized and "non-chip" not in normalized:
        final_label = "chip"
    elif "non-chip" in normalized:
        final_label = "non-chip"
    else:
        final_label = normalized  # fallback, inspect manually

    print(f"\nPredicted label: {final_label}")
