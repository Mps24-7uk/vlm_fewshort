import torch
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

# 👇 Change this to your actual local model directory
MODEL_DIR = Path(r"D:\hf_models\Qwen3-VL-8B-Instruct")

# 1. Load model from local path ONLY (no internet)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype="auto",        # or torch.bfloat16 / torch.float16
    device_map="auto",
    local_files_only=True      # 🔑 strictly use local files
)

# 2. Load processor from same local path
processor = AutoProcessor.from_pretrained(
    MODEL_DIR,
    local_files_only=True      # 🔑 strictly use local files
)

# 3. Example message with a LOCAL image instead of URL
image_path = r"D:\images\demo.jpeg"   # 👈 put your image path here
image = Image.open(image_path).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# 4. Prepare inputs
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

inputs = inputs.to(model.device)

# 5. Generate
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(output_text)
