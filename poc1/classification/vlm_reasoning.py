import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Load model
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-32B-Instruct",
    dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")

# 1 Query + 3 Reference images
query_image = "https://example.com/query.jpg"
ref_images = [
    "https://example.com/ref1.jpg",
    "https://example.com/ref2.jpg",
    "https://example.com/ref3.jpg",
]

# Multimodal prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": query_image},
            {"type": "image", "image": ref_images[0]},
            {"type": "image", "image": ref_images[1]},
            {"type": "image", "image": ref_images[2]},
            {
                "type": "text",
                "text": """
The first image is the QUERY.
The next three images are REFERENCES.

Task:
1. Compare the query image with each reference.
2. Analyze objects, layout, scene type, and visual style.
3. For each reference, give:
   - Similarity score from 0 to 100
   - Short reasoning.

Return output in this format:

Reference 1:
Score: XX/100
Reason: ...

Reference 2:
Score: XX/100
Reason: ...

Reference 3:
Score: XX/100
Reason: ...
"""
            }
        ]
    }
]

# Prepare inputs
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=512)

# Trim prompt
generated_ids_trimmed = [
    out[len(inp):] 
    for inp, out in zip(inputs.input_ids, generated_ids)
]

# Decode
output = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True
)

print(output[0])
