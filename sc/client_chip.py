import requests

URL = "http://107.99.100.53:8000/vision-fewshot-chip"

# -------- Reference set (LABELED) --------
reference_data = [
    # CHIP examples
    ("D:/chips/chip_1.jpg", "chip"),
    ("D:/chips/chip_2.jpg", "chip"),
    ("D:/chips/chip_3.jpg", "chip"),

    # NON-CHIP examples
    ("D:/chips/noise_1.jpg", "non_chip"),
    ("D:/chips/noise_2.jpg", "non_chip"),
]

query_image_path = "D:/chips/query.jpg"

# -------- Instruction prompt (CLIP-style) --------
instruction_prompt = """
You are a strict industrial vision classifier.

You have seen labeled examples of CHIP and NON_CHIP.
The last image is the QUERY image.

Compare the query image visually against BOTH classes.

Respond STRICTLY in this format:

RESULT: CHIP or NOT_A_CHIP
CONFIDENCE: 0-100
REASON: one short sentence

Rules:
- Visual similarity only
- If closer to NON_CHIP → NOT_A_CHIP
- If ambiguous → NOT_A_CHIP
"""

files = []
labels = []

for path, label in reference_data:
    files.append(("reference_images", open(path, "rb")))
    labels.append(label)

files.append(("query_image", open(query_image_path, "rb")))

response = requests.post(
    URL,
    files=files,
    data={
        "reference_labels": labels,      # 🔑 labels aligned with images
        "instruction_prompt": instruction_prompt,
        "max_new_tokens": "200",
    },
    timeout=120,
)

print(response.json()["result"])
