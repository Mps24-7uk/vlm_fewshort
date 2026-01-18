# client.py
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8000/generate"

# ================= CANONICAL PROMPT =================
CANONICAL_PROMPT = """
You are a semiconductor inspection system.

Your task is to generate a factual, structured description of a chip image.
You MUST follow the exact template below.
DO NOT add or remove sections.
DO NOT add opinions or assumptions.

TEMPLATE:

[CHIP_OVERVIEW]
- Chip type:
- View:
- Orientation:

[MATERIAL_COMPOSITION]
- Terminal material:
- Terminal finish:
- Body material:
- Body color:

[GEOMETRY]
- Shape:
- Aspect ratio:
- Edge condition:

[SURFACE_CHARACTERISTICS]
- Body surface:
- Terminals:
- Discoloration:
"""

# ================= SINGLE IMAGE =================
def generate_for_image(image_path: str):
    with open(image_path, "rb") as f:
        files = {
            "image_file": (Path(image_path).name, f, "image/jpeg")
        }
        data = {
            "prompt": CANONICAL_PROMPT,
            "max_new_tokens": 512
        }

        r = requests.post(API_URL, files=files, data=data)

    if r.status_code != 200:
        print("❌ Error:", r.text)
        return None

    return r.json()

# ================= DATASET → FILE =================
def generate_file(image_dir: str, output_jsonl: str):
    image_dir = Path(image_dir)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for img in image_dir.glob("*.*"):
            print(f"🔍 Processing {img.name}")
            res = generate_for_image(str(img))
            if not res:
                continue

            record = {
                "image_name": res["image_name"],
                "image_path": str(img),
                "description": res["result"]
            }

            f.write(
                f"{record}\n".replace("'", '"')
            )

    print(f"\n✅ Saved results to {output_jsonl}")

# ================= RUN =================
if __name__ == "__main__":
    generate_file(
        image_dir="data/chip_images",
        output_jsonl="data/outputs/chip_descriptions.jsonl"
    )
