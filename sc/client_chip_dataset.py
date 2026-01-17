
import os
import csv
import requests

URL = "http://127.0.0.1:8000/vision-fire-smoke-check"

IMAGE_DIR = r"D:/fire_images"   # 🔁 CHANGE THIS
OUTPUT_CSV = "fire_smoke_check_results.csv"

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp")

instruction_prompt = """
You are an industrial safety vision system.

Task:
Analyze the image and determine if it contains:

- FIRE
- SMOKE
- NO_FIRE_SMOKE

Respond STRICTLY with ONE of the following:

FIRE
SMOKE
NO_FIRE_SMOKE

Rules:
- If both fire and smoke are visible → FIRE
- If uncertain → NO_FIRE_SMOKE
"""

# -------- Collect images --------
image_paths = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(SUPPORTED_EXT)
]

if not image_paths:
    raise RuntimeError("No images found in directory")

print(f"📂 Found {len(image_paths)} images")

# -------- CSV setup --------
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_name", "result"])

    for img_path in image_paths:
        try:
            with open(img_path, "rb") as img_file:
                response = requests.post(
                    URL,
                    files={"image_file": img_file},
                    data={
                        "instruction_prompt": instruction_prompt,
                        "max_new_tokens": "64",
                    },
                    timeout=30,
                )

            result = response.json()["result"].strip()

            print(f"🖼️ {os.path.basename(img_path)} → {result}")

            writer.writerow([
                os.path.basename(img_path),
                result
            ])

        except Exception as e:
            print(f"❌ Failed {img_path}: {e}")
