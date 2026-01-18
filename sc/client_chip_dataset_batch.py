# client.py
import requests
from pathlib import Path
import json
from itertools import islice

API_URL = "http://127.0.0.1:8000/generate-batch"

DEFAULT_PROMPT = "Describe the chip visible in the image."
BATCH_SIZE = 8  # must match server MAX_BATCH_SIZE

# -------------------------------
def batch_iterable(iterable, size):
    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch

# -------------------------------
def generate_dataset(image_dir: str, output_jsonl: str, prompt: str = DEFAULT_PROMPT):
    image_dir = Path(image_dir)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    images = list(image_dir.glob("*.*"))

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for batch in batch_iterable(images, BATCH_SIZE):
            print(f"🚀 Sending batch of {len(batch)} images")

            files = [
                ("image_files", (img.name, open(img, "rb"), "image/jpeg"))
                for img in batch
            ]

            data = {
                "prompt": prompt,
                "max_new_tokens": 256
            }

            r = requests.post(API_URL, files=files, data=data)

            for _, file_tuple in files:
                file_tuple[1].close()

            if r.status_code != 200:
                print("❌ Batch failed:", r.text)
                continue

            for item in r.json()["results"]:
                record = {
                    "image_name": item["image_name"],
                    "image_path": str(image_dir / item["image_name"]),
                    "description": item["description"]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Dataset saved to {output_jsonl}")

# -------------------------------
if __name__ == "__main__":
    generate_dataset(
        image_dir="data/chip_images",
        output_jsonl="data/outputs/generated_descriptions.jsonl",
        prompt="Describe the chip visible in the image."
    )
