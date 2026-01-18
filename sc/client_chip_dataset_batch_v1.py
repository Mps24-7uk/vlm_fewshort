import requests
from pathlib import Path
import json
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://127.0.0.1:8000/generate-batch"
BATCH_SIZE = 16
WORKERS = 4     # tune: 2–6

def batch_iterable(iterable, size):
    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch

def send_batch(batch, image_dir, prompt):
    files = [
        ("image_files", (img.name, open(img, "rb"), "image/jpeg"))
        for img in batch
    ]

    data = {"prompt": prompt, "max_new_tokens": 256}
    r = requests.post(API_URL, files=files, data=data)

    for _, f in files:
        f[1].close()

    r.raise_for_status()
    return r.json()["results"]

def generate_dataset(image_dir, output_jsonl, prompt):
    image_dir = Path(image_dir)
    images = list(image_dir.glob("*.*"))

    with open(output_jsonl, "w", encoding="utf-8") as f, \
         ThreadPoolExecutor(max_workers=WORKERS) as pool:

        futures = []
        for batch in batch_iterable(images, BATCH_SIZE):
            futures.append(pool.submit(send_batch, batch, image_dir, prompt))

        for future in as_completed(futures):
            for item in future.result():
                f.write(json.dumps({
                    "image_name": item["image_name"],
                    "image_path": str(image_dir / item["image_name"]),
                    "description": item["description"]
                }) + "\n")

    print("✅ Dataset generation complete")

if __name__ == "__main__":
    generate_dataset(
        "data/chip_images",
        "data/outputs/generated_descriptions.jsonl",
        "Describe the chip visible in the image."
    )
