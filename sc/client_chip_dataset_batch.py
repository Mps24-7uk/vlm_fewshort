import requests
from pathlib import Path
import json
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

API_URL = "http://127.0.0.1:8000/generate-batch"
BATCH_SIZE = 16
WORKERS = 2   # tune later

def batch_iterable(iterable, size):
    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch

def send_batch(batch, prompt):
    files = [
        ("image_files", (img.name, open(img, "rb"), "image/jpeg"))
        for img in batch
    ]

    try:
        r = requests.post(
            API_URL,
            files=files,
            data={"prompt": prompt, "max_new_tokens": 256},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["results"]
    finally:
        for _, f in files:
            f[1].close()

def generate_dataset(image_dir, output_jsonl, prompt):
    image_dir = Path(image_dir)
    output_jsonl = Path(output_jsonl)

    images = list(image_dir.glob("*.*"))
    total_images = len(images)

    print(f"🖼️ Found {total_images} images")

    all_results = []

    batches = list(batch_iterable(images, BATCH_SIZE))

    with ThreadPoolExecutor(max_workers=WORKERS) as pool, \
         tqdm(total=total_images, desc="Generating", unit="img") as pbar:

        future_map = {
            pool.submit(send_batch, batch, prompt): batch
            for batch in batches
        }

        for future in as_completed(future_map):
            batch = future_map[future]
            try:
                results = future.result()
                all_results.extend(results)
                pbar.update(len(results))  # ✅ progress by images
            except Exception as e:
                print(f"\n❌ Batch failed ({len(batch)} images): {e}")
                pbar.update(len(batch))  # still advance to avoid stall

    print(f"\n✍️ Writing {len(all_results)} records")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps({
                "image_name": item["image_name"],
                "image_path": str(image_dir / item["image_name"]),
                "description": item["description"],
            }, ensure_ascii=False) + "\n")

    print("✅ Dataset generation complete")

if __name__ == "__main__":
    generate_dataset(
        image_dir="data/chip_images",
        output_jsonl="data/outputs/generated_descriptions.jsonl",
        prompt="Describe the chip visible in the image."
    )
