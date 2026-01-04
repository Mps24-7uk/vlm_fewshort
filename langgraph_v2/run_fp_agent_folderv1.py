# run_fp_agent_folder.py
import os
import csv
from tqdm import tqdm
from fp_agent_graph import build_fp_agent

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def run_fp_agent_on_folder(image_dir, csv_path="fp_agent_results.csv"):
    agent = build_fp_agent()

    images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ]

    results = []

    for img in tqdm(images, desc="FP Agent"):
        result = agent.invoke({"image_path": img})
        results.append(result)

    # ---------- WRITE CSV ----------
    fieldnames = [
        "image_name",
        "image_path",
        "predicted_class",
        "confidence",
        "fp_flag",
        "fp_score",
        "reason",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for o in results:
            writer.writerow({
                "image_name": os.path.basename(o.get("image_path", "")),
                "image_path": o.get("image_path", ""),
                "predicted_class": o.get("predicted_class", ""),
                "confidence": o.get("confidence", 0.0),
                "fp_flag": o.get("fp_flag", False),
                "fp_score": o.get("fp_score", 0.0),
                "reason": o.get("reason", ""),
            })

    print(f"✅ CSV saved at: {csv_path}")
    return results


if __name__ == "__main__":
    run_fp_agent_on_folder(
        image_dir="test_images",
        csv_path="fp_agent_results.csv"
    )
