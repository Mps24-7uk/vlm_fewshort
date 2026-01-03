import os
from tqdm import tqdm

from fp_agent_graph import build_fp_agent


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def run_fp_agent_on_folder(image_dir):
    fp_agent = build_fp_agent()

    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ]

    results = []

    for img_path in tqdm(image_paths, desc="Running FP Agent"):
        state = {
            "image_path": img_path
        }

        result = fp_agent.invoke(state)

        results.append({
            "image": os.path.basename(img_path),
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "fp_flag": result["fp_flag"],
            "reason": result["reason"]
        })

    return results


if __name__ == "__main__":
    outputs = run_fp_agent_on_folder("test_images")

    print("\n🧠 FP Agent Summary")
    print("------------------")
    for o in outputs:
        print(
            f"{o['image']} | "
            f"Class: {o['predicted_class']} | "
            f"Conf: {o['confidence']} | "
            f"FP: {o['fp_flag']} | "
            f"Reason: {o['reason']}"
        )
