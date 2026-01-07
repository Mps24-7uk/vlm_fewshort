# run_fp_agent_folder.py
import os
from tqdm import tqdm
from fp_agent_graph import build_fp_agent

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def run_fp_agent_on_folder(image_dir):
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

    return results


if __name__ == "__main__":
    outputs = run_fp_agent_on_folder("test_images")

    for o in outputs:
        print(
            f"{os.path.basename(o['image_path'])} | "
            f"Class={o['predicted_class']} | "
            f"Conf={o['confidence']} | "
            f"FP={o['fp_flag']} | "
            f"Score={o['fp_score']} | "
            f"{o['reason']}"
        )
