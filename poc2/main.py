# main.py
from yolo_roi_chip import get_horizontal_chip_rois
from embedding_res import infer_chip_rois_industrial
from vlm_reasoning import run_vlm_reasoning

IMAGE = "chips.jpg"
YOLO_FILE = "123abc.txt"

chips, yolo_stats = get_horizontal_chip_rois(IMAGE, YOLO_FILE)
print("YOLO:", yolo_stats)

results, emb_stats = infer_chip_rois_industrial(chips)
print("EMB:", emb_stats)

results, vlm_stats = run_vlm_reasoning(results)
print("VLM:", vlm_stats)

for r in results:
    print(r["status"], r.get("match"), r.get("confidence"))
