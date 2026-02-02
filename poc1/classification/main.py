from yolo_roi_chip import get_horizontal_chip_rois
from embedding_res import infer_chip_rois_industrial
from vlm_reasoning import run_vlm_reasoning

IMAGE = "chips.jpg"
YOLO_FILE = "123abc.txt"

# ---- Sensor + Geometry ----
chips, yolo_stats = get_horizontal_chip_rois(IMAGE, YOLO_FILE)

print("\nYOLO SENSOR STATS")
print(yolo_stats)

# ---- Embedding + Decision ----
results, emb_stats = infer_chip_rois_industrial(chips)

print("\nEMBEDDING RESOLUTION STATS")
print(emb_stats)

print("\nFINAL RESULTS")
for r in results:
    if r["status"] == "accepted":
        print("ACCEPTED:", r["match"], r["confidence"])
    else:
        print("REVIEW:", r["top_k"])


# ---- VLM fallback ----
results, vlm_stats = run_vlm_reasoning(results)

print("\nVLM FINAL OUTPUTS:")
for v in vlm_results:
    print(v)