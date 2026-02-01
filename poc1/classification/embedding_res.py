import faiss
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from config import *

# -------- Model --------
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()
model = model.to(DEVICE)
model.eval()

# -------- FAISS --------
index = faiss.read_index(FAISS_INDEX_PATH)
paths = np.load(FAISS_PATHS_PATH, allow_pickle=True)

# -------- Transform --------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def extract_embedding(roi_bgr):
    roi_rgb = roi_bgr[:,:,::-1]
    img = Image.fromarray(roi_rgb)
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(img)

    emb = emb.cpu().numpy().astype("float32")
    faiss.normalize_L2(emb)
    return emb


def infer_chip_rois_industrial(chips):
    outputs = []
    resolved = 0
    unresolved = 0

    for idx, chip in enumerate(chips):
        roi = chip["roi"]
        pts = chip["pts"]

        h,w = roi.shape[:2]
        if h < MIN_ROI_SIZE or w < MIN_ROI_SIZE:
            unresolved += 1
            continue

        if roi.mean() < 5:
            unresolved += 1
            continue

        emb = extract_embedding(roi)
        D, I = index.search(emb, FAISS_TOP_K)

        best_conf = float(D[0][0])
        best_match = paths[I[0][0]]

        # ---- Resolution gate ----
        if best_conf > RES_CONF_TH:
            resolved += 1
            result = {
                "status": "accepted",
                "roi": roi,
                "pts": pts,
                "match": best_match,
                "confidence": best_conf
            }
        else:
            unresolved += 1
            topk = []
            for k in range(FAISS_TOP_K):
                topk.append({
                    "match": paths[I[0][k]],
                    "confidence": float(D[0][k])
                })

            result = {
                "status": "review",
                "roi": roi,
                "pts": pts,
                "top_k": topk
            }

        outputs.append(result)

        if LOG_EVERY:
            print(f"[ROI {idx}] {result['status']} | conf={best_conf:.3f}")

    stats = {
        "total": resolved + unresolved,
        "resolved": resolved,
        "unresolved": unresolved,
        "resolution_ratio": resolved / max(1, (resolved + unresolved))
    }

    return outputs, stats
