# embedding_res.py
import torch
import numpy as np
from PIL import Image
import faiss
from models import resnet, resnet_tf, faiss_index, faiss_paths, device
from config import *

def extract_embeddings_batch(rois):
    imgs = []
    for roi in rois:
        roi_rgb = roi[:,:,::-1]
        img = Image.fromarray(roi_rgb)
        imgs.append(resnet_tf(img))

    batch = torch.stack(imgs).to(device)

    with torch.no_grad():
        emb = resnet(batch)

    emb = emb.cpu().numpy().astype("float32")
    faiss.normalize_L2(emb)
    return emb


def infer_chip_rois_industrial(chips):
    valid = []
    mapping = []

    for i, chip in enumerate(chips):
        roi = chip["roi"]
        h,w = roi.shape[:2]
        if h < MIN_ROI_SIZE or w < MIN_ROI_SIZE:
            continue
        if roi.mean() < 5:
            continue
        valid.append(roi)
        mapping.append(i)

    if not valid:
        return [], {"total":0,"resolved":0,"unresolved":0}

    embeddings = extract_embeddings_batch(valid)
    D, I = faiss_index.search(embeddings, FAISS_TOP_K)

    resolved = 0
    unresolved = 0

    for idx, orig_idx in enumerate(mapping):
        chip = chips[orig_idx]
        best_conf = float(D[idx][0])
        best_match = faiss_paths[I[idx][0]]

        if best_conf > RES_CONF_TH:
            resolved += 1
            result = {
                "status": "accepted",
                "roi": chip["roi"],
                "pts": chip["pts"],
                "match": best_match,
                "confidence": best_conf
            }
        else:
            unresolved += 1
            topk = []
            for k in range(FAISS_TOP_K):
                topk.append({
                    "match": faiss_paths[I[idx][k]],
                    "confidence": float(D[idx][k])
                })

            result = {
                "status": "review",
                "roi": chip["roi"],
                "pts": chip["pts"],
                "top_k": topk
            }

        chips[orig_idx] = result

    stats = {
        "total": resolved + unresolved,
        "resolved": resolved,
        "unresolved": unresolved
    }

    return chips, stats
