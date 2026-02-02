import cv2
import numpy as np
from config import YOLO_CONF_TH

def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_horizontal_chip_rois(image_path, yolo_txt):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    chips = []
    passed = 0
    rejected = 0

    with open(yolo_txt) as f:
        for line in f:
            vals = list(map(float, line.split()))
            x1,y1,x2,y2,x3,y3,x4,y4,conf = vals[1:]

            # ---- YOLO quality gate ----
            if conf > YOLO_CONF_TH:
                passed += 1
                continue
            else:
                rejected += 1

            pts = np.array([
                [x1*w, y1*h],
                [x2*w, y2*h],
                [x3*w, y3*h],
                [x4*w, y4*h]
            ], dtype=np.float32)

            rect = order_points(pts)

            w1 = np.linalg.norm(rect[0]-rect[1])
            w2 = np.linalg.norm(rect[2]-rect[3])
            h1 = np.linalg.norm(rect[1]-rect[2])
            h2 = np.linalg.norm(rect[0]-rect[3])

            W = int(max(w1,w2))
            H = int(max(h1,h2))

            # force horizontal
            if H > W:
                W, H = H, W
                rect = np.roll(rect,1,axis=0)

            dst = np.array([
                [0,0],[W,0],[W,H],[0,H]
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(rect, dst)
            roi = cv2.warpPerspective(img, M, (W,H))

            chips.append({
                "roi": roi,
                "pts": pts.astype(int),
                "yolo_conf": conf
            })

    stats = {
        "total": passed + rejected,
        "passed": passed,
        "rejected": rejected,
 #       "pass_ratio": passed / max(1, (passed + rejected))
    }

    return chips, stats
