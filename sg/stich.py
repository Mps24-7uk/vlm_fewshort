import torch
import cv2
import numpy as np
from PIL import Image
from transformers import SegGptImageProcessor, SegGptForImageSegmentation
from tqdm import tqdm
import math
import os

# ---------------- CONFIG ----------------
IMAGE_PATH = "input_4096x3000.jpg"

PROMPT_IMAGE_PATH = "hmbb_1.jpg"
PROMPT_MASK_PATH  = "hmbb_1_target.png"

TILE_SIZE = 1024
OVERLAP_PERCENT = 15
BATCH_SIZE = 2   # adjust based on GPU

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "BAAI/seggpt-vit-large"

# ---------------------------------------


def split_image(img, tile_size, overlap_percent):
    h, w = img.shape[:2]
    overlap = int(tile_size * overlap_percent / 100)
    stride = tile_size - overlap

    tiles = []
    coords = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)

            tile = img[y:y_end, x:x_end]

            pad_bottom = tile_size - tile.shape[0]
            pad_right = tile_size - tile.shape[1]

            if pad_bottom > 0 or pad_right > 0:
                tile = cv2.copyMakeBorder(
                    tile, 0, pad_bottom, 0, pad_right,
                    cv2.BORDER_CONSTANT, value=0
                )

            tiles.append(tile)
            coords.append((x, y, x_end, y_end))

    return tiles, coords, (h, w), stride


def run_seggpt_batch(tiles, image_processor, model):
    masks = []

    prompt_img = Image.open(PROMPT_IMAGE_PATH).convert("RGB")
    prompt_mask = Image.open(PROMPT_MASK_PATH).convert("L")

    for i in tqdm(range(0, len(tiles), BATCH_SIZE)):
        batch_tiles = tiles[i:i+BATCH_SIZE]
        batch_pil = [Image.fromarray(t).convert("RGB") for t in batch_tiles]

        inputs = image_processor(
            images=batch_pil,
            prompt_images=[prompt_img]*len(batch_pil),
            prompt_masks=[prompt_mask]*len(batch_pil),
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        batch_masks = image_processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(TILE_SIZE, TILE_SIZE)] * len(batch_pil)
        )

        for m in batch_masks:
            masks.append(m.cpu().numpy())

    return masks


def stitch_masks(masks, coords, original_shape, tile_size):
    H, W = original_shape
    final_mask = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    for mask, (x1, y1, x2, y2) in zip(masks, coords):
        mask_crop = mask[:(y2 - y1), :(x2 - x1)]

        final_mask[y1:y2, x1:x2] += mask_crop
        weight_map[y1:y2, x1:x2] += 1.0

    final_mask /= np.maximum(weight_map, 1e-6)
    final_mask = (final_mask > 0.5).astype(np.uint8) * 255

    return final_mask


# ---------------- MAIN ----------------

def main():
    print("Loading model...")
    model = SegGptForImageSegmentation.from_pretrained(CHECKPOINT).to(DEVICE)
    image_processor = SegGptImageProcessor.from_pretrained(CHECKPOINT)

    print("Reading large image...")
    img = cv2.imread(IMAGE_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("Splitting into tiles...")
    tiles, coords, original_shape, stride = split_image(
        img, TILE_SIZE, OVERLAP_PERCENT
    )
    print(f"Total tiles: {len(tiles)}")

    print("Running SegGPT on tiles...")
    masks = run_seggpt_batch(tiles, image_processor, model)

    print("Stitching masks...")
    final_mask = stitch_masks(masks, coords, original_shape, TILE_SIZE)

    cv2.imwrite("final_mask.png", final_mask)
    print("Saved: final_mask.png")


if __name__ == "__main__":
    main()
