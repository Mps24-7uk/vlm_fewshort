import os
import csv
import torch
import faiss
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import ResNetEmbedding


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
TOP_K = 5


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_classes(classes_path="classes.txt"):
    with open(classes_path) as f:
        return [line.strip() for line in f.readlines()]


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNetEmbedding().to(device).eval()
    return model, device


@torch.no_grad()
def infer_folder_to_csv(
    image_dir,
    index_path="chip_resnet.index",
    classes_path="classes.txt",
    output_csv="inference_results.csv"
):
    model, device = load_model()
    index = faiss.read_index(index_path)
    classes = load_classes(classes_path)

    images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_name",
            "predicted_class",
            "confidence",
            "top_k_scores"
        ])

        for img_path in tqdm(images, desc="Inferencing"):
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            emb = model(img).cpu().numpy().astype("float32")
            scores, ids = index.search(emb, TOP_K)

            pred_label = int(np.bincount(ids[0]).argmax())
            pred_class = classes[pred_label]
            confidence = float(scores[0].mean())

            writer.writerow([
                os.path.basename(img_path),
                pred_class,
                round(confidence, 4),
                scores[0].tolist()
            ])

    print(f"✅ CSV saved: {output_csv}")


if __name__ == "__main__":
    infer_folder_to_csv("test_images")
