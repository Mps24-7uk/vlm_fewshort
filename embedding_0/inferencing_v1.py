import os
import csv
import torch
import faiss
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import ResNetEmbedding


# ---------------------------
# Config
# ---------------------------
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
TOP_K = 5
OUTPUT_CSV = "inference_results.csv"


# ---------------------------
# Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------
# Load Model
# ---------------------------
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNetEmbedding().to(device).eval()
    return model, device


# ---------------------------
# Load FAISS + Metadata
# ---------------------------
def load_faiss_assets(index_path, embedding_npz):
    index = faiss.read_index(index_path)

    data = np.load(embedding_npz, allow_pickle=True)
    labels = data["labels"]
    classes = data["classes"]

    return index, labels, classes


# ---------------------------
# Predict Single Image
# ---------------------------
@torch.no_grad()
def predict_image(
    image_path,
    model,
    device,
    index,
    labels,
    classes,
    top_k=TOP_K
):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    emb = model(image).cpu().numpy().astype("float32")

    scores, idxs = index.search(emb, top_k)

    neighbor_labels = labels[idxs[0]]
    neighbor_scores = scores[0]

    # Majority vote
    unique, counts = np.unique(neighbor_labels, return_counts=True)
    pred_label = unique[np.argmax(counts)]
    pred_class = classes[pred_label]

    confidence = float(np.mean(neighbor_scores))

    return (
        os.path.basename(image_path),
        pred_class,
        round(confidence, 4),
        neighbor_scores.tolist()
    )


# ---------------------------
# Folder Inference + CSV
# ---------------------------
def infer_folder_to_csv(
    image_dir,
    index_path="chip_resnet.index",
    embedding_npz="chip_resnet_embeddings.npz",
    output_csv=OUTPUT_CSV
):
    model, device = load_model()
    index, labels, classes = load_faiss_assets(
        index_path, embedding_npz
    )

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

        for img_path in tqdm(images, desc="Inferencing images"):
            result = predict_image(
                img_path, model, device, index, labels, classes
            )
            writer.writerow(result)

    print(f"✅ Inference results saved to {output_csv}")


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    infer_folder_to_csv("test_images")
