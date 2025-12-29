import os
import torch
import faiss
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ResNetEmbedding
from dataloader import ChipDataset


def generate_faiss_index(
    dataset_dir,
    index_path="chip_resnet.index",
    classes_path="classes.txt",
    batch_size=32
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ChipDataset(dataset_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ResNetEmbedding().to(device).eval()

    embeddings = []
    labels = []

    for images, lbls, _ in tqdm(loader, desc="Building embeddings"):
        images = images.to(device)
        emb = model(images).cpu().numpy().astype("float32")

        embeddings.append(emb)
        labels.extend(lbls.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    dim = embeddings.shape[1]

    # 🔹 FAISS index with label IDs
    base_index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(base_index)

    index.add_with_ids(embeddings, labels)

    faiss.write_index(index, index_path)

    # 🔹 Save class names
    with open(classes_path, "w") as f:
        for cls in dataset.classes:
            f.write(f"{cls}\n")

    print(f"✅ FAISS index saved: {index_path}")
    print(f"✅ Classes saved: {classes_path}")
    print(f"✅ Total samples indexed: {index.ntotal}")


if __name__ == "__main__":
    generate_faiss_index(dataset_dir="dataset")
