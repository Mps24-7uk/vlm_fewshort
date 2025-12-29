import torch
import numpy as np
import faiss
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ResNetEmbedding
from dataloader import ChipDataset


def generate_embeddings(
    dataset_dir,
    output_file="chip_resnet_embeddings.npz",
    batch_size=32
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNetEmbedding().to(device).eval()
    dataset = ChipDataset(dataset_dir)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    labels = []
    paths = []

    for images, lbls, img_paths in tqdm(loader, desc="Embedding images"):
        images = images.to(device)

        emb = model(images)

        embeddings.append(emb.cpu().numpy())
        labels.extend(lbls.numpy())
        paths.extend(img_paths)

    embeddings = np.vstack(embeddings).astype("float32")
    labels = np.array(labels)

    # 🔹 Save class metadata
    np.savez(
        output_file,
        embeddings=embeddings,
        labels=labels,
        paths=paths,
        classes=np.array(dataset.classes),
        class_to_idx=dataset.label_map
    )

    print(f"✅ Saved embeddings: {embeddings.shape}")
    print(f"✅ Classes: {dataset.classes}")

    return embeddings, labels, paths, dataset.classes


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings)

    print(f"✅ FAISS index size: {index.ntotal}")
    return index


if __name__ == "__main__":
    embeddings, labels, paths, classes = generate_embeddings(
        dataset_dir="dataset"
    )

    index = build_faiss_index(embeddings)

    faiss.write_index(index, "chip_resnet.index")
    print("✅ FAISS index saved as chip_resnet.index")
