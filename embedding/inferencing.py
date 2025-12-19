import torch
import numpy as np
import faiss
from PIL import Image
from torchvision import transforms

from model import ResNetEmbedding


EMBEDDING_DB = "chip_resnet_embeddings.npz"
FAISS_INDEX_PATH = "chip_resnet.index"
TOP_K = 5
USE_GPU = True
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_faiss_index(index_path, use_gpu=True):
    index = faiss.read_index(index_path)

    if use_gpu and faiss.get_num_gpus() > 0:
        print("🚀 Using FAISS GPU")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        print("⚠️ Using FAISS CPU")

    return index


def infer_chip(
    image_path,
    index,
    embeddings_db_path=EMBEDDING_DB,
    top_k=TOP_K
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load ResNet embedding model
    model = ResNetEmbedding().to(device).eval()

    # Load embedding metadata
    db = np.load(embeddings_db_path, allow_pickle=True)
    labels = db["labels"]
    paths = db["paths"]

    # Load & preprocess image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Generate embedding
    with torch.no_grad():
        query_emb = model(img_tensor).cpu().numpy().astype("float32")

    # FAISS search
    scores, indices = index.search(query_emb, top_k)

    scores = scores[0]
    indices = indices[0]

    top_labels = labels[indices]
    top_paths = paths[indices]

    # Aggregate similarity
    defect_scores = scores[top_labels == 1]
    no_defect_scores = scores[top_labels == 0]

    defect_sim = defect_scores.mean() if len(defect_scores) else 0.0
    no_defect_sim = no_defect_scores.mean() if len(no_defect_scores) else 0.0

    decision = "DEFECT" if defect_sim > no_defect_sim else "NO_DEFECT"

    return {
        "decision": decision,
        "defect_similarity": float(defect_sim),
        "no_defect_similarity": float(no_defect_sim),
        "top_matches": [
            {
                "path": top_paths[i],
                "label": int(top_labels[i]),
                "similarity": float(scores[i])
            }
            for i in range(len(indices))
        ]
    }

if __name__ == "__main__":
    index = load_faiss_index(
        FAISS_INDEX_PATH,
        use_gpu=USE_GPU
    )

    result = infer_chip(
        image_path="test_chip.jpg",
        index=index
    )

    print("\n🧠 Decision:", result["decision"])
    print("Defect similarity:", result["defect_similarity"])
    print("No-defect similarity:", result["no_defect_similarity"])

    print("\n🔝 Top Matches:")
    for m in result["top_matches"]:
        print(
            f"{m['path']} | "
            f"label={m['label']} | "
            f"similarity={m['similarity']:.4f}"
        )
