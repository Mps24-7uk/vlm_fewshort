import os
import faiss
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms

# ------------------ CONFIG ------------------
IMAGE_DIR = "data/chips"
FAISS_INDEX_PATH = "chip.index"
PATHS_SAVE_PATH = "chip_paths.npy"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load ResNet50
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()
model = model.to(DEVICE)
model.eval()

def load_images(image_dir):
    paths = []
    for file in os.listdir(image_dir):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            paths.append(os.path.join(image_dir, file))
    paths.sort()
    return paths

def get_embedding_batch(image_paths):
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img = transform(img)
        imgs.append(img)

    batch = torch.stack(imgs).to(DEVICE)
    with torch.no_grad():
        emb = model(batch)
    return emb.cpu().numpy().astype("float32")

# ----------- MAIN PIPELINE -----------------

image_paths = load_images(IMAGE_DIR)
print("Total images:", len(image_paths))

# Save paths
np.save(PATHS_SAVE_PATH, np.array(image_paths))

dim = 2048
index = faiss.IndexFlatIP(dim)  # Inner Product = Cosine (after normalization)

for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[i:i+BATCH_SIZE]
    emb = get_embedding_batch(batch_paths)

    # 🔥 L2 normalize
    faiss.normalize_L2(emb)

    index.add(emb)

faiss.write_index(index, FAISS_INDEX_PATH)

print("Cosine FAISS index saved:", FAISS_INDEX_PATH)
