# models.py
import torch
import faiss
import numpy as np
from torchvision import models, transforms
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from config import *

# ---------- DEVICE ----------
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

# ---------- RESNET ----------
resnet = models.resnet50(weights="IMAGENET1K_V1")
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device).eval()

# ---------- TRANSFORM ----------
resnet_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ---------- FAISS ----------
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
faiss_paths = np.load(FAISS_PATHS_PATH, allow_pickle=True)

# ---------- QWEN ----------
qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-32B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)
qwen_processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3-VL-32B-Instruct"
)
