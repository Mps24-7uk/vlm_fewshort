# nodes/embedding_node.py
import torch
from PIL import Image
from torchvision import transforms

from model import ResNetEmbedding


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def embedding_node(state):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNetEmbedding().to(device).eval()

    image = Image.open(state["image_path"]).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(image).cpu().numpy()[0]

    state["embedding"] = embedding.tolist()
    return state
