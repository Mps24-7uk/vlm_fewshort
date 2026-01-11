# nodes/embedding_node.py
import torch
from PIL import Image
from torchvision import transforms
from model import ResNetEmbedding


_transform = transforms.Compose([
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

    img = Image.open(state["image_path"]).convert("RGB")
    img = _transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img).cpu().numpy()[0]

    state["embedding"] = emb.tolist()
    return state
