# nodes/embedding_node.py
import torch
from PIL import Image
from torchvision import transforms
from .model import ResNetEmbedding


_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Global model cache (singleton pattern)
_model_cache = {"model": None, "device": None}


def _get_model():
    """Lazy load and cache the model."""
    if _model_cache["model"] is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model_cache["device"] = device
        _model_cache["model"] = ResNetEmbedding().to(device).eval()
    return _model_cache["model"], _model_cache["device"]


def embedding_node(state):
    model, device = _get_model()

    img = Image.open(state["image_path"]).convert("RGB")
    img = _transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img).cpu().numpy()[0]

    state["embedding"] = emb.tolist()
    return state
