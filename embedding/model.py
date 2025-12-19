import torch
import torch.nn as nn
from torchvision import models


class ResNetEmbedding(nn.Module):
    """
    ResNet50 embedding model for industrial vision
    Output: 2048-D normalized embedding
    """

    def __init__(self):
        super().__init__()

        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        # Remove classifier head
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)   # (B, 2048, 1, 1)
            x = x.flatten(1)                # (B, 2048)
            x = nn.functional.normalize(x)  # cosine similarity
        return x
