import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ChipDataset(Dataset):
    """
    dataset/
      ├── class_1/
      ├── class_2/
      └── ...
    """

    def __init__(self, root_dir):
        self.samples = []

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        for cls, idx in self.class_to_idx.items():
            folder = os.path.join(root_dir, cls)
            for f in os.listdir(folder):
                if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                    self.samples.append(
                        (os.path.join(folder, f), idx)
                    )

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label, path
