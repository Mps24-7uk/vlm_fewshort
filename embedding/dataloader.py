import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ChipDataset(Dataset):
    """
    Folder structure:
    dataset/
      ├── defect/
      └── no_defect/
    """

    def __init__(self, root_dir):
        self.samples = []
        self.label_map = {
            "no_defect": 0,
            "defect": 1
        }

        for label_name, label_id in self.label_map.items():
            folder = os.path.join(root_dir, label_name)
            for file in os.listdir(folder):
                if file.lower().endswith((".jpg", ".png", ".jpeg",".bmp")):
                    self.samples.append(
                        (os.path.join(folder, file), label_id)
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
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label, img_path
