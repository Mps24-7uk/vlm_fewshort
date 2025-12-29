import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ChipDataset(Dataset):
    """
    Folder structure:
    dataset/
      ├── class_1/
      ├── class_2/
      ├── class_3/
      └── ...
    """

    def __init__(self, root_dir):
        self.samples = []

        # 🔹 Dynamically discover class folders
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # 🔹 Create dynamic label map
        self.label_map = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls_name, cls_id in self.label_map.items():
            folder = os.path.join(root_dir, cls_name)
            for file in os.listdir(folder):
                if file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                    self.samples.append(
                        (os.path.join(folder, file), cls_id)
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
