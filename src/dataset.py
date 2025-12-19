from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch

IMG_EXTS = {".jpg", ".jpeg", ".png"}

class CrackDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: lista de (img_path, label)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def build_samples(data_dir):
    """
    Devuelve lista de (img_path, label)
    """
    data_dir = Path(data_dir)

    label_map = {
        "Non-cracked": 0,
        "Cracked": 1
    }

    samples = []

    for surface_dir in data_dir.iterdir():
        if not surface_dir.is_dir():
            continue

        for label_dir in surface_dir.iterdir():
            if not label_dir.is_dir():
                continue

            label = label_map[label_dir.name]

            for img_path in label_dir.iterdir():
                if img_path.suffix.lower() in IMG_EXTS:
                    samples.append((img_path, label))

    return samples
