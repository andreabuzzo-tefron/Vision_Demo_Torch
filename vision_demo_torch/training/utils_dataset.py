import cv2
import torch
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.items = []
        self.transform = transform

        with open(csv_file, "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                filename, label = line.strip().split(",")
                self.items.append((filename, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        filename, label = self.items[idx]
        path = f"../app/data_capture/{filename}"
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (200, 50))
        img = img.astype("float32") / 255.0
        img = torch.tensor(img).unsqueeze(0)

        return img, label
