import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils_dataset import OCRDataset
from pathlib import Path

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/"
CHAR_TO_IDX = {c: i+1 for i, c in enumerate(ALPHABET)}
IDX_TO_CHAR = {i+1: c for c, i in CHAR_TO_IDX.items()}

def encode_label(s):
    return torch.tensor([CHAR_TO_IDX[c] for c in s])

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    encoded = [encode_label(l) for l in labels]
    return imgs, encoded

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.lstm = nn.LSTM(64 * 12, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c*h)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

def main():
    dataset = OCRDataset("dataset_labeled.csv")
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = CRNN(num_classes=len(ALPHABET) + 1)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=0)

    for epoch in range(30):
        for imgs, labels in loader:
            optim.zero_grad()

            logits = model(imgs)
            T = logits.size(1)
            logp = logits.log_softmax(2)

            target = torch.cat(labels)
            input_lengths = torch.full((imgs.size(0),), T, dtype=torch.long)
            target_lengths = torch.tensor([len(t) for t in labels])

            loss = criterion(logp, target, input_lengths, target_lengths)
            loss.backward()
            optim.step()

        print(f"Epoch {epoch} - loss {loss.item():.4f}")

    Path("../app/models/").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "../app/models/crnn_best.pt")
    print("✔ Training completato.")

if __name__ == "__main__":
    main()
