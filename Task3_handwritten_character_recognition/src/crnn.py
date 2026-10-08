"""Extension: CRNN (CNN + BiLSTM + CTC) for recognising a *sequence* of characters.

Demo: trains on synthetic "number strings" made by stitching 3-6 MNIST digits
side by side. The same idea works for words by stitching EMNIST letters, or for
real datasets like IAM (handwritten lines) with a larger backbone.

    python -m src.crnn --epochs 5
"""
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

MAX_LEN, IMG = 6, 28
WIDTH = MAX_LEN * IMG          # fixed canvas width; shorter strings are right-padded


class CRNN(nn.Module):
    """Conv feature extractor -> BiLSTM over the width axis -> per-timestep class scores."""

    def __init__(self, num_classes: int, hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # 14 x W/2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 7 x W/4
        )
        self.rnn = nn.LSTM(64 * 7, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, num_classes)   # num_classes includes the CTC blank (index 0)

    def forward(self, x):                      # x: (N, 1, 28, W)
        f = self.cnn(x)                        # (N, C, H, W')
        n, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(n, w, c * h)
        out, _ = self.rnn(f)
        return self.fc(out)                    # (N, T, classes)


class DigitSequences(Dataset):
    def __init__(self, train: bool, size: int, root="data"):
        self.mnist = datasets.MNIST(root, train=train, download=True)
        self.size, self.tf = size, transforms.ToTensor()

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        k = random.randint(3, MAX_LEN)
        idx = [random.randrange(len(self.mnist)) for _ in range(k)]
        imgs = [self.tf(self.mnist[j][0]) for j in idx]
        label = [self.mnist[j][1] + 1 for j in idx]           # +1: index 0 is the CTC blank
        canvas = torch.zeros(1, IMG, WIDTH)
        canvas[:, :, : k * IMG] = torch.cat(imgs, dim=2)
        return (canvas - 0.1307) / 0.3081, torch.tensor(label)


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.cat(ys), torch.tensor([len(y) for y in ys])


def greedy_decode(logits):
    """Collapse repeats and remove blanks from the argmax path."""
    best = logits.argmax(2).cpu().tolist()
    out = []
    for seq in best:
        s, prev = [], 0
        for t in seq:
            if t != prev and t != 0:
                s.append(str(t - 1))
            prev = t
        out.append("".join(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl = DataLoader(DigitSequences(True, 20000), a.batch_size, collate_fn=collate)
    test_ds = DigitSequences(False, 2000)
    test_dl = DataLoader(test_ds, 128, collate_fn=collate)

    model = CRNN(num_classes=11).to(device)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), 1e-3)

    for epoch in range(1, a.epochs + 1):
        model.train()
        for x, y, ylen in train_dl:
            x = x.to(device)
            logp = F.log_softmax(model(x), 2).permute(1, 0, 2)          # (T, N, C)
            ilen = torch.full((x.size(0),), logp.size(0), dtype=torch.long)
            loss = ctc(logp, y, ilen, ylen)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y, ylen in test_dl:
                preds = greedy_decode(model(x.to(device)))
                truths, pos = [], 0
                for l in ylen.tolist():
                    truths.append("".join(str(v - 1) for v in y[pos:pos + l].tolist())); pos += l
                correct += sum(p == t for p, t in zip(preds, truths)); total += len(truths)
        print(f"Epoch {epoch}: loss {loss.item():.3f} | sequence accuracy {correct / total:.3f}")
        print(f"   e.g. predicted {preds[0]!r} vs truth {truths[0]!r}")

    torch.save(model.state_dict(), "models/crnn_digits.pt")


if __name__ == "__main__":
    import os; os.makedirs("models", exist_ok=True)
    main()
