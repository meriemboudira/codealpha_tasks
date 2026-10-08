"""Train a CNN on MNIST / EMNIST.

Examples:
    python -m src.train --dataset mnist --epochs 10
    python -m src.train --dataset emnist_balanced --epochs 15
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.data import get_datasets
from src.model import CharCNN


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    preds, labels = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            p = out.argmax(1)
            correct += (p == y).sum().item()
            n += len(y)
            if not train:
                preds.append(p.cpu()); labels.append(y.cpu())
    result = (total_loss / n, correct / n)
    if not train:
        result += (torch.cat(preds).numpy(), torch.cat(labels).numpy())
    return result


def save_plots(history, cm, classes, out_dir, name):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"], label="train"); ax[0].plot(history["test_loss"], label="test")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["train_acc"], label="train"); ax[1].plot(history["test_acc"], label="test")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{name}_curves.png"), dpi=150); plt.close(fig)

    size = max(6, len(classes) * 0.22)
    fig, ax = plt.subplots(figsize=(size, size))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cm_norm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(list(classes), fontsize=6); ax.set_yticklabels(list(classes), fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (normalised)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{name}_confusion.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist",
                    choices=["mnist", "emnist_letters", "emnist_balanced", "emnist_byclass"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, test_ds, classes = get_datasets(args.dataset, args.data_dir, augment=not args.no_augment)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_ds, 512, num_workers=2)

    model = CharCNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)
    history = {k: [] for k in ["train_loss", "train_acc", "test_loss", "test_acc"]}
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_dl, criterion, device, optimizer)
        vl, va, preds, labels = run_epoch(model, test_dl, criterion, device)
        scheduler.step()
        for k, v in zip(history, [tl, ta, vl, va]):
            history[k].append(v)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tl:.4f} acc {ta:.4f} | test loss {vl:.4f} acc {va:.4f}")
        if va > best:
            best = va
            torch.save({"state_dict": model.state_dict(), "classes": classes,
                        "dataset": args.dataset}, f"models/{args.dataset}.pt")

    print(f"\nBest test accuracy: {best:.4f}  (model saved to models/{args.dataset}.pt)")
    cm = confusion_matrix(labels, preds)
    save_plots(history, cm, classes, "results", args.dataset)
    report = classification_report(labels, preds, target_names=list(classes), digits=3)
    with open(f"results/{args.dataset}_report.txt", "w") as f:
        f.write(f"Best test accuracy: {best:.4f}\n\n{report}")
    print("Saved plots and report to results/")


if __name__ == "__main__":
    main()
