"""Predict a character from an image file (or a PIL image).

    python -m src.predict path/to/image.png --model models/emnist_balanced.pt
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import MEAN, STD
from src.model import CharCNN


def preprocess(img: Image.Image) -> torch.Tensor:
    """Turn any handwriting image into a MNIST-style 1x1x28x28 tensor.

    Steps: grayscale -> make strokes white on black -> crop to the stroke
    bounding box -> scale longest side to 20px -> centre on a 28x28 canvas.
    """
    img = img.convert("L")
    arr = np.array(img, dtype=np.float32)
    if arr.mean() > 127:                       # dark ink on light paper -> invert
        arr = 255 - arr
    ys, xs = np.where(arr > 30)
    if len(xs) == 0:
        raise ValueError("No handwriting detected in image.")
    arr = arr[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]

    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = Image.fromarray(arr.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(small, ((28 - new_w) // 2, (28 - new_h) // 2))
    x = torch.from_numpy(np.array(canvas, dtype=np.float32) / 255.0)
    x = (x - MEAN) / STD
    return x.unsqueeze(0).unsqueeze(0)


def load_model(path: str, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = CharCNN(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["classes"]


@torch.no_grad()
def predict(model, classes, img: Image.Image, k: int = 3, device="cpu"):
    probs = F.softmax(model(preprocess(img).to(device)), dim=1)[0]
    top = probs.topk(min(k, len(classes)))
    return [(classes[i], p.item()) for p, i in zip(top.values, top.indices)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default="models/mnist.pt")
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()
    m, cls = load_model(a.model)
    for ch, p in predict(m, cls, Image.open(a.image), a.top):
        print(f"{ch!r}: {p * 100:.1f}%")
