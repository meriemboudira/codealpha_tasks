"""Word / sentence recognition by segmentation.

Idea: split the handwriting into individual characters using gaps in the
vertical ink projection, classify each with the trained CNN, and insert a
space wherever the gap is much wider than a normal letter gap.

Works for letters written separately (print style). Cursive / touching letters
need the CRNN approach (see crnn.py).
"""
import numpy as np
from PIL import Image


def segment(img: Image.Image, merge_gap: int = 8, ink_thresh: int = 30):
    """Return (ink_array, [(x_start, x_end), ...]) with white strokes on black."""
    arr = np.array(img.convert("L"), dtype=np.float32)
    if arr.mean() > 127:                         # dark ink on light paper -> invert
        arr = 255 - arr
    has_ink = (arr > ink_thresh).any(axis=0)

    runs, start = [], None
    for x, v in enumerate(has_ink):
        if v and start is None:
            start = x
        elif not v and start is not None:
            runs.append((start, x)); start = None
    if start is not None:
        runs.append((start, len(has_ink)))

    merged = []                                  # join pieces of one letter (e.g. split strokes)
    for r in runs:
        if merged and r[0] - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    return arr, merged


def recognise_text(model, classes, img: Image.Image, space_factor: float = 1.2, device="cpu"):
    """Return (text, [(char, confidence), ...])."""
    from src.predict import predict              # lazy import keeps segment() torch-free

    arr, boxes = segment(img)
    if not boxes:
        raise ValueError("No handwriting detected.")

    widths = [x1 - x0 for x0, x1 in boxes]
    space_gap = space_factor * float(np.median(widths))

    text, details = "", []
    for i, (x0, x1) in enumerate(boxes):
        if i > 0 and x0 - boxes[i - 1][1] > space_gap:
            text += " "
        crop = Image.fromarray(arr[:, x0:x1].astype(np.uint8))
        ch, conf = predict(model, classes, crop, k=1, device=device)[0]
        text += ch
        details.append((ch, conf))
    return text, details
