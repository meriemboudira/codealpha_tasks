"""Dataset helpers for MNIST and EMNIST (torchvision downloads them automatically)."""
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

DIGITS = "0123456789"
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# EMNIST "balanced" merges visually similar upper/lower pairs, keeping 11 lowercase letters
BALANCED = DIGITS + UPPER + "abdefghnqrt"
BYCLASS = DIGITS + UPPER + "abcdefghijklmnopqrstuvwxyz"

DATASETS = {
    "mnist":            dict(classes=DIGITS),
    "emnist_letters":   dict(classes=UPPER, split="letters", offset=1),   # labels are 1..26
    "emnist_balanced":  dict(classes=BALANCED, split="balanced", offset=0),
    "emnist_byclass":   dict(classes=BYCLASS, split="byclass", offset=0),
}

MEAN, STD = 0.1307, 0.3081


def _fix_emnist(img):
    """EMNIST images are stored transposed; rotate + flip to make them upright."""
    return TF.hflip(TF.rotate(img, -90))


def get_datasets(name: str, root: str = "data", augment: bool = True):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASETS)}")
    cfg = DATASETS[name]
    is_emnist = name.startswith("emnist")

    base = [transforms.Lambda(_fix_emnist)] if is_emnist else []
    aug = [transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1))] if augment else []
    tail = [transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))]
    train_tf = transforms.Compose(base + aug + tail)
    test_tf = transforms.Compose(base + tail)

    offset = cfg.get("offset", 0)
    target_tf = (lambda y: y - offset) if offset else None

    if is_emnist:
        kw = dict(root=root, split=cfg["split"], download=True, target_transform=target_tf)
        train = datasets.EMNIST(train=True, transform=train_tf, **kw)
        test = datasets.EMNIST(train=False, transform=test_tf, **kw)
    else:
        train = datasets.MNIST(root, train=True, download=True, transform=train_tf)
        test = datasets.MNIST(root, train=False, download=True, transform=test_tf)
    return train, test, cfg["classes"]
