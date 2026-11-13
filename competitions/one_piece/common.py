"""Общие пути, загрузка данных и датасеты для соревнования One Piece."""

import io
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "one-piece-classification-2026"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
CACHE_PATH = OUT_DIR / "cache_384.pkl"

with open(DATA_DIR / "labels.json") as f:
    LABELS = {int(k): v for k, v in json.load(f).items()}
CLASS_TO_LABEL = {v: k for k, v in LABELS.items()}
NUM_CLASSES = len(LABELS)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_rgb(path, max_side: int | None = None) -> Image.Image:
    """Открывает картинку в RGB; прозрачный фон заливается белым."""
    image = Image.open(path)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    image = image.convert("RGB")
    if max_side is not None and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    return image


def load_cache() -> dict:
    """Кэш из prepare.py: JPEG-байты уменьшенных картинок, метки и id теста."""
    with open(CACHE_PATH, "rb") as f:
        return pickle.load(f)


def decode(jpeg_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")


def pad_to_square(image: Image.Image, fill=(255, 255, 255)) -> Image.Image:
    """Дополняет до квадрата, чтобы при ресайзе не обрезать персонажа и не искажать пропорции."""
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


class ImageDataset(Dataset):
    def __init__(self, images: list[bytes], labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        image = decode(self.images[idx])
        if self.transform is not None:
            image = self.transform(image)
        label = -1 if self.labels is None else int(self.labels[idx])
        return image, label
