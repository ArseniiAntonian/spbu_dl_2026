"""Быстрое сравнение бэкбонов: замороженные признаки + логистическая регрессия, 5-fold macro-F1."""

import sys
import time

import numpy as np
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from torch.utils.data import DataLoader
from torchvision import transforms

from common import OUT_DIR, ImageDataset, get_device, load_cache, pad_to_square


@torch.no_grad()
def extract(model, images, transform, device) -> np.ndarray:
    loader = DataLoader(ImageDataset(images, transform=transform), batch_size=64, num_workers=4)
    feats = []
    for x, _ in loader:
        x = x.to(device)
        f = model(x) + model(torch.flip(x, dims=[3]))
        feats.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(feats)


def main() -> None:
    device = get_device()
    cache = load_cache()
    y = np.array(cache["train_labels"])
    for name in sys.argv[1:]:
        start = time.time()
        model = timm.create_model(name, pretrained=True, num_classes=0, img_size=224).eval().to(device)
        cfg = timm.data.resolve_data_config({}, model=model)
        transform = transforms.Compose([
            transforms.Lambda(pad_to_square),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ])
        train_f = extract(model, cache["train_images"], transform, device)
        test_f = extract(model, cache["test_images"], transform, device)
        np.savez(OUT_DIR / f"feats_{name}.npz", train=train_f, test=test_f)

        clf = LogisticRegression(C=10, max_iter=3000)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        pred = cross_val_predict(clf, train_f, y, cv=cv)
        f1 = f1_score(y, pred, average="macro")
        print(f"{name}: macro-F1={f1:.4f} acc={np.mean(pred == y):.4f} ({time.time() - start:.0f}s)", flush=True)
        del model
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
