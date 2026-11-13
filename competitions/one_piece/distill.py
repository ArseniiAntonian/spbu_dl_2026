"""Дистилляция дообученного учителя в собственную маленькую CNN (OnePieceNet).

Ученик видит ровно те же аугментированные картинки, на которых make_views.py посчитал логиты учителя,
и учится повторять его распределение (soft cross-entropy). Для размеченных картинок к цели учителя
подмешивается истинная метка. Неразмеченный test тоже участвует: для него цель — только учитель.
Поверх видов — mixup/cutmix, цели смешиваются с теми же весами.
"""

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from timm.utils import ModelEmaV3
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms

from common import NUM_CLASSES, OUT_DIR, ImageDataset, get_device, load_cache, pad_to_square, seed_everything
from student import OnePieceNet

STUDENT_MEAN, STUDENT_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


class Normalized(torch.nn.Module):
    """Модель + своя нормализация входа: в батче лежат картинки в [0, 1]."""

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def load_teacher(run: str, device) -> Normalized:
    args = json.loads((OUT_DIR / run / "args.json").read_text())
    model = timm.create_model(args["model"], pretrained=False, num_classes=NUM_CLASSES, img_size=args["img_size"])
    model.load_state_dict(torch.load(OUT_DIR / run / "model.pt", map_location="cpu"))
    cfg = timm.data.resolve_data_config({}, model=model)
    return Normalized(model, cfg["mean"], cfg["std"]).eval().to(device)


def mix(x: torch.Tensor, target: torch.Tensor):
    """С вероятностью 0.25 mixup, 0.25 cutmix, иначе батч без изменений."""
    r = np.random.rand()
    if r >= 0.5:
        return x, target
    perm = torch.randperm(x.shape[0], device=x.device)
    if r < 0.25:
        lam = float(np.random.beta(0.8, 0.8))
        return lam * x + (1 - lam) * x[perm], lam * target + (1 - lam) * target[perm]
    h, w = x.shape[2:]
    lam = float(np.random.beta(1.0, 1.0))
    ch, cw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
    cy, cx = np.random.randint(h), np.random.randint(w)
    y0, y1 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
    x0, x1 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
    x = x.clone()
    x[:, :, y0:y1, x0:x1] = x[perm, :, y0:y1, x0:x1]
    lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)
    return x, lam * target + (1 - lam) * target[perm]


@torch.no_grad()
def predict(model, images, img_size, device) -> np.ndarray:
    tf = transforms.Compose([
        transforms.Lambda(pad_to_square),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    model.eval()
    loader = DataLoader(ImageDataset(images, transform=tf), batch_size=128, num_workers=4)
    out = []
    for x, _ in loader:
        x = x.to(device)
        out.append(((model(x) + model(torch.flip(x, dims=[3]))) / 2).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    from make_views import ViewDataset, all_images

    parser = argparse.ArgumentParser()
    parser.add_argument("--views", nargs="+", required=True, help="файлы логитов учителя из make_views.py")
    parser.add_argument("--fold", type=int, default=-1, help="фолд, исключаемый из обучения для валидации")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--label-weight", type=float, default=0.3)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    seed_everything(0)
    device = get_device()
    run_dir = OUT_DIR / args.run
    run_dir.mkdir(exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args)))

    cache = load_cache()
    images = all_images(cache)
    n_train, n_images = len(cache["train_images"]), len(images)
    labels = np.array(cache["train_labels"])
    folds = np.load(OUT_DIR / "folds.npy")
    val_idx = np.where(folds == args.fold)[0]

    # цели для каждого (вид, картинка): softmax учителя, для train с примесью истинной метки
    teacher_probs = torch.softmax(torch.from_numpy(np.concatenate([np.load(OUT_DIR / v) for v in args.views])), dim=-1)
    n_views = teacher_probs.shape[0]
    onehot = F.one_hot(torch.from_numpy(labels), NUM_CLASSES).float() * 0.9 + 0.1 / NUM_CLASSES
    targets = teacher_probs.clone()
    targets[:, :n_train] = (1 - args.label_weight) * targets[:, :n_train] + args.label_weight * onehot
    targets = targets.reshape(-1, NUM_CLASSES)

    resize = transforms.Resize((args.img_size, args.img_size), antialias=True)
    dataset = ViewDataset(images, n_views, post_transform=resize)
    allowed = np.ones(n_images, dtype=bool)
    allowed[val_idx] = False
    allowed_items = np.where(np.tile(allowed, n_views))[0]
    subset = torch.utils.data.Subset(dataset, allowed_items)
    per_epoch = int(allowed.sum())
    loader = DataLoader(
        subset, batch_size=args.batch_size, drop_last=True, num_workers=8, persistent_workers=True,
        sampler=RandomSampler(subset, num_samples=per_epoch),
    )

    student = Normalized(OnePieceNet(NUM_CLASSES), STUDENT_MEAN, STUDENT_STD).to(device)
    ema = ModelEmaV3(student, decay=0.998)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.05)
    total_steps = args.epochs * len(loader)
    warmup = 3 * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total_steps - warmup))),
    )

    val_images = [cache["train_images"][i] for i in val_idx]
    for epoch in range(args.epochs):
        student.train()
        start, losses = time.time(), []
        for x, idx in loader:
            x, target = x.to(device), targets[idx].to(device)
            flip = torch.rand(x.shape[0], device=device) < 0.5
            x = torch.where(flip[:, None, None, None], torch.flip(x, dims=[3]), x)
            x, target = mix(x, target)
            loss = torch.sum(-target * F.log_softmax(student(x), dim=1), dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update(student)
            losses.append(loss.item())
        message = f"epoch {epoch + 1}/{args.epochs} loss={np.mean(losses):.4f} {time.time() - start:.0f}s"
        if len(val_idx) and ((epoch + 1) % 10 == 0 or epoch + 1 == args.epochs):
            for name, model in [("student", student), ("ema", ema.module)]:
                pred = predict(model, val_images, args.img_size, device).argmax(1)
                message += f" {name} F1={f1_score(labels[val_idx], pred, average='macro'):.4f}"
        print(message, flush=True)
        if (epoch + 1) % 10 == 0:
            torch.save(ema.module.model.state_dict(), run_dir / "student_ema.pt")

    final = ema.module
    torch.save(final.model.state_dict(), run_dir / "student_ema.pt")
    size_mb = (run_dir / "student_ema.pt").stat().st_size / 2**20
    test_logits = predict(final, cache["test_images"], args.img_size, device)
    np.save(run_dir / "test_logits.npy", test_logits)
    pd.DataFrame({"id": cache["test_ids"], "label": test_logits.argmax(1)}).to_csv(run_dir / "submission.csv", index=False)
    if len(val_idx):
        val_logits = predict(final, val_images, args.img_size, device)
        np.save(run_dir / "val_logits.npy", val_logits)
        print(f"final EMA val macro-F1={f1_score(labels[val_idx], val_logits.argmax(1), average='macro'):.4f}")
    print(f"weights {size_mb:.2f} MB, saved to {run_dir}")


if __name__ == "__main__":
    main()
