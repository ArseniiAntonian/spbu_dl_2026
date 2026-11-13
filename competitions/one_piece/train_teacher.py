"""Fine-tune предобученного бэкбона из timm (учитель).

--fold k  — обучение на остальных фолдах с валидацией на k;
--fold -1 — обучение на всём train (для финального сабмита).
Сохраняет веса и логиты (val / test / весь train) в outputs/<run>/.
"""

import argparse
import json
import math
import time

import numpy as np
import timm
import timm.optim
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from common import NUM_CLASSES, OUT_DIR, ImageDataset, decode, get_device, load_cache, pad_to_square, seed_everything


def build_transforms(img_size: int, mean, std):
    bicubic = transforms.InterpolationMode.BICUBIC
    train = transforms.Compose([
        transforms.Lambda(pad_to_square),
        transforms.RandomResizedCrop(img_size, scale=(0.35, 1.0), interpolation=bicubic),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)], p=0.8),
        transforms.RandomGrayscale(p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25),
    ])
    test = transforms.Compose([
        transforms.Lambda(pad_to_square),
        transforms.Resize((img_size, img_size), interpolation=bicubic),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train, test


@torch.no_grad()
def predict(model, images, transform, device, autocast) -> np.ndarray:
    """Логиты с TTA: оригинал + горизонтальное отражение."""
    model.eval()
    loader = DataLoader(ImageDataset(images, transform=transform), batch_size=64, num_workers=4)
    logits = []
    for x, _ in loader:
        x = x.to(device)
        with autocast():
            out = model(x).float() + model(torch.flip(x, dims=[3])).float()
        logits.append((out / 2).cpu().numpy())
    return np.concatenate(logits)


@torch.no_grad()
def init_head_from_linear_probe(model, name, images, labels, train_idx, transform, device) -> None:
    """LP-FT: случайная голова в начале fine-tune портит предобученные признаки, поэтому
    голова инициализируется логистической регрессией на замороженных признаках из probe.py.
    Регрессия обучена на L2-нормированных признаках, так что веса делим на среднюю норму признака."""
    feats = np.load(OUT_DIR / f"feats_{name}.npz")["train"][train_idx]
    clf = LogisticRegression(C=10, max_iter=3000).fit(feats, labels)
    model.eval()
    x = torch.stack([transform(decode(b)) for b in images[:128]]).to(device)
    mean_norm = model.forward_head(model.forward_features(x), pre_logits=True).norm(dim=1).mean()
    model.get_classifier().weight.copy_(torch.from_numpy(clf.coef_).float().to(device) / mean_norm)
    model.get_classifier().bias.copy_(torch.from_numpy(clf.intercept_).float().to(device))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_base_patch16_clip_224.openai")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--layer-decay", type=float, default=0.7)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--init-head", action="store_true", help="LP-FT: голова из линейной пробы (probe.py)")
    parser.add_argument("--max-steps", type=int, default=0, help="для замера скорости")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    seed_everything(42 + args.fold)
    device = get_device()
    run_dir = OUT_DIR / args.run
    run_dir.mkdir(exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args)))

    cache = load_cache()
    images = cache["train_images"]
    labels = np.array(cache["train_labels"])
    folds = np.load(OUT_DIR / "folds.npy")
    train_idx = np.where(folds != args.fold)[0]
    val_idx = np.where(folds == args.fold)[0]

    model = timm.create_model(
        args.model, pretrained=True, num_classes=NUM_CLASSES, img_size=args.img_size, drop_path_rate=args.drop_path
    ).to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    train_tf, test_tf = build_transforms(args.img_size, cfg["mean"], cfg["std"])

    def autocast():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    if args.init_head:
        init_head_from_linear_probe(model, args.model, [images[i] for i in train_idx], labels[train_idx], train_idx, test_tf, device)

    loader = DataLoader(
        ImageDataset([images[i] for i in train_idx], labels[train_idx], train_tf),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=6, persistent_workers=True,
    )
    optimizer = timm.optim.create_optimizer_v2(
        model, opt="adamw", lr=args.lr, weight_decay=0.05, layer_decay=args.layer_decay
    )
    total_steps = args.epochs * len(loader)
    warmup = len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total_steps - warmup))),
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    step = 0
    for epoch in range(args.epochs):
        model.train()
        start, losses = time.time(), []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with autocast():
                loss = criterion(model(x).float(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            step += 1
            if args.max_steps and step >= args.max_steps:
                print(f"{step} steps: {(time.time() - start) / step:.3f}s/step")
                return
        message = f"epoch {epoch + 1}/{args.epochs} loss={np.mean(losses):.4f} {time.time() - start:.0f}s"
        if len(val_idx) and (epoch + 1) % 2 == 0 or epoch + 1 == args.epochs:
            if len(val_idx):
                val_logits = predict(model, [images[i] for i in val_idx], test_tf, device, autocast)
                pred = val_logits.argmax(1)
                f1 = f1_score(labels[val_idx], pred, average="macro")
                message += f" val macro-F1={f1:.4f} acc={np.mean(pred == labels[val_idx]):.4f}"
        print(message, flush=True)

    torch.save(model.state_dict(), run_dir / "model.pt")
    if len(val_idx):
        np.save(run_dir / "val_idx.npy", val_idx)
        np.save(run_dir / "val_logits.npy", val_logits)
    np.save(run_dir / "test_logits.npy", predict(model, cache["test_images"], test_tf, device, autocast))
    np.save(run_dir / "train_logits.npy", predict(model, images, test_tf, device, autocast))
    print("saved to", run_dir)


if __name__ == "__main__":
    main()
