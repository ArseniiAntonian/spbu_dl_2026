"""Логиты учителя на K фиксированных аугментированных версиях каждой картинки (train + test).

Учитель (ViT-B) на MPS слишком медленный, чтобы прогонять его на каждом шаге обучения ученика,
поэтому аугментации детерминированы сидом (k, i): ученик потом воссоздаёт ровно ту же картинку.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import OUT_DIR, decode, get_device, load_cache, pad_to_square
from distill import load_teacher

VIEW_SIZE = 224


def view_transform():
    bicubic = transforms.InterpolationMode.BICUBIC
    return transforms.Compose([
        transforms.Lambda(pad_to_square),
        transforms.RandomResizedCrop(VIEW_SIZE, scale=(0.3, 1.0), interpolation=bicubic),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.05)], p=0.8),
        transforms.RandomGrayscale(p=0.15),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25),
    ])


class ViewDataset(Dataset):
    """Элемент idx = (вид k, картинка i); аугментация задаётся сидом, поэтому воспроизводима."""

    def __init__(self, images: list[bytes], n_views: int, post_transform=None):
        self.images = images
        self.n_views = n_views
        self.transform = view_transform()
        self.post_transform = post_transform

    def __len__(self) -> int:
        return self.n_views * len(self.images)

    def __getitem__(self, idx):
        k, i = divmod(idx, len(self.images))
        torch.manual_seed(1_000_003 * k + i)
        x = self.transform(decode(self.images[i]))
        if self.post_transform is not None:
            x = self.post_transform(x)
        return x, idx


def all_images(cache) -> list[bytes]:
    """Порядок картинок для видов: сначала весь train, затем test."""
    return cache["train_images"] + cache["test_images"]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teachers", nargs="+", required=True)
    parser.add_argument("--views", type=int, default=10)
    parser.add_argument("--first-view", type=int, default=0, help="чтобы досчитать виды отдельным запуском")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = get_device()
    images = all_images(load_cache())
    teachers = [load_teacher(run, device) for run in args.teachers]
    dataset = ViewDataset(images, args.first_view + args.views)
    indices = range(args.first_view * len(images), len(dataset))
    loader = DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=64, num_workers=8)

    logits = np.zeros((len(dataset), teachers[0].model.num_classes), dtype=np.float32)
    for step, (x, idx) in enumerate(loader):
        x = x.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = torch.stack([t(x).float() for t in teachers]).mean(0)
        logits[idx.numpy()] = out.cpu().numpy()
        if step % 50 == 0:
            print(f"{step}/{len(loader)}", flush=True)
    logits = logits.reshape(-1, len(images), logits.shape[1])[args.first_view:]
    np.save(OUT_DIR / f"{args.out}_views{args.first_view}-{args.first_view + args.views}.npy", logits)
    print("saved", logits.shape)


if __name__ == "__main__":
    main()
