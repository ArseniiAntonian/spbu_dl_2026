"""Собирает кэш: все картинки уменьшаются до 384 px по длинной стороне и хранятся как JPEG-байты."""

import io
import pickle
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from common import CACHE_PATH, CLASS_TO_LABEL, DATA_DIR, load_rgb

MAX_SIDE = 384


def encode(path) -> bytes:
    buffer = io.BytesIO()
    load_rgb(path, MAX_SIDE).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def main() -> None:
    train_dir = DATA_DIR / "splitted" / "train"
    test_dir = DATA_DIR / "splitted" / "test"
    train_paths = sorted(p for p in train_dir.glob("*/*") if p.is_file())
    train_labels = [CLASS_TO_LABEL[p.parent.name] for p in train_paths]

    submission = pd.read_csv(DATA_DIR / "submission.csv")
    test_files = {p.stem: p for p in test_dir.iterdir()}
    test_paths = [test_files[i] for i in submission["id"]]

    with ProcessPoolExecutor() as pool:
        train_images = list(pool.map(encode, train_paths, chunksize=32))
        test_images = list(pool.map(encode, test_paths, chunksize=32))

    cache = {
        "train_images": train_images,
        "train_labels": train_labels,
        "train_names": [f"{p.parent.name}/{p.name}" for p in train_paths],
        "test_images": test_images,
        "test_ids": list(submission["id"]),
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    size_mb = CACHE_PATH.stat().st_size / 2**20
    print(f"train={len(train_images)} test={len(test_images)} cache={size_mb:.0f} MB")


if __name__ == "__main__":
    main()
