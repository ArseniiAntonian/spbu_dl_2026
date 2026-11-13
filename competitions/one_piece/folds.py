"""Разбиение train на фолды так, чтобы почти-дубликаты попадали в один фолд (иначе валидация завышена)."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

from common import OUT_DIR, load_cache

N_FOLDS = 5
DUP_THRESHOLD = 0.95


def main() -> None:
    labels = np.array(load_cache()["train_labels"])
    feats = np.load(OUT_DIR / "feats_vit_base_patch16_clip_224.openai.npz")["train"]
    similar = csr_matrix(feats @ feats.T > DUP_THRESHOLD)
    n_groups, groups = connected_components(similar, directed=False)

    folds = np.zeros(len(labels), dtype=int)
    splitter = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=0)
    for fold, (_, val_idx) in enumerate(splitter.split(feats, labels, groups)):
        folds[val_idx] = fold
    np.save(OUT_DIR / "folds.npy", folds)
    print(f"{len(labels)} images, {n_groups} duplicate groups, fold sizes {np.bincount(folds)}")


if __name__ == "__main__":
    main()
