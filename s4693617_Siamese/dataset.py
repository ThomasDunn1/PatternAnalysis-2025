# dataset.py
# Minimal ISIC-2020 dataset (train/val transforms), no custom sampler yet.

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Dict
import math, numpy as np
from torch.utils.data import Sampler

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class ISIC2020Dataset(Dataset):
    """
    Expects a CSV with columns: image_path,label,patient_id  (headers case-insensitive).
      - image_path: absolute or relative path to the image file
      - label: 0 (benign/normal) or 1 (melanoma)
      - patient_id: string patient identifier (used later for leakage checks / samplers)

    Args
    ----
    csv_path : str | Path
        Path to the split CSV (train_fold*.csv or val_fold*.csv)
    root_dir : str | Path | None
        Optional directory to prepend if image_path in CSV is relative
    image_size : int
        Final square size after crops/resizes
    mode : {"train","val"}
        Controls augmentations (train = heavier; val = center crop only)
    """

    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path | None = None,
        image_size: int = 320,
        mode: str = "train",
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir) if root_dir else None
        self.mode = mode.lower().strip()
        if self.mode not in {"train", "val"}:
            raise ValueError("mode must be 'train' or 'val'")

        df = pd.read_csv(self.csv_path)
        cols = {c.lower(): c for c in df.columns}
        self.paths = df[cols.get("image_path", list(df.columns)[0])].astype(str).tolist()
        self.labels = df[cols.get("label")].astype(int).tolist()
        self.pids = df[cols.get("patient_id")].astype(str).tolist()

        # Augmentations (kept realistic for dermoscopy)
        if self.mode == "train":
            self.tf = T.Compose(
                [
                    T.Resize(int(image_size * 1.15)),
                    T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomRotation(180),
                    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:  # val
            self.tf = T.Compose(
                [
                    T.Resize(image_size),
                    T.CenterCrop(image_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self) -> int:
        return len(self.paths)

    def _resolve_path(self, p: str) -> Path:
        pth = Path(p)
        if pth.is_absolute() or self.root_dir is None:
            return pth
        return (self.root_dir / pth).resolve()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path = self._resolve_path(self.paths[idx])
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        y = int(self.labels[idx])
        pid = self.pids[idx]
        return x, y, pid

class PKSampler(Sampler[List[int]]):
    """
    Finite-epoch P×K sampler: each epoch yields ~N // (P*K) batches (drop_last=True)
    or ceil(N/(P*K)) when drop_last=False. Uses wrap-around within classes to keep
    batches full, but limits the number of batches per epoch to 'steps'.
    """
    def __init__(self, labels: List[int], batch_p: int = 8, batch_k: int = 4, drop_last: bool = True):
        self.labels = np.asarray(labels, dtype=int)
        self.P = int(batch_p)
        self.K = int(batch_k)
        self.drop_last = bool(drop_last)

        self.idxs_by_class: Dict[int, np.ndarray] = {
            int(c): np.where(self.labels == c)[0] for c in np.unique(self.labels)
        }
        for c, arr in self.idxs_by_class.items():
            if arr.size == 0:
                raise ValueError(f"class {c} has no samples; cannot build P×K batches")

        self._n = int(self.labels.size)
        self._bsz = self.P * self.K
        self._steps = (self._n // self._bsz) if self.drop_last else int(math.ceil(self._n / self._bsz))

    def __len__(self) -> int:
        return self._steps

    def __iter__(self):
        rng = np.random.default_rng()
        # shuffle per-class pools and init pointers each epoch
        pools = {c: rng.permutation(v.copy()) for c, v in self.idxs_by_class.items()}
        ptrs  = {c: 0 for c in pools}
        classes = list(pools.keys())

        for _ in range(self._steps):
            # choose P classes; for binary ensure both if possible
            if set(classes) == {0, 1} and self.P >= 2:
                chosen = [0, 1]
                if self.P > 2:
                    extra = rng.choice(classes, size=self.P - 2, replace=True).tolist()
                    chosen += extra
            else:
                chosen = rng.choice(classes, size=self.P, replace=True).tolist()

            batch: List[int] = []
            for c in chosen:
                for _k in range(self.K):
                    if ptrs[c] >= pools[c].size:
                        pools[c] = rng.permutation(pools[c])
                        ptrs[c] = 0
                    batch.append(int(pools[c][ptrs[c]]))
                    ptrs[c] += 1

            yield batch
