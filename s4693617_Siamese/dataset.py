# dataset.py
# Minimal ISIC-2020 dataset (train/val transforms), no custom sampler yet.

from __future__ import annotations
from pathlib import Path
from typing import Tuple

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
