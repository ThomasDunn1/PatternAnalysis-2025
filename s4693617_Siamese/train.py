# train.py
# Minimal training loop (no validation yet): PKSampler → SiameseEncoder → TripletLoss → AdamW.
# Logs per-epoch train loss to a small JSON file.
#
# Example (1 epoch, tiny subset for speed):
#   python train.py \
#     --train_csv data/splits/train_fold0.csv \
#     --out_dir runs/mvp_step6 \
#     --epochs 1 --limit_train 512 --image_size 224 --batch_p 4 --batch_k 4 --freeze_backbone

from __future__ import annotations
import argparse, json, time
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader, Subset

from dataset import ISIC2020Dataset, PKSampler
from modules import SiameseEncoder
from losses import TripletLoss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, type=str, help="CSV with image_path,label,patient_id")
    ap.add_argument("--out_dir", required=True, type=str, help="Directory to write logs/ckpts")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--image_size", type=int, default=320)
    ap.add_argument("--batch_p", type=int, default=8, help="P: classes per batch")
    ap.add_argument("--batch_k", type=int, default=4, help="K: samples per class")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--pretrained", action="store_true", help="use ImageNet weights for backbone")
    ap.add_argument("--freeze_backbone", action="store_true", help="freeze backbone; train head only")
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    ap.add_argument("--margin", type=float, default=0.3, help="triplet margin")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--limit_train", type=int, default=0, help="limit training set size (0 = use all)")
    return ap.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    random.seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = True  # keep True for speed (we’re not going for exact determinism)


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    # -------- Dataset & Loader (train-only) --------
    train_ds = ISIC2020Dataset(args.train_csv, image_size=args.image_size, mode="train")

    if args.limit_train and args.limit_train > 0:
        idx = torch.randperm(len(train_ds))[: args.limit_train]
        train_ds = Subset(train_ds, idx.tolist())
        # Subset removes .labels; grab from original dataset using indices
        # build labels list aligned to Subset order:
        base = train_ds.dataset
        labels = [base.labels[i] for i in train_ds.indices]
    else:
        labels = getattr(train_ds, "labels", None)
        if labels is None:
            raise RuntimeError("Dataset must expose `labels` for PKSampler.")

    sampler = PKSampler(labels, batch_p=args.batch_p, batch_k=args.batch_k)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # -------- Model, Loss, Optim --------
    model = SiameseEncoder(
        embedding_dim=args.embed_dim,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    lossfn = TripletLoss(margin=args.margin, metric="euclidean")
    optimzr = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    # -------- Train loop --------
    history = {"epoch": [], "train_loss": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, steps = 0.0, 0
        t0 = time.time()

        for batch in train_loader:
            x, y, _ = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device)

            optimzr.zero_grad(set_to_none=True)

            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    z = model(x)
                    loss = lossfn(z, y)
                scaler.scale(loss).backward()
                scaler.step(optimzr)
                scaler.update()
            else:
                z = model(x)
                loss = lossfn(z, y)
                loss.backward()
                optimzr.step()

            running += loss.item()
            steps += 1

            # (Optional) modest print frequency
            if steps % 50 == 0:
                print(f"[epoch {epoch}] step {steps}  loss {running/steps:.4f}")

        epoch_loss = running / max(steps, 1)
        dt = time.time() - t0
        print(f"[epoch {epoch}] train_loss={epoch_loss:.4f}  ({dt:.1f}s)")

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)

        # Save small rolling history
        with open(out_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"[done] wrote history → {out_dir/'training_history.json'}")


if __name__ == "__main__":
    main()
