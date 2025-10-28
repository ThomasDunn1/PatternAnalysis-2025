# train.py
# Train + Validate each epoch.
#
# Example:
#   python train.py \
#     --train_csv data/splits/train_fold0.csv \
#     --val_csv   data/splits/val_fold0.csv \
#     --out_dir   runs/step7_val \
#     --epochs 2 --limit_train 2048 \
#     --image_size 224 --batch_p 4 --batch_k 4 \
#     --pretrained --freeze_backbone --num_workers 0

from __future__ import annotations
import argparse, json, time
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader, Subset

from dataset import ISIC2020Dataset, PKSampler
from modules import SiameseEncoder
from losses import TripletLoss
from utils import (
    embed_dataset, evaluate_prototypes, plot_training_curves, plot_val_curves
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, type=str)
    ap.add_argument("--val_csv",   required=True, type=str)
    ap.add_argument("--out_dir",   required=True, type=str)

    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=320)
    ap.add_argument("--batch_p", type=int, default=8)
    ap.add_argument("--batch_k", type=int, default=4)
    ap.add_argument("--val_batch", type=int, default=64, help="plain batch size for val embed pass")
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--limit_train", type=int, default=0)
    return ap.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    random.seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


def make_val_loader(csv_path: str, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    ds = ISIC2020Dataset(csv_path, image_size=image_size, mode="val")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    # -------- Datasets & Loaders --------
    # Train
    train_ds = ISIC2020Dataset(args.train_csv, image_size=args.image_size, mode="train")
    if args.limit_train and args.limit_train > 0:
        idx = torch.randperm(len(train_ds))[: args.limit_train]
        train_ds = Subset(train_ds, idx.tolist())
        base = train_ds.dataset
        labels = [base.labels[i] for i in train_ds.indices]
    else:
        labels = getattr(train_ds, "labels", None)
        if labels is None:
            raise RuntimeError("Dataset must expose `labels` for PKSampler.")

    sampler = PKSampler(labels, batch_p=args.batch_p, batch_k=args.batch_k)
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # Val (plain sequential batching)
    val_loader = make_val_loader(args.val_csv, args.image_size, args.val_batch, args.num_workers)

    # -------- Model, Loss, Optim --------
    model = SiameseEncoder(
        embedding_dim=args.embed_dim,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    lossfn = TripletLoss(margin=args.margin, metric="euclidean")
    optimzr = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    # -------- Training + Validation --------
    history = {"epoch": [], "train_loss": [], "val_auc": [], "val_ap": [], "val_acc": [], "val_thr": []}

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
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
            if steps % 50 == 0:
                print(f"[epoch {epoch}] step {steps}  loss {running/steps:.4f}")

        epoch_loss = running / max(steps, 1)
        dt = time.time() - t0
        print(f"[epoch {epoch}] train_loss={epoch_loss:.4f}  ({dt:.1f}s)")

        # ---- validate ----
        eval_res = evaluate_prototypes(model, val_loader, device)
        print(f"[epoch {epoch}] val_auc={eval_res.auc:.4f}  val_ap={eval_res.ap:.4f}  "
              f"val_acc={eval_res.acc:.4f}  thr={eval_res.thr:.3f}  counts={eval_res.counts}")

        # Persist curves & history
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_auc"].append(eval_res.auc)
        history["val_ap"].append(eval_res.ap)
        history["val_acc"].append(eval_res.acc)
        history["val_thr"].append(eval_res.thr)

        with open(out_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Save ROC/PR curves of this epoch
        # (Re-embed to get scores for plotting)
        with torch.no_grad():
            emb, lab = embed_dataset(model, val_loader, device)
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
            protos = compute_prototypes(emb, lab)
            score = score_by_prototypes(emb, protos)
        plot_training_curves(history, out_dir / "training_curves.png")
        plot_val_curves(lab, score, out_dir)

    print(f"[done] wrote history → {out_dir/'training_history.json'}  & plots → {out_dir}")


if __name__ == "__main__":
    # numpy is used in val plotting; import here to keep top clean
    import numpy as np
    from utils import compute_prototypes, score_by_prototypes  # used in plotting block
    main()
