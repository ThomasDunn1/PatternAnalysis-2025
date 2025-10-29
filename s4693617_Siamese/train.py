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
# train.py — Step 8 (checkpointing + best selection + final report)

from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
import os, torch.multiprocessing as mp, platform
mp.set_sharing_strategy("file_system")

from dataset import ISIC2020Dataset, PKSampler
from modules import SiameseEncoder
from losses import TripletLoss
from utils import (
    embed_dataset, evaluate_prototypes, plot_training_curves, plot_val_curves,
    compute_prototypes, score_by_prototypes, save_config
)

# Safety when on WSL
def on_drvfs(path: str) -> bool:
    # Heuristic: WSL Windows mount (e.g., /mnt/c, /mnt/d, /mnt/e)
    return path.startswith("/mnt/")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default=None)
    ap.add_argument("--val_csv", type=str, default=None)
    ap.add_argument("--out_dir",   required=True, type=str)

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--image_size", type=int, default=320)
    ap.add_argument("--batch_p", type=int, default=8)
    ap.add_argument("--batch_k", type=int, default=4)
    ap.add_argument("--val_batch", type=int, default=64)
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
    ap.add_argument("--limit_val", type=int, default=0, help="limit val set size for fast smoke runs (0=all)")

    ap.add_argument("--early_stop", action="store_true", help="enable early stop on val AUC")
    ap.add_argument("--patience", type=int, default=3, help="epochs without AUC improvement")
    
    ap.add_argument("--fold", type=int, default=None, help="if set, use train_fold{fold}.csv / val_fold{fold}.csv")
    ap.add_argument("--splits_dir", type=str, default="data/splits", help="directory holding fold CSVs")

    return ap.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    random.seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


def make_val_loader(csv_path: str, image_size: int, batch_size: int, num_workers: int, limit_val: int = 0) -> DataLoader:
    ds = ISIC2020Dataset(csv_path, image_size=image_size, mode="val")
    if limit_val and limit_val > 0:
        from torch.utils.data import Subset
        idx = torch.arange(min(len(ds), limit_val))
        ds = Subset(ds, idx.tolist())
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )


def main():
    args = parse_args()
    
    splits_dir = Path(args.splits_dir)

    def _auto_csvs():
        if args.fold is None:
            return args.train_csv, args.val_csv
        tr = splits_dir / f"train_fold{args.fold}.csv"
        va = splits_dir / f"val_fold{args.fold}.csv"
        if not tr.exists() or not va.exists():
            raise FileNotFoundError(f"Could not find fold CSVs: {tr} / {va}")
        return str(tr), str(va)

    # accept either: (A) --fold or (B) explicit CSVs
    if args.fold is not None:
        train_csv_resolved, val_csv_resolved = _auto_csvs()
    elif args.train_csv is not None and args.val_csv is not None:
        train_csv_resolved, val_csv_resolved = args.train_csv, args.val_csv
    else:
        raise ValueError("Either pass --fold (with --splits_dir) OR both --train_csv and --val_csv.")

    print(f"[fold] using train={train_csv_resolved}  val={val_csv_resolved}")

    
    train_on_drvfs = on_drvfs(os.path.abspath(train_csv_resolved)) or on_drvfs(os.path.abspath(args.out_dir))
    safe_num_workers = 0 if train_on_drvfs else args.num_workers
    pin = (torch.cuda.is_available() and not train_on_drvfs)

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    # -------- Datasets & Loaders --------
    train_ds = ISIC2020Dataset(train_csv_resolved, image_size=args.image_size, mode="train")
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
        train_ds,
        batch_sampler=sampler,
        num_workers=safe_num_workers,
        pin_memory=pin,
        persistent_workers=False,   # keep False on WSL/drvfs
    )

    val_loader = make_val_loader(val_csv_resolved, args.image_size, args.val_batch, safe_num_workers, limit_val=args.limit_val)

    # -------- Model, Loss, Optim --------
    model = SiameseEncoder(
        embedding_dim=args.embed_dim,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    lossfn = TripletLoss(margin=args.margin, metric="euclidean")
    optimzr = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    # -------- History / Best tracking --------
    history = {"epoch": [], "train_loss": [], "val_auc": [], "val_ap": [], "val_acc": [], "val_thr": []}
    best_auc, best_epoch = -1.0, -1
    bad_epochs = 0
    best_path = out_dir / "best.pt"

    # Save a simple config snapshot once
    cfg = {
        "train_csv": train_csv_resolved,
        "val_csv": train_csv_resolved,
        "image_size": args.image_size,
        "embed_dim": args.embed_dim,
        "margin": args.margin,
        "batch_p": args.batch_p,
        "batch_k": args.batch_k,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pretrained": args.pretrained,
        "freeze_backbone": args.freeze_backbone,
        "amp": args.amp,
        "seed": args.seed,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # -------- Training + Validation --------
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

        # Update history
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_auc"].append(eval_res.auc)
        history["val_ap"].append(eval_res.ap)
        history["val_acc"].append(eval_res.acc)
        history["val_thr"].append(eval_res.thr)

        with open(out_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Save curves (also store epoch ROC/PR)
        with torch.no_grad():
            emb, lab = embed_dataset(model, val_loader, device)
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
            protos = compute_prototypes(emb, lab)
            score = score_by_prototypes(emb, protos)
        plot_training_curves(history, out_dir / "training_curves.png")
        plot_val_curves(lab, score, out_dir)

        # ---- checkpoint on AUC improvement ----
        if eval_res.auc > best_auc:
            best_auc, best_epoch = eval_res.auc, epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict()}, best_path)
            print(f"[epoch {epoch}] ↑ best AUC {best_auc:.4f} — saved {best_path.name}")
        else:
            bad_epochs += 1
            if args.early_stop and bad_epochs >= args.patience:
                print(f"[early stop] no AUC improvement for {bad_epochs} epochs (patience={args.patience})")
                break

    # -------- Final report: reload best.pt and evaluate once more --------
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        final_res = evaluate_prototypes(model, val_loader, device)
        rep = (
            f"best_epoch={best_epoch}\n"
            f"val_auc={final_res.auc:.4f}\n"
            f"val_ap={final_res.ap:.4f}\n"
            f"val_acc={final_res.acc:.4f}\n"
            f"thr={final_res.thr:.6f}\n"
            f"class_counts={final_res.counts}\n"
        )
        (out_dir / "test_stats.txt").write_text(rep)
        print("[final] wrote", (out_dir / "test_stats.txt").as_posix())
    else:
        print("[warn] best.pt not found; no final report generated")

    print(f"[done] history → {out_dir/'training_history.json'}  best → {best_path if best_path.exists() else 'n/a'}")


if __name__ == "__main__":
    main()
