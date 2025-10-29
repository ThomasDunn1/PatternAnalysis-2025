# train.py
# Train + Validate each epoch — memory-safe version.
# Writes JSON/TXT artifacts; no per-epoch plots (optional one-shot plots at end).
#
# Example (debug-fast):
#   python train.py \
#     --fold 0 --splits_dir data/splits \
#     --out_dir runs/debug_fold0 \
#     --epochs 2 --limit_train 4000 --limit_val 2000 --max_train_steps 64 \
#     --image_size 224 --batch_p 4 --batch_k 4 --val_batch 256 \
#     --pretrained --freeze_backbone --num_workers 2 --amp \
#     --early_stop --patience 2 \
#     --log_file runs/debug_fold0/train.log
#
# Example (normal):
#   python train.py \
#     --fold 0 --splits_dir data/splits \
#     --out_dir runs/folds/fold0 \
#     --epochs 5 --image_size 320 --batch_p 8 --batch_k 4 --val_batch 256 \
#     --pretrained --freeze_backbone --num_workers 6 --amp \
#     --early_stop --patience 3

from __future__ import annotations
import argparse, json, time, os, gc, datetime, psutil, platform
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

from dataset import ISIC2020Dataset, PKSampler
from modules import SiameseEncoder
from losses import TripletLoss
from utils import (
    evaluate_prototypes,   # must stream val once; should NOT retain arrays internally
    # The following are used only if --plots_at_end is set
    embed_dataset, compute_prototypes, score_by_prototypes,
    plot_training_curves, plot_val_curves,
)

# -------------------- utilities --------------------

def _now(): return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def mem_snapshot(device: torch.device) -> str:
    rss = psutil.Process().memory_info().rss / (1024**3)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / (1024**3)
        return f"CPU {rss:.2f} GiB | GPU {used:.2f} / {total/(1024**3):.0f} GiB"
    return f"CPU {rss:.2f} GiB"

def log(msg: str, log_path: str | None = None):
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

def on_drvfs(path: str) -> bool:
    # Detect WSL Windows mounts (/mnt/c, /mnt/d, ...)
    return path.startswith("/mnt/")

def set_seed(seed: int):
    import random
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    random.seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

# -------------------- args --------------------

def parse_args():
    ap = argparse.ArgumentParser()
    # explicit paths (optional if --fold is used)
    ap.add_argument("--train_csv", type=str, default=None)
    ap.add_argument("--val_csv",   type=str, default=None)
    ap.add_argument("--out_dir",   required=True, type=str)

    # convenience fold selection
    ap.add_argument("--fold", type=int, default=None,
                    help="if set, use train_fold{fold}.csv / val_fold{fold}.csv")
    ap.add_argument("--splits_dir", type=str, default="data/splits",
                    help="directory holding fold CSVs")

    # training knobs
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
    ap.add_argument("--limit_val", type=int, default=0,
                    help="limit val set size for fast smoke runs (0=all)")
    ap.add_argument("--max_train_steps", type=int, default=0,
                    help="cap train steps per epoch (0 = unlimited)")

    ap.add_argument("--early_stop", action="store_true",
                    help="enable early stop on val AUC")
    ap.add_argument("--patience", type=int, default=3,
                    help="epochs without AUC improvement")

    # logging/plots policy
    ap.add_argument("--log_file", type=str, default="",
                    help="path to rolling log file ('' disables)")
    ap.add_argument("--plots_at_end", action="store_true",
                    help="after training only: embed val once to produce plots safely")

    return ap.parse_args()

# -------------------- dataloaders --------------------

def make_val_loader(csv_path: str, image_size: int, batch_size: int,
                    num_workers: int, limit_val: int, pin: bool,
                    persistent: bool) -> DataLoader:
    ds = ISIC2020Dataset(csv_path, image_size=image_size, mode="val")
    if limit_val and limit_val > 0:
        idx = torch.arange(min(len(ds), limit_val))
        ds = Subset(ds, idx.tolist())
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )

# -------------------- main --------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    # Resolve CSVs (either --fold or explicit paths)
    splits_dir = Path(args.splits_dir)
    def _auto_csvs():
        if args.fold is None:
            return args.train_csv, args.val_csv
        tr = splits_dir / f"train_fold{args.fold}.csv"
        va = splits_dir / f"val_fold{args.fold}.csv"
        if not tr.exists() or not va.exists():
            raise FileNotFoundError(f"Could not find fold CSVs: {tr} / {va}")
        return str(tr), str(va)

    if args.fold is not None:
        train_csv_resolved, val_csv_resolved = _auto_csvs()
    elif args.train_csv and args.val_csv:
        train_csv_resolved, val_csv_resolved = args.train_csv, args.val_csv
    else:
        raise ValueError("Either pass --fold (with --splits_dir) OR both --train_csv and --val_csv.")

    print(f"[fold] using train={train_csv_resolved}  val={val_csv_resolved}")

    # Environment capabilities
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    # Worker policy: disable on DRVFS/WSL to avoid shared-mem/FD issues.
    train_on_drvfs = on_drvfs(os.path.abspath(train_csv_resolved)) or on_drvfs(os.path.abspath(args.out_dir))
    safe_num_workers = 0 if train_on_drvfs else args.num_workers
    pin = (device.type == "cuda") and (not train_on_drvfs)
    persistent = False  # keep False for stability (can turn True later on native Linux)

    log_path = args.log_file if args.log_file else None
    log(f"device={device} | {mem_snapshot(device)}", log_path)

    # ---------------- datasets / loaders ----------------
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
        persistent_workers=persistent,
    )

    val_loader = make_val_loader(
        val_csv_resolved, args.image_size, args.val_batch,
        safe_num_workers, args.limit_val, pin, persistent
    )

    # ---------------- model / optim ----------------
    model = SiameseEncoder(
        embedding_dim=args.embed_dim,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    lossfn = TripletLoss(margin=args.margin, metric="euclidean")
    optimzr = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    # ---------------- bookkeeping ----------------
    history = {"epoch": [], "train_loss": [], "val_auc": [], "val_ap": [], "val_acc": [], "val_thr": []}
    best_auc, best_epoch = -1.0, -1
    bad_epochs = 0
    best_path = out_dir / "best.pt"

    # save config snapshot (correct val_csv)
    cfg = {
        "train_csv": train_csv_resolved,
        "val_csv":   val_csv_resolved,
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
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # ---------------- epoch loop ----------------
    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        running, steps = 0.0, 0
        t0 = time.time()

        max_steps = args.max_train_steps if args.max_train_steps and args.max_train_steps > 0 else None
        for batch in train_loader:
            x, y, _ = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

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

            running += float(loss.detach().cpu())
            steps += 1
            if steps % 50 == 0:
                print(f"[epoch {epoch}] step {steps}  loss {running/steps:.4f}")
            if steps % 25 == 0:
                log(f"epoch {epoch} step {steps} loss={running/steps:.4f} | {mem_snapshot(device)}", log_path)
            if max_steps and steps >= max_steps:
                log(f"epoch {epoch} reached max_train_steps={max_steps}", log_path)
                break

        epoch_loss = running / max(steps, 1)
        dt = time.time() - t0
        print(f"[epoch {epoch}] train_loss={epoch_loss:.4f}  ({dt:.1f}s)")

        # ---- validate (single pass) ----
        log(f"epoch {epoch} validating... | {mem_snapshot(device)}", log_path)
        eval_res = evaluate_prototypes(model, val_loader, device)
        print(f"[epoch {epoch}] val_auc={eval_res.auc:.4f}  val_ap={eval_res.ap:.4f}  "
              f"val_acc={eval_res.acc:.4f}  thr={eval_res.thr:.3f}  counts={eval_res.counts}")

        # Record & persist history (no plots here)
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_auc"].append(eval_res.auc)
        history["val_ap"].append(eval_res.ap)
        history["val_acc"].append(eval_res.acc)
        history["val_thr"].append(eval_res.thr)
        (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))

        # Checkpoint on AUC
        if eval_res.auc > best_auc:
            best_auc, best_epoch = eval_res.auc, epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict()}, best_path)
            print(f"[epoch {epoch}] ↑ best AUC {best_auc:.4f} — saved {best_path.name}")
        else:
            bad_epochs += 1
            if args.early_stop and bad_epochs >= args.patience:
                print(f"[early stop] no AUC improvement for {bad_epochs} epochs (patience={args.patience})")
                log(f"early stop at epoch {epoch} | {mem_snapshot(device)}", log_path)
                break

        # cleanup after val (release any transient graph buffers)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log(f"epoch {epoch} done val | {mem_snapshot(device)}", log_path)

    # ---------------- final report ----------------
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

    # --------- optional one-shot plots at end (safe) ---------
    if args.plots_at_end:
        with torch.no_grad():
            val_emb, val_lab = embed_dataset(model, val_loader, device,
                                             use_amp=(args.amp and device.type == "cuda"))
            val_emb = val_emb / (np.linalg.norm(val_emb, axis=1, keepdims=True) + 1e-9)
            protos_plot = compute_prototypes(val_emb, val_lab)
            val_scores  = score_by_prototypes(val_emb, protos_plot)
        plot_training_curves(history, out_dir / "training_curves.png")
        plot_val_curves(val_lab, val_scores, out_dir)
        # tidy
        del val_emb, val_lab, protos_plot, val_scores
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"[done] history → {out_dir/'training_history.json'}  best → {best_path if best_path.exists() else 'n/a'}")

if __name__ == "__main__":
    main()
