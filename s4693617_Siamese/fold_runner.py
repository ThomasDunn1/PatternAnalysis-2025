#!/usr/bin/env python
# fold_runner.py
# Launch 5 runs of train.py (folds 0..4), collect metrics from test_stats.txt, and write reports/folds.csv.

from __future__ import annotations
import argparse, subprocess, json, time
from pathlib import Path
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_script", type=str, default="train.py")
    ap.add_argument("--splits_dir", type=str, default="data/splits")
    ap.add_argument("--out_root", type=str, default="runs/folds")
    ap.add_argument("--folds", type=int, default=5)
    # common training knobs (override as you like)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--image_size", type=int, default=320)
    ap.add_argument("--batch_p", type=int, default=8)
    ap.add_argument("--batch_k", type=int, default=4)
    ap.add_argument("--val_batch", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--patience", type=int, default=3)
    return ap.parse_args()

def read_test_stats(path: Path) -> dict:
    d = {"val_auc": None, "val_ap": None, "val_acc": None, "thr": None, "best_epoch": None}
    if not path.exists():
        return d
    lines = path.read_text().strip().splitlines()
    for ln in lines:
        if ln.startswith("best_epoch="): d["best_epoch"] = int(ln.split("=",1)[1])
        elif ln.startswith("val_auc="):  d["val_auc"]  = float(ln.split("=",1)[1])
        elif ln.startswith("val_ap="):   d["val_ap"]   = float(ln.split("=",1)[1])
        elif ln.startswith("val_acc="):  d["val_acc"]  = float(ln.split("=",1)[1])
        elif ln.startswith("thr="):      d["thr"]      = float(ln.split("=",1)[1])
    return d

def main():
    args = parse_args()
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for fold in range(args.folds):
        run_dir = out_root / f"fold{fold}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python", args.train_script,
            "--fold", str(fold),
            "--splits_dir", args.splits_dir,
            "--out_dir", str(run_dir),
            "--epochs", str(args.epochs),
            "--image_size", str(args.image_size),
            "--batch_p", str(args.batch_p),
            "--batch_k", str(args.batch_k),
            "--val_batch", str(args.val_batch),
            "--num_workers", str(args.num_workers),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--embed_dim", str(args.embed_dim),
            "--margin", str(args.margin),
        ]
        if args.pretrained: cmd.append("--pretrained")
        if args.freeze_backbone: cmd.append("--freeze_backbone")
        if args.amp: cmd.append("--amp")
        if args.early_stop: cmd += ["--early_stop", "--patience", str(args.patience)]

        print(f"[runner] fold {fold}: {' '.join(cmd)}")
        t0 = time.time()
        subprocess.run(cmd, check=True)
        dt = time.time() - t0

        stats = read_test_stats(run_dir / "test_stats.txt")
        stats.update({"fold": fold, "secs": int(dt)})
        rows.append(stats)

    df = pd.DataFrame(rows).sort_values("fold")
    rep_dir = Path("reports"); rep_dir.mkdir(parents=True, exist_ok=True)
    out_csv = rep_dir / "folds.csv"
    df.to_csv(out_csv, index=False)
    print(f"[runner] wrote {out_csv.as_posix()}\n{df}")

if __name__ == "__main__":
    main()
