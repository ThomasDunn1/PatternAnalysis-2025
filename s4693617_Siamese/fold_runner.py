# fold_runner.py
# Run K folds, collect per-fold metrics, and append a final mean±std row.
#
# Example:
#   python fold_runner.py --pretrained --freeze_backbone --amp \
#     --epochs 8 --image_size 320 --batch_p 8 --batch_k 4 --val_batch 256 --num_workers 6 \
#     --scheduler onecycle --max_lr 1e-3 --backbone_lr_scale 0.25 --final_div_factor 1e3 \
#     --head_lr 1e-3 --backbone_lr 1e-4 --unfreeze_epoch 2 \
#     --log_root runs/folds_logs
#
from __future__ import annotations
import argparse, csv, json, math, re, subprocess, sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5, help="number of folds (0..folds-1)")
    ap.add_argument("--splits_dir", type=str, default="data/splits")

    # training knobs (mirrors train.py; pass through)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=320)
    ap.add_argument("--batch_p", type=int, default=8)
    ap.add_argument("--batch_k", type=int, default=4)
    ap.add_argument("--val_batch", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=6)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1337)

    # staged unfreeze
    ap.add_argument("--unfreeze_epoch", type=int, default=2)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--backbone_lr", type=float, default=1e-4)

    # scheduler
    ap.add_argument("--scheduler", type=str, default="onecycle", choices=["none","cosine","onecycle"])
    ap.add_argument("--warmup_epochs", type=int, default=0)
    ap.add_argument("--max_lr", type=float, default=1e-3)
    ap.add_argument("--final_div_factor", type=float, default=1e3)
    ap.add_argument("--backbone_lr_scale", type=float, default=0.25)

    # stability
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--patience", type=int, default=3)

    # logging / outputs
    ap.add_argument("--out_root", type=str, default="runs/folds")
    ap.add_argument("--log_root", type=str, default="runs/folds_logs")
    ap.add_argument("--reports_dir", type=str, default="reports")
    return ap.parse_args()

def parse_test_stats(path: Path) -> Dict[str, Any]:
    """Parse key metrics from test_stats.txt written by train.py."""
    stats = {
        "best_epoch": math.nan,
        "val_auc": math.nan,
        "val_ap": math.nan,
        "val_acc": math.nan,
        "thr": math.nan,
        "count_neg": math.nan,
        "count_pos": math.nan,
    }
    if not path.exists():
        return stats
    text = path.read_text().strip().splitlines()
    for line in text:
        line = line.strip()
        if line.startswith("best_epoch="):
            stats["best_epoch"] = int(line.split("=",1)[1])
        elif line.startswith("val_auc="):
            stats["val_auc"] = float(line.split("=",1)[1])
        elif line.startswith("val_ap="):
            stats["val_ap"]  = float(line.split("=",1)[1])
        elif line.startswith("val_acc="):
            stats["val_acc"] = float(line.split("=",1)[1])
        elif line.startswith("thr="):
            stats["thr"]     = float(line.split("=",1)[1])
        elif line.startswith("class_counts="):
            # e.g. class_counts={0: 1963, 1: 37}
            m0 = re.search(r"{\s*0:\s*([0-9]+)", line)
            m1 = re.search(r"1:\s*([0-9]+)", line)
            if m0: stats["count_neg"] = int(m0.group(1))
            if m1: stats["count_pos"] = int(m1.group(1))
    return stats

def mean_std_fmt(values: List[float]) -> str:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "nan±nan"
    return f"{arr.mean():.4f}±{arr.std(ddof=1):.4f}"  # sample std

def main():
    args = parse_args()
    splits_dir = Path(args.splits_dir)
    out_root   = Path(args.out_root)
    log_root   = Path(args.log_root)
    reports    = Path(args.reports_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    per_fold_rows: List[Dict[str, Any]] = []

    for fold in range(args.folds):
        out_dir = out_root / f"fold{fold}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_root / f"fold{fold}" / "train.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "train.py",
            "--fold", str(fold),
            "--splits_dir", str(splits_dir),
            "--out_dir", str(out_dir),
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
            "--seed", str(args.seed),
            "--unfreeze_epoch", str(args.unfreeze_epoch),
            "--head_lr", str(args.head_lr),
            "--backbone_lr", str(args.backbone_lr),
            "--scheduler", args.scheduler,
            "--warmup_epochs", str(args.warmup_epochs),
            "--max_lr", str(args.max_lr),
            "--final_div_factor", str(args.final_div_factor),
            "--backbone_lr_scale", str(args.backbone_lr_scale),
            "--grad_clip", str(args.grad_clip),
            "--patience", str(args.patience),
            "--log_file", str(log_file),
        ]
        if args.pretrained:      cmd.append("--pretrained")
        if args.freeze_backbone: cmd.append("--freeze_backbone")
        if args.amp:             cmd.append("--amp")
        if args.early_stop:      cmd.append("--early_stop")

        print(f"[runner] fold {fold}: {' '.join(cmd)}")
        # Stream output to console; logs also go to train.log via train.py itself.
        subprocess.run(cmd, check=True)

        # parse metrics
        stats_path = out_dir / "test_stats.txt"
        stats = parse_test_stats(stats_path)
        row = {
            "fold": fold,
            "best_epoch": stats["best_epoch"],
            "val_auc": stats["val_auc"],
            "val_ap": stats["val_ap"],
            "val_acc": stats["val_acc"],
            "thr": stats["thr"],
            "count_neg": stats["count_neg"],
            "count_pos": stats["count_pos"],
            "out_dir": str(out_dir),
        }
        per_fold_rows.append(row)
        print(f"[runner] fold {fold} metrics:", json.dumps(row, indent=2))

    # write folds.csv
    folds_csv = reports / "folds.csv"
    with open(folds_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fold","best_epoch","val_auc","val_ap","val_acc","thr","count_neg","count_pos","out_dir"]
        )
        writer.writeheader()
        for r in per_fold_rows:
            writer.writerow(r)

        # final mean±std row
        aucs = [r["val_auc"] for r in per_fold_rows]
        aps  = [r["val_ap"]  for r in per_fold_rows]
        accs = [r["val_acc"] for r in per_fold_rows]
        writer.writerow({
            "fold": "mean±std",
            "best_epoch": "",
            "val_auc": mean_std_fmt(aucs),
            "val_ap":  mean_std_fmt(aps),
            "val_acc": mean_std_fmt(accs),
            "thr": "",
            "count_neg": sum([r["count_neg"] for r in per_fold_rows if isinstance(r['count_neg'], (int,float)) and math.isfinite(r['count_neg'])]),
            "count_pos": sum([r["count_pos"] for r in per_fold_rows if isinstance(r['count_pos'], (int,float)) and math.isfinite(r['count_pos'])]),
            "out_dir": ""
        })

    # summary.txt (human readable)
    summary_txt = reports / "summary.txt"
    summary_txt.write_text(
        "\n".join([
            f"folds: 0..{args.folds-1}",
            f"AUC: {mean_std_fmt([r['val_auc'] for r in per_fold_rows])}",
            f"AP : {mean_std_fmt([r['val_ap']  for r in per_fold_rows])}",
            f"ACC: {mean_std_fmt([r['val_acc'] for r in per_fold_rows])}",
            f"Total counts: neg={sum([r['count_neg'] for r in per_fold_rows if isinstance(r['count_neg'], (int,float)) and math.isfinite(r['count_neg'])])}  "
            f"pos={sum([r['count_pos'] for r in per_fold_rows if isinstance(r['count_pos'], (int,float)) and math.isfinite(r['count_pos'])])}"
        ])
    )
    print(f"[runner] wrote {folds_csv} and {summary_txt}")

if __name__ == "__main__":
    main()
