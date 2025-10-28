# predict.py
# Prototype-based inference for the Siamese pipeline.
# - Loads best.pt (SiameseEncoder) and config.json to recover model args
# - Builds class prototypes from a labeled "support" CSV
# - Scores any CSV (labels optional); optional TTA averages embeddings
#
# Example:
#   python predict.py \
#     --ckpt runs/step8_ckpt/best.pt \
#     --out_csv runs/step9_pred/pred_val_fold0.csv \
#     --support_csv data/splits/val_fold0.csv \
#     --pred_csv    data/splits/val_fold0.csv \
#     --tta

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import ISIC2020Dataset
from modules import SiameseEncoder
from utils import compute_prototypes, score_by_prototypes

from tqdm import tqdm
from torch import amp

# ---- helpers ----

def load_config_from_ckpt(ckpt_path: Path) -> dict:
    """Load config.json from the checkpoint directory."""
    cfg_path = ckpt_path.parent / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found next to ckpt: {cfg_path}")
    with open(cfg_path, "r") as f:
        return json.load(f)

def on_drvfs(path: str) -> bool:
    # WSL Windows mounts like /mnt/c, /mnt/d, /mnt/e
    return path.startswith("/mnt/")

@torch.no_grad()
# --- progress + AMP-enabled embedding (with TTA) ---
@torch.no_grad()
def embed_loader(model, loader: DataLoader, device: torch.device, tta: bool = False, desc: str = "embed"):
    model.eval()
    all_emb, all_lab = [], []

    def _tta(x: torch.Tensor) -> torch.Tensor:
        if not tta:                # single forward
            return model(x)
        outs = [
            model(x),
            model(torch.flip(x, dims=[-1])),                     # hflip
            model(torch.flip(x, dims=[-2])),                     # vflip
            model(torch.rot90(x, k=1, dims=(-2, -1))),           # rot90
            model(torch.rot90(x, k=2, dims=(-2, -1))),           # rot180
            model(torch.rot90(x, k=3, dims=(-2, -1))),           # rot270
        ]
        return torch.stack(outs, dim=0).mean(dim=0)

    use_amp = (device.type == "cuda")
    it = tqdm(loader, desc=desc, total=len(loader), leave=False, dynamic_ncols=True)
    for x, y, _ in it:
        x = x.to(device, non_blocking=True)
        if use_amp:
            with amp.autocast(device_type="cuda"):
                z = _tta(x)
        else:
            z = _tta(x)
        all_emb.append(z.detach().cpu().numpy())
        all_lab.append(y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y))
    return np.concatenate(all_emb, 0), np.concatenate(all_lab, 0)

def read_csv_keep_order(csv_path: Path) -> dict:
    """Read a split CSV and return {'paths': [...], 'labels': [... or None]} in file order."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    paths = df[cols.get("image_path", list(df.columns)[0])].astype(str).tolist()
    labels = df[cols["label"]].astype(int).tolist() if "label" in cols else None
    return {"paths": paths, "labels": labels}

def on_native_linux() -> bool:
    import platform
    return platform.system().lower() == "linux"

# --- change build_val_loader to enable perf knobs on native Linux ---
def build_val_loader(csv_path: Path, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    ds = ISIC2020Dataset(str(csv_path), image_size=image_size, mode="val")
    native = on_native_linux() and (not on_drvfs(os.path.abspath(str(csv_path))))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if native else 0,
        pin_memory=True if native else False,
        persistent_workers=True if (native and num_workers > 0) else False,
        prefetch_factor=2 if (native and num_workers > 0) else None,
    )

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--support_csv", required=True, type=str)
    ap.add_argument("--pred_csv", required=True, type=str)
    ap.add_argument("--out_csv", required=True, type=str)
    ap.add_argument("--image_size", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=256)        # bigger default on Linux
    ap.add_argument("--num_workers", type=int, default=4)         # use workers now
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--limit_support", type=int, default=0, help="cap support rows for quick smoke")
    ap.add_argument("--limit_pred", type=int, default=0, help="cap pred rows for quick smoke")
    return ap.parse_args()

def _cap_csv(csv_path: Path, limit: int) -> Path:
    if not limit or limit <= 0:
        return csv_path
    import pandas as pd
    df = pd.read_csv(csv_path).head(limit)
    tmp = csv_path.parent / f"__tmp_{csv_path.stem}_cap{limit}.csv"
    df.to_csv(tmp, index=False)
    return tmp

def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    cfg = load_config_from_ckpt(ckpt_path)

    # reconstruct model from config
    embed_dim = int(cfg.get("embed_dim", 256))
    pretrained = bool(cfg.get("pretrained", False))
    freeze_bb  = bool(cfg.get("freeze_backbone", False))
    image_size = int(args.image_size or cfg.get("image_size", 320))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseEncoder(embedding_dim=embed_dim, pretrained=pretrained, freeze_backbone=freeze_bb).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    support_csv = Path(args.support_csv)
    pred_csv    = Path(args.pred_csv)
    # apply optional caps (for smoke)
    support_csv = _cap_csv(support_csv, args.limit_support)
    pred_csv    = _cap_csv(pred_csv, args.limit_pred)

    # loaders (val-mode transforms)
    support_csv = Path(args.support_csv)
    pred_csv    = Path(args.pred_csv)
    supp_loader = build_val_loader(support_csv, image_size, args.batch_size, args.num_workers)
    pred_loader = build_val_loader(pred_csv,    image_size, args.batch_size, args.num_workers)

    # embed support and build prototypes
    supp_meta = read_csv_keep_order(support_csv)
    supp_emb, supp_lab = embed_loader(model, supp_loader, device, tta=args.tta, desc="support")
    # normalize embeddings
    supp_emb = supp_emb / (np.linalg.norm(supp_emb, axis=1, keepdims=True) + 1e-9)

    if supp_meta["labels"] is None:
        raise ValueError("support_csv must include labels to build class prototypes.")
    protos = compute_prototypes(supp_emb, np.asarray(supp_meta["labels"], dtype=int))

    # embed pred set and score
    pred_meta = read_csv_keep_order(pred_csv)
    pred_emb, pred_lab = embed_loader(model, pred_loader, device, tta=args.tta, desc="predict")
    pred_emb = pred_emb / (np.linalg.norm(pred_emb, axis=1, keepdims=True) + 1e-9)

    scores = score_by_prototypes(pred_emb, protos)          # larger ⇒ more melanoma-like
    preds  = (scores >= 0.0).astype(int)                    # default 0-threshold; caller can post-filter

    # if pred labels exist, compute simple metrics & best threshold (same as training)
    metrics_txt = None
    if pred_meta["labels"] is not None:
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, roc_curve
        y_true = np.asarray(pred_meta["labels"], dtype=int)
        auc = float(roc_auc_score(y_true, scores))
        ap  = float(average_precision_score(y_true, scores))
        # pick threshold by max balanced accuracy
        fpr, tpr, thr = roc_curve(y_true, scores)
        acc_bal = (tpr + (1 - fpr)) / 2.0
        j = int(np.argmax(acc_bal))
        thr_star = float(thr[j])
        y_pred = (scores >= thr_star).astype(int)
        acc = float(accuracy_score(y_true, y_pred))
        metrics_txt = f"val_auc={auc:.4f}\nval_ap={ap:.4f}\nval_acc@thr*={acc:.4f}\nthr*={thr_star:.6f}\n"

    # write predictions.csv
    import pandas as pd
    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "image_path": pred_meta["paths"],
        "score": scores.astype(float),
        "pred": preds.astype(int),
    }
    if pred_meta["labels"] is not None:
        data["label"] = pred_meta["labels"]
    pd.DataFrame(data).to_csv(out, index=False)
    print(f"[done] wrote predictions → {out.as_posix()}")
    if metrics_txt:
        (out.parent / "pred_metrics.txt").write_text(metrics_txt)
        print("[metrics]\n" + metrics_txt)

if __name__ == "__main__":
    # keep WSL safe
    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")
    main()
