# utils.py
# Small helpers: embedding pass, prototype building, metrics, and plotting.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, accuracy_score
import matplotlib.pyplot as plt


@dataclass
class EvalResult:
    auc: float
    ap: float
    acc: float
    thr: float
    counts: Dict[int, int]


@torch.no_grad()
def embed_dataset(model, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Run model over loader and collect (embeddings, labels)."""
    model.eval()
    embs, labs = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        z = model(x)                      # [B, D] already L2-normalized
        embs.append(z.detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())
    return np.concatenate(embs, axis=0), np.concatenate(labs, axis=0)


def compute_prototypes(emb: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    """Compute unit-norm class prototypes from (emb, labels)."""
    protos = {}
    for c in np.unique(labels):
        m = emb[labels == c].mean(axis=0)
        m /= (np.linalg.norm(m) + 1e-9)
        protos[int(c)] = m
    return protos


def score_by_prototypes(emb: np.ndarray, protos: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Convert embeddings → scalar 'melanoma score'.
    We use (dist_to_neg - dist_to_pos): higher ⇒ closer to melanoma (class 1).
    """
    # cosine distance on unit-norm vectors: d = 1 - cos_sim
    p1 = protos.get(1)
    p0 = protos.get(0)
    if p1 is None or p0 is None:
        raise ValueError("Both class 0 and 1 prototypes are required for scoring.")
    # dot products
    s1 = emb @ p1
    s0 = emb @ p0
    d1 = 1.0 - s1
    d0 = 1.0 - s0
    return (d0 - d1)  # larger ⇒ more melanoma-like


def pick_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    Choose a reasonable threshold by maximizing balanced accuracy over the PR or ROC sweep.
    Returns (best_thr, best_acc).
    """
    fpr, tpr, thr_roc = roc_curve(y_true, y_score)
    acc = (tpr + (1 - fpr)) / 2.0
    j = int(np.argmax(acc))
    return float(thr_roc[j]), float(acc[j])


def evaluate_prototypes(
    model,
    val_loader: DataLoader,
    device: torch.device,
) -> EvalResult:
    """Embed val set, build prototypes (on val itself), score, and compute metrics."""
    emb, lab = embed_dataset(model, val_loader, device)
    # L2 normalize defensively
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    protos = compute_prototypes(emb, lab)
    score = score_by_prototypes(emb, protos)  # higher => melanoma

    auc = float(roc_auc_score(lab, score))
    ap  = float(average_precision_score(lab, score))

    thr, acc = pick_threshold(lab, score)
    pred = (score >= thr).astype(int)
    acc = float(accuracy_score(lab, pred))

    counts = {int(c): int((lab == c).sum()) for c in np.unique(lab)}
    return EvalResult(auc=auc, ap=ap, acc=acc, thr=thr, counts=counts)


def plot_training_curves(history: dict, out_png: Path):
    """Plot train/val curves from history dict."""
    out_png = Path(out_png)
    plt.figure(figsize=(6.5, 4.0))
    if "train_loss" in history:
        plt.plot(history["epoch"], history["train_loss"], label="train loss")
    if "val_auc" in history:
        plt.plot(history["epoch"], history["val_auc"], label="val AUC")
    if "val_ap" in history:
        plt.plot(history["epoch"], history["val_ap"], label="val AP")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_val_curves(y_true: np.ndarray, y_score: np.ndarray, out_dir: Path):
    """Save ROC and PR curves for the current evaluation."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(5.0, 4.0))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], ls="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.tight_layout(); plt.savefig(out_dir / "val_roc.png", dpi=160); plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(5.0, 4.0))
    plt.plot(rec, prec, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR")
    plt.tight_layout(); plt.savefig(out_dir / "val_pr.png", dpi=160); plt.close()
