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
from pathlib import Path

import json
from dataclasses import asdict

@dataclass
class EvalResult:
    auc: float
    ap: float
    acc: float
    thr: float
    counts: Dict[int, int]


@torch.no_grad()
def embed_dataset(model, loader: DataLoader, device: torch.device, progress_every: int = 50):
    model.eval()
    embs, labs = [], []
    for i, (x, y, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        z = model(x)
        embs.append(z.detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[val] embedded {i+1} batches")
    import numpy as _np
    return _np.concatenate(embs, axis=0), _np.concatenate(labs, axis=0)


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
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    epochs = history.get("epoch", [])
    n = len(epochs)

    plt.figure(figsize=(6.5, 4.0), facecolor="white")

    def style_for_series():
        # With 1–2 epochs, show markers so it isn't an invisible tiny line.
        return "o-" if n <= 2 else "-"

    if "train_loss" in history and len(history["train_loss"]) > 0:
        plt.plot(epochs, history["train_loss"], style_for_series(), label="train loss")
    if "val_auc" in history and len(history["val_auc"]) > 0:
        plt.plot(epochs, history["val_auc"], style_for_series(), label="val AUC")
    if "val_ap" in history and len(history["val_ap"]) > 0:
        plt.plot(epochs, history["val_ap"], style_for_series(), label="val AP")

    plt.xlabel("epoch"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=160, facecolor="white"); plt.close()


def plot_val_curves(y_true, y_score, out_dir: Path, title_suffix: str = ""):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ROC with legend
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(5.0, 4.0), facecolor="white")
    plt.plot(fpr, tpr, lw=2, label=f"Model (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], ls="--", label="Chance")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC{title_suffix}")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "val_roc.png", dpi=160, facecolor="white"); plt.close()

    # PR with legend
    ap = average_precision_score(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(5.0, 4.0), facecolor="white")
    plt.plot(rec, prec, lw=2, label=f"Model (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR{title_suffix}")
    plt.legend(loc="lower left"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "val_pr.png", dpi=160, facecolor="white"); plt.close()


def save_config(cfg_obj, path):
    """Save a dataclass or dict to JSON."""
    path = Path(path)
    data = asdict(cfg_obj) if hasattr(cfg_obj, "__dataclass_fields__") else dict(cfg_obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_config(path) -> dict:
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)