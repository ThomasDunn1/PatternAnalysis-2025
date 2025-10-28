# losses.py
# Triplet loss with in-batch semi-hard negative mining for L2-normalized embeddings.

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_dist(emb: torch.Tensor, metric: str = "euclidean") -> torch.Tensor:
    """
    Pairwise distance matrix for a batch of unit-normalized embeddings.
    Returns a [B, B] matrix of distances (smaller = closer).
    """
    if metric == "euclidean":
        # For unit-norm vectors: ||a-b||^2 = 2 - 2 * (a·b)
        sim = emb @ emb.t()                   # [-1..1]
        dist = 2 - 2 * sim                    # [0..4], 0 = identical
        return dist.clamp_min(0)
    elif metric == "cosine":
        sim = emb @ emb.t()
        dist = 1 - sim                        # [0..2]
        return dist.clamp_min(0)
    else:
        raise ValueError("metric must be 'euclidean' or 'cosine'")


class TripletLoss(nn.Module):
    """
    Margin triplet loss with semi-hard negative mining (within each batch).

    L = max(0, d(a,p) - d(a,n) + margin)
    """

    def __init__(self, margin: float = 0.3, metric: str = "euclidean"):
        super().__init__()
        self.margin = float(margin)
        self.metric = metric

    @torch.no_grad()
    def _mine_triplets(self, emb: torch.Tensor, labels: torch.Tensor):
        """
        Mining uses a non-grad copy of emb to select indices only.
        We DO NOT return distances from here, to avoid detaching the graph.
        """
        D = _pairwise_dist(emb, self.metric)  # detached because of no_grad
        B = emb.shape[0]
        eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        pos_mask = eq & (~torch.eye(B, dtype=torch.bool, device=labels.device))
        neg_mask = ~eq

        triplets = []
        for i in range(B):
            pos_idx = torch.where(pos_mask[i])[0]
            neg_idx = torch.where(neg_mask[i])[0]
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            d_ap = D[i, pos_idx]
            d_an = D[i, neg_idx]

            # hardest positive for anchor i
            p = pos_idx[torch.argmax(d_ap)]

            # semi-hard negatives band
            d_ap_min = d_ap.min()
            semi = neg_idx[(d_an > d_ap_min) & (d_an < d_ap_min + self.margin)]
            if len(semi) == 0:
                n = neg_idx[torch.argmin(d_an)]  # fallback: closest negative
            else:
                n = semi[torch.argmax(D[i, semi])]  # closest to violating the margin
            triplets.append((i, p.item(), n.item()))

        return triplets

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # normalize defensively (ok if already normalized)
        emb = F.normalize(embeddings, p=2, dim=1)

        # 1) mine on a detached view
        with torch.no_grad():
            triplets = self._mine_triplets(emb.detach(), labels)

        if len(triplets) == 0:
            # No valid triplets (e.g., batch is single-class). Return a zero that still backprops.
            return torch.zeros([], device=embeddings.device, requires_grad=True)

        # 2) recompute DIFFERENTIABLE distances
        D = _pairwise_dist(emb, self.metric)  # this has grad

        a = torch.tensor([t[0] for t in triplets], device=embeddings.device, dtype=torch.long)
        p = torch.tensor([t[1] for t in triplets], device=embeddings.device, dtype=torch.long)
        n = torch.tensor([t[2] for t in triplets], device=embeddings.device, dtype=torch.long)

        loss = F.relu(D[a, p] - D[a, n] + self.margin).mean()
        return loss