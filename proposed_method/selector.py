"""
selector.py — Adaptive Point Selector.

Given per-point importance scores, selects M points from N using an
adaptive strategy that splits the budget M into:

    M_c = round(alpha * M)        — top-score "critical" points
    M_f = M - M_c                 — randomly sampled "fill" points
                                     from the remaining non-critical pool

This hybrid strategy guarantees coverage of geometrically important
regions while avoiding degenerate collapsed sets.

Class:
    AdaptiveSelector
"""

import torch
import torch.nn as nn
from torch import Tensor


class AdaptiveSelector(nn.Module):
    """Adaptive point selector combining score-based and random sampling.

    Args:
        M:         Target number of output points (simplified cloud size).
        alpha:     Fraction of M allocated to top-score critical points.
                   Default 0.7.
        threshold: Score threshold used to separate critical vs. non-critical
                   pool (informational / for analysis). Default 0.5.
    """

    def __init__(
        self,
        M:         int   = 1024,
        alpha:     float = 0.7,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.M         = M
        self.alpha     = alpha
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, P: Tensor, score: Tensor) -> Tensor:
        """Select M points from P based on importance scores.

        Args:
            P:     Input point cloud        (B, N, 3).
            score: Per-point importance     (B, N) in [0, 1].

        Returns:
            idx: Selected point indices (B, M) — LongTensor.
                 Use `index_points(P, idx)` in model.py to gather coords.
        """
        B, N, _ = P.shape
        M       = self.M
        device  = P.device

        M_c = round(self.alpha * M)   # critical (top-score) budget
        M_f = M - M_c                 # fill (random) budget

        idx_list = []

        for b in range(B):
            s = score[b]                                        # (N,)

            # Sort descending by score
            sorted_idx = torch.argsort(s, descending=True)     # (N,)

            # Critical pool: top M_c
            crit_idx = sorted_idx[:M_c]                        # (M_c,)

            # Fill pool: everything below top M_c, sample M_f randomly
            fill_pool = sorted_idx[M_c:]                       # (N - M_c,)
            if fill_pool.numel() >= M_f:
                perm     = torch.randperm(fill_pool.numel(), device=device)
                fill_idx = fill_pool[perm[:M_f]]               # (M_f,)
            else:
                # Edge case: not enough points → pad with repetition
                fill_idx = fill_pool.repeat(
                    (M_f // fill_pool.numel()) + 1
                )[:M_f]

            combined = torch.cat([crit_idx, fill_idx], dim=0)  # (M,)
            idx_list.append(combined)

        idx = torch.stack(idx_list, dim=0)   # (B, M)
        return idx
