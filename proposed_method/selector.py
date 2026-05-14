"""
selector.py — Adaptive Geometry-Balanced Selector.

Implements Eq. (10)–(14) from the proposed method.

The selector partitions the point cloud into:
    • contour points (high NC score)
    • flat points    (low NC score)

Then independently selects:
    • top-M_c contour points
    • top-M_f flat points

using importance scores, and merges them into the final
simplified point cloud.

Class:
    AdaptiveSelector
"""

import torch
import torch.nn as nn
from torch import Tensor


class AdaptiveSelector(nn.Module):
    """
    Adaptive Geometry-Balanced Selector.

    Proposed-method version:
        • contour pool selection
        • flat pool selection
        • NC-guided balancing

    Args:
        M:
            Number of output simplified points.

        alpha:
            Fraction allocated to contour region.
            Default:
                alpha = 0.7

        threshold:
            NC threshold separating:
                contour vs flat

            Default:
                threshold = 0.5
    """

    def __init__(
        self,
        M: int = 1024,
        alpha: float = 0.7,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()

        self.M = M
        self.alpha = alpha
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    # ini bagian yang diubah
    def forward(
        self,
        P: Tensor,
        score: Tensor,
        nc_score: Tensor,
    ) -> Tensor:
        """
        Select M points using geometry-balanced selection.

        Args:
            P:
                Input point cloud
                shape = (B, N, 3)

            score:
                Importance score
                shape = (B, N)

            nc_score:
                Normalized NC score
                shape = (B, N)

        Returns:
            idx:
                Selected point indices
                shape = (B, M)
        """

        B, N, _ = P.shape
        M = self.M
        device = P.device

        # --------------------------------------------------------------
        # Eq. (11)
        # M_c = contour budget
        # M_f = flat budget
        # --------------------------------------------------------------

        # ini bagian yang diubah
        M_c = int(self.alpha * M)

        # ini bagian yang diubah
        M_f = M - M_c

        idx_list = []

        for b in range(B):

            # ----------------------------------------------------------
            # Current batch scores
            # ----------------------------------------------------------

            # ini bagian yang diubah
            s = score[b]

            # ini bagian yang diubah
            nc = nc_score[b]

            # ----------------------------------------------------------
            # Eq. (10)
            # Split contour vs flat
            # ----------------------------------------------------------

            # ini bagian yang diubah
            contour_mask = nc >= self.threshold

            # ini bagian yang diubah
            flat_mask = nc < self.threshold

            # ini bagian yang diubah
            contour_idx = torch.where(contour_mask)[0]

            # ini bagian yang diubah
            flat_idx = torch.where(flat_mask)[0]

            # ----------------------------------------------------------
            # Edge-case fallback
            # Prevent empty pools
            # ----------------------------------------------------------

            # ini bagian yang ditambahkan
            if contour_idx.numel() == 0:
                contour_idx = torch.arange(N, device=device)

            # ini bagian yang ditambahkan
            if flat_idx.numel() == 0:
                flat_idx = torch.arange(N, device=device)

            # ----------------------------------------------------------
            # Gather scores from each pool
            # ----------------------------------------------------------

            # ini bagian yang ditambahkan
            contour_scores = s[contour_idx]

            # ini bagian yang ditambahkan
            flat_scores = s[flat_idx]

            # ----------------------------------------------------------
            # Eq. (12)
            # Top-k contour selection
            # ----------------------------------------------------------

            # ini bagian yang diubah
            top_contour = contour_scores.topk(
                min(M_c, contour_scores.numel()),
                largest=True
            ).indices

            # ini bagian yang ditambahkan
            selected_contour = contour_idx[top_contour]

            # ----------------------------------------------------------
            # Eq. (13)
            # Top-k flat selection
            # ----------------------------------------------------------

            # ini bagian yang diubah
            top_flat = flat_scores.topk(
                min(M_f, flat_scores.numel()),
                largest=True
            ).indices

            # ini bagian yang ditambahkan
            selected_flat = flat_idx[top_flat]

            # ----------------------------------------------------------
            # Eq. (14)
            # Merge contour + flat
            # ----------------------------------------------------------

            # ini bagian yang diubah
            combined = torch.cat(
                [
                    selected_contour,
                    selected_flat
                ],
                dim=0
            )

            # ----------------------------------------------------------
            # Padding if selected points < M
            # ----------------------------------------------------------

            # ini bagian yang ditambahkan
            if combined.numel() < M:

                pad = combined.repeat(
                    (M // combined.numel()) + 1
                )[:M - combined.numel()]

                combined = torch.cat(
                    [combined, pad],
                    dim=0
                )

            idx_list.append(combined[:M])

        idx = torch.stack(idx_list, dim=0)

        return idx