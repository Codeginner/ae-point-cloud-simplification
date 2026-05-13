"""
scoring.py — Importance Scoring MLP.

Fuses per-point DGCNN features f_i ∈ R^448 with NC score s_i ∈ R^1
to produce a scalar importance score per point:

    concat(f_i, s_i) ∈ R^449 → MLP → score_i ∈ [0, 1]

Architecture: 449 → 128 → 64 → 1 → Sigmoid

Class:
    ImportanceScoringMLP
"""

import torch
import torch.nn as nn
from torch import Tensor


class ImportanceScoringMLP(nn.Module):
    """Per-point importance scorer.

    Takes concatenated DGCNN features and NC score, outputs a scalar
    importance value in [0, 1] for each point.

    Args:
        in_dim:    Input dimension (448 features + 1 NC score = 449).
        hidden_dims: Tuple of hidden layer widths. Default (128, 64).
        dropout:   Dropout probability applied after each hidden layer.
                   Default 0.1.
    """

    def __init__(
        self,
        in_dim:      int        = 1025,   # 1024 (emb_dims) + 1 (NC score)
        hidden_dims: tuple      = (128, 64),
        dropout:     float      = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim

        # Build MLP: in_dim → 128 → 64 → 1 → Sigmoid
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),      # applied after reshaping to (B*N, h)
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]

        self.layers = nn.Sequential(*layers)

    def forward(self, f_i: Tensor, s_i: Tensor) -> Tensor:
        """
        Args:
            f_i: Per-point DGCNN features  (B, N, 448).
            s_i: Per-point NC scores       (B, N)  — will be unsqueezed.

        Returns:
            score: Per-point importance scores (B, N) in [0, 1].
        """
        B, N, _ = f_i.shape

        # Concatenate along feature dimension
        s_i_expanded = s_i.unsqueeze(-1)              # (B, N, 1)
        x = torch.cat([f_i, s_i_expanded], dim=-1)   # (B, N, 449)

        # Flatten batch + point dims for Linear / BatchNorm
        x = x.reshape(B * N, -1)                     # (B*N, 449)
        x = self.layers(x)                            # (B*N, 1)
        score = x.reshape(B, N)                       # (B, N)
        return score
