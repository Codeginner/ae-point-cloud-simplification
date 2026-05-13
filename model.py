"""
model.py — PointCloudSimplifier: main orchestrator.

Full forward pipeline:

    P (B,N,3)
    ├── DGCNNEncoder          → f_i   (B, N, 448)
    ├── NCScoreModule         → s_i   (B, N)          [no learned params]
    ├── ImportanceScoringMLP  → score (B, N)
    ├── AdaptiveSelector      → idx   (B, M)
    │    └── gather P, f_i → P_s (B,M,3), f_s (B,M,448)
    ├── FoldingNetDecoder     → P_recon (B, M, 3)
    └── GeometryAwareLoss     → loss_dict

Class:
    PointCloudSimplifier
"""

import torch
import torch.nn as nn
from torch import Tensor

from .encoder  import DGCNNEncoder
from .nc_score import NCScoreModule
from .scoring  import ImportanceScoringMLP
from .selector import AdaptiveSelector
from .decoder  import FoldingNetDecoder
from .loss     import GeometryAwareLoss
from .utils    import index_points


class PointCloudSimplifier(nn.Module):
    """End-to-end point cloud simplification network.

    Args:
        M:         Number of output (simplified) points.
        k:         KNN for DGCNN and NC score. Default 20.
        alpha:     AdaptiveSelector critical-point fraction. Default 0.7.
        threshold: AdaptiveSelector score threshold. Default 0.5.
        latent_dim: FoldingNetDecoder latent dimension. Default 1024.
        lambda_1:  Chamfer loss weight.              Default 1.0.
        lambda_2:  Normal consistency loss weight.   Default 0.5.
        lambda_3:  NC score preservation loss weight. Default 0.3.
    """

    def __init__(
        self,
        M:          int   = 1024,
        k:          int   = 20,
        alpha:      float = 0.7,
        threshold:  float = 0.5,
        latent_dim: int   = 1024,
        lambda_1:   float = 1.0,
        lambda_2:   float = 0.5,
        lambda_3:   float = 0.3,
    ) -> None:
        super().__init__()
        self.M = M

        # --- Sub-modules ---
        self.encoder   = DGCNNEncoder(k=k)
        self.nc_module = NCScoreModule(k=k)

        self.scorer    = ImportanceScoringMLP(
            in_dim=1025,         # 1024 (emb_dims) + 1 (NC score)
        )

        self.selector  = AdaptiveSelector(
            M=M,
            alpha=alpha,
            threshold=threshold,
        )

        self.decoder   = FoldingNetDecoder(
            M=M,
            in_dim=1024,         # matches DGCNNEncoder emb_dims
            latent_dim=latent_dim,
        )

        self.loss_fn   = GeometryAwareLoss(
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        P: Tensor,
        compute_loss: bool = True,
    ) -> dict[str, Tensor]:
        """
        Args:
            P:            Input point cloud (B, N, 3).
            compute_loss: Whether to compute and return GeometryAwareLoss.
                          Set False during inference for speed. Default True.

        Returns:
            out: {
                'P_simplified': (B, M, 3)  — simplified point cloud,
                'P_recon':      (B, M, 3)  — reconstructed point cloud,
                'score':        (B, N)     — per-point importance scores,
                'idx':          (B, M)     — selected point indices,
                'loss':         loss_dict  — only if compute_loss=True,
            }
        """
        # 1. Per-point DGCNN features
        f_i = self.encoder(P)                          # (B, N, 448)

        # 2. NC scores (geometry-based, no grad needed here)
        with torch.no_grad():
            s_i = self.nc_module(P)                    # (B, N)

        # 3. Importance score fusion
        score = self.scorer(f_i, s_i)                  # (B, N)

        # 4. Adaptive selection
        idx = self.selector(P, score)                  # (B, M)

        # 5. Gather simplified point cloud and its features
        P_s = index_points(P, idx)                     # (B, M, 3)
        f_s = index_points(f_i, idx)                   # (B, M, 1024)

        # 6. Reconstruct via FoldingNet
        P_recon = self.decoder(P_s, f_s)               # (B, M, 3)

        out = {
            "P_simplified": P_s,
            "P_recon":      P_recon,
            "score":        score,
            "idx":          idx,
        }

        # 7. Geometry-aware loss
        if compute_loss:
            loss_dict = self.loss_fn(P_recon, P)
            out["loss"] = loss_dict

        return out
