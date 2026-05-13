"""
decoder.py — FoldingNet-based point cloud decoder.

Given a simplified point cloud P_s, encodes a global latent vector z,
tiles it with a 2D grid, and applies two folding stages to reconstruct
a dense point cloud P_recon.

Class:
    FoldingNetDecoder
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class FoldingNetDecoder(nn.Module):
    """Two-stage FoldingNet decoder.

    Pipeline:
        1. Global feature:  MaxPool over P_s  → z ∈ R^{latent_dim}
        2. 2D grid:         Uniform grid √M × √M tiled with z
        3. Fold 1:          MLP([grid_2d || z]) → 3D surface approx
        4. Fold 2:          MLP([fold1_out || z]) → P_recon ∈ R^{M×3}

    Args:
        M:          Number of output reconstruction points.
        in_dim:     Feature dimension of the input simplified cloud P_s.
                    Should match DGCNNEncoder output (448). Default 448.
        latent_dim: Global feature / latent code dimension. Default 1024.
    """

    def __init__(
        self,
        M:          int = 1024,
        in_dim:     int = 448,
        latent_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.M          = M
        self.latent_dim = latent_dim

        # Global feature extractor: simple MLP on per-point features
        self.global_mlp = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.ReLU(inplace=True),
        )

        # 2D grid size: √M (will be rounded up; M points sampled from grid)
        self.grid_size = math.ceil(math.sqrt(M))

        grid_dim = 2  # (u, v)

        # Fold stage 1: concat(grid_2d, z) → intermediate 3D
        self.fold_layers = nn.ModuleList([
            nn.Sequential(                       # fold 1
                nn.Conv1d(grid_dim + latent_dim, 512, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 3, 1),
            ),
            nn.Sequential(                       # fold 2
                nn.Conv1d(3 + latent_dim, 512, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 3, 1),
            ),
        ])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_grid(self, B: int, device: torch.device) -> Tensor:
        """Build a 2D regular grid and flatten to (B, 2, M).

        Returns:
            grid: (B, 2, M)
        """
        gs  = self.grid_size
        lin = torch.linspace(-0.5, 0.5, gs, device=device)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")     # each (gs, gs)
        grid = torch.stack([gx.flatten(), gy.flatten()], dim=0)  # (2, gs²)
        grid = grid[:, : self.M]                              # (2, M)
        grid = grid.unsqueeze(0).expand(B, -1, -1)           # (B, 2, M)
        return grid

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, P_s: Tensor, f_s: Tensor) -> Tensor:
        """
        Args:
            P_s: Simplified point cloud        (B, M, 3)  [not used directly].
            f_s: Per-point features of P_s     (B, M, 448) from DGCNNEncoder.

        Returns:
            P_recon: Reconstructed point cloud (B, M, 3).
        """
        B, M, _ = f_s.shape
        device  = f_s.device

        # 1. Global feature via max-pooling over M points
        z = self.global_mlp(f_s)                           # (B, M, latent_dim)
        z = z.max(dim=1).values                            # (B, latent_dim)

        # 2. Tile z to match grid size: (B, latent_dim, M)
        z_tiled = z.unsqueeze(-1).expand(-1, -1, self.M)  # (B, latent_dim, M)

        # 3. 2D grid: (B, 2, M)
        grid = self._build_grid(B, device)                 # (B, 2, M)

        # 4. Fold stage 1
        fold1_input = torch.cat([grid, z_tiled], dim=1)   # (B, 2+latent, M)
        fold1_out   = self.fold_layers[0](fold1_input)     # (B, 3, M)

        # 5. Fold stage 2
        fold2_input = torch.cat([fold1_out, z_tiled], dim=1)  # (B, 3+latent, M)
        fold2_out   = self.fold_layers[1](fold2_input)         # (B, 3, M)

        P_recon = fold2_out.permute(0, 2, 1)               # (B, M, 3)
        return P_recon
