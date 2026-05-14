"""
decoder.py — FoldingNet-based point cloud decoder.

Given:
    • simplified point cloud P_s
    • simplified point features f_s

the decoder:
    1. extracts a global latent vector z
    2. tiles z with a 2D grid
    3. applies two folding stages
    4. reconstructs the point cloud P_recon

Class:
    FoldingNetDecoder
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class FoldingNetDecoder(nn.Module):
    """
    Two-stage FoldingNet decoder.

    Proposed-method version:

        P_s + f_s
            ↓
        global latent vector z
            ↓
        2D grid folding
            ↓
        reconstructed point cloud

    Pipeline:
        1. MaxPool over simplified features
        2. Build 2D grid
        3. Folding stage 1
        4. Folding stage 2

    Args:
        M:
            Number of reconstruction points.

        in_dim:
            Simplified feature dimension.

            Proposed method:
                448

        latent_dim:
            Global latent vector dimension.

            Proposed method:
                1024
    """

    def __init__(
        self,

        M: int = 1024,

        # ini bagian yang diubah
        in_dim: int = 448,

        latent_dim: int = 1024,
    ) -> None:

        super().__init__()

        self.M = M
        self.latent_dim = latent_dim

        # --------------------------------------------------------------
        # Global feature encoder
        # (B,M,448) -> (B,M,1024)
        # --------------------------------------------------------------

        self.global_mlp = nn.Sequential(

            # ini bagian yang diubah
            nn.Linear(in_dim, latent_dim),

            nn.ReLU(inplace=True),
        )

        # --------------------------------------------------------------
        # 2D grid
        # --------------------------------------------------------------

        self.grid_size = math.ceil(math.sqrt(M))

        grid_dim = 2

        # --------------------------------------------------------------
        # Folding stages
        # --------------------------------------------------------------

        self.fold_layers = nn.ModuleList([

            # ----------------------------------------------------------
            # Fold Stage 1
            # concat(grid, z)
            # ----------------------------------------------------------

            nn.Sequential(

                nn.Conv1d(
                    grid_dim + latent_dim,
                    512,
                    1
                ),

                nn.ReLU(inplace=True),

                nn.Conv1d(
                    512,
                    512,
                    1
                ),

                nn.ReLU(inplace=True),

                nn.Conv1d(
                    512,
                    3,
                    1
                ),
            ),

            # ----------------------------------------------------------
            # Fold Stage 2
            # concat(fold1, z)
            # ----------------------------------------------------------

            nn.Sequential(

                nn.Conv1d(
                    3 + latent_dim,
                    512,
                    1
                ),

                nn.ReLU(inplace=True),

                nn.Conv1d(
                    512,
                    512,
                    1
                ),

                nn.ReLU(inplace=True),

                nn.Conv1d(
                    512,
                    3,
                    1
                ),
            ),
        ])

    # ------------------------------------------------------------------
    # Build 2D Grid
    # ------------------------------------------------------------------

    def _build_grid(
        self,
        B: int,
        device: torch.device,
    ) -> Tensor:
        """
        Build 2D regular grid.

        Returns:
            grid:
                shape = (B,2,M)
        """

        gs = self.grid_size

        lin = torch.linspace(
            -0.5,
            0.5,
            gs,
            device=device
        )

        gy, gx = torch.meshgrid(
            lin,
            lin,
            indexing="ij"
        )

        grid = torch.stack(
            [
                gx.flatten(),
                gy.flatten()
            ],
            dim=0
        )

        grid = grid[:, :self.M]

        grid = grid.unsqueeze(0).expand(
            B,
            -1,
            -1
        )

        return grid

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        P_s: Tensor,
        f_s: Tensor,
    ) -> Tensor:
        """
        Args:
            P_s:
                Simplified point cloud
                shape = (B,M,3)

            f_s:
                Simplified point features
                shape = (B,M,448)

        Returns:
            P_recon:
                Reconstructed point cloud
                shape = (B,M,3)
        """

        B, M, _ = f_s.shape

        device = f_s.device

        # --------------------------------------------------------------
        # 1. Global latent vector
        # Eq. (15)
        # --------------------------------------------------------------

        # ini bagian yang diubah
        z = self.global_mlp(f_s)

        # MaxPool:
        # (B,M,1024) -> (B,1024)

        # ini bagian yang diubah
        z = z.max(dim=1).values

        # --------------------------------------------------------------
        # 2. Tile latent vector
        # (B,1024) -> (B,1024,M)
        # --------------------------------------------------------------

        # ini bagian yang diubah
        z_tiled = z.unsqueeze(-1).expand(
            -1,
            -1,
            self.M
        )

        # --------------------------------------------------------------
        # 3. Build 2D grid
        # --------------------------------------------------------------

        grid = self._build_grid(
            B,
            device
        )

        # --------------------------------------------------------------
        # 4. Folding Stage 1
        # --------------------------------------------------------------

        fold1_input = torch.cat(
            [
                grid,
                z_tiled
            ],
            dim=1
        )

        fold1_out = self.fold_layers[0](
            fold1_input
        )

        # --------------------------------------------------------------
        # 5. Folding Stage 2
        # --------------------------------------------------------------

        fold2_input = torch.cat(
            [
                fold1_out,
                z_tiled
            ],
            dim=1
        )

        fold2_out = self.fold_layers[1](
            fold2_input
        )

        # --------------------------------------------------------------
        # Final reconstructed point cloud
        # --------------------------------------------------------------

        P_recon = fold2_out.permute(
            0,
            2,
            1
        )

        return P_recon