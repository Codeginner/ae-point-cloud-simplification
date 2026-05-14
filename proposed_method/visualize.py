"""
visualize.py — Point cloud visualization utilities.

Visualisasi:
    - Original point cloud
    - Simplified point cloud
    - Reconstruction point cloud (optional)

Usage:
    from .visualize import visualize_point_clouds
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _to_numpy(pc: Tensor):
    """Convert tensor → numpy safely."""
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()
    return pc


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------


def visualize_point_clouds(
    original: Tensor,
    simplified: Tensor,
    reconstructed: Optional[Tensor] = None,
    save_path: Optional[str] = None,
    elev: int = 25,
    azim: int = 45,
) -> None:
    """
    Visualize original vs simplified point cloud.

    Args:
        original:      (N,3)
        simplified:    (M,3)
        reconstructed: (M,3), optional
        save_path:     Optional output image path
        elev:          Camera elevation
        azim:          Camera azimuth
    """

    original   = _to_numpy(original)
    simplified = _to_numpy(simplified)

    if reconstructed is not None:
        reconstructed = _to_numpy(reconstructed)

    # Figure layout
    ncols = 3 if reconstructed is not None else 2

    fig = plt.figure(figsize=(6 * ncols, 6))

    # ------------------------------------------------------------------
    # Original
    # ------------------------------------------------------------------

    ax1 = fig.add_subplot(1, ncols, 1, projection='3d')

    ax1.scatter(
        original[:, 0],
        original[:, 1],
        original[:, 2],
        s=2,
    )

    ax1.set_title("Original Point Cloud")
    ax1.view_init(elev=elev, azim=azim)

    # ------------------------------------------------------------------
    # Simplified
    # ------------------------------------------------------------------

    ax2 = fig.add_subplot(1, ncols, 2, projection='3d')

    ax2.scatter(
        simplified[:, 0],
        simplified[:, 1],
        simplified[:, 2],
        s=6,
    )

    ax2.set_title("Simplified Point Cloud")
    ax2.view_init(elev=elev, azim=azim)

    # ------------------------------------------------------------------
    # Reconstruction (optional)
    # ------------------------------------------------------------------

    if reconstructed is not None:

        ax3 = fig.add_subplot(1, ncols, 3, projection='3d')

        ax3.scatter(
            reconstructed[:, 0],
            reconstructed[:, 1],
            reconstructed[:, 2],
            s=6,
        )

        ax3.set_title("Reconstructed Point Cloud")
        ax3.view_init(elev=elev, azim=azim)

    # ------------------------------------------------------------------
    # Clean axes
    # ------------------------------------------------------------------

    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()