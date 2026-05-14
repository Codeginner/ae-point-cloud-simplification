import sys
import torch
import numpy as np
import open3d as o3d

# ----------------------------------------------------------
# Add project path
# ----------------------------------------------------------

from .model import PointCloudSimplifier


# ----------------------------------------------------------
# Initialize model
# ----------------------------------------------------------

model = PointCloudSimplifier(
    M=512
)

model.eval()


# ----------------------------------------------------------
# Dummy point cloud
# Replace later with real dataset sample
# Shape = (1, N, 3)
# ----------------------------------------------------------

P = torch.randn(1, 2048, 3)


# ----------------------------------------------------------
# Run full pipeline
# ----------------------------------------------------------

with torch.no_grad():

    out = model(P)

    P_original = P[0].cpu().numpy()

    P_simplified = out["P_simplified"][0].cpu().numpy()

    P_recon = out["P_recon"][0].cpu().numpy()


# ----------------------------------------------------------
# Helper: numpy -> Open3D point cloud
# ----------------------------------------------------------

def to_o3d(points, color):

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.tile(
        np.array(color),
        (points.shape[0], 1)
    )

    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


# ----------------------------------------------------------
# Create point clouds
# ----------------------------------------------------------

# Original = blue
pcd_original = to_o3d(
    P_original,
    [0.2, 0.4, 1.0]
)

# Simplified = red
pcd_simplified = to_o3d(
    P_simplified,
    [1.0, 0.2, 0.2]
)

# Reconstructed = green
pcd_recon = to_o3d(
    P_recon,
    [0.2, 1.0, 0.2]
)


# ----------------------------------------------------------
# Shift clouds for side-by-side visualization
# ----------------------------------------------------------

pcd_original.translate((-2.5, 0, 0))

pcd_recon.translate((2.5, 0, 0))


# ----------------------------------------------------------
# Visualize
# ----------------------------------------------------------

o3d.visualization.draw_geometries(
    [
        pcd_original,
        pcd_simplified,
        pcd_recon
    ],
    window_name="Original | Simplified | Reconstructed",
    width=1400,
    height=700
)