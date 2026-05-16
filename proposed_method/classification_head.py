"""
classification_head.py

Geometry-Aware Relational Classification Head
for Simplified / Reconstructed Point Clouds

Input:
    P_recon : (B, M, 3)
    f_s     : (B, M, C)

Output:
    logits  : (B, num_classes)

Author:
    designed for point cloud simplification AE pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# KNN
# =========================================================

def knn(x, k):
    """
    x : (B, C, N)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)

    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]

    return idx


# =========================================================
# Graph Feature
# =========================================================

def get_graph_feature(x, k=16, idx=None):
    """
    x : (B, C, N)
    """

    batch_size = x.size(0)
    num_points = x.size(2)

    if idx is None:
        idx = knn(x, k=k)

    device = x.device

    idx_base = torch.arange(
        0,
        batch_size,
        device=device
    ).view(-1, 1, 1) * num_points

    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(
        batch_size,
        num_points,
        k,
        num_dims
    )

    x = x.view(batch_size, num_points, 1, num_dims)
    x = x.repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3)

    return feature.permute(0, 3, 1, 2).contiguous()


# =========================================================
# EdgeConv Block
# =========================================================

class EdgeConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()

        self.k = k

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        """
        x : (B, C, N)
        """

        x = get_graph_feature(x, k=self.k)

        x = self.conv(x)

        x = x.max(dim=-1)[0]

        return x


# =========================================================
# Cross Attention Fusion
# =========================================================

class CrossAttentionFusion(nn.Module):

    def __init__(self, dim, num_heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, recon_feat, selected_feat):
        """
        recon_feat   : (B, N, C)
        selected_feat: (B, N, C)
        """

        attn_out, _ = self.attn(
            query=recon_feat,
            key=selected_feat,
            value=selected_feat
        )

        out = self.norm(recon_feat + attn_out)

        return out


# =========================================================
# Residual MLP Head
# =========================================================

class ResidualMLP(nn.Module):

    def __init__(self, dim, dropout=0.3):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)

        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        identity = x

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)

        x = x + identity

        x = F.gelu(x)

        return x


# =========================================================
# Main Classification Head
# =========================================================

class GeometryAwareClassifier(nn.Module):

    def __init__(
        self,
        feature_dim=448,
        emb_dim=256,
        num_classes=40,
        k=16
    ):
        super().__init__()

        # -----------------------------------------
        # Point Encoding
        # -----------------------------------------

        self.input_proj = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )

        self.edgeconv1 = EdgeConvBlock(
            64,
            128,
            k=k
        )

        self.edgeconv2 = EdgeConvBlock(
            128,
            emb_dim,
            k=k
        )

        # -----------------------------------------
        # Selected Feature Projection
        # -----------------------------------------

        self.selected_proj = nn.Sequential(
            nn.Linear(feature_dim, emb_dim),
            nn.GELU()
        )

        # -----------------------------------------
        # Cross Attention
        # -----------------------------------------

        self.cross_attn = CrossAttentionFusion(
            dim=emb_dim,
            num_heads=4
        )

        # -----------------------------------------
        # Classifier
        # -----------------------------------------

        fusion_dim = emb_dim * 2

        self.mlp = ResidualMLP(fusion_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(256, num_classes)
        )

    def forward(self, p_recon, f_s):

        """
        p_recon : (B, M, 3)
        f_s     : (B, M, feature_dim)
        """

        # -----------------------------------------
        # Point Feature Encoding
        # -----------------------------------------

        x = p_recon.transpose(1, 2)

        x = self.input_proj(x)

        x = self.edgeconv1(x)

        x = self.edgeconv2(x)

        recon_feat = x.transpose(1, 2)

        # -----------------------------------------
        # Selected Feature Projection
        # -----------------------------------------

        selected_feat = self.selected_proj(f_s)

        # -----------------------------------------
        # Cross Attention Fusion
        # -----------------------------------------

        fused_feat = self.cross_attn(
            recon_feat,
            selected_feat
        )

        # -----------------------------------------
        # Dual Global Pooling
        # -----------------------------------------

        fused_feat_t = fused_feat.transpose(1, 2)

        max_pool = F.adaptive_max_pool1d(
            fused_feat_t,
            1
        ).squeeze(-1)

        avg_pool = F.adaptive_avg_pool1d(
            fused_feat_t,
            1
        ).squeeze(-1)

        global_feat = torch.cat(
            [max_pool, avg_pool],
            dim=1
        )

        # -----------------------------------------
        # Residual MLP
        # -----------------------------------------

        global_feat = self.mlp(global_feat)

        # -----------------------------------------
        # Classification
        # -----------------------------------------

        logits = self.classifier(global_feat)

        return logits
