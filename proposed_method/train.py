"""
train.py — Training loop for PointCloudSimplifier.

Usage:
    python -m proposed_method.train --data_root /path/to/dataset
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .model import PointCloudSimplifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Placeholder dataset  — replace with your actual dataset class
# ---------------------------------------------------------------------------

class PointCloudDataset(Dataset):
    """Dummy dataset returning random point clouds.

    Replace with your actual ModelNet40 / ShapeNet / custom loader.
    """

    def __init__(self, n_samples: int = 1024, n_points: int = 2048) -> None:
        self.n_samples = n_samples
        self.n_points  = n_points

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Returns: (N, 3)  — single point cloud
        return torch.randn(self.n_points, 3)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:      PointCloudSimplifier,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    epoch:      int,
) -> dict[str, float]:
    """Run one full training epoch.

    Args:
        model:     The PointCloudSimplifier.
        loader:    DataLoader yielding (B, N, 3) tensors.
        optimizer: Torch optimiser.
        device:    Compute device.
        epoch:     Current epoch index (for logging).

    Returns:
        avg_losses: Dict of average loss values for this epoch.
    """
    model.train()
    totals: dict[str, float] = {"total": 0.0, "chamfer": 0.0, "normal": 0.0, "nc": 0.0}

    for step, batch in enumerate(loader):
        P: torch.Tensor = batch.to(device)      # (B, N, 3)

        optimizer.zero_grad()

        out   = model(P, compute_loss=True)
        loss  = out["loss"]

        total_loss = loss["total"].mean()
        total_loss.backward()
        optimizer.step()

        for k, v in loss.items():
            totals[k] += v.item()

        if step % 50 == 0:
            logger.info(
                f"Epoch {epoch}  step {step}/{len(loader)}  "
                f"loss={loss['total'].mean().item():.4f}  "
                f"cd={loss['chamfer'].mean().item():.4f}  "
                f"n={loss['normal'].mean().item():.4f}  "
                f"nc={loss['nc'].mean().item():.4f}"
            )

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:  PointCloudSimplifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate on validation set.

    Args:
        model:  The PointCloudSimplifier in eval mode.
        loader: Validation DataLoader.
        device: Compute device.

    Returns:
        avg_losses: Dict of average validation losses.
    """
    model.eval()
    totals: dict[str, float] = {"total": 0.0, "chamfer": 0.0, "normal": 0.0, "nc": 0.0}

    for batch in loader:
        P: torch.Tensor = batch.to(device)
        out  = model(P, compute_loss=True)
        loss = out["loss"]
        for k, v in loss.items():
            totals[k] += v.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PointCloudSimplifier")
    parser.add_argument("--data_root",    type=str,   default="./data")
    parser.add_argument("--M",            type=int,   default=1024,  help="Simplified cloud size")
    parser.add_argument("--k",            type=int,   default=20,    help="KNN neighbours")
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint",   type=str,   default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to checkpoint to resume training from")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    train_ds = PointCloudDataset()          # TODO: replace with real dataset
    val_ds   = PointCloudDataset(n_samples=256)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ── Model ─────────────────────────────────────────────────────────
    model = PointCloudSimplifier(M=args.M, k=args.k).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Pakai {torch.cuda.device_count()} GPU")
        model = torch.nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0

    # Resume from checkpoint
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    ckpt_dir = Path(args.checkpoint)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        train_losses = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_losses   = validate(model, val_loader, device)

        scheduler.step()

        logger.info(
            f"[Epoch {epoch}]  "
            f"train_loss={train_losses['total']:.4f}  "
            f"val_loss={val_losses['total']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Save latest checkpoint
        ckpt_path = ckpt_dir / "latest.pth"
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss":  val_losses,
        }, ckpt_path)

        # Save best checkpoint
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            logger.info(f"  → New best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
