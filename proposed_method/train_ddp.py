"""
train_ddp.py — DistributedDataParallel training + evaluation script.

Lebih efisien dari DataParallel karena setiap GPU punya proses sendiri
(tidak ada GIL bottleneck, gradient sync lebih cepat).

Usage — single node, 2 GPU (Kaggle T4 x2):
    torchrun --nproc_per_node=2 proposed_method/train_ddp.py [args]

    atau:
    python -m torch.distributed.run --nproc_per_node=2 proposed_method/train_ddp.py [args]

Usage — test / eval only (1 proses, tidak perlu torchrun):
    python proposed_method/train_ddp.py --mode test --resume ./checkpoints/best.pth

Args penting:
    --mode        train | test          (default: train)
    --data_root   path ke folder data
    --resume      path ke checkpoint .pth
    --n_points    jumlah point per sampel
    --M           jumlah simplified points
    --batch_size  per GPU (total = batch_size × n_gpu)
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Import dari package — support run sebagai script langsung maupun modul
if __package__:
    from .model   import PointCloudSimplifier
    from .train   import PointCloudDataset        # reuse dataset dari train.py
else:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from proposed_method.model import PointCloudSimplifier
    from proposed_method.train import PointCloudDataset


# ---------------------------------------------------------------------------
# Logging — hanya rank 0 yang print supaya tidak berantakan
# ---------------------------------------------------------------------------

def setup_logger(rank: int) -> logging.Logger:
    logger = logging.getLogger("train_ddp")
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [rank0] %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        logging.disable(logging.CRITICAL)   # silence non-rank-0
    return logger


# ---------------------------------------------------------------------------
# DDP setup / cleanup
# ---------------------------------------------------------------------------

def setup_ddp(rank: int, world_size: int) -> None:
    """Inisialisasi process group DDP."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend="nccl",          # nccl = best untuk GPU-GPU communication
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Build DataLoader dengan DistributedSampler
# ---------------------------------------------------------------------------

def build_loaders(args: argparse.Namespace, rank: int, world_size: int):
    train_ds = PointCloudDataset(
        data_root=args.data_root, mode='train',
        n_points=args.n_points,   augment=True,
    )
    val_ds = PointCloudDataset(
        data_root=args.data_root, mode='test',
        n_points=args.n_points,   augment=False,
    )

    # DistributedSampler: tiap GPU hanya lihat subset data-nya sendiri
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,      # per GPU
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# Reduce loss tensors across all ranks → rata-rata global
# ---------------------------------------------------------------------------

def reduce_dict(loss_dict: dict, world_size: int) -> dict:
    """All-reduce loss values dari semua GPU, return rata-rata."""
    reduced = {}
    for k, v in loss_dict.items():
        t = v.clone() if isinstance(v, torch.Tensor) else torch.tensor(v)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        reduced[k] = (t / world_size).item()
    return reduced


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:       DDP,
    loader:      DataLoader,
    sampler:     DistributedSampler,
    optimizer:   torch.optim.Optimizer,
    device:      torch.device,
    epoch:       int,
    rank:        int,
    world_size:  int,
    logger:      logging.Logger,
) -> dict[str, float]:
    model.train()
    sampler.set_epoch(epoch)     # penting: biar shuffle berbeda tiap epoch

    totals = {"total": 0.0, "chamfer": 0.0, "normal": 0.0, "nc": 0.0}

    for step, (P, _) in enumerate(loader):
        P = P.to(device, non_blocking=True)      # (B, N, 3)

        optimizer.zero_grad()

        out  = model(P, compute_loss=True)
        loss = out["loss"]

        loss["total"].backward()

        # Gradient clipping — mencegah exploding gradient
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate (nilai sudah di-sync otomatis oleh DDP via backward)
        for k, v in loss.items():
            totals[k] += v.item()

        if rank == 0 and step % 50 == 0:
            logger.info(
                f"Epoch {epoch}  step {step}/{len(loader)}  "
                f"loss={loss['total'].item():.4f}  "
                f"cd={loss['chamfer'].item():.4f}  "
                f"n={loss['normal'].item():.4f}  "
                f"nc={loss['nc'].item():.4f}"
            )

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:      DDP,
    loader:     DataLoader,
    device:     torch.device,
    world_size: int,
) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "chamfer": 0.0, "normal": 0.0, "nc": 0.0}

    for P, _ in loader:
        P    = P.to(device, non_blocking=True)
        out  = model(P, compute_loss=True)
        loss = out["loss"]
        for k, v in loss.items():
            totals[k] += v.item()

    # All-reduce agar semua rank punya angka yang sama
    n = len(loader)
    local_avgs = {k: torch.tensor(v / n, device=device) for k, v in totals.items()}
    reduced    = reduce_dict(local_avgs, world_size)
    return reduced


# ---------------------------------------------------------------------------
# Test / Evaluation only
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_test(args: argparse.Namespace) -> None:
    """Jalankan evaluasi tanpa DDP (single process)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] device={device}")

    val_ds = PointCloudDataset(
        data_root=args.data_root, mode='test',
        n_points=args.n_points,   augment=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    model = PointCloudSimplifier(M=args.M, k=args.k).to(device)

    assert args.resume is not None, "Test mode butuh --resume path/ke/checkpoint.pth"
    ckpt = torch.load(args.resume, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"[test] Loaded checkpoint dari {args.resume}")

    model.eval()
    totals = {"total": 0.0, "chamfer": 0.0, "normal": 0.0, "nc": 0.0}

    for P, _ in val_loader:
        P    = P.to(device)
        out  = model(P, compute_loss=True)
        loss = out["loss"]
        for k, v in loss.items():
            totals[k] += v.item()

    n = len(val_loader)
    print("\n=== TEST RESULTS ===")
    for k, v in totals.items():
        print(f"  {k:10s}: {v / n:.6f}")


# ---------------------------------------------------------------------------
# Main DDP worker (dipanggil oleh torchrun per GPU)
# ---------------------------------------------------------------------------

def ddp_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    logger = setup_logger(rank)

    if rank == 0:
        logger.info(f"DDP world_size={world_size}  batch_size per GPU={args.batch_size}"
                    f"  total effective batch={args.batch_size * world_size}")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, train_sampler = build_loaders(args, rank, world_size)

    if rank == 0:
        logger.info(f"Train: {len(train_loader.dataset)} samples | "
                    f"Val: {len(val_loader.dataset)} samples")

    # ── Model ─────────────────────────────────────────────────────────
    model = PointCloudSimplifier(M=args.M, k=args.k).to(device)

    # Sync BatchNorm: biar BN statistics di-aggregate dari semua GPU
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_loss = float("inf")

    # ── Resume ────────────────────────────────────────────────────────
    if args.resume is not None:
        # Load di rank 0, broadcast ke semua rank
        map_loc = {"cuda:0": f"cuda:{rank}"}
        ckpt    = torch.load(args.resume, map_location=map_loc)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if rank == 0:
            logger.info(f"Resumed dari epoch {start_epoch}")

    ckpt_dir = Path(args.checkpoint)
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        train_losses = train_one_epoch(
            model, train_loader, train_sampler,
            optimizer, device, epoch, rank, world_size, logger,
        )
        val_losses = validate(model, val_loader, device, world_size)

        scheduler.step()

        if rank == 0:
            logger.info(
                f"[Epoch {epoch}/{args.epochs}]  "
                f"train={train_losses['total']:.4f}  "
                f"val={val_losses['total']:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

            # Save checkpoint — hanya rank 0
            state_dict = model.module.state_dict()
            torch.save({
                "epoch":          epoch,
                "model":          state_dict,
                "optimizer":      optimizer.state_dict(),
                "scheduler":      scheduler.state_dict(),
                "val_loss":       val_losses,
                "best_val_loss":  best_val_loss,
                "args":           vars(args),
            }, ckpt_dir / "latest.pth")

            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                torch.save(state_dict, ckpt_dir / "best.pth")
                logger.info(f"  → New best val loss: {best_val_loss:.4f}")

        # Barrier: tunggu semua rank selesai sebelum epoch berikutnya
        dist.barrier()

    cleanup_ddp()


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP Training — PointCloudSimplifier")
    parser.add_argument("--mode",         type=str,   default="train",
                        choices=["train", "test"],
                        help="'train' untuk DDP training, 'test' untuk evaluasi single-GPU")
    parser.add_argument("--data_root",    type=str,   default="./data")
    parser.add_argument("--n_points",     type=int,   default=1024)
    parser.add_argument("--M",            type=int,   default=512,
                        help="Jumlah simplified output points")
    parser.add_argument("--k",            type=int,   default=20)
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--batch_size",   type=int,   default=16,
                        help="Batch size PER GPU")
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--checkpoint",   type=str,   default="./checkpoints")
    parser.add_argument("--resume",       type=str,   default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if args.mode == "test":
        # Test mode: single process, tidak perlu torchrun
        run_test(args)

    else:
        # Train mode: torchrun inject RANK, LOCAL_RANK, WORLD_SIZE otomatis
        rank       = int(os.environ.get("RANK",       0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Kalau WORLD_SIZE == 1 (tidak di-launch via torchrun), warning dulu
        if world_size == 1 and torch.cuda.device_count() > 1:
            print(
                "[WARNING] Terdeteksi >1 GPU tapi WORLD_SIZE=1.\n"
                "Jalankan via torchrun biar DDP aktif:\n"
                f"  torchrun --nproc_per_node={torch.cuda.device_count()} "
                f"proposed_method/train_ddp.py [args]"
            )

        ddp_worker(rank, world_size, args)