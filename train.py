#!/usr/bin/env python3
"""
train.py — Distributed Data Parallel training for the SMILES Transformer.

Launch with:
    torchrun --standalone --nproc_per_node=3 train.py
    torchrun --standalone --nproc_per_node=3 train.py --config config/my_run.yaml

DDP notes:
  • Each GPU runs an identical copy of this script as a separate process.
  • torchrun sets LOCAL_RANK / RANK / WORLD_SIZE env vars automatically.
  • Only rank 0 logs, saves checkpoints, and runs validity sampling.
  • Checkpoints save model.module.state_dict() — portable to single-GPU inference.
  • Training loss is all-reduced across ranks so the logged value is exact.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
import yaml
from tqdm import tqdm

from dataset import build_dataloaders
from model import SmilesTransformer
from tokenizer import SmilesTokenizer


# ======================================================================
# DDP helpers — call these before anything else
# ======================================================================

def init_distributed() -> tuple[int, int, int]:
    """
    Initialise the process group using env vars set by torchrun.
    Returns (rank, local_rank, world_size).
    """
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup() -> None:
    dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


# ======================================================================
# Logging — only rank 0 should emit anything
# ======================================================================

def setup_logging(rank: int) -> logging.Logger:
    logger = logging.getLogger("train")
    if is_main(rank):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        logging.disable(logging.CRITICAL)   # silence all non-rank-0 loggers
    return logger


# ======================================================================
# Utilities
# ======================================================================

def set_seed(seed: int, rank: int) -> None:
    """Each rank gets a different seed so data shuffling stays independent."""
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)


def get_device(local_rank: int) -> torch.device:
    return torch.device(f"cuda:{local_rank}")


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr: float):
    """Cosine schedule with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        base_lr = optimizer.defaults["lr"]
        return min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ======================================================================
# DDP-aware data loading
# ======================================================================

def build_distributed_dataloaders(cfg: dict, tokenizer: SmilesTokenizer, rank: int, world_size: int):
    """
    Wraps build_dataloaders() to replace standard samplers with
    DistributedSampler. Each rank sees a non-overlapping shard of the data.

    The val loader intentionally uses DistributedSampler too so that
    val loss is computed consistently across ranks and can be all-reduced.
    """
    # build_dataloaders builds vocab and returns loaders — we need the
    # underlying datasets, so we call it on rank 0 first to build the vocab,
    # broadcast the tokenizer, then all ranks build their own samplers.
    #
    # Simpler approach: call build_dataloaders on every rank (they all read
    # the same parquet file) but replace the DataLoader portion ourselves.

    train_loader, val_loader, tokenizer = build_dataloaders(cfg, tokenizer)

    # Extract the underlying datasets from the loaders
    train_ds = train_loader.dataset
    val_ds   = val_loader.dataset
    ds_cfg   = cfg["dataset"]
    tcfg     = cfg["training"]

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        sampler=train_sampler,          # DistributedSampler replaces shuffle=True
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        sampler=val_sampler,
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader, tokenizer, train_sampler


# ======================================================================
# Loss averaging across ranks
# ======================================================================

def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-reduce a scalar tensor and return the mean across all ranks."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size


# ======================================================================
# Chemistry validation — rank 0 only
# ======================================================================

def check_validity(
    model,                   # the raw (unwrapped) model, or DDP wrapper — both work
    tokenizer: SmilesTokenizer,
    device: torch.device,
    num_samples: int = 512,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> dict:
    """
    Sample molecules and compute RDKit validity + uniqueness.
    Called on rank 0 only — model.generate() runs on a single GPU.
    """
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return {}

    # Unwrap DDP if needed so .generate() works cleanly
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    with torch.no_grad():
        ids = raw_model.generate(
            bos_idx=tokenizer.bos_idx,
            eos_idx=tokenizer.eos_idx,
            max_new_tokens=tokenizer.max_len - 2,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            batch_size=num_samples,
            device=device,
        )

    smiles_list = [tokenizer.decode(seq.tolist()) for seq in ids]

    valid = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid.append(Chem.MolToSmiles(mol))

    n_valid  = len(valid)
    n_unique = len(set(valid))
    metrics  = {
        "chem/validity":   n_valid  / max(1, num_samples),
        "chem/uniqueness": n_unique / max(1, n_valid) if n_valid else 0.0,
        "chem/num_valid":  n_valid,
        "chem/num_unique": n_unique,
    }
    tqdm.write(
        f"  ↳ Validity: {metrics['chem/validity']:.2%} | "
        f"Uniqueness: {metrics['chem/uniqueness']:.2%} | "
        f"Examples: {valid[:4]}"
    )
    return metrics


# ======================================================================
# Checkpoint management — rank 0 only
# ======================================================================

def save_checkpoint(
    model: DDP,
    optimizer,
    scheduler,
    tokenizer: SmilesTokenizer,
    epoch: int,
    global_step: int,
    cfg: dict,
    checkpoint_dir: Path,
    keep_last_k: int = 3,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"checkpoint_epoch{epoch:04d}.pt"

    # Save model.module.state_dict() — the unwrapped weights.
    # This checkpoint loads cleanly on a single GPU without DDP.
    torch.save(
        {
            "epoch":                epoch,
            "global_step":          global_step,
            "model_state_dict":     model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config":               cfg,
        },
        ckpt_path,
    )
    tokenizer.save(checkpoint_dir / "tokenizer.json")
    tqdm.write(f"[rank 0] Saved checkpoint → {ckpt_path}")

    # Prune old checkpoints, keeping the last k
    ckpts = sorted(checkpoint_dir.glob("checkpoint_epoch*.pt"))
    while len(ckpts) > keep_last_k:
        old = ckpts.pop(0)
        old.unlink()
        tqdm.write(f"[rank 0] Removed old checkpoint {old.name}")


def load_checkpoint(
    path: Path,
    model: DDP,
    optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, int]:
    """
    Load a checkpoint saved by save_checkpoint().
    Returns (start_epoch, global_step).
    """
    ckpt = torch.load(path, map_location=device)
    model.module.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"] + 1, ckpt["global_step"]


# ======================================================================
# Main training loop
# ======================================================================

def train(cfg: dict) -> None:
    # ---- DDP init — must be first ----
    rank, local_rank, world_size = init_distributed()
    logger = setup_logging(rank)
    device = get_device(local_rank)

    logger.info(f"DDP ready: rank {rank}/{world_size} on {device}")

    set_seed(cfg["training"]["seed"], rank)

    # ---- Precision ----
    precision = cfg["device"].get("precision", "fp32")
    use_amp   = precision in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    # ---- Tokenizer ----
    tok_cfg   = cfg["tokenizer"]
    tokenizer = SmilesTokenizer(
        pad_token=tok_cfg["pad_token"],
        bos_token=tok_cfg["bos_token"],
        eos_token=tok_cfg["eos_token"],
        unk_token=tok_cfg["unk_token"],
        max_len=tok_cfg["max_len"],
    )

    # ---- Data ----
    # All ranks call this — each gets its own non-overlapping shard.
    train_loader, val_loader, tokenizer, train_sampler = build_distributed_dataloaders(
        cfg, tokenizer, rank, world_size
    )
    logger.info(
        f"Data: {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} val samples "
        f"(each rank sees 1/{world_size})"
    )

    # ---- Model ----
    model = SmilesTransformer.from_config(cfg["model"], tokenizer).to(device)

    # Sync BatchNorm across GPUs (no-op for standard LN but good practice)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap in DDP — gradients are automatically averaged across GPUs
    # find_unused_parameters=False is faster; set True only if your model
    # has conditional branches that leave some params unused some steps.
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    logger.info(f"Model parameters: {model.module.num_parameters():,}")

    # ---- Optimizer ----
    tcfg    = cfg["training"]
    opt_cls = torch.optim.AdamW if tcfg["optimizer"] == "adamw" else torch.optim.Adam
    optimizer = opt_cls(
        model.parameters(),
        lr=tcfg["lr"],
        betas=tuple(tcfg["betas"]),
        eps=tcfg["eps"],
        weight_decay=tcfg.get("weight_decay", 0.0),
    )

    # ---- Scheduler ----
    # total_steps is per-rank — DDP doesn't change the step count since
    # each rank processes its own shard, so one "step" still = one optimizer update.
    total_steps = (
        len(train_loader) // tcfg.get("gradient_accumulation_steps", 1)
    ) * tcfg["epochs"]

    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=tcfg["scheduler"]["warmup_steps"],
        total_steps=total_steps,
        min_lr=tcfg["scheduler"]["min_lr"],
    )

    # ---- AMP scaler ----
    # bf16 doesn't need a scaler (it has fp32 exponent range), but we still
    # create it for fp16 compatibility. enabled=False means it's a no-op.
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # ---- Resume from checkpoint ----
    start_epoch  = 1
    global_step  = 0
    ckpt_dir     = Path(tcfg["checkpoint_dir"])
    resume_path  = ckpt_dir / "best.pt"
    if resume_path.exists():
        logger.info(f"Resuming from {resume_path}")
        # All ranks must load the checkpoint — they all need the same weights
        start_epoch, global_step = load_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
        logger.info(f"Resumed at epoch {start_epoch}, step {global_step}")

    # ---- WandB — rank 0 only ----
    log_cfg   = cfg["logging"]
    wandb_run = None
    if is_main(rank) and log_cfg.get("use_wandb"):
        try:
            import wandb
            wandb_run = wandb.init(
                project=log_cfg.get("wandb_project", "smiles-transformer"),
                entity=log_cfg.get("wandb_entity"),
                name=log_cfg.get("wandb_run_name"),
                config=cfg,
            )
            logger.info(f"WandB run: {wandb_run.url}")
        except ImportError:
            logger.warning("wandb not installed — logging disabled")

    # ---- Validation config ----
    val_cfg    = cfg["validation"]
    accum_steps = tcfg.get("gradient_accumulation_steps", 1)
    best_val_loss = float("inf")

    # ---- Training loop ----
    for epoch in range(start_epoch, tcfg["epochs"] + 1):

        # DistributedSampler must know the epoch to re-shuffle differently
        # each epoch. Without this all ranks see the same order every epoch.
        train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss   = 0.0
        steps_in_epoch = 0
        t0 = time.time()

        train_pbar = (
            tqdm(train_loader, desc=f"Epoch {epoch}/{tcfg['epochs']} [Train]", leave=False)
            if is_main(rank) else train_loader
        )

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_pbar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels    = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                out  = model(input_ids, labels=labels)
                loss = out["loss"] / accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("max_grad_norm", 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                # All-reduce the loss so rank 0 logs the true mean across all GPUs.
                # detach() + clone() avoids holding the compute graph in the tensor.
                loss_tensor = (loss * accum_steps).detach().clone()
                reduce_tensor(loss_tensor, world_size)   # in-place mean
                reduced_loss = loss_tensor.item()

                epoch_loss     += reduced_loss
                steps_in_epoch += 1

                if is_main(rank):
                    lr = scheduler.get_last_lr()[0]
                    train_pbar.set_postfix(loss=f"{reduced_loss:.4f}", lr=f"{lr:.2e}")

                    if global_step % log_cfg.get("log_every_n_steps", 50) == 0:
                        tqdm.write(
                            f"Epoch {epoch} | Step {global_step} | "
                            f"Loss {reduced_loss:.4f} | LR {lr:.2e}"
                        )
                        if wandb_run:
                            import wandb
                            wandb.log(
                                {"train/loss": reduced_loss, "train/lr": lr},
                                step=global_step,
                            )

                    # ---- Validity check — rank 0 only ----
                    if (
                        val_cfg["validity_check_every_n_steps"] > 0
                        and global_step % val_cfg["validity_check_every_n_steps"] == 0
                    ):
                        chem_metrics = check_validity(
                            model, tokenizer, device,
                            num_samples=val_cfg["num_samples_for_validity"],
                            temperature=val_cfg["temperature"],
                            top_k=val_cfg["top_k"],
                            top_p=val_cfg["top_p"],
                        )
                        if wandb_run and chem_metrics:
                            import wandb
                            wandb.log(chem_metrics, step=global_step)

                        # Put model back in train mode after sampling
                        model.train()

        # ---- Epoch-level validation ----
        model.eval()
        val_loss    = torch.tensor(0.0, device=device)
        val_batches = torch.tensor(0,   device=device)

        val_pbar = (
            tqdm(val_loader, desc=f"Epoch {epoch}/{tcfg['epochs']} [Val]", leave=False)
            if is_main(rank) else val_loader
        )

        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels    = batch["labels"].to(device, non_blocking=True)
                with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                    out = model(input_ids, labels=labels)
                val_loss    += out["loss"].detach()
                val_batches += 1

        # Sum across all ranks, then divide — gives the true global val loss
        dist.all_reduce(val_loss,    op=dist.ReduceOp.SUM)
        dist.all_reduce(val_batches, op=dist.ReduceOp.SUM)
        val_loss = (val_loss / val_batches.clamp(min=1)).item()

        elapsed = time.time() - t0

        if is_main(rank):
            avg_train_loss = epoch_loss / max(1, steps_in_epoch)
            tqdm.write(
                f"Epoch {epoch}/{tcfg['epochs']} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Time: {elapsed:.1f}s"
            )
            if wandb_run:
                import wandb
                wandb.log(
                    {"val/loss": val_loss, "train/epoch_loss": avg_train_loss, "epoch": epoch},
                    step=global_step,
                )

        # ---- Checkpointing — rank 0 only ----
        # Barrier ensures all ranks finish validation before rank 0 saves.
        dist.barrier()

        if is_main(rank):
            if epoch % tcfg.get("save_every_n_epochs", 5) == 0 or val_loss < best_val_loss:
                save_checkpoint(
                    model, optimizer, scheduler, tokenizer,
                    epoch, global_step, cfg, ckpt_dir,
                    keep_last_k=tcfg.get("keep_last_k", 3),
                )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = ckpt_dir / "best.pt"
                shutil.copy2(ckpt_dir / f"checkpoint_epoch{epoch:04d}.pt", best_path)
                tqdm.write(f"New best model (val_loss={val_loss:.4f}) → {best_path}")

        # Barrier after checkpointing — all ranks stay in sync before
        # the next epoch's train_sampler.set_epoch() call.
        dist.barrier()

    # ---- Cleanup ----
    if is_main(rank):
        logger.info("Training complete!")
        if wandb_run:
            import wandb
            wandb.finish()

    cleanup()


# ======================================================================
# CLI
# ======================================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train SMILES Transformer (DDP)")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--max_samples", type=int,   default=None)
    parser.add_argument("--no_wandb",    action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs      is not None: cfg["training"]["epochs"]           = args.epochs
    if args.batch_size  is not None: cfg["training"]["batch_size"]       = args.batch_size
    if args.lr          is not None: cfg["training"]["lr"]               = args.lr
    if args.max_samples is not None: cfg["dataset"]["max_samples"]       = args.max_samples
    if args.no_wandb:                cfg["logging"]["use_wandb"]         = False

    train(cfg)


if __name__ == "__main__":
    main()