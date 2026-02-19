#!/usr/bin/env python3
"""
train.py — Training loop for the SMILES Transformer.

Features:
  • Cosine-warmup learning rate schedule
  • Gradient accumulation & clipping
  • Periodic RDKit validity checks on sampled molecules
  • Weights & Biases logging
  • Checkpoint saving with keep-last-k

Usage:
    python train.py                              # uses config/default.yaml
    python train.py --config config/my_run.yaml  # custom config
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
from typing import Optional

import numpy as np
import torch
import yaml

from dataset import build_dataloaders
from model import SmilesTransformer
from tokenizer import SmilesTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ======================================================================
# Utilities
# ======================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    acc = cfg["device"]["accelerator"]
    if acc == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(acc)


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr: float):
    """Cosine schedule with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale between min_lr and base_lr
        base_lr = optimizer.defaults["lr"]
        target = min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine
        return target

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ======================================================================
# Chemistry validation
# ======================================================================

def check_validity(
    model: SmilesTransformer,
    tokenizer: SmilesTokenizer,
    device: torch.device,
    num_samples: int = 256,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
) -> dict:
    """Sample molecules and check RDKit validity + uniqueness."""
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        logger.warning("RDKit not installed — skipping validity check")
        return {}

    model.eval()
    ids = model.generate(
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

    n_valid = len(valid)
    n_unique = len(set(valid))
    metrics = {
        "chem/validity": n_valid / max(1, num_samples),
        "chem/uniqueness": n_unique / max(1, n_valid) if n_valid else 0.0,
        "chem/num_valid": n_valid,
        "chem/num_unique": n_unique,
    }
    # Log a few examples
    examples = valid[:8] if valid else smiles_list[:8]
    logger.info(
        f"Validity: {metrics['chem/validity']:.2%} | "
        f"Uniqueness: {metrics['chem/uniqueness']:.2%} | "
        f"Examples: {examples[:4]}"
    )
    return metrics


# ======================================================================
# Checkpoint management
# ======================================================================

def save_checkpoint(
    model: SmilesTransformer,
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
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": cfg,
        },
        ckpt_path,
    )
    tokenizer.save(checkpoint_dir / "tokenizer.json")
    logger.info(f"Saved checkpoint → {ckpt_path}")

    # Prune old checkpoints
    ckpts = sorted(checkpoint_dir.glob("checkpoint_epoch*.pt"))
    while len(ckpts) > keep_last_k:
        old = ckpts.pop(0)
        old.unlink()
        logger.info(f"Removed old checkpoint {old.name}")


# ======================================================================
# Main training loop
# ======================================================================

def train(cfg: dict) -> None:
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg)
    logger.info(f"Device: {device}")

    # ---- Precision ----
    precision = cfg["device"].get("precision", "fp32")
    use_amp = precision in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    # ---- Tokenizer ----
    tok_cfg = cfg["tokenizer"]
    tokenizer = SmilesTokenizer(
        pad_token=tok_cfg["pad_token"],
        bos_token=tok_cfg["bos_token"],
        eos_token=tok_cfg["eos_token"],
        unk_token=tok_cfg["unk_token"],
        max_len=tok_cfg["max_len"],
    )

    # ---- Data ----
    train_loader, val_loader, tokenizer = build_dataloaders(cfg, tokenizer)

    # ---- Model ----
    model = SmilesTransformer.from_config(cfg["model"], tokenizer).to(device)
    logger.info(f"Model parameters: {model.num_parameters():,}")

    # ---- Optimizer ----
    tcfg = cfg["training"]
    opt_cls = torch.optim.AdamW if tcfg["optimizer"] == "adamw" else torch.optim.Adam
    optimizer = opt_cls(
        model.parameters(),
        lr=tcfg["lr"],
        betas=tuple(tcfg["betas"]),
        eps=tcfg["eps"],
        weight_decay=tcfg.get("weight_decay", 0.0),
    )

    # ---- Scheduler ----
    total_steps = (
        len(train_loader) // tcfg.get("gradient_accumulation_steps", 1)
    ) * tcfg["epochs"]
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=cfg["training"]["scheduler"]["warmup_steps"],
        total_steps=total_steps,
        min_lr=cfg["training"]["scheduler"]["min_lr"],
    )

    # ---- AMP ----
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- WandB ----
    log_cfg = cfg["logging"]
    wandb_run = None
    if log_cfg.get("use_wandb"):
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
    val_cfg = cfg["validation"]
    ckpt_dir = Path(tcfg["checkpoint_dir"])

    # ---- Train ----
    global_step = 0
    accum_steps = tcfg.get("gradient_accumulation_steps", 1)
    best_val_loss = float("inf")

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype if use_amp else None):
                out = model(input_ids, labels=labels)
                loss = out["loss"] / accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), tcfg.get("max_grad_norm", 1.0)
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                epoch_loss += loss.item() * accum_steps

                # ---- Step-level logging ----
                if global_step % log_cfg.get("log_every_n_steps", 50) == 0:
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Epoch {epoch} | Step {global_step} | "
                        f"Loss {loss.item() * accum_steps:.4f} | LR {lr:.2e}"
                    )
                    if wandb_run:
                        wandb.log(
                            {"train/loss": loss.item() * accum_steps, "train/lr": lr},
                            step=global_step,
                        )

                # ---- Periodic validity check ----
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
                        wandb.log(chem_metrics, step=global_step)
                    model.train()

        # ---- Epoch-level validation ----
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                out = model(input_ids, labels=labels)
                val_loss += out["loss"].item()
                val_batches += 1
        val_loss /= max(1, val_batches)

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch}/{tcfg['epochs']} done in {elapsed:.1f}s | "
            f"Val Loss: {val_loss:.4f}"
        )
        if wandb_run:
            wandb.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)

        # ---- Checkpointing ----
        if epoch % tcfg.get("save_every_n_epochs", 5) == 0 or val_loss < best_val_loss:
            save_checkpoint(
                model, optimizer, scheduler, tokenizer,
                epoch, global_step, cfg, ckpt_dir,
                keep_last_k=tcfg.get("keep_last_k", 3),
            )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Also save as "best"
            best_path = ckpt_dir / "best.pt"
            shutil.copy2(
                ckpt_dir / f"checkpoint_epoch{epoch:04d}.pt", best_path
            )
            logger.info(f"New best model (val_loss={val_loss:.4f}) → {best_path}")

    logger.info("Training complete!")
    if wandb_run:
        wandb.finish()


# ======================================================================
# CLI
# ======================================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train SMILES Transformer")
    parser.add_argument(
        "--config", type=str, default='configs/train_config.yaml',
        help="Path to YAML config file",
    )
    # Allow overrides via CLI
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.max_samples is not None:
        cfg["dataset"]["max_samples"] = args.max_samples
    if args.no_wandb:
        cfg["logging"]["use_wandb"] = False

    train(cfg)


if __name__ == "__main__":
    main()