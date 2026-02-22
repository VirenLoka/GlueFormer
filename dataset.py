"""
SMILES dataset with support for:
  - ZINC (auto-download via tdc)
  - Custom CSV / TXT files
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from tokenizer import SmilesTokenizer
import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# Core Dataset
# ======================================================================
class SmilesDataset(Dataset):
    """
    A PyTorch Dataset that lazily encodes SMILES sequences on access.

    Stores only raw strings in memory. Tokenization and padding happen
    in __getitem__, so shared memory usage scales with batch size, not
    dataset size.

    Each item is a dict with:
        input_ids : LongTensor [max_len-1]
        labels    : LongTensor [max_len-1]
    """

    def __init__(
        self,
        smiles_list: List[str],
        tokenizer: SmilesTokenizer,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        # Store only raw strings — no pre-encoding
        self.smiles_list: List[str] = smiles_list

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> dict:
        smi = self.smiles_list[idx]
        ids = self.tokenizer.encode(smi, add_special=True)
        ids = self.tokenizer.pad_sequence(ids)
        ids = torch.tensor(ids, dtype=torch.long)

        input_ids = ids[:-1].clone()
        labels = ids[1:].clone()
        labels[labels == self.tokenizer.pad_idx] = -100
        return {"input_ids": input_ids, "labels": labels}


# ======================================================================
# Data loading helpers
# ======================================================================

def _load_zinc(max_samples: Optional[int] = None) -> List[str]:
    df = pd.read_parquet('data/zinc_250k.parquet')
    smiles = df["smiles"].dropna().tolist()
    if max_samples is not None:
        smiles = smiles[:max_samples]
    logger.info(f"Loaded {len(smiles)} SMILES from ZINC")
    return smiles


def _load_custom(path: str, smiles_column: str = "smiles") -> List[str]:
    """Load SMILES from a CSV or a plain-text file (one SMILES per line)."""
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
        return df[smiles_column].dropna().tolist()
    else:
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def _validate_smiles(smiles_list: List[str]) -> List[str]:
    """Keep only RDKit-parseable SMILES."""
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        logger.warning("RDKit not installed — skipping SMILES validation filter.")
        return smiles_list

    valid = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid.append(Chem.MolToSmiles(mol))
    logger.info(
        f"SMILES validation: {len(valid)}/{len(smiles_list)} valid "
        f"({len(valid)/len(smiles_list)*100:.1f}%)"
    )
    return valid


# ======================================================================
# Public API
# ======================================================================

def build_dataloaders(
    cfg: dict,
    tokenizer: SmilesTokenizer,
) -> Tuple[DataLoader, DataLoader, SmilesTokenizer]:
    """
    Build train & val DataLoaders from the config dict.

    Returns (train_loader, val_loader, tokenizer).
    The tokenizer's vocab is built here.
    """
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"].lower()

    # ---- Load raw SMILES ----
    if name == "zinc":
        smiles = _load_zinc(max_samples=ds_cfg.get("max_samples"))
    elif name == "custom":
        train_path = ds_cfg.get("custom_train_path")
        assert train_path, "custom_train_path must be set for custom dataset"
        smiles = _load_custom(train_path, ds_cfg.get("smiles_column", "smiles"))
        if ds_cfg.get("max_samples"):
            smiles = smiles[: ds_cfg["max_samples"]]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # ---- Filter to valid SMILES ----
    smiles = _validate_smiles(smiles)

    # ---- Build tokenizer vocab ----
    tokenizer.build_vocab(smiles)
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")

    # ---- Create dataset (stores only raw strings) ----
    full_ds = SmilesDataset(smiles, tokenizer)

    # ---- Train / val split ----
    val_frac = ds_cfg.get("val_split", 0.1)
    val_size = int(len(full_ds) * val_frac)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["training"]["seed"]),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,   # avoids worker respawn overhead each epoch
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=True,
    )

    logger.info(f"Train samples: {train_size} | Val samples: {val_size}")
    return train_loader, val_loader, tokenizer