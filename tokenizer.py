"""
Character-level SMILES tokenizer with support for multi-character tokens
like Br, Cl, @@, etc.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

# Regex that splits SMILES into chemically meaningful tokens.
# Handles: Br, Cl, @@, multi-digit ring-closures (%NN), and single chars.
SMILES_REGEX = re.compile(
    r"(\[[^\]]+\]"        # bracketed atoms  [nH], [C@@H], [O-]
    r"|Br|Cl"             # two-letter organic atoms
    r"|@@|@"              # chirality
    r"|%\d{2}"            # ring-closure  %10 … %99
    r"|[A-Z][a-z]?"       # remaining organic atoms  C, N, O, S, P, …
    r"|[^A-Za-z])"        # everything else: digits, (, ), =, #, +, -, etc.
)


class SmilesTokenizer:
    """Deterministic, regex-based SMILES tokenizer."""

    SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __init__(
        self,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        max_len: int = 128,
    ):
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.max_len = max_len

        # Will be populated by build_vocab
        self.token2idx: dict[str, int] = {}
        self.idx2token: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------
    def build_vocab(self, smiles_list: List[str]) -> None:
        """Build vocabulary from a list of SMILES strings."""
        token_set: set[str] = set()
        for smi in smiles_list:
            token_set.update(self.tokenize(smi))

        # Deterministic ordering: specials first, then sorted tokens
        tokens = list(self.SPECIAL_TOKENS) + sorted(token_set)
        self.token2idx = {t: i for i, t in enumerate(tokens)}
        self.idx2token = {i: t for t, i in self.token2idx.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token2idx)

    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.pad_token]

    @property
    def bos_idx(self) -> int:
        return self.token2idx[self.bos_token]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[self.eos_token]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.unk_token]

    # ------------------------------------------------------------------
    # Tokenise / encode / decode
    # ------------------------------------------------------------------
    @staticmethod
    def tokenize(smiles: str) -> List[str]:
        """Split a SMILES string into a list of tokens."""
        return SMILES_REGEX.findall(smiles)

    def encode(self, smiles: str, add_special: bool = True) -> List[int]:
        """SMILES string  →  list of token ids."""
        tokens = self.tokenize(smiles)
        ids = [self.token2idx.get(t, self.unk_idx) for t in tokens]
        if add_special:
            ids = [self.bos_idx] + ids + [self.eos_idx]
        # Truncate
        ids = ids[: self.max_len]
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        """List of token ids  →  SMILES string."""
        tokens = [self.idx2token.get(i, self.unk_token) for i in ids]
        if strip_special:
            tokens = [
                t for t in tokens if t not in self.SPECIAL_TOKENS
            ]
        return "".join(tokens)

    def pad_sequence(self, ids: List[int]) -> List[int]:
        """Pad (or truncate) to self.max_len."""
        ids = ids[: self.max_len]
        return ids + [self.pad_idx] * (self.max_len - len(ids))

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "token2idx": self.token2idx,
            "max_len": self.max_len,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SmilesTokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls(
            pad_token=data["pad_token"],
            bos_token=data["bos_token"],
            eos_token=data["eos_token"],
            unk_token=data["unk_token"],
            max_len=data["max_len"],
        )
        tok.token2idx = data["token2idx"]
        tok.idx2token = {int(i): t for t, i in tok.token2idx.items()}
        return tok