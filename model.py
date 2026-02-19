"""
GPT-style autoregressive Transformer for SMILES generation.

Architecture:
    Token Embedding + Learned Positional Embedding
    → N × TransformerDecoderBlock (causal self-attention + FFN)
    → LayerNorm → Linear head (logits over vocabulary)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Building blocks
# ======================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, max_len: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask (upper-triangular = -inf)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len), diagonal=1).bool(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # (3, B, H, T, D)
        q, k, v = qkv.unbind(0)                   # each (B, H, T, D)

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, H, T, T)
        attn = attn.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.out_proj(out))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str):
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int,
        dropout: float, activation: str, max_len: int,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ======================================================================
# Full model
# ======================================================================

class SmilesTransformer(nn.Module):
    """
    Autoregressive (GPT-style) Transformer for SMILES generation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_len: int = 128,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, activation, max_len)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            input_ids: (B, T) token indices
            labels:    (B, T) target indices (-100 for ignore)

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, V)

        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        bos_idx: int,
        eos_idx: int,
        max_new_tokens: int = 120,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Autoregressively sample `batch_size` sequences.

        Returns:
            ids: (batch_size, seq_len) — variable length, padded with eos_idx
        """
        device = device or next(self.parameters()).device
        self.eval()

        ids = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # Truncate context to max_len - 1 (input is one shorter than full seq)
            context = ids[:, -(self.max_len - 1):]
            logits = self(context)["logits"][:, -1, :]  # (B, V)
            logits = logits / max(temperature, 1e-8)

            # Top-k
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, -1:]] = float("-inf")

            # Top-p (nucleus)
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Finished sequences keep producing eos
            next_id[finished] = eos_idx
            ids = torch.cat([ids, next_id], dim=1)
            finished = finished | (next_id.squeeze(-1) == eos_idx)

            if finished.all():
                break

        return ids

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, model_cfg: dict, tokenizer) -> "SmilesTransformer":
        """Instantiate from a config dict + tokenizer."""
        return cls(
            vocab_size=tokenizer.vocab_size,
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_layers"],
            d_ff=model_cfg["d_ff"],
            dropout=model_cfg["dropout"],
            activation=model_cfg.get("activation", "gelu"),
            max_len=tokenizer.max_len,
            pad_idx=0,
        )

    # ------------------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if not trainable_only or p.requires_grad
        )