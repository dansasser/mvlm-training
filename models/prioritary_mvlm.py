from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class PrioritaryConfig:
    """Configuration for :class:`PrioritaryMVLM`."""

    vocab_size: int = 8000
    max_position_embeddings: int = 512
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    dropout: float = 0.1


class CustomAttention(nn.Module):
    """Multi-head self-attention with causal masking."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        bsz, seq_len, _ = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = attn_probs @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        return attn_output


class GatedFFN(nn.Module):
    """Feed-forward network with gating mechanism."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_size, intermediate_size)
        self.fc_gate = nn.Linear(hidden_size, intermediate_size)
        self.proj = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        hidden = F.relu(self.fc(x))
        gate = torch.sigmoid(self.fc_gate(x))
        hidden = hidden * gate
        hidden = self.dropout(hidden)
        hidden = self.proj(hidden)
        hidden = self.dropout(hidden)
        return hidden


class PrioritaryBlock(nn.Module):
    """Single transformer block used by :class:`PrioritaryMVLM`."""

    def __init__(self, config: PrioritaryConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = CustomAttention(
            config.hidden_size, config.num_attention_heads, config.dropout
        )
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.ffn = GatedFFN(
            config.hidden_size, config.intermediate_size, config.dropout
        )

    def forward(self, x: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        attn_out = self.attn(self.ln1(x), attention_mask)
        x = x + attn_out
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out
        return x


class PrioritaryMVLM(nn.Module):
    """Minimal transformer language model with custom components."""

    def __init__(self, config: PrioritaryConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_position_embeddings, config.hidden_size)
        )
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [PrioritaryBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        seq_len = input_ids.size(1)
        if seq_len > self.config.max_position_embeddings:
            raise ValueError("Sequence length exceeds model's maximum")

        pos_emb = self.pos_embed[:, :seq_len, :]
        x = self.embed_tokens(input_ids) + pos_emb
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
