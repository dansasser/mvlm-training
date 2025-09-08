import torch
import torch.nn as nn
import torch.nn.functional as F

class SIMONEBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Governance heads
        self.policy_gate = nn.Linear(hidden_dim, hidden_dim)
        self.logit_gate = nn.Linear(hidden_dim, hidden_dim)
        self.trace_head = nn.Linear(hidden_dim, hidden_dim)
        self.memory_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None, policy_mask=None):
        attn_out, attn_weights = self.attn(x, x, x, attn_mask=mask)
        if policy_mask is not None:
            gated_weights = attn_weights * policy_mask
            attn_out = torch.bmm(gated_weights, x)

        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        policy_logits = self.policy_gate(x)
        memory_signals = self.memory_head(x)
        trace = self.trace_head(x)
        gated_output = self.logit_gate(x)

        return gated_output, {
            "policy_logits": policy_logits,
            "memory_signals": memory_signals,
            "trace": trace,
            "attn_weights": attn_weights
        }


class SIMONEModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim=512,
        num_heads=8,
        ff_dim=2048,
        num_layers=6,
        max_seq_len=512,
        dropout=0.1,
        emb_dropout=0.1,
        block_dropout=0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.layers = nn.ModuleList([
            SIMONEBlock(hidden_dim, num_heads, ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.block_dropout = nn.Dropout(block_dropout)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Tie token embedding and LM head weights
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids, mask=None, policy_mask=None):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_dropout(x)

        policy_logits_all, memory_signals_all, trace_all = [], [], []

        for layer in self.layers:
            x, outputs = layer(x, mask=mask, policy_mask=policy_mask)
            x = self.block_dropout(x)
            policy_logits_all.append(outputs["policy_logits"])
            memory_signals_all.append(outputs["memory_signals"])
            trace_all.append(outputs["trace"])

        logits = self.lm_head(x)

        return logits, {
            "policy_logits": policy_logits_all,
            "memory_signals": memory_signals_all,
            "trace": trace_all
        }
