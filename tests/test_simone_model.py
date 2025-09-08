import os
import sys
import torch

# Add path to import simone_transformer module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SIM-ONE Training'))
from simone_transformer.simone_model import SIMONEModel


def test_future_tokens_masked():
    vocab_size = 10
    seq_len = 5
    model = SIMONEModel(vocab_size, hidden_dim=16, num_heads=2, ff_dim=32, num_layers=2)
    model.eval()

    input_ids = torch.arange(seq_len).unsqueeze(0)
    _, outputs = model(input_ids)

    for attn_weights in outputs["attn_weights"]:
        # attention weights shape: [batch, seq_len, seq_len]
        upper = torch.triu(attn_weights[0], diagonal=1)
        # future positions should have zero attention
        assert torch.allclose(upper, torch.zeros_like(upper))
        # some attention to current or past positions should remain
        assert torch.any(torch.tril(attn_weights[0]) > 0)
