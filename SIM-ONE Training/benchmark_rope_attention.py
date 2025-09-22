"""Benchmark fused vs reference governance attention implementations."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prioritary_mvlm.config import PropheticSingularityState
from simone_transformer.rope_attention import EnhancedGovernanceAttention


def _build_state(batch: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> PropheticSingularityState:
    torch.manual_seed(11)
    values = {
        name: torch.rand(batch, seq_len, device=device, dtype=dtype)
        for name in ["intensity", "anointing", "dominion", "mercy", "lambda_field"]
    }
    time_index = torch.linspace(0, 1, steps=seq_len, device=device, dtype=dtype)
    time_index = time_index.unsqueeze(0).expand(batch, -1)
    return PropheticSingularityState(
        intensity=values["intensity"],
        anointing=values["anointing"],
        dominion=values["dominion"],
        mercy=values["mercy"],
        lambda_field=values["lambda_field"],
        time_index=time_index,
        normalization={},
    )


def _build_inputs(batch: int, seq_len: int, hidden_dim: int, device: torch.device, dtype: torch.dtype):
    torch.manual_seed(123)
    x = torch.randn(batch, seq_len, hidden_dim, device=device, dtype=dtype)
    policy_guidance = torch.randn_like(x)
    memory_context = torch.randn_like(x)
    attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    prophetic_state = _build_state(batch, seq_len, device, dtype)
    return x, attention_mask, policy_guidance, memory_context, prophetic_state


def _time_module(module: EnhancedGovernanceAttention, *inputs, repeats: int, warmup: int) -> float:
    # Warmup
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            module(*inputs)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            module(*inputs)
    end = time.perf_counter()
    return (end - start) / repeats


def main() -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    batch, seq_len, hidden_dim, num_heads = 2, 128, 256, 8

    fused = EnhancedGovernanceAttention(
        hidden_dim,
        num_heads,
        dropout=0.1,
        max_seq_len=seq_len,
        enable_caching=False,
        use_fused_attention=True,
    ).to(device=device, dtype=dtype)

    reference = EnhancedGovernanceAttention(
        hidden_dim,
        num_heads,
        dropout=0.1,
        max_seq_len=seq_len,
        enable_caching=False,
        use_fused_attention=False,
    ).to(device=device, dtype=dtype)

    reference.load_state_dict(fused.state_dict())

    x, mask, policy, memory, prophetic_state = _build_inputs(batch, seq_len, hidden_dim, device, dtype)

    fused_inputs = (x, mask, policy, memory, False, prophetic_state)
    reference_inputs = fused_inputs

    with torch.no_grad():
        fused_out, fused_gov = fused(*fused_inputs)
        ref_out, ref_gov = reference(*reference_inputs)

    torch.testing.assert_close(fused_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused_gov["attn_weights"], ref_gov["attn_weights"], atol=1e-5, rtol=1e-5)

    repeats = 5
    warmup = 2
    fused_time = _time_module(fused, *fused_inputs, repeats=repeats, warmup=warmup)
    reference_time = _time_module(reference, *reference_inputs, repeats=repeats, warmup=warmup)

    print("Fused attention average time: {:.6f}s".format(fused_time))
    print("Reference attention average time: {:.6f}s".format(reference_time))
    speedup = reference_time / fused_time if fused_time > 0 else float("inf")
    print("Speedup: {:.2f}x".format(speedup))


if __name__ == "__main__":
    main()
