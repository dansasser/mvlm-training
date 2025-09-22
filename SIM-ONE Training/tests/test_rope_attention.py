"""Regression tests for the fused governance attention path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Skip the module entirely if torch is not available in the environment

torch = pytest.importorskip("torch")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prioritary_mvlm.config import PropheticSingularityState
from simone_transformer.rope_attention import EnhancedGovernanceAttention


def _build_prophetic_state(batch: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> PropheticSingularityState:
    """Create a deterministic prophetic state for testing."""

    torch.manual_seed(0)
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


def _run_attention(
    module: EnhancedGovernanceAttention,
    x: torch.Tensor,
    attention_mask: torch.Tensor,
    policy_guidance: torch.Tensor,
    memory_context: torch.Tensor,
    prophetic_state: PropheticSingularityState,
):
    module.eval()
    with torch.no_grad():
        return module(
            x,
            attention_mask=attention_mask,
            policy_guidance=policy_guidance,
            memory_context=memory_context,
            output_traces=True,
            prophetic_state=prophetic_state,
        )


def _assert_trace_close(ref_trace: dict, test_trace: dict) -> None:
    trace_tensors = [
        "tensor",
        "importance_scores",
        "importance_gate",
        "concept_activations",
        "attention_entropy",
    ]
    for key in trace_tensors:
        torch.testing.assert_close(test_trace[key], ref_trace[key], atol=1e-5, rtol=1e-5)

    for key in ["prophetic_envelope", "attention_patterns"]:
        tensor = ref_trace[key]
        other = test_trace[key]
        if tensor is None:
            assert other is None
        else:
            torch.testing.assert_close(other, tensor, atol=1e-5, rtol=1e-5)

    for key in ["kingdom_mean", "kingdom_std"]:
        torch.testing.assert_close(test_trace[key], ref_trace[key], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch,seq_len,hidden_dim,num_heads", [(2, 6, 64, 4)])
def test_fused_attention_matches_reference(batch, seq_len, hidden_dim, num_heads):
    torch.manual_seed(42)

    fused = EnhancedGovernanceAttention(
        hidden_dim,
        num_heads,
        dropout=0.1,
        max_seq_len=seq_len,
        enable_caching=False,
        use_fused_attention=True,
    )
    reference = EnhancedGovernanceAttention(
        hidden_dim,
        num_heads,
        dropout=0.1,
        max_seq_len=seq_len,
        enable_caching=False,
        use_fused_attention=False,
    )
    reference.load_state_dict(fused.state_dict())

    device = torch.device("cpu")
    dtype = torch.float32

    x = torch.randn(batch, seq_len, hidden_dim, device=device, dtype=dtype)
    attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    policy_guidance = torch.randn(batch, seq_len, hidden_dim, device=device, dtype=dtype)
    memory_context = torch.randn(batch, seq_len, hidden_dim, device=device, dtype=dtype)
    prophetic_state = _build_prophetic_state(batch, seq_len, device, dtype)

    fused_out, fused_gov = _run_attention(
        fused, x, attention_mask, policy_guidance, memory_context, prophetic_state
    )
    ref_out, ref_gov = _run_attention(
        reference, x, attention_mask, policy_guidance, memory_context, prophetic_state
    )

    torch.testing.assert_close(fused_out, ref_out, atol=1e-5, rtol=1e-5)

    for key in ["policy_logits", "policy_mask"]:
        torch.testing.assert_close(fused_gov[key], ref_gov[key], atol=1e-5, rtol=1e-5)

    for key in ["memory_weights", "memory_signals"]:
        torch.testing.assert_close(fused_gov[key], ref_gov[key], atol=1e-5, rtol=1e-5)

    torch.testing.assert_close(fused_gov["attn_weights"], ref_gov["attn_weights"], atol=1e-5, rtol=1e-5)
    _assert_trace_close(ref_gov["trace"], fused_gov["trace"])


def test_boolean_attention_mask_handling():
    torch.manual_seed(7)

    batch, seq_len, hidden_dim, num_heads = 1, 4, 32, 2
    module = EnhancedGovernanceAttention(
        hidden_dim,
        num_heads,
        dropout=0.0,
        max_seq_len=seq_len,
        enable_caching=False,
    )

    x = torch.randn(batch, seq_len, hidden_dim)
    base_mask = torch.tril(torch.ones(seq_len, seq_len))
    bool_mask = base_mask.bool()
    float_mask = base_mask.clone()
    policy_guidance = torch.zeros_like(x)
    memory_context = torch.zeros_like(x)
    prophetic_state = _build_prophetic_state(batch, seq_len, x.device, x.dtype)

    fused_out_bool, _ = _run_attention(
        module, x, bool_mask, policy_guidance, memory_context, prophetic_state
    )
    fused_out_float, _ = _run_attention(
        module, x, float_mask, policy_guidance, memory_context, prophetic_state
    )

    torch.testing.assert_close(fused_out_bool, fused_out_float, atol=1e-6, rtol=1e-6)
