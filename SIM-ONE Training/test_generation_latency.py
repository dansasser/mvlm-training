import time
from typing import List

import pytest
import torch

from simone_transformer.enhanced_model import EnhancedSIMONEModel


@pytest.mark.integration
@torch.no_grad()
def test_generation_latency_scales_linearly():
    """Generation should scale linearly with the number of new tokens."""
    torch.manual_seed(0)

    model = EnhancedSIMONEModel(
        vocab_size=512,
        hidden_dim=128,
        num_heads=4,
        ff_dim=512,
        num_layers=2,
        max_seq_len=128,
        dropout=0.0,
    )
    model.eval()

    prompt = torch.randint(0, model.vocab_size, (1, 8))

    additional_tokens: List[int] = [4, 8, 12]
    max_lengths = [prompt.shape[1] + n for n in additional_tokens]

    # Warmup to avoid including startup overhead in timings
    _ = model.generate(prompt.clone(), max_length=prompt.shape[1] + 1, do_sample=False)

    runtimes = []
    for max_len in max_lengths:
        measurements = []
        for _ in range(3):
            start = time.perf_counter()
            _ = model.generate(prompt.clone(), max_length=max_len, do_sample=False)
            measurements.append(time.perf_counter() - start)
        runtimes.append(sum(measurements) / len(measurements))

    # Ensure runtime increases with requested output length (allowing small jitter)
    for prev, curr in zip(runtimes, runtimes[1:]):
        assert curr >= prev * 0.9, "Generation runtime should not drop significantly for longer sequences"
    assert runtimes[-1] > runtimes[0], "Longest generation should take more time than shortest"

    per_token = [runtime / tokens for runtime, tokens in zip(runtimes, additional_tokens)]
    min_time = min(per_token)
    max_time = max(per_token)

    # Allow modest tolerance because very small models have small absolute runtimes
    assert max_time / min_time < 1.35, "Per-token generation time should stay approximately constant"
