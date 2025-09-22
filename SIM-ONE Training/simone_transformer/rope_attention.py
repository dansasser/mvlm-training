566        ).mean(dim=1)  # [batch, seq_len]
567
568    trace_info = {
569        'tensor': trace_tensor,  # [batch, seq_len, hidden_dim]
570        'importance_scores': importance_scores.squeeze(-1),  # [batch, seq_len]
571        'importance_gate': importance_gate.squeeze(-1),  # [batch, seq_len]
572        'concept_activations': concept_activations,  # [batch, seq_len, 64]
573        'attention_entropy': attention_entropy,  # [batch, seq_len]
574        'attention_patterns': avg_attention  # [batch, num_heads, seq_len]
575    }
576
577    if prophetic_state is not None:
578        aligned_state = prophetic_state.align_to_length(seq_len).to(x.device, x.dtype)
579        envelope = aligned_state.compute_trace_envelope(seq_len)
580        summary = aligned_state.summary()
581        trace_info['prophetic_envelope'] = envelope
582        trace_info['kingdom_mean'] = summary['kingdom']['mean']
583        trace_info['kingdom_std'] = summary['kingdom']['std']
584
585    return trace_info
586
587
588def create_causal_mask(
589    seq_len: int,
590    device: torch.device,
591    kv_len: Optional[int] = None,
592    dtype: Optional[torch.dtype] = None
593) -> torch.Tensor:
594    """Create causal attention mask supporting cached KV (q_len x kv_len)."""
595    kv_len = kv_len or seq_len
596    dtype = dtype or torch.float32
597
598    if kv_len == seq_len:
599        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
600    else:
601        past = torch.ones(seq_len, kv_len - seq_len, device=device, dtype=dtype)
602        curr = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
603        mask = torch.cat([past, curr], dim=-1)
604
605    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, kv_len]
606
607
608if __name__ == "__main__":
609    # Test the enhanced attention mechanism
610    batch_size, seq_len, hidden_dim = 2, 128, 512
611    num_heads = 8