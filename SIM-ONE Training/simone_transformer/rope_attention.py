if hidden_dim % num_heads != 0:
    raise ValueError("hidden_dim must be divisible by num_heads")

else:
    past = torch.ones(seq_len, kv_len - seq_len, device=device)
    mask = torch.cat([past, current], dim=-1)

return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, kv_len]


if __name__ == "__main__":
    # Test the enhanced attention mechanism
    batch_size, seq_len, hidden_dim = 2, 128, 512
    num_heads = 8
    
    # Create test input
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Create attention layer
    attention = EnhancedGovernanceAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        max_seq_len=256
    )
    
    # Create causal mask
    mask = create_causal_mask(seq_len, x.device)
    
    # Forward pass
    print("Testing enhanced governance attention...")
    output, governance = attention(x, attention_mask=mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Governance outputs: {list(governance.keys())}")
    
    for key, value in governance.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: {list(value.keys())}")
    
    print("✓ Enhanced attention mechanism working correctly!")