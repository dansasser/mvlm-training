#!/usr/bin/env python3
"""
Test suite for Phase 2 architectural optimizations.
Validates shared governance backbone, optimized MoE, and attention caching.
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simone_transformer.enhanced_model import EnhancedSIMONEModel
from simone_transformer.shared_governance import SharedGovernanceBackbone
from simone_transformer.modern_layers import MoELayer
from simone_transformer.attention_cache import AttentionPatternCache, CachedAttentionMixin
from simone_transformer.rope_attention import EnhancedGovernanceAttention
from prioritary_mvlm.config import PropheticSingularityState


def test_shared_governance_backbone():
    """Test the shared governance backbone functionality."""
    print("🧪 Testing Shared Governance Backbone...")
    
    batch_size, seq_len, hidden_dim = 2, 64, 512
    num_heads = 8
    
    # Create test input
    x = torch.randn(batch_size, seq_len, hidden_dim)
    attention_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
    attention_weights = F.softmax(attention_weights, dim=-1)
    
    # Create shared governance backbone
    backbone = SharedGovernanceBackbone(hidden_dim, governance_dim=256, num_heads=num_heads)
    
    # Test forward pass
    governance_outputs = backbone(
        x,
        attention_weights=attention_weights,
        output_traces=True
    )
    
    # Validate outputs
    assert 'policy_logits' in governance_outputs, "Missing policy_logits"
    assert 'policy_mask' in governance_outputs, "Missing policy_mask"
    assert 'memory_weights' in governance_outputs, "Missing memory_weights"
    assert 'memory_signals' in governance_outputs, "Missing memory_signals"
    assert 'trace' in governance_outputs, "Missing trace"
    assert 'shared_governance_features' in governance_outputs, "Missing shared_governance_features"
    
    # Check shapes
    assert governance_outputs['policy_logits'].shape == (batch_size, seq_len, 256), "Wrong policy_logits shape"
    assert governance_outputs['policy_mask'].shape == (batch_size, num_heads, seq_len, seq_len), "Wrong policy_mask shape"
    assert governance_outputs['memory_weights'].shape == (batch_size, num_heads, 1, seq_len), "Wrong memory_weights shape"
    assert governance_outputs['memory_signals'].shape == (batch_size, seq_len, 256), "Wrong memory_signals shape"
    
    print("✅ Shared Governance Backbone test passed!")


def test_optimized_moe():
    """Test the optimized MoE layer."""
    print("🧪 Testing Optimized MoE Layer...")
    
    batch_size, seq_len, dim = 2, 64, 512
    num_experts = 8
    
    # Create test input
    x = torch.randn(batch_size, seq_len, dim)
    
    # Create optimized MoE layer
    moe = MoELayer(
        dim=dim,
        num_experts=num_experts,
        num_experts_per_token=2,
        load_balancing_weight=0.01
    )
    
    # Test forward pass
    output = moe(x)
    
    # Validate output
    assert output.shape == x.shape, f"Wrong output shape: {output.shape} vs {x.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    
    # Test load balancing
    moe.train()
    _ = moe(x)  # Trigger load balancing update
    
    load_balance_loss = moe.get_load_balancing_loss()
    assert load_balance_loss.item() >= 0, "Load balancing loss should be non-negative"
    
    # Test statistics reset
    moe.reset_load_balancing_stats()
    assert moe.expert_usage.sum().item() == 0, "Expert usage should be reset"
    assert moe.total_tokens.item() == 0, "Total tokens should be reset"
    
    print("✅ Optimized MoE Layer test passed!")


def test_attention_cache():
    """Test the attention pattern caching system."""
    print("🧪 Testing Attention Pattern Cache...")
    
    cache = AttentionPatternCache(max_cache_size=10)
    
    # Create test data
    seq_len, num_heads = 64, 8
    pattern = torch.randn(2, num_heads, seq_len, seq_len)
    governance_outputs = {
        'policy_logits': torch.randn(2, seq_len, 512),
        'memory_signals': torch.randn(2, seq_len, 512)
    }
    
    # Test cache miss
    cached = cache.get_pattern(seq_len, num_heads, governance_outputs)
    assert cached is None, "Should be cache miss"
    
    # Store pattern
    cache.store_pattern(pattern, seq_len, num_heads, governance_outputs)
    
    # Test cache hit
    cached = cache.get_pattern(seq_len, num_heads, governance_outputs)
    assert cached is not None, "Should be cache hit"
    assert torch.allclose(cached, pattern, atol=1e-6), "Cached pattern should match"
    
    # Test statistics
    stats = cache.get_stats()
    assert stats['hits'] == 1, "Should have 1 hit"
    assert stats['misses'] == 1, "Should have 1 miss"
    assert stats['hit_rate'] == 0.5, "Hit rate should be 0.5"
    
    # Test cache eviction
    for i in range(15):  # Exceed cache size
        test_pattern = torch.randn(2, num_heads, seq_len, seq_len)
        test_governance = {
            'policy_logits': torch.randn(2, seq_len, 512) + i,  # Make unique
            'memory_signals': torch.randn(2, seq_len, 512) + i
        }
        cache.store_pattern(test_pattern, seq_len, num_heads, test_governance)
    
    assert len(cache.cache) <= cache.max_cache_size, "Cache size should not exceed maximum"
    assert cache.evictions > 0, "Should have evictions"
    
    print("✅ Attention Pattern Cache test passed!")


def test_cached_attention_integration():
    """Test attention caching integration with EnhancedGovernanceAttention."""
    print("🧪 Testing Cached Attention Integration...")
    
    batch_size, seq_len, hidden_dim = 2, 64, 512
    num_heads = 8
    
    # Create test input
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Create attention layer with caching enabled
    attention = EnhancedGovernanceAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        enable_caching=True,
        cache_size=100
    )
    
    # Create causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    # Test forward pass (should be cache miss)
    attention.eval()  # Enable caching (only works in eval mode)
    output1, governance1 = attention(x, attention_mask=mask)
    
    # Test second forward pass with same input (should be cache hit)
    output2, governance2 = attention(x, attention_mask=mask)
    
    # Outputs should be identical for cached computation
    assert torch.allclose(output1, output2, atol=1e-6), "Cached outputs should match"
    
    # Check cache statistics
    cache_stats = attention.get_cache_stats()
    assert cache_stats is not None, "Should have cache statistics"
    assert cache_stats['hits'] >= 1, "Should have at least 1 cache hit"
    
    # Test cache clearing
    attention.clear_attention_cache()
    cache_stats_after_clear = attention.get_cache_stats()
    assert cache_stats_after_clear['hits'] == 0, "Cache should be cleared"
    
    print("✅ Cached Attention Integration test passed!")


def test_enhanced_model_with_phase2():
    """Test the enhanced model with Phase 2 optimizations."""
    print("🧪 Testing Enhanced Model with Phase 2 Optimizations...")
    
    # Model configuration
    config = {
        'vocab_size': 1000,  # Smaller for testing
        'hidden_dim': 256,
        'num_heads': 4,
        'ff_dim': 1024,
        'num_layers': 2,  # Smaller for testing
        'max_seq_len': 128,
        'dropout': 0.1,
        'use_moe': True,  # Enable MoE
        'num_experts': 4
    }
    
    # Create model
    model = EnhancedSIMONEModel(**config)
    
    # Test input
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    # Create prophetic state
    prophetic_state = PropheticSingularityState.default(
        batch_size, seq_len
    )
    
    # Test forward pass
    model.eval()  # Enable caching
    logits, governance, _ = model(
        input_ids,
        prophetic_state=prophetic_state,
        output_governance=True
    )
    
    # Validate outputs
    assert logits.shape == (batch_size, seq_len, config['vocab_size']), "Wrong logits shape"
    assert governance is not None, "Should have governance outputs"
    assert 'policy_logits' in governance, "Missing policy outputs"
    assert 'memory_signals' in governance, "Missing memory outputs"
    assert 'trace' in governance, "Missing trace outputs"
    
    # Test generation
    prompt = torch.randint(0, config['vocab_size'], (1, 10))
    generated = model.generate(
        prompt, 
        max_length=20, 
        temperature=0.8,
        prophetic_state=prophetic_state
    )
    
    assert generated.shape[0] == 1, "Wrong batch size"
    assert generated.shape[1] >= 10, "Should generate at least input length"
    
    print("✅ Enhanced Model with Phase 2 test passed!")


def test_governance_preservation():
    """Test that governance functionality is preserved after optimizations."""
    print("🧪 Testing Governance Functionality Preservation...")
    
    batch_size, seq_len, hidden_dim = 2, 32, 256
    num_heads = 4
    
    # Create test inputs
    x = torch.randn(batch_size, seq_len, hidden_dim)
    prophetic_state = PropheticSingularityState.default(batch_size, seq_len)
    
    # Create attention layer
    attention = EnhancedGovernanceAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        enable_caching=True
    )
    
    # Test with prophetic state
    output, governance = attention(
        x, 
        prophetic_state=prophetic_state,
        output_traces=True
    )
    
    # Validate governance outputs
    assert 'policy_logits' in governance, "Missing policy outputs"
    assert 'memory_signals' in governance, "Missing memory outputs"
    assert 'trace' in governance, "Missing trace outputs"
    assert 'shared_governance_features' in governance, "Missing shared features"
    
    # Check prophetic state integration
    if 'prophetic_mask' in governance:
        assert governance['prophetic_mask'].shape[0] == batch_size, "Wrong prophetic mask batch size"
    
    # Test policy mask shape
    if 'policy_mask' in governance:
        policy_mask = governance['policy_mask']
        assert policy_mask.shape == (batch_size, num_heads, seq_len, seq_len), "Wrong policy mask shape"
    
    # Test memory weights shape
    if 'memory_weights' in governance:
        memory_weights = governance['memory_weights']
        assert memory_weights.shape == (batch_size, num_heads, 1, seq_len), "Wrong memory weights shape"
    
    # Test trace information
    if 'trace' in governance and isinstance(governance['trace'], dict):
        trace = governance['trace']
        assert 'tensor' in trace, "Missing trace tensor"
        assert 'importance_scores' in trace, "Missing importance scores"
        assert 'concept_activations' in trace, "Missing concept activations"
    
    print("✅ Governance Functionality Preservation test passed!")


def run_all_tests():
    """Run all Phase 2 optimization tests."""
    print("🚀 Running Phase 2 Optimization Tests")
    print("=" * 50)
    
    try:
        test_shared_governance_backbone()
        test_optimized_moe()
        test_attention_cache()
        test_cached_attention_integration()
        test_enhanced_model_with_phase2()
        test_governance_preservation()
        
        print("\n" + "=" * 50)
        print("🎉 All Phase 2 tests passed successfully!")
        print("✅ Shared governance backbone working correctly")
        print("✅ Optimized MoE routing functional")
        print("✅ Attention pattern caching operational")
        print("✅ Enhanced model integration successful")
        print("✅ Governance functionality preserved")
        print("\n🚀 Phase 2 optimizations are ready for benchmarking!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)