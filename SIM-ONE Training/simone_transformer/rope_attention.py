"""
RoPE-based Attention Mechanism with Enhanced Governance for SIM-ONE Transformer
Implements Rotary Position Embedding and improved governance integration
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from prioritary_mvlm.config import PropheticSingularityState
from .attention_cache import CachedAttentionMixin


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    Provides better position encoding than learned embeddings.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Cache for computed embeddings
        self._cached_embeddings = None
        self._cached_seq_len = 0
        self._cached_device: Optional[torch.device] = None
        self._cached_dtype: Optional[torch.dtype] = None

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch, seq_len, dim]
            seq_len: Sequence length
            
        Returns:
            cos, sin tensors for rotary embedding
        """
        total_len = offset + seq_len
        cache_miss = (
            self._cached_embeddings is None
            or total_len > self._cached_seq_len
            or self._cached_device != x.device
            or self._cached_dtype != x.dtype
        )

        if cache_miss:
            self._compute_embeddings(total_len, x.device, x.dtype)

        return (
            self._cached_embeddings['cos'][offset:offset + seq_len],
            self._cached_embeddings['sin'][offset:offset + seq_len]
        )

    def _compute_embeddings(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Precompute embeddings for efficiency."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device=device))

        # Create rotation matrices
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)

        self._cached_embeddings = {'cos': cos, 'sin': sin}
        self._cached_seq_len = seq_len
        self._cached_device = device
        self._cached_dtype = dtype


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to input tensor."""
    # Split last dimension in half
    x1, x2 = x.chunk(2, dim=-1)
    
    # Apply rotation
    return torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], dim=-1)


class EnhancedGovernanceAttention(CachedAttentionMixin, nn.Module):
    """
    Multi-head attention with RoPE and enhanced governance mechanisms.
    Features:
    - Rotary position embeddings
    - Policy-controlled attention patterns
    - Memory-aware attention weighting
    - Trace generation for interpretability
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        governance_strength: float = 0.1,
        enable_caching: bool = True,
        cache_size: int = 1000
    ):
        super().__init__(enable_caching=enable_caching, cache_size=cache_size)

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE (hidden_dim/num_heads)")
        self.scale = self.head_dim ** -0.5
        self.governance_strength = governance_strength
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # RoPE for position encoding
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        
        # OPTIMIZED: Shared governance backbone instead of individual components
        from .shared_governance import SharedGovernanceBackbone
        self.governance_backbone = SharedGovernanceBackbone(
            hidden_dim, governance_dim=hidden_dim//2, num_heads=num_heads
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _compute_combined_attention_bias(
        self, 
        prophetic_mask: Optional[torch.Tensor], 
        policy_mask: Optional[torch.Tensor], 
        memory_weights: Optional[torch.Tensor],
        aligned_state: Optional[PropheticSingularityState] = None
    ) -> torch.Tensor:
        """
        Pre-compute combined attention modifications for efficiency.
        Combines prophetic, policy, and memory biases in single operation.
        """
        combined_bias = None
        
        # Add prophetic mask contribution
        if prophetic_mask is not None:
            combined_bias = prophetic_mask * (self.governance_strength * 0.5)
        
        # Add policy mask contribution
        if policy_mask is not None:
            if combined_bias is None:
                combined_bias = policy_mask * self.governance_strength
            else:
                combined_bias = combined_bias + policy_mask * self.governance_strength
        
        # Add memory weights contribution (convert multiplicative to additive)
        if memory_weights is not None:
            memory_bias = torch.log(1 + memory_weights * self.governance_strength + 1e-8)
            if combined_bias is None:
                combined_bias = memory_bias
            else:
                combined_bias = combined_bias + memory_bias
        
        return combined_bias if combined_bias is not None else 0.0
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        policy_guidance: Optional[torch.Tensor] = None,
        memory_context: Optional[torch.Tensor] = None,
        output_traces: bool = True,
        prophetic_state: Optional[PropheticSingularityState] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            attention_mask: Causal mask [seq_len, seq_len] 
            policy_guidance: External policy guidance [batch, seq_len, hidden_dim]
            memory_context: Memory context from previous layers
            output_traces: Whether to generate interpretability traces
            
        Returns:
            output: Attention output [batch, seq_len, hidden_dim]
            governance_outputs: Dict containing policy, memory, and trace outputs
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Generate Q, K, V for current tokens
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        past_k = None
        past_v = None
        past_length = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k is not None and past_v is not None:
                past_length = past_k.size(-2)

        # Apply RoPE to Q and K with offset for cached tokens
        cos, sin = self.rope(x, seq_len, offset=past_length)
        rope_cos = cos.unsqueeze(0).unsqueeze(2)
        rope_sin = sin.unsqueeze(0).unsqueeze(2)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_k is not None and past_v is not None:
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present_key_value = (k, v) if use_cache else None

        q_len = q.size(-2)
        kv_len = k.size(-2)
        past_kv_len = max(kv_len - q_len, 0)

        # Compute base attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        aligned_state = None
        prophetic_mask = None
        if prophetic_state is not None:
            aligned_state = prophetic_state.align_to_length(seq_len).to(x.device, x.dtype)
            prophetic_mask = aligned_state.compute_policy_mask(self.num_heads, seq_len)
            decay_gate = aligned_state.compute_memory_decay(self.num_heads, seq_len).unsqueeze(-2)

        # OPTIMIZED: Apply governance mechanisms using shared backbone
        governance_outputs = self.governance_backbone(
            x,
            attention_scores=scores,
            attention_weights=None,  # Will be computed after
            attention_output=None,   # Will be computed after
            policy_guidance=policy_guidance,
            memory_context=memory_context,
            prophetic_state=aligned_state,
            output_traces=output_traces
        )
        
        # Extract components for attention modification
        policy_mask = governance_outputs.get('policy_mask')
        memory_weights = governance_outputs.get('memory_weights')

        def _ensure_attention_mask(mask: Optional[torch.Tensor]) -> torch.Tensor:
            if mask is None:
                mask = create_causal_mask(q_len, q.device, kv_len=kv_len)
            else:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                elif mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                elif mask.dim() != 4:
                    raise ValueError("attention_mask must have 2, 3, or 4 dimensions")

                mask = mask.to(device=q.device)

                if mask.shape[-2] != q_len:
                    if mask.shape[-2] > q_len:
                        mask = mask[..., -q_len:, :]
                    else:
                        raise ValueError("attention_mask has insufficient query length")

                if mask.shape[-1] != kv_len:
                    if mask.shape[-1] > kv_len:
                        mask = mask[..., -kv_len:]
                    else:
                        pad_size = kv_len - mask.shape[-1]
                        pad_shape = mask.shape[:-1] + (pad_size,)
                        pad = mask.new_ones(pad_shape)
                        mask = torch.cat([pad, mask], dim=-1)

                mask = mask[:, :1, :, :]
            if mask.size(0) != batch_size:
                mask = mask.expand(batch_size, -1, -1, -1)
            return mask

        def _pad_additive_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if mask is None or past_kv_len == 0:
                return mask
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() != 4:
                raise ValueError("governance masks must have 3 or 4 dimensions")

            mask = mask.to(device=q.device)

            if mask.shape[-2] != q_len:
                if mask.shape[-2] > q_len:
                    mask = mask[..., -q_len:, :]
                else:
                    raise ValueError("governance mask has insufficient query length")

            if mask.shape[-1] != kv_len:
                if mask.shape[-1] > kv_len:
                    mask = mask[..., -kv_len:]
                else:
                    pad_shape = mask.shape[:-1] + (past_kv_len,)
                    pad = mask.new_zeros(pad_shape)
                    mask = torch.cat([pad, mask], dim=-1)

            return mask

        def _pad_memory_weights(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if mask is None or past_kv_len == 0:
                return mask
            if mask.dim() != 4:
                raise ValueError("memory weights must have 4 dimensions")

            mask = mask.to(device=q.device)

            if mask.shape[-1] != kv_len:
                if mask.shape[-1] > kv_len:
                    mask = mask[..., -kv_len:]
                else:
                    pad_shape = mask.shape[:-1] + (past_kv_len,)
                    pad = mask.new_zeros(pad_shape)
                    mask = torch.cat([pad, mask], dim=-1)

            return mask

        attention_mask = _ensure_attention_mask(attention_mask)
        prophetic_mask = _pad_additive_mask(prophetic_mask)
        policy_mask = _pad_additive_mask(policy_mask)
        if policy_mask is not None:
            governance_outputs['policy_mask'] = policy_mask
        memory_weights = _pad_memory_weights(memory_weights)
        if memory_weights is not None:
            governance_outputs['memory_weights'] = memory_weights

        # OPTIMIZED: Try to get cached attention weights first
        cached_attn_weights = self._try_get_cached_attention(
            seq_len, self.num_heads, governance_outputs, attention_mask, aligned_state
        )
        
        if cached_attn_weights is not None:
            attn_weights = cached_attn_weights
        else:
            # Compute attention weights normally
            combined_bias = self._compute_combined_attention_bias(
                prophetic_mask, policy_mask, memory_weights, aligned_state
            )
            scores = scores + combined_bias
            
            # Apply causal mask
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float('-inf'))
            
            # Compute attention weights
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Cache the computed attention weights
            self._cache_attention_pattern(
                attn_weights, seq_len, self.num_heads, governance_outputs, 
                attention_mask, aligned_state
            )
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        out = self.out_proj(out)
        
        # Update trace generation with final attention weights and output (if traces requested)
        if output_traces and 'trace' in governance_outputs:
            # Update the trace with final attention information
            updated_trace = self.governance_backbone.trace_head(
                governance_outputs['shared_governance_features'],
                attention_weights=attn_weights,
                attention_output=out,
                prophetic_state=aligned_state
            )
            governance_outputs.update(updated_trace)

        governance_outputs['attn_weights'] = attn_weights
        if prophetic_mask is not None:
            governance_outputs['prophetic_mask'] = prophetic_mask

        if aligned_state is not None:
            governance_outputs['prophetic_decay'] = aligned_state.compute_memory_decay(self.num_heads, seq_len)

        return out, governance_outputs, present_key_value


class PolicyController(nn.Module):
    """
    Controls attention patterns based on learned policies.
    Helps guide the model's attention to relevant information.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Attention pattern controllers for each head
        self.pattern_controllers = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_heads)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        attention_scores: torch.Tensor,
        policy_guidance: Optional[torch.Tensor] = None,
        prophetic_state: Optional[PropheticSingularityState] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate policy-based attention modifications.
        
        Args:
            x: Input representations [batch, seq_len, hidden_dim]
            attention_scores: Current attention scores [batch, num_heads, seq_len, seq_len]
            policy_guidance: External policy guidance
            
        Returns:
            policy_logits: Policy predictions for governance loss
            policy_mask: Attention score modifications
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Generate policy representation
        if policy_guidance is not None:
            # Combine input with external guidance
            policy_input = x + policy_guidance
        else:
            policy_input = x
            
        policy_logits = self.policy_net(policy_input)

        aligned_state = None
        if prophetic_state is not None:
            aligned_state = prophetic_state.align_to_length(seq_len).to(x.device, x.dtype)
            policy_logits = policy_logits + aligned_state.kingdom_flow.unsqueeze(-1)
            time_gate = aligned_state.time_index.unsqueeze(-1)
            policy_logits = policy_logits * (1.0 + (time_gate - 0.5) * 0.2)
        
        # Generate head-specific attention modifications
        policy_masks = []
        for i, controller in enumerate(self.pattern_controllers):
            # Create policy mask for this head
            head_policy = controller(policy_logits).squeeze(-1)  # [batch, seq_len]
            
            # Convert to attention mask
            # Encourage attention to high-policy positions
            policy_mask = head_policy.unsqueeze(1) + head_policy.unsqueeze(2)  # [batch, seq_len, seq_len]
            policy_masks.append(policy_mask)
        
        # Stack for all heads
        policy_mask = torch.stack(policy_masks, dim=1)  # [batch, num_heads, seq_len, seq_len]

        if prophetic_state is not None and aligned_state is not None:
            additional_mask = aligned_state.compute_policy_mask(self.num_heads, seq_len)
            policy_mask = policy_mask + additional_mask

        return policy_logits, policy_mask


class MemoryManager(nn.Module):
    """
    Manages memory context and influences attention based on memory state.
    Helps maintain coherence across long sequences.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, memory_dim: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.memory_dim = memory_dim or hidden_dim
        
        # Memory processing networks
        self.memory_encoder = nn.Sequential(
            nn.Linear(hidden_dim, self.memory_dim),
            nn.ReLU(),
            nn.Linear(self.memory_dim, self.memory_dim)
        )
        
        self.context_integrator = nn.MultiheadAttention(
            self.memory_dim, num_heads=4, batch_first=True
        )
        
        self.memory_to_weights = nn.Linear(self.memory_dim, num_heads)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_scores: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None,
        prophetic_state: Optional[PropheticSingularityState] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Generate memory-based attention modifications.
        
        Args:
            x: Input representations [batch, seq_len, hidden_dim]
            attention_scores: Current attention scores
            memory_context: Previous memory state
            
        Returns:
            memory_weights: Attention weight modifications
            memory_signals: New memory signals for next layer
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Encode current input to memory space
        current_memory = self.memory_encoder(x)

        aligned_state = None
        if memory_context is not None:
            # Integrate with previous memory context
            integrated_memory, _ = self.context_integrator(
                current_memory, memory_context, memory_context
            )
        else:
            integrated_memory = current_memory

        if prophetic_state is not None:
            aligned_state = prophetic_state.align_to_length(seq_len).to(x.device, x.dtype)
            mercy_gate = 1.0 + (aligned_state.mercy - 0.5) * 0.3
            integrated_memory = integrated_memory * mercy_gate.unsqueeze(-1)

        # Generate attention weight modifications
        memory_weights = self.memory_to_weights(integrated_memory)  # [batch, seq_len, num_heads]
        memory_weights = memory_weights.transpose(1, 2).unsqueeze(-1)  # [batch, num_heads, 1, seq_len]

        # Memory weights influence how much attention each position should receive
        memory_weights = torch.tanh(memory_weights)

        if prophetic_state is not None and aligned_state is not None:
            decay = aligned_state.compute_memory_decay(self.num_heads, seq_len).unsqueeze(-2)
            memory_weights = memory_weights * decay

        return memory_weights, integrated_memory


class TraceGenerator(nn.Module):
    """
    Generates interpretability traces for understanding model behavior.
    Helps analyze what the model is focusing on and why.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Trace analysis networks
        self.attention_analyzer = nn.Linear(num_heads, hidden_dim)
        self.importance_scorer = nn.Linear(hidden_dim, 1)
        self.concept_dim = 64
        self.concept_detector = nn.Linear(hidden_dim, self.concept_dim)  # Detect 64 different concepts
        self.trace_projector = nn.Linear(hidden_dim + self.concept_dim + 1, hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_weights: torch.Tensor,
        attention_output: torch.Tensor,
        prophetic_state: Optional[PropheticSingularityState] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate interpretability traces.
        
        Args:
            x: Input representations [batch, seq_len, hidden_dim]
            attention_weights: Attention weights [batch, num_heads, seq_len, seq_len]
            attention_output: Attention output [batch, num_heads, seq_len, head_dim]
            
        Returns:
            trace_info: Dictionary of trace information
        """
        batch_size, seq_len = x.shape[:2]
        
        # Analyze attention patterns
        # Average attention weights across sequence dimension
        avg_attention = attention_weights.mean(dim=-1)  # [batch, num_heads, seq_len]
        
        # Convert attention patterns to hidden space
        attention_features = self.attention_analyzer(avg_attention.transpose(1, 2))
        
        # Score token importance
        importance_scores = self.importance_scorer(x + attention_features)
        importance_gate = torch.sigmoid(importance_scores)

        # Detect concepts being processed
        concept_activations = torch.sigmoid(self.concept_detector(attention_features))

        # Build a trace representation that captures salience and concept activations
        trace_features = torch.cat(
            [attention_features, importance_gate, concept_activations],
            dim=-1
        )
        trace_tensor = torch.tanh(self.trace_projector(trace_features))
        trace_tensor = trace_tensor * importance_gate + x

        # Compute attention entropy (measure of attention spread)
        attention_entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + 1e-8),
            dim=-1
        ).mean(dim=1)  # [batch, seq_len]

        trace_info = {
            'tensor': trace_tensor,  # [batch, seq_len, hidden_dim]
            'importance_scores': importance_scores.squeeze(-1),  # [batch, seq_len]
            'importance_gate': importance_gate.squeeze(-1),  # [batch, seq_len]
            'concept_activations': concept_activations,  # [batch, seq_len, 64]
            'attention_entropy': attention_entropy,  # [batch, seq_len]
            'attention_patterns': avg_attention  # [batch, num_heads, seq_len]
        }

        if prophetic_state is not None:
            aligned_state = prophetic_state.align_to_length(seq_len).to(x.device, x.dtype)
            envelope = aligned_state.compute_trace_envelope(seq_len)
            summary = aligned_state.summary()
            trace_info['prophetic_envelope'] = envelope
            trace_info['kingdom_mean'] = summary['kingdom']['mean']
            trace_info['kingdom_std'] = summary['kingdom']['std']

        return trace_info


def create_causal_mask(
    seq_len: int,
    device: torch.device,
    kv_len: Optional[int] = None
) -> torch.Tensor:
    """Create causal attention mask supporting cached key/value pairs."""
    kv_len = kv_len or seq_len
    if kv_len < seq_len:
        raise ValueError("kv_len must be at least as large as seq_len")

    current = torch.tril(torch.ones(seq_len, seq_len, device=device))
    if kv_len == seq_len:
        mask = current
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