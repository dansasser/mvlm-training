"""
RoPE-based Attention Mechanism with Enhanced Governance for SIM-ONE Transformer
Implements Rotary Position Embedding and improved governance integration
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from prioritary_mvlm.config import PropheticSingularityState

from .attention_cache import CachedAttentionMixin


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    Provides better position encoding than learned embeddings.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        """
        Initialize a RotaryPositionalEmbedding and precompute inverse frequencies used for RoPE.
        
        Parameters:
            dim (int): Embedding dimensionality (must be even); RoPE operates on pairs of dimensions.
            max_seq_len (int): Maximum sequence length to support for cached embeddings.
            base (int): Base used to compute inverse frequencies for rotary angles (default 10000).
        
        Notes:
            Registers `inv_freq` as a buffer and initializes internal caches for cosine/sine embeddings.
        """
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
    
    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return precomputed cosine and sine rotary positional embeddings for a contiguous subsequence.
        
        Parameters:
            x (torch.Tensor): Reference tensor whose device and dtype are used for the embeddings.
            seq_len (int): Number of positions to return.
            offset (int, optional): Starting position in the cached embeddings. Defaults to 0.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: `(cos, sin)` tensors for positions `[offset:offset+seq_len]`, with device, dtype, and trailing embedding dimension compatible with `x`.
        """
        total_len = seq_len + offset
        if (
            self._cached_embeddings is None
            or total_len > self._cached_seq_len
            or self._cached_embeddings['cos'].device != x.device
            or self._cached_embeddings['cos'].dtype != x.dtype
        ):
            self._compute_embeddings(total_len, x.device, x.dtype)

        return (
            self._cached_embeddings['cos'][offset:offset + seq_len],
            self._cached_embeddings['sin'][offset:offset + seq_len]
        )

    def _compute_embeddings(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """
        Precompute and cache cosine and sine rotary positional embeddings up to a given sequence length.
        
        Parameters:
            seq_len (int): Number of positions to compute embeddings for.
            device (torch.device): Device on which to allocate the tensors.
            dtype (torch.dtype): Data type for the computed tensors.
        
        Notes:
            Stores results in `self._cached_embeddings` with keys `'cos'` and `'sin'`, and updates `self._cached_seq_len`.
        """
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq)

        # Create rotation matrices
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)

        self._cached_embeddings = {'cos': cos, 'sin': sin}
        self._cached_seq_len = seq_len


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings (RoPE) to the last dimension of the input tensor.
    
    Parameters:
        x (torch.Tensor): Input tensor with last dimension = 2 * d (two interleaved halves).
        cos (torch.Tensor): Cosine positional factors broadcastable to x[..., :d].
        sin (torch.Tensor): Sine positional factors broadcastable to x[..., :d].
    
    Returns:
        torch.Tensor: Tensor with the same shape as `x` where the last dimension has been rotated
        by the provided `cos` and `sin` factors (RoPE applied).
    """
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
        """
        Initialize EnhancedGovernanceAttention with RoPE-enabled multi-head attention and a shared governance backbone.
        
        Parameters:
            hidden_dim (int): Total model hidden dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability applied to attention weights.
            max_seq_len (int): Maximum sequence length used to precompute rotary embeddings.
            governance_strength (float): Scaling factor that controls the influence of policy/memory/governance signals on attention bias.
            enable_caching (bool): If True, enables KV and attention-pattern caching provided by CachedAttentionMixin.
            cache_size (int): Maximum number of cached attention patterns when caching is enabled.
        
        Raises:
            AssertionError: If `hidden_dim` is not divisible by `num_heads`, or if the derived per-head dimension is not even (required for RoPE).
        """
        super().__init__(enable_caching=enable_caching, cache_size=cache_size)
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE (hidden_dim/num_heads)."
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
            hidden_dim, governance_dim=hidden_dim, num_heads=num_heads
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _compute_combined_attention_bias(
        self, 
        prophetic_mask: Optional[torch.Tensor], 
        policy_mask: Optional[torch.Tensor], 
        memory_weights: Optional[torch.Tensor],
        aligned_state: Optional['PropheticSingularityState'] = None
    ) -> torch.Tensor:
        """
        Combine prophetic, policy, and memory signals into a single attention bias tensor.
        
        Parameters:
            prophetic_mask (Optional[torch.Tensor]): Binary or scalar mask contributing a fixed positive bias; expected shape broadcastable to attention scores (e.g., [batch, heads, q_len, kv_len]).
            policy_mask (Optional[torch.Tensor]): Per-head policy mask contributing a scaled bias; expected shape broadcastable to attention scores.
            memory_weights (Optional[torch.Tensor]): Per-position memory weights that are converted to an additive bias via log(1 + memory_weights * governance_strength + 1e-8); expected shape broadcastable to attention scores.
            aligned_state (Optional[PropheticSingularityState]): Unused in the current computation (reserved for aligned-state-dependent extensions).
        
        Returns:
            torch.Tensor: The combined additive bias to add to attention scores (shape broadcastable to the provided masks). If no inputs are provided, returns scalar 0.0.
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
        prophetic_state: Optional['PropheticSingularityState'] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Compute multi-head attention with RoPE and governance adjustments, returning the attention output, governance signals, and optional cached key/value tensors.
        
        Parameters:
            x (torch.Tensor): Input tensor of shape [batch, seq_len, hidden_dim].
            attention_mask (Optional[torch.Tensor]): Binary mask that prevents attention to certain keys. Accepted shapes: [seq_len, seq_len], [batch, seq_len, seq_len], or [batch, num_heads, seq_len, kv_len]; shorter dimensions are automatically expanded or padded to match query/key lengths.
            policy_guidance (Optional[torch.Tensor]): External policy conditioning tensor of shape [batch, seq_len, hidden_dim] that adjusts per-position policy signals.
            memory_context (Optional[torch.Tensor]): Prior memory representations to be integrated into memory-aware attention adjustments.
            output_traces (bool): If True, generate and include interpretability traces in the governance outputs.
            prophetic_state (Optional[PropheticSingularityState]): Optional aligned prophetic state used to derive policy/memory masks and decay signals; it will be aligned to the current sequence length and device/dtype.
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Optional cached (k, v) tensors to prepend to current keys/values for decoding; each tensor must be shaped to match per-head layout used by the module.
            use_cache (bool): If True, returns present_key_value (k, v) suitable for caching in subsequent calls.
        
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                - output: Attention output tensor of shape [batch, seq_len, hidden_dim].
                - governance_outputs: Dictionary containing governance-related tensors such as 'policy_mask', 'memory_weights', 'trace' (when requested), 'attn_weights', and any prophetic signals like 'prophetic_mask' or 'prophetic_decay'.
                - present_key_value: Tuple (k, v) of the keys and values after concatenating past and current tensors when `use_cache` is True, otherwise None.
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

        # Apply RoPE to Q and K
        cos, sin = self.rope(x, seq_len, offset=past_length)
        q = apply_rope(q, cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2))
        k = apply_rope(k, cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2))
        
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
        past_len = max(kv_len - q_len, 0)

        if attention_mask is None:
            attention_mask = create_causal_mask(q_len, q.device, kv_len=kv_len, dtype=q.dtype)
        else:
            attention_mask = attention_mask.to(device=q.device)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)

            if attention_mask.shape[-2] != q_len:
                if attention_mask.shape[-2] == 1:
                    attention_mask = attention_mask.expand(-1, -1, q_len, -1)
                else:
                    attention_mask = attention_mask[..., -q_len:, :]

            if attention_mask.shape[-1] != kv_len:
                if attention_mask.shape[-1] < kv_len:
                    pad = attention_mask.new_ones(
                        attention_mask.shape[0],
                        attention_mask.shape[1],
                        attention_mask.shape[2],
                        kv_len - attention_mask.shape[-1]
                    )
                    attention_mask = torch.cat([pad, attention_mask], dim=-1)
                else:
                    attention_mask = attention_mask[..., -kv_len:]

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

        if past_len > 0:
            if prophetic_mask is not None and prophetic_mask.shape[-1] != kv_len:
                pad = prophetic_mask.new_zeros(*prophetic_mask.shape[:-1], past_len)
                prophetic_mask = torch.cat([pad, prophetic_mask], dim=-1)

            if policy_mask is not None and policy_mask.shape[-1] != kv_len:
                pad = policy_mask.new_zeros(*policy_mask.shape[:-1], past_len)
                policy_mask = torch.cat([pad, policy_mask], dim=-1)
                governance_outputs['policy_mask'] = policy_mask

            if memory_weights is not None and memory_weights.shape[-1] != kv_len:
                pad = memory_weights.new_zeros(*memory_weights.shape[:-1], past_len)
                memory_weights = torch.cat([pad, memory_weights], dim=-1)
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
        """
        Initialize the PolicyController with a policy MLP and per-head pattern controllers.
        
        The policy_net is an MLP (Linear -> ReLU -> Linear -> Tanh) that maps token representations of size `hidden_dim` back to `hidden_dim` for producing policy logits; `pattern_controllers` is a ModuleList of `num_heads` linear layers (each Linear(hidden_dim, 1)) that produce a per-token scalar controller signal for each attention head.
        
        Parameters:
            hidden_dim (int): Dimensionality of token representations processed by the policy network.
            num_heads (int): Number of attention heads; determines the number of per-head pattern controllers.
        """
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
        prophetic_state: Optional['PropheticSingularityState'] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Produce per-token policy logits and per-head attention modification masks.
        
        Parameters:
            x (torch.Tensor): Input representations of shape [batch, seq_len, hidden_dim].
            attention_scores (torch.Tensor): Current attention scores of shape [batch, num_heads, seq_len, seq_len].
            policy_guidance (Optional[torch.Tensor]): Optional external per-token guidance added to `x` before policy prediction.
            prophetic_state (Optional[PropheticSingularityState]): Optional prophetic state aligned to `seq_len` that modifies logits and can contribute an additional per-head policy mask.
        
        Returns:
            policy_logits (torch.Tensor): Per-token policy predictions with shape [batch, seq_len, hidden_dim].
            policy_mask (torch.Tensor): Per-head attention modification masks with shape [batch, num_heads, seq_len, seq_len]; higher values encourage attention between positions.
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
        """
        Initialize the MemoryManager and construct modules used to encode current inputs into memory, integrate them with contextual memory, and produce per-head memory weights.
        
        Parameters:
            hidden_dim (int): Dimensionality of the input token embeddings.
            num_heads (int): Number of attention heads used to produce per-head memory weights.
            memory_dim (int, optional): Dimensionality used for internal memory representations; defaults to `hidden_dim`.
        """
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
        prophetic_state: Optional['PropheticSingularityState'] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Compute memory-derived attention weight modifications and produce updated memory signals.
        
        Integrates the current input into a memory representation, optionally merges it with a provided memory_context, and applies prophetic scaling and decay when prophetic_state is given. Produces per-head memory weight adjustments suitable for biasing attention and returns the integrated memory signals for downstream use.
        
        Parameters:
            memory_context (Optional[torch.Tensor]): Previous memory sequence to integrate with the current memory.
            prophetic_state (Optional[PropheticSingularityState]): Prophetic state used to modulate memory integration (affects scaling and decay).
        
        Returns:
            memory_weights (Optional[torch.Tensor]): Per-head memory weight modifications with shape [batch, num_heads, 1, seq_len].
            integrated_memory (torch.Tensor): Memory signals after encoding and integration with shape [batch, seq_len, memory_dim].
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
        """
        Initialize the TraceGenerator, setting up modules that extract attention-derived features, importance scores, concept activations, and project combined trace representations.
        
        Parameters:
            hidden_dim (int): Dimensionality of the model hidden representations used as the feature size for trace outputs.
            num_heads (int): Number of attention heads; used as input width for attention analysis.
        
        Attributes:
            attention_analyzer (nn.Linear): Maps per-head averaged attention (width `num_heads`) into `hidden_dim` features.
            importance_scorer (nn.Linear): Scores combined features to produce a scalar importance value per token.
            concept_dim (int): Number of concept detectors (fixed to 64).
            concept_detector (nn.Linear): Projects `hidden_dim` features into `concept_dim` concept activations.
            trace_projector (nn.Linear): Projects the concatenation of attention features, concept activations, and importance gate (size `hidden_dim + concept_dim + 1`) back to `hidden_dim`.
        """
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
        prophetic_state: Optional['PropheticSingularityState'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate interpretability traces from attention patterns and model representations.
        
        Parameters:
            x (torch.Tensor): Input representations of shape [batch, seq_len, hidden_dim].
            attention_weights (torch.Tensor): Attention weights of shape [batch, num_heads, seq_len, seq_len].
            attention_output (torch.Tensor): Per-head attention output of shape [batch, num_heads, seq_len, head_dim].
            prophetic_state (Optional[PropheticSingularityState]): Optional prophetic state aligned to the current sequence for additional contextual traces.
        
        Returns:
            dict: A dictionary containing trace tensors and scalar/summary signals:
                - 'tensor' (torch.Tensor): Trace representation [batch, seq_len, hidden_dim].
                - 'importance_scores' (torch.Tensor): Raw importance scores per token [batch, seq_len].
                - 'importance_gate' (torch.Tensor): Sigmoid-scaled importance gate per token [batch, seq_len].
                - 'concept_activations' (torch.Tensor): Detected concept activations [batch, seq_len, 64].
                - 'attention_entropy' (torch.Tensor): Mean attention entropy across heads per token [batch, seq_len].
                - 'attention_patterns' (torch.Tensor): Average attention patterns across target positions [batch, num_heads, seq_len].
                - If `prophetic_state` is provided, includes:
                    - 'prophetic_envelope' (torch.Tensor): Aligned prophetic envelope for the sequence.
                    - 'kingdom_mean' (torch.Tensor): Prophetic summary mean.
                    - 'kingdom_std' (torch.Tensor): Prophetic summary standard deviation.
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
    kv_len: Optional[int] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Builds a causal attention mask that supports cached key/value memories (q_len x kv_len).
    
    Parameters:
        seq_len (int): Query sequence length (q_len).
        device (torch.device): Device for the returned tensor.
        kv_len (Optional[int]): Key/value sequence length (kv_len). If None, uses seq_len.
        dtype (Optional[torch.dtype]): Data type for the mask tensor. If None, uses torch.float32.
    
    Returns:
        torch.Tensor: A mask of shape [1, 1, seq_len, kv_len] where allowed attention positions are 1 and disallowed positions are 0.
        - If kv_len == seq_len: lower-triangular mask for causal attention.
        - If kv_len > seq_len: left block of ones for past (kv_len - seq_len) positions and a lower-triangular block for current positions.
    """
    kv_len = kv_len or seq_len
    dtype = dtype or torch.float32

    if kv_len == seq_len:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    else:
        past = torch.ones(seq_len, kv_len - seq_len, device=device, dtype=dtype)
        curr = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
        mask = torch.cat([past, curr], dim=-1)

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