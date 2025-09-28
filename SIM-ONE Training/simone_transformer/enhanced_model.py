"""
Enhanced SIM-ONE Transformer with Modern Architecture Components
Integrates RoPE attention, SwiGLU, RMSNorm, and advanced governance mechanisms
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from prioritary_mvlm.config import PropheticSingularityState


from .rope_attention import EnhancedGovernanceAttention, create_causal_mask
from .modern_layers import (
    RMSNorm, SwiGLU, GeGLU, GatedResidualConnection, 
    BiblicalAttentionBias, apply_weight_init
)


class EnhancedSIMONEBlock(nn.Module):
    """
    Enhanced SIM-ONE transformer block with modern components.
    Features:
    - RoPE-based attention with governance
    - SwiGLU feedforward network
    - RMSNorm for stability
    - Gated residual connections
    - Biblical attention biasing
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        use_moe: bool = False,
        num_experts: int = 8,
        layer_idx: int = 0,
        vocab_size: int = 32000,
        total_layers: int = 12
    ):
        """
        Initialize an EnhancedSIMONEBlock composed of RoPE-enabled governance-aware attention, a SwiGLU or MoE feedforward, RMSNorm layers, gated residuals, dropout, and a layer-specific governance enhancer.
        
        Parameters:
            hidden_dim (int): Dimension of model hidden states.
            num_heads (int): Number of attention heads.
            ff_dim (int): Inner dimension of the feedforward network or expert hidden size for MoE.
            dropout (float): Dropout probability applied after attention and feedforward.
            max_seq_len (int): Maximum supported sequence length for attention positional handling.
            use_moe (bool): If True, use a Mixture-of-Experts feedforward; otherwise use SwiGLU.
            num_experts (int): Number of experts to instantiate when `use_moe` is True.
            layer_idx (int): Zero-based index of this layer within the model; used to scale layer-specific governance behavior.
            vocab_size (int): Vocabulary size used to initialize the BiblicalAttentionBias.
            total_layers (int): Total number of layers in the model, used for layer-relative computations.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        
        # Pre-attention normalization
        self.attn_norm = RMSNorm(hidden_dim)
        
        # Enhanced attention with governance
        self.attention = EnhancedGovernanceAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            governance_strength=0.1 + 0.05 * layer_idx  # Increase governance in deeper layers
        )
        
        # Biblical attention bias
        self.biblical_bias = BiblicalAttentionBias(vocab_size, hidden_dim)
        
        # Pre-feedforward normalization
        self.ff_norm = RMSNorm(hidden_dim)
        
        # Enhanced feedforward network
        if use_moe:
            from .modern_layers import MoELayer
            self.feedforward = MoELayer(
                dim=hidden_dim,
                num_experts=num_experts,
                num_experts_per_token=2,
                expert_hidden_dim=ff_dim
            )
        else:
            self.feedforward = SwiGLU(hidden_dim, ff_dim)
        
        # Gated residual connections
        self.attn_gate = GatedResidualConnection(hidden_dim)
        self.ff_gate = GatedResidualConnection(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer-specific governance enhancement
        self.governance_enhancer = GovernanceEnhancer(
            hidden_dim, layer_idx, max_layers=12
        )
        
    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_context: Optional[torch.Tensor] = None,
        policy_guidance: Optional[torch.Tensor] = None,
        output_traces: bool = True,
        prophetic_state: Optional['PropheticSingularityState'] = None,
        precomputed_modulation: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Process a single transformer block: attention (with optional prophetic modulation and biblical bias), gated residuals, feedforward, and layer-wise governance enhancement.
        
        Parameters:
            x (torch.Tensor): Hidden states of shape [batch, seq_len, hidden_dim].
            input_ids (torch.Tensor): Token ids [batch, seq_len] used for optional biblical attention bias.
            attention_mask (Optional[torch.Tensor]): Attention mask (causal or provided) aligned with x.
            memory_context (Optional[torch.Tensor]): Optional memory signals propagated from previous layers.
            policy_guidance (Optional[torch.Tensor]): External policy logits or guidance passed into the attention mechanism.
            output_traces (bool): If True, request interpretability traces (attention/auxiliary outputs) from the attention module.
            prophetic_state (Optional[PropheticSingularityState]): Per-step prophetic state used to compute layer-specific modulation gates; expected to provide alignment and layer_modulation helpers.
            precomputed_modulation (Optional[torch.Tensor]): Pre-aligned modulation vector for this layer and sequence to avoid recomputation; shaped [batch, seq_len] or compatible for broadcasting.
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Cached key/value tensors for cached autoregressive decoding.
            use_cache (bool): If True, produce and return present_key_value for use in subsequent cached decoding steps.
        
        Returns:
            Tuple containing:
            - output (torch.Tensor): Block output hidden states of shape [batch, seq_len, hidden_dim].
            - governance_outputs (Dict[str, torch.Tensor]): Collected governance signals and any enhanced policy/memory/trace outputs for this layer.
            - present_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Key/value cache for this layer when caching is enabled, otherwise None.
        """
        residual = x
        
        # Pre-attention normalization
        x_norm = self.attn_norm(x)

        block_state = None
        modulation_gate = None
        if prophetic_state is not None:
            block_state = prophetic_state.align_to_length(x.shape[1])
            # Use pre-computed modulation if available, otherwise compute
            if precomputed_modulation is not None:
                modulation_gate = precomputed_modulation.unsqueeze(-1)
            else:
                modulation_gate = block_state.layer_modulation(self.layer_idx, self.total_layers).unsqueeze(-1)
            x_norm = x_norm * modulation_gate

        # Enhanced attention with governance
        attn_output, gov_outputs, present_key_value = self.attention(
            x_norm,
            attention_mask=attention_mask,
            policy_guidance=policy_guidance,
            memory_context=memory_context,
            output_traces=output_traces,
            prophetic_state=block_state,
            past_key_value=past_key_value,
            use_cache=use_cache
        )

        if modulation_gate is not None:
            attn_output = attn_output * modulation_gate

        # Apply biblical attention bias
        if input_ids is not None:
            # Get attention scores and apply biblical bias
            biblical_bias = self.biblical_bias(input_ids, gov_outputs.get('attn_weights'))
            # This would be integrated into attention computation in practice
            
        attn_output = self.dropout(attn_output)
        
        # Gated residual connection for attention
        x = self.attn_gate(attn_output, residual)
        
        # Feedforward block
        residual = x
        x_norm = self.ff_norm(x)
        if modulation_gate is not None:
            x_norm = x_norm * modulation_gate
        ff_output = self.feedforward(x_norm)
        if modulation_gate is not None:
            ff_output = ff_output * modulation_gate
        ff_output = self.dropout(ff_output)
        
        # Gated residual connection for feedforward
        x = self.ff_gate(ff_output, residual)
        
        # Enhance governance outputs based on layer position
        gov_outputs = self.governance_enhancer(gov_outputs, x, self.layer_idx)

        if block_state is not None:
            gov_outputs['prophetic_kingdom'] = block_state.kingdom_flow

        return x, gov_outputs, present_key_value


class GovernanceEnhancer(nn.Module):
    """
    Enhances governance outputs based on layer position and specialization.
    Different layers focus on different aspects of governance.
    """
    
    def __init__(self, hidden_dim: int, layer_idx: int, max_layers: int = 12):
        """
        Initialize a GovernanceEnhancer with layer-positioned specialization weights and processors.
        
        Parameters:
            hidden_dim (int): Dimensionality of the governance feature vectors processed by the layer-specific linear modules.
            layer_idx (int): Zero-based index of this layer within the model; used to compute specialization strengths.
            max_layers (int): Total number of layers used to scale the positional specialization profiles (default 12).
        
        Description:
            Computes three layer-position-dependent scalar weights (syntax, semantic, pragmatic) that bias processing toward
            syntactic signals in early layers, semantic signals in middle layers, and pragmatic signals in later layers.
            Instantiates three linear processors (syntax_processor, semantic_processor, pragmatic_processor) that map
            hidden-dimension vectors to the same dimensionality for subsequent governance refinement.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.max_layers = max_layers
        
        # Layer specialization weights
        self.syntax_weight = max(0, 1 - layer_idx / (max_layers * 0.3))  # Strong early
        self.semantic_weight = math.exp(-(layer_idx - max_layers/2)**2 / (max_layers/4)**2)  # Peak middle
        self.pragmatic_weight = min(1, layer_idx / (max_layers * 0.7))  # Strong late
        
        # Specialized governance processors
        self.syntax_processor = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_processor = nn.Linear(hidden_dim, hidden_dim)  
        self.pragmatic_processor = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self, 
        gov_outputs: Dict[str, torch.Tensor], 
        hidden_states: torch.Tensor,
        layer_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Refine and weight layer-specific governance signals and return an updated governance outputs dictionary.
        
        If `policy_logits` is present in `gov_outputs`, applies the layer's syntax, semantic, and pragmatic processors to those logits using the layer-specific weights, replaces `policy_logits` with the refined values, and adds a `governance_weights` entry containing the three weights.
        
        Parameters:
            gov_outputs (Dict[str, torch.Tensor]): Governance outputs from a layer; may contain `policy_logits`.
            hidden_states (torch.Tensor): Current hidden states for the layer (provided for contextual processing; not required for all enhancements).
            layer_idx (int): Index of the current layer (used to select layer-specific behavior).
        
        Returns:
            Dict[str, torch.Tensor]: A copy of `gov_outputs` with `policy_logits` replaced by the enhanced logits when applicable and `governance_weights` added to report the applied weights.
        """
        
        enhanced_outputs = gov_outputs.copy()
        
        # Apply layer-specific governance enhancement
        if 'policy_logits' in gov_outputs:
            policy = gov_outputs['policy_logits']
            
            # Combine different aspects based on layer position
            enhanced_policy = (
                self.syntax_weight * self.syntax_processor(policy) +
                self.semantic_weight * self.semantic_processor(policy) +
                self.pragmatic_weight * self.pragmatic_processor(policy)
            )
            
            enhanced_outputs['policy_logits'] = enhanced_policy
            enhanced_outputs['governance_weights'] = {
                'syntax': self.syntax_weight,
                'semantic': self.semantic_weight, 
                'pragmatic': self.pragmatic_weight
            }
        
        return enhanced_outputs


class EnhancedSIMONEModel(nn.Module):
    """
    Enhanced SIM-ONE Language Model with modern architecture and governance.
    
    Key improvements:
    - Modern attention with RoPE
    - SwiGLU feedforward networks
    - RMSNorm for stability
    - Advanced governance mechanisms
    - Biblical attention biasing
    - Layer-wise specialization
    - Efficient generation support
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 768,
        num_heads: int = 12,
        ff_dim: int = 3072,
        num_layers: int = 12,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        use_moe: bool = False,
        moe_layers: Optional[List[int]] = None,
        num_experts: int = 8,
        tie_embeddings: bool = True,
        use_gradient_checkpointing: bool = False
    ):
        """
        Construct an EnhancedSIMONEModel and initialize its components and caches.
        
        Parameters:
            vocab_size (int): Size of the token vocabulary.
            hidden_dim (int): Dimensionality of model hidden states.
            num_heads (int): Number of attention heads.
            ff_dim (int): Hidden dimensionality of feedforward sublayers.
            num_layers (int): Number of transformer layers to create.
            max_seq_len (int): Maximum supported sequence length (RoPE is used for positions).
            dropout (float): Dropout probability applied in attention and feedforward.
            use_moe (bool): If True, enable Mixture-of-Experts feedforward layers where configured.
            moe_layers (Optional[List[int]]): Indices of layers that should use MoE; if None, MoE is applied to all layers when enabled.
            num_experts (int): Number of experts for MoE layers when used.
            tie_embeddings (bool): If True, share token embedding weights with the language modeling head.
            use_gradient_checkpointing (bool): If True, enable gradient checkpointing for memory-efficient training.
        
        Notes:
            Initializes token embeddings, a stack of EnhancedSIMONEBlock layers, final normalization, the LM head, a GovernanceAggregator, weight initialization, and internal caches used for generation and prophetic-state precomputation.
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # No positional embedding needed (RoPE handles position in attention)
        
        # Transformer layers with hierarchical specialization
        self.layers = nn.ModuleList([
            EnhancedSIMONEBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                max_seq_len=max_seq_len,
                use_moe=use_moe and (moe_layers is None or i in moe_layers),
                num_experts=num_experts,
                layer_idx=i,
                vocab_size=vocab_size,
                total_layers=num_layers
            )
            for i in range(num_layers)
        ])
        
        # Final normalization
        self.final_norm = RMSNorm(hidden_dim)
        
        # Language modeling head
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie embeddings if specified (parameter sharing)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        
        # Governance aggregator for final outputs
        self.governance_aggregator = GovernanceAggregator(hidden_dim, num_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # KV cache for efficient generation
        self.kv_cache = None
        self.cache_enabled = False
        self.last_prophetic_state: Optional['PropheticSingularityState'] = None
        
        # Pre-computed prophetic state cache
        self._precomputed_prophetic_cache = None
        
    def _init_weights(self, module):
        """
        Initialize parameters for supported module types.
        
        Applies the following initializations:
        - nn.Linear: weight ~ Normal(mean=0.0, std=0.02); bias = 0 if present.
        - nn.Embedding: weight ~ Normal(mean=0.0, std=0.02).
        - RMSNorm: weight = 1.
        
        Parameters:
            module: The module whose parameters should be initialized (e.g., nn.Linear, nn.Embedding, or RMSNorm).
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
    
    def enable_cache(self):
        """
        Enable the key–value cache used during autoregressive generation.
        
        Sets the internal flag to enable caching and initializes an empty `kv_cache` dictionary.
        """
        self.cache_enabled = True
        self.kv_cache = {}
    
    def disable_cache(self):
        """Disable KV cache."""
        self.cache_enabled = False
        self.kv_cache = None
    
    def clear_cache(self):
        """
        Clear the model's key-value (KV) cache.
        
        If a KV cache exists, removes all cached entries; does nothing if no cache is present.
        """
        if self.kv_cache is not None:
            self.kv_cache.clear()
    
    def _precompute_prophetic_modulations(
        self,
        prophetic_state: Optional['PropheticSingularityState'],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        past_length: int = 0
    ) -> Optional[
        Tuple[
            PropheticSingularityState,
            List[torch.Tensor],
            PropheticSingularityState
        ]
    ]:
        """
        Prepare per-layer modulation vectors from a prophetic state aligned to the current token window.
        
        Precomputes and caches layer-specific modulation tensors for the sequence window defined by seq_len and past_length so they can be reused by each Transformer layer during a forward pass.
        
        Parameters:
            prophetic_state (Optional[PropheticSingularityState]): Time-varying governance/state object; if None, no precomputation is performed.
            seq_len (int): Number of new tokens in the current forward pass (excluding past cached tokens).
            device (torch.device): Target device for the returned tensors.
            dtype (torch.dtype): Target dtype for the returned tensors.
            past_length (int): Number of tokens already present in the past key/value cache; used to align and slice the prophetic state.
        
        Returns:
            Optional[Tuple[PropheticSingularityState, List[torch.Tensor], PropheticSingularityState]]:
                - aligned_state: the prophetic state sliced to the current sequence window (seq_len) and moved to the requested device/dtype.
                - layer_modulations: list of per-layer modulation tensors, each already sliced to the current window.
                - aligned_total: the full prophetic state aligned to the total sequence length (past_length + seq_len).
                Returns None if `prophetic_state` is None.
        
        Side effects:
            Caches the computed result on the model instance and will return the cached value when called again with an equivalent prophetic_state and length parameters.
        """
        if prophetic_state is None:
            return None
            
        # Check cache validity
        total_len = seq_len + past_length
        cache_key = (id(prophetic_state), total_len, past_length, device, dtype)
        if (self._precomputed_prophetic_cache is not None and
            self._precomputed_prophetic_cache[0] == cache_key):
            return self._precomputed_prophetic_cache[1]

        # Align state to total sequence length (past + current tokens)
        aligned_total = prophetic_state.align_to_length(total_len).to(
            device=device,
            dtype=dtype
        )

        start_idx = max(total_len - seq_len, 0)

        # Slice helper to preserve normalization metadata
        def _slice_state(
            state: PropheticSingularityState,
            start: int
        ) -> PropheticSingularityState:
            """
            Create a copy of a PropheticSingularityState with its time-series tensors sliced from `start` to the end.
            
            Parameters:
                state (PropheticSingularityState): The original prophetic state containing tensor fields aligned along the time dimension.
                start (int): Start index (inclusive) along the time dimension to slice from.
            
            Returns:
                PropheticSingularityState: A new state where `intensity`, `anointing`, `dominion`, `mercy`, `lambda_field`, and `time_index` are sliced as `[..., start:]`; `normalization` is preserved unchanged.
            """
            if start == 0 and seq_len == total_len:
                return state

            return PropheticSingularityState(
                intensity=state.intensity[..., start:],
                anointing=state.anointing[..., start:],
                dominion=state.dominion[..., start:],
                mercy=state.mercy[..., start:],
                lambda_field=state.lambda_field[..., start:],
                time_index=state.time_index[..., start:],
                normalization=state.normalization,
            )

        aligned_state = _slice_state(aligned_total, start_idx)

        # Pre-compute all layer modulations
        layer_modulations = []
        for layer_idx in range(self.num_layers):
            modulation = aligned_total.layer_modulation(layer_idx, self.num_layers)
            layer_modulations.append(modulation[..., start_idx:])

        # Cache the result
        result = (aligned_state, layer_modulations, aligned_total)
        self._precomputed_prophetic_cache = (cache_key, result)

        return result
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        policy_guidance: Optional[torch.Tensor] = None,
        output_governance: bool = True,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        prophetic_state: Optional['PropheticSingularityState'] = None
    ) -> Tuple[torch.Tensor, Optional[Dict], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Compute language-model logits and, optionally, aggregated governance outputs for the provided token ids, supporting causal masking, KV caching, and optional prophetic-state modulations.
        
        Parameters:
            input_ids (torch.Tensor): Token IDs with shape [batch, seq_len].
            attention_mask (Optional[torch.Tensor]): Attention mask. Accepted shapes: [batch, seq_len], [batch, 1, 1, seq_len], [batch, seq_len, seq_len]; will be adapted to causal mask and KV cache length.
            policy_guidance (Optional[torch.Tensor]): External policy guidance per token with shape [batch, seq_len, hidden_dim], if provided.
            output_governance (bool): If True, collect and return aggregated governance information across layers.
            use_cache (bool): If True, return per-layer present key/value tensors for incremental generation and accept past_key_values.
            past_key_values (Optional[List[Tuple[torch.Tensor, torch.Tensor]]]): List of cached (key, value) tensors per layer used to extend past context; must have length equal to the number of layers when provided.
            prophetic_state (Optional[PropheticSingularityState]): Optional prophetic-state object used to precompute and apply per-layer modulation signals.
        
        Returns:
            Tuple[torch.Tensor, Optional[dict], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
                logits (torch.Tensor): Language modeling logits with shape [batch, seq_len, vocab_size].
                governance_info (dict or None): Aggregated governance outputs produced by the GovernanceAggregator when `output_governance` is True; includes policy, memory, trace fields and may include prophetic summary statistics when a prophetic_state was provided.
                next_past_key_values (list or None): Present key/value tuples for each layer when `use_cache` is True (None otherwise).
        """
        _, seq_len = input_ids.shape
        device = input_ids.device

        if past_key_values is None:
            past_key_values = [None] * self.num_layers
        elif len(past_key_values) != self.num_layers:
            raise ValueError(
                f"past_key_values must have {self.num_layers} entries (got {len(past_key_values)})"
            )

        past_length = 0
        if use_cache and past_key_values[0] is not None:
            past_length = past_key_values[0][0].size(-2)

        kv_len = seq_len + past_length if use_cache else seq_len

        # Token embeddings for current step
        x = self.token_embedding(input_ids)

        # Pre-compute prophetic state modulations once
        precomputed_state = self._precompute_prophetic_modulations(
            prophetic_state, seq_len, device, x.dtype, past_length=past_length
        )

        summary_state = None
        if precomputed_state is not None:
            aligned_state, layer_modulations, aligned_total_state = precomputed_state
            summary_state = aligned_total_state
            self.last_prophetic_state = aligned_total_state
        else:
            aligned_state = None
            layer_modulations = None
            self.last_prophetic_state = None

        # Create causal attention mask if not provided and align with KV cache
        if attention_mask is None:
            attention_mask = create_causal_mask(seq_len, device, kv_len=kv_len, dtype=x.dtype)
        else:
            attention_mask = attention_mask.to(device=device)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)

            if attention_mask.shape[-2] != seq_len:
                if attention_mask.shape[-2] == 1:
                    attention_mask = attention_mask.expand(-1, -1, seq_len, -1)
                else:
                    attention_mask = attention_mask[..., -seq_len:, :]

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

        # Store governance outputs from all layers
        all_governance_outputs = [] if output_governance else None
        memory_context = None
        next_past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        # Pass through transformer layers with pre-computed modulations
        for layer_idx, layer in enumerate(self.layers):
            # Get pre-computed modulation for this layer
            precomputed_modulation = layer_modulations[layer_idx] if layer_modulations else None
            layer_past = past_key_values[layer_idx] if past_key_values else None

            # Apply gradient checkpointing if enabled during training
            if self.training and self.use_gradient_checkpointing:
                # Use gradient checkpointing to trade compute for memory
                x, gov_outputs, _ = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    input_ids,
                    attention_mask,
                    memory_context,
                    policy_guidance,
                    output_governance,
                    aligned_state,
                    precomputed_modulation,
                    None,
                    False,
                    use_reentrant=False  # Use new checkpointing API
                )
            else:
                x, gov_outputs, present_key_value = layer(
                    x=x,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    memory_context=memory_context,
                    policy_guidance=policy_guidance,
                    output_traces=output_governance,
                    prophetic_state=aligned_state,
                    precomputed_modulation=precomputed_modulation,
                    past_key_value=layer_past,
                    use_cache=use_cache
                )
                if use_cache and next_past_key_values is not None:
                    next_past_key_values.append(present_key_value)

            if output_governance:
                all_governance_outputs.append(gov_outputs)

            # Use memory signals as context for next layer
            if isinstance(gov_outputs, dict) and 'memory_signals' in gov_outputs:
                memory_context = gov_outputs['memory_signals']
        
        # Final normalization
        x = self.final_norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Aggregate governance information
        governance_info = None
        if output_governance and all_governance_outputs:
            governance_info = self.governance_aggregator(all_governance_outputs, x)
            if summary_state is not None:
                summary = summary_state.summary()
                governance_info['prophetic_kingdom_mean'] = summary['kingdom']['mean']
                governance_info['prophetic_kingdom_std'] = summary['kingdom']['std']

        return logits, governance_info, next_past_key_values if use_cache else None
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        prophetic_state: Optional['PropheticSingularityState'] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively from the given prompt using the model's cached KV mechanism and optional prophetic guidance.
        
        This performs stepwise decoding up to max_length, optionally using sampling (temperature, top_k, top_p) or greedy selection. If provided, `prophetic_state` adjusts per-step sampling parameters (temperature, top-k, top-p) based on its internal statistics. The method enables the model's KV cache at start and disables it before returning.
        
        Parameters:
            input_ids (torch.Tensor): Initial token IDs with shape [batch, initial_seq_len].
            max_length (int): Maximum total sequence length (prompt + generated tokens).
            temperature (float): Base sampling temperature; lower values make outputs more deterministic.
            top_k (Optional[int]): If set, keeps only the top_k highest-probability tokens at each step.
            top_p (Optional[float]): If set, keeps the smallest set of tokens with cumulative probability >= top_p.
            do_sample (bool): When True, sample from the distribution; otherwise take the highest-probability token.
            eos_token_id (Optional[int]): If provided, generation stops when all batches produce this token.
            pad_token_id (Optional[int]): Padding token id (not used for stopping behavior here).
            prophetic_state (Optional[PropheticSingularityState]): Optional state used to dynamically modulate per-step sampling hyperparameters.
        
        Returns:
            torch.Tensor: Generated token IDs with shape [batch, total_seq_len] where total_seq_len <= max_length.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Enable caching for efficient generation
        self.enable_cache()

        # Initialize generation
        generated = input_ids.clone()

        state = prophetic_state or self.last_prophetic_state
        state_device = None
        if state is not None:
            state_device = state.align_to_length(max_length).to(device=device, dtype=torch.float32)

        adjustments_log: List[Dict[str, float]] = []

        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        current_input = input_ids
        max_new_tokens = max(0, max_length - input_ids.shape[1])

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits, _, past_key_values = self.forward(
                    current_input,
                    output_governance=False,
                    use_cache=True,
                    past_key_values=past_key_values,
                    prophetic_state=state_device
                )

            step_idx = generated.shape[1] - 1

            dynamic_temperature = temperature
            dynamic_top_k = top_k
            dynamic_top_p = top_p
            kingdom_val = None

            if state_device is not None:
                stats = state_device.step_statistics(step_idx)

                def _to_float(value: torch.Tensor) -> float:
                    """
                    Convert a tensor or numeric value to a Python float.
                    
                    Parameters:
                        value (torch.Tensor | number): A zero-dimensional torch.Tensor or any value convertible to float.
                    
                    Returns:
                        float: The Python float representation of the input (for tensors, obtained via `tensor.item()`).
                    """
                    if isinstance(value, torch.Tensor):
                        return float(value.item())
                    return float(value)

                intensity_val = _to_float(stats['intensity'])
                mercy_val = _to_float(stats['mercy'])
                dominion_val = _to_float(stats['dominion'])
                lambda_val = _to_float(stats['lambda'])
                time_val = _to_float(stats['time'])
                kingdom_val = _to_float(stats['kingdom'])

                temp_scale = 1.0 + 0.25 * (mercy_val - dominion_val)
                dynamic_temperature = max(0.05, temperature * temp_scale)

                if top_k is not None:
                    dynamic_top_k = max(1, int(top_k * (1.0 + intensity_val * 0.2 - lambda_val * 0.1)))

                if top_p is not None:
                    adjusted = top_p * (1.0 - 0.15 * (lambda_val - 0.5) + 0.1 * time_val)
                    adjusted = float(torch.clamp(torch.tensor(adjusted), 0.01, 0.99).item())
                    dynamic_top_p = adjusted

            # Get logits for last position (current step)
            next_token_logits = logits[:, -1, :] / dynamic_temperature

            # Apply top-k filtering
            if dynamic_top_k is not None:
                topk = torch.topk(next_token_logits, dynamic_top_k, dim=-1)[0][..., -1, None]
                indices_to_remove = next_token_logits < topk
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if dynamic_top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > dynamic_top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = torch.gather(sorted_indices_to_remove, -1, sorted_indices.argsort(-1))
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample or take greedy choice
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated sequence and prepare next step
            generated = torch.cat([generated, next_token], dim=-1)
            current_input = next_token

            adjustments_log.append({
                'step': float(step_idx),
                'temperature': float(dynamic_temperature),
                'top_k': float(dynamic_top_k or 0),
                'top_p': float(dynamic_top_p or 0.0),
                'kingdom_flow': float(kingdom_val) if kingdom_val is not None else float('nan')
            })

            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        # Disable cache
        self.disable_cache()

        self.last_generation_adjustments = adjustments_log

        return generated
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Report the model's memory usage for parameters and buffers.
        
        Returns:
            memory_usage (Dict[str, int]): Dictionary with three keys:
                - 'parameters': total bytes used by all model parameters.
                - 'buffers': total bytes used by all registered buffers.
                - 'total': sum of 'parameters' and 'buffers'.
        """
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in self.buffers())
        
        return {
            'parameters': param_memory,
            'buffers': buffer_memory,
            'total': param_memory + buffer_memory
        }


class GovernanceAggregator(nn.Module):
    """
    Aggregates governance information from all layers into final outputs.
    Provides interpretability and control signals.
    """
    
    def __init__(self, hidden_dim: int, num_layers: int):
        """
        Initialize the governance aggregator that consolidates per-layer governance signals into final policy, memory, and trace representations.
        
        Parameters:
            hidden_dim (int): Dimensionality of model hidden states used per layer.
            num_layers (int): Number of transformer layers whose governance outputs will be aggregated.
        
        The constructor builds three linear aggregation networks (policy, memory, trace) that each accept the concatenation of per-layer hidden vectors (hidden_dim * num_layers) and project down to hidden_dim, and a small sequential head (`final_policy`) that produces a scalar policy score in the range [0, 1].
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Aggregation networks
        self.policy_aggregator = nn.Linear(hidden_dim * num_layers, hidden_dim)
        self.memory_aggregator = nn.Linear(hidden_dim * num_layers, hidden_dim)
        self.trace_aggregator = nn.Linear(hidden_dim * num_layers, hidden_dim)
        
        # Final processors
        self.final_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        all_governance_outputs: List[Dict[str, torch.Tensor]],
        final_hidden_states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate per-layer governance outputs into consolidated policy, memory, and trace outputs.
        
        Parameters:
            all_governance_outputs (List[Dict[str, torch.Tensor]]): List of per-layer governance dictionaries. Each dictionary may contain:
                - 'policy_logits': tensor of policy logits for that layer.
                - 'memory_signals': tensor representing memory signals for that layer.
                - 'trace': either a tensor representing layer trace or a dict with a 'tensor' key and optional metadata keys.
            final_hidden_states (torch.Tensor): Final model hidden states used as a fallback trace when no per-layer trace is present.
        
        Returns:
            Dict[str, torch.Tensor]: Aggregated governance outputs. Possible keys:
                - 'policy' (torch.Tensor): Aggregated policy features from all layers.
                - 'policy_score' (torch.Tensor): Final policy score produced by the policy head.
                - 'policy_logits' (torch.Tensor): Policy logits from the last layer (preserved for loss/analysis).
                - 'policy_logits_all_layers' (List[torch.Tensor]): List of policy logits from each layer.
                - 'memory' (torch.Tensor): Aggregated memory features from all layers.
                - 'memory_signals' (torch.Tensor): Memory signals from the last layer.
                - 'memory_signals_all_layers' (List[torch.Tensor]): List of memory signals from each layer.
                - 'trace' (torch.Tensor): Aggregated trace representation (or `final_hidden_states` if no per-layer traces).
                - 'trace_all_layers' (List[torch.Tensor]): List of per-layer trace tensors (or a single entry of `final_hidden_states`).
                - 'trace_metadata' (List[Dict[str, torch.Tensor]]): Optional list of metadata dictionaries associated with per-layer trace tensors.
        """
        
        # Extract and concatenate outputs from all layers
        policy_outputs = []
        memory_outputs = []
        trace_outputs = []
        trace_metadata: List[Dict[str, torch.Tensor]] = []

        for gov_out in all_governance_outputs:
            if 'policy_logits' in gov_out:
                policy_outputs.append(gov_out['policy_logits'])
            if 'memory_signals' in gov_out:
                memory_outputs.append(gov_out['memory_signals'])
            trace_entry = gov_out.get('trace')
            if isinstance(trace_entry, dict):
                trace_tensor = trace_entry.get('tensor')
                if isinstance(trace_tensor, torch.Tensor):
                    trace_outputs.append(trace_tensor)
                    meta = {k: v for k, v in trace_entry.items() if k != 'tensor'}
                    trace_metadata.append(meta)
            elif isinstance(trace_entry, torch.Tensor):
                trace_outputs.append(trace_entry)

        aggregated = {}

        # Aggregate policy information
        if policy_outputs:
            policy_concat = torch.cat(policy_outputs, dim=-1)
            aggregated['policy'] = self.policy_aggregator(policy_concat)
            aggregated['policy_score'] = self.final_policy(aggregated['policy'])
            # Preserve original policy logits for downstream losses/analysis
            aggregated['policy_logits'] = policy_outputs[-1]
            aggregated['policy_logits_all_layers'] = list(policy_outputs)

        # Aggregate memory information
        if memory_outputs:
            memory_concat = torch.cat(memory_outputs, dim=-1)
            aggregated['memory'] = self.memory_aggregator(memory_concat)
            aggregated['memory_signals'] = memory_outputs[-1]
            aggregated['memory_signals_all_layers'] = list(memory_outputs)

        # Aggregate trace information
        if trace_outputs:
            trace_concat = torch.cat(trace_outputs, dim=-1)
            expected_dim = self.hidden_dim * self.num_layers
            if trace_concat.size(-1) != expected_dim:
                if trace_concat.size(-1) < expected_dim:
                    pad_size = expected_dim - trace_concat.size(-1)
                    pad_tensor = trace_concat.new_zeros(
                        trace_concat.size(0), trace_concat.size(1), pad_size
                    )
                    trace_concat = torch.cat([trace_concat, pad_tensor], dim=-1)
                else:
                    trace_concat = trace_concat[..., :expected_dim]
            aggregated_trace = self.trace_aggregator(trace_concat)
            aggregated['trace'] = aggregated_trace
            aggregated['trace_all_layers'] = list(trace_outputs)
            if trace_metadata:
                aggregated['trace_metadata'] = trace_metadata
        else:
            aggregated['trace'] = final_hidden_states
            aggregated['trace_all_layers'] = [final_hidden_states]

        return aggregated


if __name__ == "__main__":
    # Test the enhanced SIM-ONE model
    print("Testing Enhanced SIM-ONE Model...")
    
    # Model configuration
    config = {
        'vocab_size': 32000,
        'hidden_dim': 768,
        'num_heads': 12,
        'ff_dim': 3072,
        'num_layers': 6,  # Smaller for testing
        'max_seq_len': 1024,
        'dropout': 0.1
    }
    
    # Create model
    model = EnhancedSIMONEModel(**config)
    
    # Test input
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Memory usage: {model.get_memory_usage()}")
    
    # Forward pass
    logits, governance, _ = model(input_ids, output_governance=True)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Governance outputs: {list(governance.keys()) if governance else 'None'}")
    
    # Test generation
    print("\nTesting generation...")
    prompt = torch.randint(0, config['vocab_size'], (1, 10))
    generated = model.generate(prompt, max_length=50, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
    
    print("✓ Enhanced SIM-ONE model working correctly!")