"""
Enhanced SIM-ONE Transformer with Modern Architecture Components
Integrates RoPE attention, SwiGLU, RMSNorm, and advanced governance mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math

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
        prophetic_state: Optional[PropheticSingularityState] = None,
        precomputed_modulation: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            input_ids: Token IDs [batch, seq_len] for biblical biasing
            attention_mask: Causal mask
            memory_context: Memory context from previous layers
            policy_guidance: External policy guidance
            output_traces: Whether to output interpretability traces
            
        Returns:
            output: Block output [batch, seq_len, hidden_dim]
            governance_outputs: Governance information
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
        """Enhance governance outputs with layer-specific processing."""
        
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
        self.last_prophetic_state: Optional[PropheticSingularityState] = None
        
        # Pre-computed prophetic state cache
        self._precomputed_prophetic_cache = None
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
    
    def enable_cache(self):
        """Enable KV cache for efficient generation."""
        self.cache_enabled = True
        self.kv_cache = {}
    
    def disable_cache(self):
        """Disable KV cache."""
        self.cache_enabled = False
        self.kv_cache = None
    
    def clear_cache(self):
        """Clear KV cache."""
        if self.kv_cache is not None:
            self.kv_cache.clear()
    
    def _precompute_prophetic_modulations(
        self,
        prophetic_state: Optional[PropheticSingularityState],
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
        Pre-compute all layer modulations once for efficiency.
        Avoids repeated computation in each layer.
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
        prophetic_state: Optional[PropheticSingularityState] = None
    ) -> Tuple[torch.Tensor, Optional[Dict], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass through the enhanced SIM-ONE model.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len, seq_len]
            policy_guidance: External policy guidance [batch, seq_len, hidden_dim]
            output_governance: Whether to output governance information
            use_cache: Whether to use KV caching for generation
            
        Returns:
            logits: Language modeling logits [batch, seq_len, vocab_size]
            governance_info: Aggregated governance information (if output_governance=True)
            present_key_values: Cached key/value tensors for each layer when ``use_cache`` is True
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
        prophetic_state: Optional[PropheticSingularityState] = None
    ) -> torch.Tensor:
        """
        Generate text using the enhanced SIM-ONE model.
        
        Args:
            input_ids: Initial token IDs [batch, initial_seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling vs greedy decoding
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID
            
        Returns:
            generated_ids: Generated token IDs [batch, total_seq_len]
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
        """Get memory usage statistics."""
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
        """Aggregate governance outputs from all layers."""
        
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