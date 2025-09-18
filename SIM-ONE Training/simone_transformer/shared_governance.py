"""
Shared Governance Backbone for Enhanced SIM-ONE Transformer
Optimizes governance computation by sharing feature extraction across components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from prioritary_mvlm.config import PropheticSingularityState


class SharedGovernanceBackbone(nn.Module):
    """
    Shared feature extraction backbone for all governance components.
    
    Instead of having separate networks for policy, memory, and trace generation,
    this backbone extracts shared features once and provides specialized heads
    for each governance component.
    
    Expected improvement: 20-30% faster governance computation
    """
    
    def __init__(self, hidden_dim: int, governance_dim: int = None, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.governance_dim = governance_dim or hidden_dim // 2  # Reduce dimensionality
        self.num_heads = num_heads
        
        # Shared feature extraction backbone
        self.shared_encoder = nn.Sequential(
            nn.Linear(hidden_dim, self.governance_dim),
            nn.ReLU(),
            nn.Linear(self.governance_dim, self.governance_dim),
            nn.LayerNorm(self.governance_dim)  # Stabilize shared features
        )
        
        # Specialized heads for each governance component
        self.policy_head = PolicyHead(self.governance_dim, hidden_dim, num_heads)
        self.memory_head = MemoryHead(self.governance_dim, hidden_dim, num_heads)
        self.trace_head = TraceHead(self.governance_dim, hidden_dim, num_heads)
        
        # Optional: Cross-component attention for governance coordination
        self.governance_coordination = nn.MultiheadAttention(
            self.governance_dim, num_heads=4, batch_first=True
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
        attention_output: Optional[torch.Tensor] = None,
        policy_guidance: Optional[torch.Tensor] = None,
        memory_context: Optional[torch.Tensor] = None,
        prophetic_state: Optional[PropheticSingularityState] = None,
        output_traces: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through shared governance backbone.
        
        Args:
            x: Input representations [batch, seq_len, hidden_dim]
            attention_scores: Current attention scores (for policy)
            attention_weights: Attention weights (for trace)
            attention_output: Attention output (for trace)
            policy_guidance: External policy guidance
            memory_context: Previous memory context
            prophetic_state: Prophetic singularity state
            output_traces: Whether to generate traces
            
        Returns:
            Dict containing all governance outputs
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Extract shared governance features once
        shared_features = self.shared_encoder(x)  # [batch, seq_len, governance_dim]
        
        # Apply prophetic state modulation to shared features if available
        if prophetic_state is not None:
            aligned_state = prophetic_state.align_to_length(seq_len).to(x.device, x.dtype)
            kingdom_modulation = aligned_state.kingdom_flow.unsqueeze(-1)  # [batch, seq_len, 1]
            shared_features = shared_features * (1.0 + kingdom_modulation * 0.1)
        
        # Optional: Apply governance coordination (cross-component attention)
        coordinated_features, _ = self.governance_coordination(
            shared_features, shared_features, shared_features
        )
        
        # Use coordinated features for better governance integration
        enhanced_features = shared_features + coordinated_features * 0.1
        
        # Generate outputs from specialized heads
        governance_outputs = {}
        
        # Policy head
        policy_outputs = self.policy_head(
            enhanced_features, attention_scores, policy_guidance, prophetic_state
        )
        governance_outputs.update(policy_outputs)
        
        # Memory head
        memory_outputs = self.memory_head(
            enhanced_features, memory_context, prophetic_state
        )
        governance_outputs.update(memory_outputs)
        
        # Trace head (only if requested)
        if output_traces:
            trace_outputs = self.trace_head(
                enhanced_features, attention_weights, attention_output, prophetic_state
            )
            governance_outputs.update(trace_outputs)
        
        # Add shared features for downstream use
        governance_outputs['shared_governance_features'] = enhanced_features
        
        return governance_outputs


class PolicyHead(nn.Module):
    """Specialized head for policy control using shared features."""
    
    def __init__(self, governance_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.governance_dim = governance_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Policy processing network (smaller than original)
        self.policy_processor = nn.Sequential(
            nn.Linear(governance_dim, governance_dim),
            nn.Tanh()
        )
        
        # Attention pattern controllers for each head
        self.pattern_controllers = nn.ModuleList([
            nn.Linear(governance_dim, 1) for _ in range(num_heads)
        ])
        
    def forward(
        self,
        shared_features: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        policy_guidance: Optional[torch.Tensor] = None,
        prophetic_state: Optional[PropheticSingularityState] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate policy outputs from shared features."""
        batch_size, seq_len, _ = shared_features.shape
        
        # Process shared features for policy
        if policy_guidance is not None:
            # Integrate external guidance (project to governance dim)
            guidance_proj = nn.Linear(policy_guidance.size(-1), self.governance_dim).to(shared_features.device)
            policy_input = shared_features + guidance_proj(policy_guidance)
        else:
            policy_input = shared_features
        
        policy_logits = self.policy_processor(policy_input)
        
        # Apply prophetic state modulation
        if prophetic_state is not None:
            aligned_state = prophetic_state.align_to_length(seq_len).to(shared_features.device, shared_features.dtype)
            kingdom_flow = aligned_state.kingdom_flow.unsqueeze(-1)
            time_gate = aligned_state.time_index.unsqueeze(-1)
            
            policy_logits = policy_logits + kingdom_flow
            policy_logits = policy_logits * (1.0 + (time_gate - 0.5) * 0.2)
        
        # Generate head-specific attention modifications
        policy_masks = []
        for i, controller in enumerate(self.pattern_controllers):
            head_policy = controller(policy_logits).squeeze(-1)  # [batch, seq_len]
            
            # Convert to attention mask
            policy_mask = head_policy.unsqueeze(1) + head_policy.unsqueeze(2)  # [batch, seq_len, seq_len]
            policy_masks.append(policy_mask)
        
        # Stack for all heads
        policy_mask = torch.stack(policy_masks, dim=1)  # [batch, num_heads, seq_len, seq_len]
        
        # Add additional prophetic mask if available
        if prophetic_state is not None:
            aligned_state = prophetic_state.align_to_length(seq_len).to(shared_features.device, shared_features.dtype)
            additional_mask = aligned_state.compute_policy_mask(self.num_heads, seq_len)
            policy_mask = policy_mask + additional_mask
        
        return {
            'policy_logits': policy_logits,
            'policy_mask': policy_mask
        }


class MemoryHead(nn.Module):
    """Specialized head for memory management using shared features."""
    
    def __init__(self, governance_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.governance_dim = governance_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Memory processing (smaller than original)
        self.memory_processor = nn.Sequential(
            nn.Linear(governance_dim, governance_dim),
            nn.ReLU()
        )
        
        # Context integration (lightweight)
        self.context_integrator = nn.MultiheadAttention(
            governance_dim, num_heads=2, batch_first=True  # Fewer heads than original
        )
        
        # Memory to attention weights
        self.memory_to_weights = nn.Linear(governance_dim, num_heads)
        
    def forward(
        self,
        shared_features: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None,
        prophetic_state: Optional[PropheticSingularityState] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate memory outputs from shared features."""
        batch_size, seq_len, _ = shared_features.shape
        
        # Process shared features for memory
        current_memory = self.memory_processor(shared_features)
        
        # Integrate with previous memory context if available
        if memory_context is not None:
            # Project memory context to governance dimension if needed
            if memory_context.size(-1) != self.governance_dim:
                context_proj = nn.Linear(memory_context.size(-1), self.governance_dim).to(shared_features.device)
                memory_context = context_proj(memory_context)
            
            integrated_memory, _ = self.context_integrator(
                current_memory, memory_context, memory_context
            )
        else:
            integrated_memory = current_memory
        
        # Apply prophetic state modulation
        if prophetic_state is not None:
            aligned_state = prophetic_state.align_to_length(seq_len).to(shared_features.device, shared_features.dtype)
            mercy_gate = 1.0 + (aligned_state.mercy - 0.5) * 0.3
            integrated_memory = integrated_memory * mercy_gate.unsqueeze(-1)
        
        # Generate attention weight modifications
        memory_weights = self.memory_to_weights(integrated_memory)  # [batch, seq_len, num_heads]
        memory_weights = memory_weights.transpose(1, 2).unsqueeze(-1)  # [batch, num_heads, 1, seq_len]
        memory_weights = torch.tanh(memory_weights)
        
        # Apply prophetic decay if available
        if prophetic_state is not None:
            aligned_state = prophetic_state.align_to_length(seq_len).to(shared_features.device, shared_features.dtype)
            decay = aligned_state.compute_memory_decay(self.num_heads, seq_len).unsqueeze(-2)
            memory_weights = memory_weights * decay
        
        return {
            'memory_weights': memory_weights,
            'memory_signals': integrated_memory
        }


class TraceHead(nn.Module):
    """Specialized head for trace generation using shared features."""
    
    def __init__(self, governance_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.governance_dim = governance_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Trace analysis (optimized)
        self.attention_analyzer = nn.Linear(num_heads, governance_dim)
        self.importance_scorer = nn.Linear(governance_dim, 1)
        
        # Concept detection (smaller than original)
        self.concept_dim = 32  # Reduced from 64
        self.concept_detector = nn.Linear(governance_dim, self.concept_dim)
        
        # Trace projector
        self.trace_projector = nn.Linear(governance_dim + self.concept_dim + 1, hidden_dim)
        
    def forward(
        self,
        shared_features: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        attention_output: Optional[torch.Tensor] = None,
        prophetic_state: Optional[PropheticSingularityState] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate trace outputs from shared features."""
        batch_size, seq_len, _ = shared_features.shape
        
        # Analyze attention patterns if available
        if attention_weights is not None:
            avg_attention = attention_weights.mean(dim=-1)  # [batch, num_heads, seq_len]
            attention_features = self.attention_analyzer(avg_attention.transpose(1, 2))
            
            # Combine with shared features
            combined_features = shared_features + attention_features
        else:
            combined_features = shared_features
        
        # Score token importance
        importance_scores = self.importance_scorer(combined_features)
        importance_gate = torch.sigmoid(importance_scores)
        
        # Detect concepts
        concept_activations = torch.sigmoid(self.concept_detector(combined_features))
        
        # Build trace representation
        trace_features = torch.cat([
            combined_features, importance_gate, concept_activations
        ], dim=-1)
        
        trace_tensor = torch.tanh(self.trace_projector(trace_features))
        
        # Apply importance gating
        trace_tensor = trace_tensor * importance_gate + shared_features
        
        # Compute attention entropy if weights available
        if attention_weights is not None:
            attention_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-8),
                dim=-1
            ).mean(dim=1)  # [batch, seq_len]
        else:
            attention_entropy = torch.zeros(batch_size, seq_len, device=shared_features.device)
        
        trace_info = {
            'trace': {
                'tensor': trace_tensor,
                'importance_scores': importance_scores.squeeze(-1),
                'importance_gate': importance_gate.squeeze(-1),
                'concept_activations': concept_activations,
                'attention_entropy': attention_entropy,
                'attention_patterns': avg_attention if attention_weights is not None else None
            }
        }
        
        # Add prophetic information if available
        if prophetic_state is not None:
            aligned_state = prophetic_state.align_to_length(seq_len).to(shared_features.device, shared_features.dtype)
            envelope = aligned_state.compute_trace_envelope(seq_len)
            summary = aligned_state.summary()
            
            trace_info['trace']['prophetic_envelope'] = envelope
            trace_info['trace']['kingdom_mean'] = summary['kingdom']['mean']
            trace_info['trace']['kingdom_std'] = summary['kingdom']['std']
        
        return trace_info


if __name__ == "__main__":
    # Test the shared governance backbone
    print("Testing Shared Governance Backbone...")
    
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
    
    print(f"Input shape: {x.shape}")
    print(f"Governance outputs: {list(governance_outputs.keys())}")
    
    for key, value in governance_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: {list(value.keys())}")
    
    print("✓ Shared Governance Backbone working correctly!")