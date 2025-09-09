"""
Modern Neural Network Layers for SIM-ONE Transformer
Implements SwiGLU, RMSNorm, and other state-of-the-art components
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More stable and efficient than LayerNorm, used in modern LLMs.
    """
    
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU activation function: SiLU(x * W1) * (x * W2)
    Shown to be more effective than ReLU in language models.
    """
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, bias: bool = False):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8/3)  # Common scaling factor
        hidden_dim = int(2 * hidden_dim / 3)  # Adjust for GLU
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(x @ W1) * (x @ W2) @ W3
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GeGLU(nn.Module):
    """
    GeGLU activation: GELU(x * W1) * (x * W2)
    Alternative to SwiGLU with GELU activation.
    """
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, bias: bool = False):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8/3)
        hidden_dim = int(2 * hidden_dim / 3)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.gelu(self.w1(x)) * self.w2(x))


class MoELayer(nn.Module):
    """
    Mixture of Experts layer for increased model capacity without proportional compute increase.
    Routes tokens to different expert networks based on learned routing.
    """
    
    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        expert_hidden_dim: Optional[int] = None
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        expert_hidden_dim = expert_hidden_dim or dim * 4
        
        # Router network
        self.router = nn.Linear(dim, num_experts, bias=False)
        
        # Expert networks
        self.experts = nn.ModuleList([
            SwiGLU(dim, expert_hidden_dim) for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # [batch_size * seq_len, dim]
        
        # Router logits
        router_logits = self.router(x_flat)  # [batch_size * seq_len, num_experts]
        
        # Get top-k experts for each token
        top_k_logits, top_k_indices = torch.topk(
            router_logits, self.num_experts_per_token, dim=-1
        )
        
        # Softmax over selected experts
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # Route to experts
        output = torch.zeros_like(x_flat)
        
        for i in range(self.num_experts_per_token):
            expert_indices = top_k_indices[:, i]
            weights = top_k_weights[:, i].unsqueeze(-1)
            
            # Process tokens for each expert
            for expert_idx in range(self.num_experts):
                mask = expert_indices == expert_idx
                if mask.any():
                    tokens_for_expert = x_flat[mask]
                    expert_output = self.experts[expert_idx](tokens_for_expert)
                    output[mask] += weights[mask] * expert_output
        
        return output.view(batch_size, seq_len, dim)


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization that can modulate normalization based on conditioning.
    Useful for incorporating external signals into normalization.
    """
    
    def __init__(self, dim: int, condition_dim: int):
        super().__init__()
        self.dim = dim
        self.condition_dim = condition_dim
        
        # Base normalization parameters
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        
        # Conditioning networks
        self.gamma_net = nn.Linear(condition_dim, dim)
        self.beta_net = nn.Linear(condition_dim, dim)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # Standard layer norm
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + 1e-8)
        
        # Adaptive modulation
        gamma = 1 + self.gamma_net(condition)
        beta = self.beta_net(condition)
        
        return self.weight * normalized * gamma + self.bias + beta


class PositionalEmbedding(nn.Module):
    """
    Advanced positional embedding with multiple encoding schemes.
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        embedding_type: str = "learned"  # "learned", "sinusoidal", "rope"
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.embedding_type = embedding_type
        
        if embedding_type == "learned":
            self.pos_emb = nn.Embedding(max_seq_len, dim)
        elif embedding_type == "sinusoidal":
            self.register_buffer("pos_emb", self._create_sinusoidal_embeddings())
        # RoPE is handled separately in the attention mechanism
        
    def _create_sinusoidal_embeddings(self) -> torch.Tensor:
        """Create sinusoidal positional embeddings."""
        pe = torch.zeros(self.max_seq_len, self.dim)
        position = torch.arange(0, self.max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, self.dim, 2).float() * 
            -(math.log(10000.0) / self.dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        
        if self.embedding_type == "learned":
            positions = torch.arange(seq_len, device=x.device)
            return self.pos_emb(positions)
        elif self.embedding_type == "sinusoidal":
            return self.pos_emb[:seq_len].to(x.device)
        else:
            # For RoPE, return zeros (handled in attention)
            return torch.zeros_like(x)


class GatedResidualConnection(nn.Module):
    """
    Gated residual connection that can modulate the residual flow.
    Helps with training stability and allows dynamic residual weighting.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),  # Input: concat of residual and main
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # Compute gate weights
        concat_input = torch.cat([x, residual], dim=-1)
        gate_weights = self.gate(concat_input)
        
        # Gated residual connection
        return x + gate_weights * residual


class ScaledDotProductAttention(nn.Module):
    """
    Optimized scaled dot-product attention with optional improvements.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_flash_attention: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attention = use_flash_attention
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized flash attention if available
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0
            )
            # Flash attention doesn't return weights, so we compute them separately if needed
            with torch.no_grad():
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, float('-inf'))
                attn_weights = F.softmax(scores, dim=-1)
            
            return attn_output, attn_weights
        else:
            # Standard attention computation
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, v)
            
            return attn_output, attn_weights


class BiblicalAttentionBias(nn.Module):
    """
    Attention bias that encourages attention to biblical concepts and themes.
    Helps the model focus on theologically important elements.
    """
    
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Biblical concept embeddings
        self.biblical_concepts = nn.Embedding(vocab_size, hidden_dim)
        
        # Concept importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize with higher weights for known biblical terms
        self._init_biblical_concepts()
        
    def _init_biblical_concepts(self):
        """Initialize embeddings with special treatment for biblical concepts."""
        # This would be initialized based on biblical vocabulary analysis
        # For now, use standard initialization
        nn.init.xavier_uniform_(self.biblical_concepts.weight)
        
    def forward(self, input_ids: torch.Tensor, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_scores: Current attention scores [batch, num_heads, seq_len, seq_len]
            
        Returns:
            bias: Attention bias to add to scores
        """
        batch_size, seq_len = input_ids.shape
        
        # Get biblical concept embeddings for input tokens
        concept_embeds = self.biblical_concepts(input_ids)  # [batch, seq_len, hidden_dim]
        
        # Score importance of each token
        importance = self.importance_scorer(concept_embeds).squeeze(-1)  # [batch, seq_len]
        
        # Create attention bias - encourage attention TO important tokens
        # Broadcasting: [batch, 1, 1, seq_len] to match attention_scores shape
        attention_bias = importance.unsqueeze(1).unsqueeze(1)
        
        # Scale bias
        attention_bias = attention_bias * 0.1  # Small bias to not overwhelm learned attention
        
        return attention_bias


# Utility functions

def get_activation_fn(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'swish': nn.SiLU(),
        'silu': nn.SiLU(),
        'mish': nn.Mish(),
    }
    
    if name.lower() in activations:
        return activations[name.lower()]
    else:
        raise ValueError(f"Unknown activation: {name}")


def apply_weight_init(module: nn.Module, init_type: str = "xavier"):
    """Apply weight initialization to a module."""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        if init_type == "xavier":
            nn.init.xavier_uniform_(module.weight)
        elif init_type == "kaiming":
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif init_type == "normal":
            nn.init.normal_(module.weight, std=0.02)
            
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
    elif isinstance(module, (nn.LayerNorm, RMSNorm)):
        if hasattr(module, 'weight'):
            nn.init.ones_(module.weight)
        if hasattr(module, 'bias'):
            nn.init.zeros_(module.bias)


if __name__ == "__main__":
    # Test the modern layers
    batch_size, seq_len, dim = 2, 64, 512
    
    print("Testing modern neural network layers...")
    
    # Test RMSNorm
    x = torch.randn(batch_size, seq_len, dim)
    rms_norm = RMSNorm(dim)
    normalized = rms_norm(x)
    print(f"✓ RMSNorm: {x.shape} -> {normalized.shape}")
    
    # Test SwiGLU
    swiglu = SwiGLU(dim)
    swiglu_output = swiglu(x)
    print(f"✓ SwiGLU: {x.shape} -> {swiglu_output.shape}")
    
    # Test GeGLU
    geglu = GeGLU(dim)
    geglu_output = geglu(x)
    print(f"✓ GeGLU: {x.shape} -> {geglu_output.shape}")
    
    # Test MoE Layer
    moe = MoELayer(dim, num_experts=4, num_experts_per_token=2)
    moe_output = moe(x)
    print(f"✓ MoE Layer: {x.shape} -> {moe_output.shape}")
    
    # Test Adaptive LayerNorm
    condition = torch.randn(batch_size, seq_len, 128)
    adaptive_norm = AdaptiveLayerNorm(dim, 128)
    adaptive_output = adaptive_norm(x, condition)
    print(f"✓ Adaptive LayerNorm: {x.shape} -> {adaptive_output.shape}")
    
    print("All modern layers working correctly! ✓")