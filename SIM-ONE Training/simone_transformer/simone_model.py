"""
SIM-ONE Transformer Model
Delegates to the working implementation in the base models
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import sys
from pathlib import Path

# Add parent directory to path for base model imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from simone_training.models.base import EnhancedSIMONEWrapper, MVLMAdapter
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

from .shared_governance import GovernanceAggregator


class SIMONEModel(nn.Module):
    """
    SIM-ONE Transformer Model - delegates to Enhanced SIM-ONE implementation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if ENHANCED_AVAILABLE:
            self.enhanced_wrapper = EnhancedSIMONEWrapper(*args, **kwargs)
        else:
            raise ImportError("Enhanced SIM-ONE implementation not available")

    def forward(self, *args, **kwargs):
        return self.enhanced_wrapper(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.enhanced_wrapper.generate(*args, **kwargs)

    def get_num_params(self):
        return self.enhanced_wrapper.get_num_params()

    def get_memory_usage(self):
        return self.enhanced_wrapper.get_memory_usage()


class SIMONEBlock(nn.Module):
    """
    SIM-ONE Transformer Block - basic implementation for compatibility.
    """

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        # Basic transformer block components
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # Feedforward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


# Aliases for backward compatibility
EnhancedSIMONEModel = SIMONEModel
EnhancedSIMONEBlock = SIMONEBlock

__all__ = ["SIMONEModel", "SIMONEBlock", "EnhancedSIMONEModel", "EnhancedSIMONEBlock", "GovernanceAggregator"]
