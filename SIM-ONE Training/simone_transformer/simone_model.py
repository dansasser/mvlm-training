"""
SIM-ONE Transformer Model
Delegates to the working implementation in the base models
"""

import importlib
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Add parent directory to path for base model imports (needed for runtime delegation)
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

ENHANCED_AVAILABLE = False
_ENHANCED_IMPORT_ERROR: Optional[ImportError] = None
_ENHANCED_COMPONENTS: Optional[Tuple[object, object]] = None


def _load_enhanced_components():
    """Lazy import to avoid circular dependency during module import."""

    global ENHANCED_AVAILABLE, _ENHANCED_IMPORT_ERROR, _ENHANCED_COMPONENTS

    if _ENHANCED_COMPONENTS is not None:
        return _ENHANCED_COMPONENTS

    try:
        module = importlib.import_module("simone_training.models.base")
        components = (module.EnhancedSIMONEWrapper, module.MVLMAdapter)
        _ENHANCED_COMPONENTS = components
        ENHANCED_AVAILABLE = True
        return components
    except ImportError as exc:
        _ENHANCED_IMPORT_ERROR = exc
        ENHANCED_AVAILABLE = False
        raise


if "simone_training.models.base" in sys.modules:
    existing_module = sys.modules["simone_training.models.base"]
    if hasattr(existing_module, "EnhancedSIMONEWrapper") and hasattr(existing_module, "MVLMAdapter"):
        _ENHANCED_COMPONENTS = (
            existing_module.EnhancedSIMONEWrapper,
            existing_module.MVLMAdapter,
        )
        ENHANCED_AVAILABLE = True

from .shared_governance import GovernanceAggregator


class SIMONEModel(nn.Module):
    """
    SIM-ONE Transformer Model - delegates to Enhanced SIM-ONE implementation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        try:
            enhanced_wrapper_cls, _ = _load_enhanced_components()
        except ImportError as exc:
            origin = _ENHANCED_IMPORT_ERROR or exc
            raise ImportError("Enhanced SIM-ONE implementation not available") from origin

        self.enhanced_wrapper = enhanced_wrapper_cls(*args, **kwargs)

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
