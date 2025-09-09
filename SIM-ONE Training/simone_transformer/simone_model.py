"""
Enhanced SIM-ONE Transformer Model
Modern architecture with RoPE, SwiGLU, RMSNorm, and advanced governance
"""

# Import the enhanced model as the main implementation
from .enhanced_model import EnhancedSIMONEModel, EnhancedSIMONEBlock, GovernanceAggregator

# Main exports - only Enhanced version
SIMONEModel = EnhancedSIMONEModel
SIMONEBlock = EnhancedSIMONEBlock

__all__ = ["SIMONEModel", "SIMONEBlock", "EnhancedSIMONEModel", "EnhancedSIMONEBlock", "GovernanceAggregator"]
