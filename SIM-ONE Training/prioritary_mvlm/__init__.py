# Import models from sibling directory (conditional to avoid circular imports)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Conditional imports to avoid circular dependencies
try:
    from simone_transformer import SIMONEModel, EnhancedSIMONEModel
    TRANSFORMER_IMPORTS_AVAILABLE = True
except ImportError:
    TRANSFORMER_IMPORTS_AVAILABLE = False
    SIMONEModel = None
    EnhancedSIMONEModel = None
from .advanced_tokenizer import BiblicalBPETokenizer, train_biblical_tokenizer
from .enhanced_trainer import EnhancedPrioritaryTrainer
from .config import PrioritaryConfig, PropheticSingularityState
from .dataset import WeightedTextDataset
from .advanced_losses import (
    BiblicalAlignmentLoss,
    TheologicalCoherenceLoss,
    ScriptureReferenceLoss,
    StyleConsistencyLoss,
    ComprehensiveBiblicalLoss,
    create_biblical_metadata
)

# Build __all__ list conditionally
__all__ = [
    # Tokenizer (BPE only)
    "BiblicalBPETokenizer",
    "train_biblical_tokenizer",

    # Training components (Enhanced only)
    "EnhancedPrioritaryTrainer",
    "PrioritaryConfig",
    "PropheticSingularityState",
    "WeightedTextDataset",

    # Advanced losses
    "BiblicalAlignmentLoss",
    "TheologicalCoherenceLoss",
    "ScriptureReferenceLoss",
    "StyleConsistencyLoss",
    "ComprehensiveBiblicalLoss",
    "create_biblical_metadata"
]

# Add models if available
if TRANSFORMER_IMPORTS_AVAILABLE:
    __all__.extend(["SIMONEModel", "EnhancedSIMONEModel"])
