# Import models from sibling directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from simone_transformer import SIMONEModel, EnhancedSIMONEModel
from .advanced_tokenizer import BiblicalBPETokenizer, train_biblical_tokenizer
from .enhanced_trainer import EnhancedPrioritaryTrainer
from .config import PrioritaryConfig
from .dataset import WeightedTextDataset
from .advanced_losses import (
    BiblicalAlignmentLoss,
    TheologicalCoherenceLoss,
    ScriptureReferenceLoss,
    StyleConsistencyLoss,
    ComprehensiveBiblicalLoss,
    create_biblical_metadata
)

__all__ = [
    # Models (Enhanced only)
    "SIMONEModel",
    "EnhancedSIMONEModel",
    
    # Tokenizer (BPE only)
    "BiblicalBPETokenizer",
    "train_biblical_tokenizer",
    
    # Training components (Enhanced only)
    "EnhancedPrioritaryTrainer",
    "PrioritaryConfig", 
    "WeightedTextDataset",
    
    # Advanced losses
    "BiblicalAlignmentLoss",
    "TheologicalCoherenceLoss",
    "ScriptureReferenceLoss",
    "StyleConsistencyLoss", 
    "ComprehensiveBiblicalLoss",
    "create_biblical_metadata"
]
