"""
Compatibility layer for existing code.
Provides imports and wrappers to maintain backward compatibility.
"""

import sys
import warnings
from pathlib import Path

# Add paths for existing modules
SIM_ONE_PATH = Path(__file__).parent.parent / "SIM-ONE Training"
sys.path.insert(0, str(SIM_ONE_PATH))

def deprecated_import(old_name: str, new_name: str):
    """Issue deprecation warning for old imports."""
    warnings.warn(
        f"Importing from '{old_name}' is deprecated. Use '{new_name}' instead.",
        DeprecationWarning,
        stacklevel=3
    )

# Existing imports with compatibility
try:
    from prioritary_mvlm import (
        PrioritaryConfig as _PrioritaryConfig,
        EnhancedPrioritaryTrainer as _EnhancedPrioritaryTrainer,
        WeightedTextDataset as _WeightedTextDataset,
        BiblicalBPETokenizer as _BiblicalBPETokenizer,
    )
    LEGACY_IMPORTS_AVAILABLE = True
except ImportError:
    LEGACY_IMPORTS_AVAILABLE = False
    _PrioritaryConfig = None
    _EnhancedPrioritaryTrainer = None
    _WeightedTextDataset = None
    _BiblicalBPETokenizer = None

try:
    from simone_transformer import (
        SIMONEModel as _SIMONEModel,
        EnhancedSIMONEModel as _EnhancedSIMONEModel
    )
    TRANSFORMER_IMPORTS_AVAILABLE = True
except ImportError:
    TRANSFORMER_IMPORTS_AVAILABLE = False
    _SIMONEModel = None
    _EnhancedSIMONEModel = None


# Compatibility wrappers
class PrioritaryConfig:
    """Compatibility wrapper for PrioritaryConfig."""
    def __new__(cls, *args, **kwargs):
        deprecated_import('simone_training.compat.PrioritaryConfig', 'simone_training.config.EnhancedConfig')
        if _PrioritaryConfig:
            return _PrioritaryConfig(*args, **kwargs)
        else:
            from .config import EnhancedConfig
            return EnhancedConfig(*args, **kwargs).to_prioritary_config()


class EnhancedPrioritaryTrainer:
    """Compatibility wrapper for EnhancedPrioritaryTrainer."""
    def __new__(cls, *args, **kwargs):
        deprecated_import('simone_training.compat.EnhancedPrioritaryTrainer', 'simone_training.models.enhanced.EnhancedSIMONETrainer')
        if _EnhancedPrioritaryTrainer:
            return _EnhancedPrioritaryTrainer(*args, **kwargs)
        else:
            raise ImportError("Legacy EnhancedPrioritaryTrainer not available")


class WeightedTextDataset:
    """Compatibility wrapper for WeightedTextDataset."""
    def __new__(cls, *args, **kwargs):
        deprecated_import('simone_training.compat.WeightedTextDataset', 'simone_training.data.WeightedTextDataset')
        if _WeightedTextDataset:
            return _WeightedTextDataset(*args, **kwargs)
        else:
            raise ImportError("Legacy WeightedTextDataset not available")


class BiblicalBPETokenizer:
    """Compatibility wrapper for BiblicalBPETokenizer."""
    def __new__(cls, *args, **kwargs):
        deprecated_import('simone_training.compat.BiblicalBPETokenizer', 'simone_training.data.tokenizers.BiblicalBPETokenizer')
        if _BiblicalBPETokenizer:
            return _BiblicalBPETokenizer(*args, **kwargs)
        else:
            raise ImportError("Legacy BiblicalBPETokenizer not available")


# Model compatibility
class SIMONEModel:
    """Compatibility wrapper for SIMONEModel."""
    def __new__(cls, *args, **kwargs):
        deprecated_import('simone_training.compat.SIMONEModel', 'simone_training.models.enhanced.SIMONEModel')
        if _SIMONEModel:
            return _SIMONEModel(*args, **kwargs)
        else:
            raise ImportError("Legacy SIMONEModel not available")


class EnhancedSIMONEModel:
    """Compatibility wrapper for EnhancedSIMONEModel."""
    def __new__(cls, *args, **kwargs):
        deprecated_import('simone_training.compat.EnhancedSIMONEModel', 'simone_training.models.enhanced.EnhancedSIMONEModel')
        if _EnhancedSIMONEModel:
            return _EnhancedSIMONEModel(*args, **kwargs)
        else:
            raise ImportError("Legacy EnhancedSIMONEModel not available")


# Utility functions
def check_legacy_compatibility():
    """Check if legacy imports are working."""
    issues = []
    
    if not LEGACY_IMPORTS_AVAILABLE:
        issues.append("Legacy prioritary_mvlm imports not available")
    
    if not TRANSFORMER_IMPORTS_AVAILABLE:
        issues.append("Legacy simone_transformer imports not available")
    
    return issues


def migrate_legacy_code_hints():
    """Provide hints for migrating legacy code."""
    return [
        "Replace 'from prioritary_mvlm import PrioritaryConfig' with 'from simone_training.config import EnhancedConfig'",
        "Replace 'from prioritary_mvlm.enhanced_trainer import EnhancedPrioritaryTrainer' with 'from simone_training.models.enhanced import EnhancedSIMONETrainer'", 
        "Replace 'from simone_transformer import EnhancedSIMONEModel' with 'from simone_training.models.enhanced import EnhancedSIMONEModel'",
        "Use 'from simone_training import *' for new code"
    ]