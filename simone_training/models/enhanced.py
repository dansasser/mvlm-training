"""Enhanced SIM-ONE model and trainer - compatibility with existing enhanced components."""

import sys
from pathlib import Path

# Add SIM-ONE Training path
SIM_ONE_PATH = Path(__file__).parent.parent.parent / "SIM-ONE Training"
sys.path.insert(0, str(SIM_ONE_PATH))

# Import existing enhanced components for compatibility
try:
    from simone_transformer import EnhancedSIMONEModel as _EnhancedSIMONEModel
    from simone_transformer import SIMONEModel as _SIMONEModel
    from prioritary_mvlm.enhanced_trainer import EnhancedPrioritaryTrainer as _EnhancedPrioritaryTrainer
    ENHANCED_IMPORTS_AVAILABLE = True
except ImportError as e:
    ENHANCED_IMPORTS_AVAILABLE = False
    _import_error = e


class EnhancedSIMONEModel:
    """Enhanced SIM-ONE model - delegates to existing implementation."""
    
    def __new__(cls, *args, **kwargs):
        if ENHANCED_IMPORTS_AVAILABLE:
            return _EnhancedSIMONEModel(*args, **kwargs)
        else:
            raise ImportError(f"Could not import enhanced model: {_import_error}")


class SIMONEModel:
    """Base SIM-ONE model - delegates to existing implementation."""
    
    def __new__(cls, *args, **kwargs):
        if ENHANCED_IMPORTS_AVAILABLE:
            return _SIMONEModel(*args, **kwargs)
        else:
            raise ImportError(f"Could not import SIM-ONE model: {_import_error}")


class EnhancedSIMONETrainer:
    """Enhanced SIM-ONE trainer - delegates to existing implementation."""
    
    def __new__(cls, *args, **kwargs):
        if ENHANCED_IMPORTS_AVAILABLE:
            return _EnhancedPrioritaryTrainer(*args, **kwargs)
        else:
            raise ImportError(f"Could not import enhanced trainer: {_import_error}")


# Convenience imports for existing code
EnhancedPrioritaryTrainer = EnhancedSIMONETrainer  # Alias for backward compatibility