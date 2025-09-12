"""Dataset implementation - compatibility with existing WeightedTextDataset."""

import sys
from pathlib import Path

# Add SIM-ONE Training path
SIM_ONE_PATH = Path(__file__).parent.parent.parent / "SIM-ONE Training"
sys.path.insert(0, str(SIM_ONE_PATH))

# Import existing dataset for compatibility
try:
    from prioritary_mvlm.dataset import WeightedTextDataset as _WeightedTextDataset
    DATASET_AVAILABLE = True
except ImportError as e:
    DATASET_AVAILABLE = False
    _import_error = e


class WeightedTextDataset:
    """Weighted text dataset - delegates to existing implementation."""
    
    def __new__(cls, *args, **kwargs):
        if DATASET_AVAILABLE:
            return _WeightedTextDataset(*args, **kwargs)
        else:
            raise ImportError(f"Could not import WeightedTextDataset: {_import_error}")