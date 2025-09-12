"""Biblical BPE tokenizer - compatibility with existing implementation."""

import sys
from pathlib import Path

# Add SIM-ONE Training path
SIM_ONE_PATH = Path(__file__).parent.parent.parent.parent / "SIM-ONE Training"
sys.path.insert(0, str(SIM_ONE_PATH))

# Import existing tokenizer for compatibility
try:
    from prioritary_mvlm.advanced_tokenizer import BiblicalBPETokenizer as _BiblicalBPETokenizer
    from prioritary_mvlm.advanced_tokenizer import train_biblical_tokenizer
    TOKENIZER_AVAILABLE = True
except ImportError as e:
    TOKENIZER_AVAILABLE = False
    _import_error = e


class BiblicalBPETokenizer:
    """Biblical BPE tokenizer - delegates to existing implementation."""
    
    def __new__(cls, *args, **kwargs):
        if TOKENIZER_AVAILABLE:
            return _BiblicalBPETokenizer(*args, **kwargs)
        else:
            raise ImportError(f"Could not import BiblicalBPETokenizer: {_import_error}")


# Export training function
def train_biblical_bpe_tokenizer(*args, **kwargs):
    """Train biblical BPE tokenizer - delegates to existing function."""
    if TOKENIZER_AVAILABLE:
        return train_biblical_tokenizer(*args, **kwargs)
    else:
        raise ImportError(f"Could not import train_biblical_tokenizer: {_import_error}")