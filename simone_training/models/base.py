"""Base MVLM-GPT2 model and trainer - compatibility with existing mvlm_trainer.py."""

import sys
from pathlib import Path

# Add root path to import existing mvlm_trainer
ROOT_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_PATH))

# For now, create stubs that will delegate to existing implementation
class MVLMModel:
    """Stub for MVLM model - delegates to existing GPT2 implementation."""
    
    def __init__(self, *args, **kwargs):
        # TODO: Import and wrap existing GPT2LMHeadModel from mvlm_trainer.py
        raise NotImplementedError("MVLMModel migration not yet complete")


class MVLMTrainer:
    """Stub for MVLM trainer - delegates to existing mvlm_trainer.py."""
    
    def __init__(self, *args, **kwargs):
        # TODO: Import and wrap existing MVLMTrainer from mvlm_trainer.py  
        raise NotImplementedError("MVLMTrainer migration not yet complete")