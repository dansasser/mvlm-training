"""
SIM-ONE Training Package - Reorganized Structure

This package provides a clean, organized structure for both MVLM-GPT2 and Enhanced SIM-ONE training.
Maintains backward compatibility with existing code while improving maintainability.
"""

# Core models
from .models.base import MVLMModel, MVLMTrainer
from .models.enhanced import EnhancedSIMONEModel, EnhancedSIMONETrainer

# Configuration
from .config import BaseConfig, MVLMConfig, EnhancedConfig

# Data components  
from .data import WeightedTextDataset
from .data.tokenizers import BaseTokenizer, BiblicalBPETokenizer

# Training utilities
from .training import TrainingManager
from .utils import setup_logging, validate_environment

__version__ = "1.0.0"

__all__ = [
    # Models
    "MVLMModel", "EnhancedSIMONEModel",
    
    # Trainers
    "MVLMTrainer", "EnhancedSIMONETrainer", 
    "TrainingManager",
    
    # Configuration
    "BaseConfig", "MVLMConfig", "EnhancedConfig",
    
    # Data
    "WeightedTextDataset", "BaseTokenizer", "BiblicalBPETokenizer",
    
    # Utils
    "setup_logging", "validate_environment"
]