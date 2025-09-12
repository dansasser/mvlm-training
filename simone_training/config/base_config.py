"""Base configuration class for all SIM-ONE models."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
from pathlib import Path


@dataclass
class BaseConfig:
    """Base configuration for SIM-ONE training."""
    
    # Data paths
    data_dir: str = "mvlm_training_dataset_complete"
    output_dir: str = "models/output"
    
    # Model architecture basics
    vocab_size: int = 32000
    max_length: int = 1024
    
    # Training basics
    batch_size: int = 8
    learning_rate: float = 3e-4
    num_epochs: int = 3
    
    # Hardware optimization
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Logging and checkpoints
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 1000
    
    # Environment
    device: str = "auto"  # auto, cuda, cpu
    seed: int = 42
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Convert string paths to Path objects
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device if needed
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    def save(self, path: str):
        """Save configuration to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def validate(self) -> bool:
        """Validate configuration."""
        # Check data directory exists
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # Check positive values
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
            
        return True