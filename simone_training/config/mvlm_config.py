"""MVLM-GPT2 configuration - bridges to existing MVLMConfig."""

from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class MVLMConfig(BaseConfig):
    """MVLM-GPT2 configuration matching existing trainer."""
    
    # GPT-2 specific architecture
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    n_positions: int = 1024  # Context length
    n_embd: int = 768       # Embedding dimension
    n_layer: int = 12       # Number of transformer layers
    n_head: int = 12        # Number of attention heads
    n_inner: int = 3072     # Feed-forward dimension
    
    # Training parameters (matching existing MVLM defaults)
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # GPT-2 specific
    use_gpt2_tokenizer: bool = True
    model_type: str = "gpt2"
    
    def to_mvlm_dict(self) -> dict:
        """Convert to format expected by existing mvlm_trainer.py."""
        return {
            "vocab_size": self.vocab_size,
            "n_positions": self.n_positions, 
            "n_embd": self.n_embd,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_inner": self.n_inner,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "num_epochs": self.num_epochs,
            "max_length": self.max_length,
            "mixed_precision": self.mixed_precision,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm
        }